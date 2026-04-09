#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import numpy as np
import cv2

import rosbag
from cv_bridge import CvBridge


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def make_object_points(board_w, board_h, square_size):
    """
    board_w, board_h: 棋盘格内角点数量 (cols, rows)
    square_size: 每格边长（米/毫米都行，但要一致）
    返回: (N,3) 物理坐标，z=0
    """
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size
    return objp


def draw_and_save(vis_dir, idx, img, corners, board_size, ok):
    vis = img.copy()
    if ok and corners is not None:
        cv2.drawChessboardCorners(vis, board_size, corners, ok)
    out_path = os.path.join(vis_dir, f"frame_{idx:06d}.png")
    cv2.imwrite(out_path, vis)


def calibrate_pinhole(objpoints, imgpoints, image_size):
    """
    OpenCV pinhole 标定
    """
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=flags, criteria=criteria
    )
    return ret, K, dist, rvecs, tvecs


def calibrate_fisheye(objpoints, imgpoints, image_size):
    """
    OpenCV fisheye 标定（注意 fisheye 的数据结构要求更严格）
    """
    # fisheye 需要每张图的点是 (N,1,2) float64
    objpoints_ = []
    imgpoints_ = []
    for op, ip in zip(objpoints, imgpoints):
        objpoints_.append(op.reshape(-1, 1, 3).astype(np.float64))
        imgpoints_.append(ip.reshape(-1, 1, 2).astype(np.float64))

    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints_, imgpoints_, image_size, K, D,
        None, None, flags=flags, criteria=criteria
    )
    return rms, K, D, rvecs, tvecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to rosbag file")
    ap.add_argument("--topic", default="/usb_cam_left/image_raw", help="image topic name")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--cols", type=int, required=True, help="chessboard inner corners cols (width)")
    ap.add_argument("--rows", type=int, required=True, help="chessboard inner corners rows (height)")
    ap.add_argument("--square", type=float, required=True, help="square size (e.g. 0.03 for 30mm)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 means no limit; otherwise cap number of processed frames")
    ap.add_argument("--stride", type=int, default=1, help="process every N frames")
    ap.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole", help="camera model for calibration")
    ap.add_argument("--subpix", action="store_true", help="use cornerSubPix refinement")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    vis_dir = os.path.join(args.out_dir, "vis")
    ensure_dir(vis_dir)

    board_size = (args.cols, args.rows)
    objp_template = make_object_points(args.cols, args.rows, args.square)

    bridge = CvBridge()
    objpoints = []
    imgpoints = []

    processed = 0
    used = 0
    image_size = None

    # 棋盘格检测参数：对真实场景更稳一些
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE)

    # 如果你是很规整的打印板，有时加 FAST_CHECK 会更快，但可能漏检：
    # cb_flags |= cv2.CALIB_CB_FAST_CHECK

    print(f"[INFO] Reading bag: {args.bag}")
    print(f"[INFO] Topic: {args.topic}")
    print(f"[INFO] Board: cols={args.cols}, rows={args.rows}, square={args.square}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {args.out_dir}")

    with rosbag.Bag(args.bag, "r") as bag:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[args.topic])):
            if args.stride > 1 and (processed % args.stride != 0):
                processed += 1
                continue
            processed += 1

            if args.max_frames > 0 and processed > args.max_frames:
                break

            # 转 OpenCV 图像（尽量保持原始）
            try:
                # 常见：bgr8 / mono8
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except Exception as e:
                print(f"[WARN] cv_bridge convert failed at frame {processed}: {e}")
                continue

            # 统一到灰度用于检测
            if cv_img.ndim == 3:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                vis_src = cv_img
            else:
                gray = cv_img
                vis_src = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])  # (w,h)
                print(f"[INFO] Image size: {image_size}")

            ok, corners = cv2.findChessboardCorners(gray, board_size, cb_flags)

            if ok and corners is not None:
                if args.subpix:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                    corners = cv2.cornerSubPix(
                        gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
                    )

                objpoints.append(objp_template.copy())
                imgpoints.append(corners.copy())
                used += 1

            # 保存可视化结果（每帧都存，方便你回看漏检/误检）
            draw_and_save(vis_dir, processed, vis_src, corners, board_size, ok)

            if processed % 50 == 0:
                print(f"[INFO] processed={processed}, used={used}")

    print(f"[INFO] Done reading. processed={processed}, used(valid)={used}")

    if used < 8:
        print("[ERROR] 有效标定帧太少（<8）。建议：多拍一些不同姿态/距离/角度的图，或检查 cols/rows 是否填对。")
        sys.exit(1)

    # 标定
    if args.model == "pinhole":
        rms, K, dist, rvecs, tvecs = calibrate_pinhole(objpoints, imgpoints, image_size)
        calib = {
            "model": "pinhole",
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "K": K.tolist(),
            "dist": dist.reshape(-1).tolist(),
            "rms_reproj_error": float(rms),
            "used_frames": int(used),
            "board": {"cols": args.cols, "rows": args.rows, "square": float(args.square)},
        }
    else:
        rms, K, D, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, image_size)
        calib = {
            "model": "fisheye",
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "K": K.tolist(),
            "dist": D.reshape(-1).tolist(),  # fisheye: k1..k4
            "rms_reproj_error": float(rms),
            "used_frames": int(used),
            "board": {"cols": args.cols, "rows": args.rows, "square": float(args.square)},
        }

    out_yaml = os.path.join(args.out_dir, "calib.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(calib, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Saved calibration to: {out_yaml}")
    print(f"[INFO] Saved detections visualization to: {vis_dir}")


if __name__ == "__main__":
    main()
