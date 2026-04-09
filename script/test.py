#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
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


def calibrate_fisheye(objpoints, imgpoints, image_size, max_drop=500):
    """
    OpenCV fisheye 标定（鲁棒版）：
    若出现
      CALIB_CHECK_COND - Ill-conditioned matrix for input array XXX
    则自动剔除该帧（pop XXX）并重试，直到成功或剔除过多/帧不足。

    注意：成功后会把剔除后的集合同步回传入的 objpoints/imgpoints（原地修改）。
    """
    # 用副本循环剔除，成功后再同步回去
    obj_list = list(objpoints)
    img_list = list(imgpoints)

    dropped = 0

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    while True:
        if len(obj_list) < 8:
            raise RuntimeError(f"Too few valid frames left for fisheye calibration: {len(obj_list)}")

        # fisheye 需要每张图的点是 (N,1,2) float64；物点 (N,1,3) float64
        objpoints_ = []
        imgpoints_ = []
        for op, ip in zip(obj_list, img_list):
            objpoints_.append(op.reshape(-1, 1, 3).astype(np.float64))
            imgpoints_.append(ip.reshape(-1, 1, 2).astype(np.float64))

        # 每次重试都重新初始化（更稳）
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints_, imgpoints_, image_size, K, D,
                None, None, flags=flags, criteria=criteria
            )

            # ✅ 成功：把剔除后的集合写回原始输入（原地修改）
            objpoints[:] = obj_list
            imgpoints[:] = img_list

            return rms, K, D, rvecs, tvecs

        except cv2.error as e:
            msg = str(e)

            # 解析 bad_idx：兼容两种常见报错文本
            m = re.search(r"Ill-conditioned matrix for input array\s+(\d+)", msg)
            if m is None:
                m = re.search(r"input array\s+(\d+)", msg)

            if m is None:
                # 不是我们预期的“坏帧导致的病态”错误，直接抛出
                raise

            bad_idx = int(m.group(1))
            if not (0 <= bad_idx < len(obj_list)):
                raise

            # ✅ 剔除该坏帧并重试
            obj_list.pop(bad_idx)
            img_list.pop(bad_idx)
            dropped += 1

            print(f"[WARN] fisheye ill-conditioned -> pop index {bad_idx}, "
                  f"dropped={dropped}, remain={len(obj_list)}")

            if dropped >= max_drop:
                raise RuntimeError(
                    f"Too many bad frames popped: {dropped} (max_drop={max_drop}). "
                    f"Consider using fewer/ more diverse calibration frames."
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to rosbag file")
    ap.add_argument("--topic", default="/usb_cam_left/image_raw", help="image topic name")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--cols", type=int, required=True, help="chessboard inner corners cols (width)")
    ap.add_argument("--rows", type=int, required=True, help="chessboard inner corners rows (height)")
    ap.add_argument("--square", type=float, required=True, help="square size (e.g. 0.03 for 30mm)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 means no limit; otherwise cap number of processed frames")
    ap.add_argument("--stride", type=int, default=1, help="process every N frames (>=1)")
    ap.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole", help="camera model for calibration")
    ap.add_argument("--subpix", action="store_true", help="use cornerSubPix refinement")
    ap.add_argument("--max_drop", type=int, default=500, help="max bad frames to pop in fisheye calibration")
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

    print(f"[INFO] Reading bag: {args.bag}")
    print(f"[INFO] Topic: {args.topic}")
    print(f"[INFO] Board: cols={args.cols}, rows={args.rows}, square={args.square}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {args.out_dir}")
    print(f"[INFO] Stride: {args.stride}")

    with rosbag.Bag(args.bag, "r") as bag:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[args.topic])):
            # 先根据 stride 决定是否处理该帧（按“原始帧序号 i”更直观）
            if args.stride > 1 and (i % args.stride != 0):
                continue

            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            # 转 OpenCV 图像（尽量保持原始）
            try:
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

            # 保存可视化结果（每处理帧都存）
            draw_and_save(vis_dir, processed, vis_src, corners, board_size, ok)

            if processed % 50 == 0:
                print(f"[INFO] processed={processed}, used(valid)={used}")

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
            "used_frames": int(len(objpoints)),
            "stride": int(args.stride),
            "board": {"cols": args.cols, "rows": args.rows, "square": float(args.square)},
        }
    else:
        rms, K, D, rvecs, tvecs = calibrate_fisheye(
            objpoints, imgpoints, image_size, max_drop=args.max_drop
        )
        calib = {
            "model": "fisheye",
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "K": K.tolist(),
            "dist": D.reshape(-1).tolist(),  # fisheye: k1..k4
            "rms_reproj_error": float(rms),
            "used_frames": int(len(objpoints)),  # 剔除坏帧后的真实数量
            "stride": int(args.stride),
            "max_drop": int(args.max_drop),
            "board": {"cols": args.cols, "rows": args.rows, "square": float(args.square)},
        }

    out_yaml = os.path.join(args.out_dir, "calib.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(calib, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Saved calibration to: {out_yaml}")
    print(f"[INFO] Saved detections visualization to: {vis_dir}")


if __name__ == "__main__":
    main()
