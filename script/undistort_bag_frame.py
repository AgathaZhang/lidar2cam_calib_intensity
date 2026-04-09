#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import cv2
import yaml

import rosbag
from cv_bridge import CvBridge


def load_kd_from_yaml(yaml_path: str, cam_key: str = "cam0"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if cam_key not in data:
        raise KeyError(f"YAML里找不到键 '{cam_key}'，实际keys={list(data.keys())}")

    cam = data[cam_key]
    intr = cam["intrinsics"]  # [fx, fy, cx, cy]
    dist = cam["distortion_coeffs"]  # [k1, k2, k3, k4]
    res = cam.get("resolution", None)

    if len(intr) != 4:
        raise ValueError(f"intrinsics长度应为4 (fx,fy,cx,cy)，实际={intr}")
    if len(dist) != 4:
        raise ValueError(f"fisheye equidistant畸变系数应为4 (k1..k4)，实际={dist}")

    fx, fy, cx, cy = [float(x) for x in intr]
    k1, k2, k3, k4 = [float(x) for x in dist]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)

    return K, D, res


def find_nth_image_message(bag_path: str, topic: str, index: int):
    if index < 0:
        raise ValueError("--index 必须 >= 0")

    count = 0
    with rosbag.Bag(bag_path, "r") as bag:
        for tpc, msg, stamp in bag.read_messages(topics=[topic]):
            # 只处理 sensor_msgs/Image
            if msg._type != "sensor_msgs/Image":
                continue
            if count == index:
                return msg, stamp, count
            count += 1

    raise IndexError(f"在topic={topic}里找不到第 {index} 帧（0-based）。该topic总帧数可能是 {count}。")


def undistort_fisheye(img_bgr_or_gray: np.ndarray, K: np.ndarray, D: np.ndarray):
    h, w = img_bgr_or_gray.shape[:2]

    # 这里默认输出同分辨率、同视场（newK = K）
    newK = K.copy()

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3, dtype=np.float64), newK, (w, h), cv2.CV_16SC2
    )
    undist = cv2.remap(img_bgr_or_gray, map1, map2, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    return undist


def main():
    parser = argparse.ArgumentParser(description="Undistort N-th frame from rosbag using fisheye equidistant intrinsics YAML.")
    parser.add_argument("--bag", required=True, help="rosbag文件路径")
    parser.add_argument("--yaml", required=True, help="相机内参yaml路径（含cam0.intrinsics与distortion_coeffs）")
    parser.add_argument("--topic", default="/usb_cam_front/image_raw", help="图像topic，默认 /usb_cam_front/image_raw")
    parser.add_argument("--index", type=int, required=True, help="取topic中的第几帧（0-based）")
    parser.add_argument("--cam_key", default="cam0", help="yaml里相机key，默认 cam0")
    parser.add_argument("--out", default="undistorted.png", help="输出图片路径（png/jpg均可）")
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"[ERROR] bag不存在: {args.bag}")
        sys.exit(1)
    if not os.path.isfile(args.yaml):
        print(f"[ERROR] yaml不存在: {args.yaml}")
        sys.exit(1)

    K, D, res = load_kd_from_yaml(args.yaml, args.cam_key)
    print("[INFO] Loaded intrinsics:")
    print("       K=\n", K)
    print("       D=", D.tolist())
    if res is not None:
        print("       resolution(yaml)=", res)

    msg, stamp, real_index = find_nth_image_message(args.bag, args.topic, args.index)
    print(f"[INFO] Found frame index={real_index}, stamp={stamp.to_sec():.6f}, encoding={msg.encoding}, size={msg.width}x{msg.height}")

    bridge = CvBridge()

    # 让cv_bridge尽量按原编码转；如果是mono8就得到灰度，否则一般是bgr8/rgb8等
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as e:
        print(f"[ERROR] cv_bridge转换失败: {e}")
        sys.exit(2)

    # 如果是RGB，转BGR方便opencv保存显示一致
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
        if msg.encoding.lower() in ["rgb8", "rgb16"]:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    undist = undistort_fisheye(cv_img, K, D)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(args.out, undist)
    if not ok:
        print(f"[ERROR] 写文件失败: {args.out}")
        sys.exit(3)

    print(f"[INFO] Saved undistorted image: {args.out}")


if __name__ == "__main__":
    main()
