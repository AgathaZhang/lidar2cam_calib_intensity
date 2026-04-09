#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import rosbag
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2


def compressed_to_raw(msg: CompressedImage, encoding: str) -> Image | None:
    """
    把 sensor_msgs/CompressedImage 解码成 sensor_msgs/Image
    encoding: "bgr8" or "mono8"
    """
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if cv_img is None:
        return None

    # 统一到指定 encoding
    if encoding == "mono8":
        if cv_img.ndim == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        if cv_img.dtype != np.uint8:
            cv_img = cv_img.astype(np.uint8)
        h, w = cv_img.shape[:2]
        step = w
        data = cv_img.tobytes()
    else:  # bgr8
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        if cv_img.dtype != np.uint8:
            cv_img = cv_img.astype(np.uint8)
        h, w = cv_img.shape[:2]
        step = w * 3
        data = cv_img.tobytes()

    img_msg = Image()
    img_msg.header = msg.header
    img_msg.height = h
    img_msg.width = w
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.step = step
    img_msg.data = data
    return img_msg


def derive_raw_topic(compressed_topic: str) -> str:
    """
    默认：把末尾的 /compressed 去掉
    /usb_cam_front/image_raw/compressed -> /usb_cam_front/image_raw
    """
    suffix = "/compressed"
    if compressed_topic.endswith(suffix):
        return compressed_topic[: -len(suffix)]
    # 极少数情况不是以 /compressed 结尾，就给个后缀
    return compressed_topic + "_raw"


def build_topic_map(in_bag_path: str, only: list[str], exclude: list[str]) -> dict[str, str]:
    """
    扫描 bag 的连接信息，找出所有 CompressedImage topic，并生成映射表：
    {compressed_topic: raw_topic}
    """
    topic_map = {}
    with rosbag.Bag(in_bag_path, "r") as bag:
        info = bag.get_type_and_topic_info()
        topics = info.topics  # dict: topic_name -> TopicTuple(msg_type, count, ...)

        for topic_name, topic_info in topics.items():
            if topic_info.msg_type != "sensor_msgs/CompressedImage":
                continue

            if only and topic_name not in only:
                continue
            if exclude and topic_name in exclude:
                continue

            topic_map[topic_name] = derive_raw_topic(topic_name)

    return topic_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert sensor_msgs/CompressedImage topics in a rosbag into sensor_msgs/Image (raw)."
    )
    parser.add_argument("--in", dest="in_bag", required=True, help="input .bag path")
    parser.add_argument("--out", dest="out_bag", required=True, help="output .bag path")
    parser.add_argument("--encoding", default="bgr8", choices=["bgr8", "mono8"],
                        help="output raw image encoding")
    parser.add_argument("--only", nargs="*", default=[],
                        help="only convert these compressed topics (exact match). e.g. /usb_cam_front/image_raw/compressed")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="exclude these compressed topics (exact match).")
    parser.add_argument("--keep-compressed", action="store_true",
                        help="also keep original compressed topics in output (default: drop them).")
    args = parser.parse_args()

    in_bag = args.in_bag
    out_bag = args.out_bag

    print(f"Reading from : {in_bag}")
    print(f"Writing to   : {out_bag}")
    print(f"Encoding     : {args.encoding}")

    topic_map = build_topic_map(in_bag, args.only, args.exclude)

    if not topic_map:
        print("[WARN] No sensor_msgs/CompressedImage topics matched. Nothing to convert.")
        print("       Tip: check 'rosbag info <bag>' and/or use --only to specify a topic.")
        return

    print("Will convert topics:")
    for c, r in topic_map.items():
        print(f"  {c}  -->  {r}")

    total_msgs = 0
    total_converted_in = 0
    total_converted_out = 0
    failed = 0

    per_topic_in = {k: 0 for k in topic_map.keys()}
    per_topic_out = {k: 0 for k in topic_map.keys()}

    with rosbag.Bag(in_bag, "r") as ib, rosbag.Bag(out_bag, "w") as ob:
        for topic, msg, t in ib.read_messages():
            total_msgs += 1

            # 压缩图像：解码并写入 raw topic
            if topic in topic_map and msg._type == "sensor_msgs/CompressedImage":
                per_topic_in[topic] += 1
                total_converted_in += 1

                raw_msg = compressed_to_raw(msg, args.encoding)
                if raw_msg is None:
                    failed += 1
                    print(f"[WARN] decode failed: topic={topic} time={t.to_sec()}")
                    if args.keep_compressed:
                        ob.write(topic, msg, t)
                    continue

                ob.write(topic_map[topic], raw_msg, t)
                per_topic_out[topic] += 1
                total_converted_out += 1

                # 是否保留原 compressed
                if args.keep_compressed:
                    ob.write(topic, msg, t)
                continue

            # 其它 topic 原样拷贝
            ob.write(topic, msg, t)

    print("\nDone.")
    print(f"Total messages           : {total_msgs}")
    print(f"Compressed frames in     : {total_converted_in}")
    print(f"Raw frames written       : {total_converted_out}")
    print(f"Decode failed            : {failed}")
    print("Per-topic stats:")
    for topic in topic_map.keys():
        print(f"  {topic}: in={per_topic_in[topic]} out={per_topic_out[topic]}  ->  {topic_map[topic]}")


if __name__ == "__main__":
    main()
