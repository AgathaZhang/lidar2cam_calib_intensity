#!/usr/bin/env python3
import rosbag
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_bag", required=True)
parser.add_argument("--out", dest="out_bag", required=True)
args = parser.parse_args()

IN_BAG = args.in_bag
OUT_BAG = args.out_bag

# ===== 输入/输出 bag 路径（绝对路径，避免 cwd 混乱） =====
# IN_BAG  = "/home/kilox/workspace/kalibr_ws/src/kalibr/aslam_offline_calibration/kalibr/data/left1.bag"
# OUT_BAG = "/home/kilox/workspace/kalibr_ws/src/kalibr/aslam_offline_calibration/kalibr/data/left1_raw.bag"

# IN_BAG  = "/home/kilox/workspace/lidar2cam_calib_ws/data/left/left33.bag"
# OUT_BAG = "/home/kilox/workspace/lidar2cam_calib_ws/data/left/left33_raw.bag"

# IN_BAG  = "/home/kilox/workspace/calibr_data/left1.bag"
# OUT_BAG = "/home/kilox/workspace/calibr_data/left11_raw.bag"

# ===== 需要转换的 topic 映射：压缩 → raw =====
# LEFT_COMPRESSED  = "/usb_cam_front/image_raw/compressed"
LEFT_COMPRESSED  = "/usb_cam_left/image_raw/compressed"
RIGHT_COMPRESSED = "/usb_cam_right/image_raw/compressed"
FRONT_COMPRESSED = "/usb_cam_front/image_raw/compressed"

LEFT_RAW_TOPIC   = "/usb_cam_left/image_raw"
RIGHT_RAW_TOPIC  = "/usb_cam_right/image_raw"
FRONT_RAW_TOPIC  = "/usb_cam_front/image_raw"

print(f"Reading from: {IN_BAG}")
print(f"Writing to  : {OUT_BAG}")

def compressed_to_raw(msg: CompressedImage) -> Image:
    """把 CompressedImage 解码成 Image（bgr8）"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        return None

    h, w = cv_img.shape[:2]

    img_msg = Image()
    img_msg.header = msg.header
    img_msg.height = h
    img_msg.width = w
    img_msg.encoding = "bgr8"   # Kalibr 对 bgr8/rgb8 都可以
    img_msg.is_bigendian = 0
    img_msg.step = w * 3
    img_msg.data = cv_img.tobytes()
    return img_msg


with rosbag.Bag(IN_BAG, "r") as in_bag, rosbag.Bag(OUT_BAG, "w") as out_bag:
    count_in_left = 0
    count_out_left = 0
    count_in_right = 0
    count_out_right = 0
    count_in_front = 0
    count_out_front = 0
    total_msgs = 0

    for topic, msg, t in in_bag.read_messages():
        total_msgs += 1

        # 1) 左相机 compressed → raw
        if topic == LEFT_COMPRESSED and msg._type == "sensor_msgs/CompressedImage":
            count_in_left += 1
            raw_msg = compressed_to_raw(msg)
            if raw_msg is None:
                print(f"[WARN] failed to decode LEFT frame at time {t.to_sec()}")
                continue
            out_bag.write(LEFT_RAW_TOPIC, raw_msg, t)
            count_out_left += 1
            continue  # 不再写入原 compressed topic

        # 2) 右相机 compressed → raw
        if topic == RIGHT_COMPRESSED and msg._type == "sensor_msgs/CompressedImage":
            count_in_right += 1
            raw_msg = compressed_to_raw(msg)
            if raw_msg is None:
                print(f"[WARN] failed to decode RIGHT frame at time {t.to_sec()}")
                continue
            out_bag.write(RIGHT_RAW_TOPIC, raw_msg, t)
            count_out_right += 1
            continue  # 不再写入原 compressed topic

        # TODO 这里应该识别前目 2026.01.13
        # 3) 前相机 compressed → raw
        if topic == FRONT_COMPRESSED and msg._type == "sensor_msgs/CompressedImage":
            count_in_front += 1
            raw_msg = compressed_to_raw(msg)
            if raw_msg is None:
                print(f"[WARN] failed to decode FRONT frame at time {t.to_sec()}")
                continue
            out_bag.write(FRONT_RAW_TOPIC, raw_msg, t)
            count_out_right += 1
            continue  # 不再写入原 compressed topic

        # 4) 其他所有 topic（IMU, lidar 等）原样拷贝
        out_bag.write(topic, msg, t)

    print("Done.")
    print(f"Total messages        : {total_msgs}")
    print(f"Left  compressed in   : {count_in_left}")
    print(f"Left  raw written     : {count_out_left}")
    print(f"Right compressed in   : {count_in_right}")
    print(f"Right raw written     : {count_out_right}")
    # print(f"Front compressed in   : {count_in_front}")
    # print(f"Front raw written     : {count_out_front}")

# #!/usr/bin/env python3
# import rosbag
# from sensor_msgs.msg import CompressedImage, Image
# import numpy as np
# import cv2

# # ===== 输入/输出 bag 路径 =====
# IN_BAG  = "/home/kilox/workspace/kalibr_ws/src/kalibr/aslam_offline_calibration/kalibr/data/left1.bag"         # 原始含 compressed 的包
# OUT_BAG = "/home/kilox/workspace/kalibr_ws/src/kalibr/aslam_offline_calibration/kalibr/data/left1_raw.bag"     # 转换后含 raw 的包

# # ===== 需要转换的 topic 映射：压缩 → raw =====
# TOPIC_MAP = {
#     "/usb_cam_left/image_raw/compressed":  "/usb_cam_left/image_raw",
#     "/usb_cam_right/image_raw/compressed": "/usb_cam_right/image_raw",
# }

# print(f"Reading from: {IN_BAG}")
# print(f"Writing to  : {OUT_BAG}")

# def compressed_to_raw(msg: CompressedImage) -> Image:
#     """把 CompressedImage 解码成 Image（bgr8）"""
#     np_arr = np.frombuffer(msg.data, np.uint8)
#     cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if cv_img is None:
#         return None

#     h, w = cv_img.shape[:2]

#     img_msg = Image()
#     img_msg.header = msg.header
#     img_msg.height = h
#     img_msg.width = w
#     img_msg.encoding = "bgr8"   # Kalibr 对 bgr8/rgb8 都可以
#     img_msg.is_bigendian = 0
#     img_msg.step = w * 3
#     img_msg.data = cv_img.tobytes()
#     return img_msg


# with rosbag.Bag(IN_BAG, "r") as in_bag, rosbag.Bag(OUT_BAG, "w") as out_bag:
#     count_in_left = 0
#     count_out_left = 0
#     count_in_right = 0
#     count_out_right = 0
#     total_msgs = 0

#     for topic, msg, t in in_bag.read_messages():
#         total_msgs += 1

#         # 1) 左右相机的 compressed → raw
#         if topic in TOPIC_MAP and isinstance(msg, CompressedImage):
#             out_topic = TOPIC_MAP[topic]

#             if "left" in topic:
#                 count_in_left += 1
#             else:
#                 count_in_right += 1

#             raw_msg = compressed_to_raw(msg)
#             if raw_msg is None:
#                 print(f"[WARN] failed to decode frame on topic {topic} at time {t.to_sec()}")
#                 continue

#             out_bag.write(out_topic, raw_msg, t)

#             if "left" in topic:
#                 count_out_left += 1
#             else:
#                 count_out_right += 1

#         # 2) 其他所有 topic（IMU, lidar 等）原样拷贝
#         else:
#             # 注意：这里不会再写入原来的 compressed 话题
#             # 因为 TOPIC_MAP 中的两个已经在上面处理掉了
#             out_bag.write(topic, msg, t)

#     print("Done.")
#     print(f"Total messages        : {total_msgs}")
#     print(f"Left  compressed in   : {count_in_left}")
#     print(f"Left  raw written     : {count_out_left}")
#     print(f"Right compressed in   : {count_in_right}")
#     print(f"Right raw written     : {count_out_right}")
