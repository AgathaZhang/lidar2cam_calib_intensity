#!/usr/bin/env python3
import argparse, os, sys, subprocess, shlex
from datetime import datetime

def run(cmd, log_path, cwd=None, env=None):
    """运行命令并把 stdout/stderr 全部写入 log，同时在终端打印"""
    print(f"\n>>> RUN: {cmd}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"[{datetime.now().isoformat()}] CMD: {cmd}\n")
        f.flush()

        p = subprocess.Popen(
            cmd, shell=True, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        rc = p.wait()
        f.write(f"\n[exit code] {rc}\n")
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {cmd}")

def bag_has_topic(bag_path, topic):
    """用 rosbag info 简单判断 topic 是否存在"""
    cmd = f"rosbag info {shlex.quote(bag_path)}"
    out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    return topic in out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--checkerboard", required=True)
    ap.add_argument("--kilox_map", required=True)
    ap.add_argument("--out_dir", default="/work/output")
    ap.add_argument("--tmp_dir", default="/work/tmp")
    ap.add_argument("--log_dir", default="/work/logs")

    # 你这次的 topic（按需改）
    ap.add_argument("--left_compressed", default="/usb_cam_left/image_raw/compressed")
    ap.add_argument("--right_compressed", default="/usb_cam_right/image_raw/compressed")
    ap.add_argument("--left_raw", default="/usb_cam_left/image_raw")
    ap.add_argument("--right_raw", default="/usb_cam_right/image_raw")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, "pipeline.log")

    bag_in = args.bag
    bag_for_kalibr = bag_in

    # ---------- Stage 1: decompress if needed ----------
    need_decompress = (
        bag_has_topic(bag_in, args.left_compressed) or bag_has_topic(bag_in, args.right_compressed)
    ) and (not bag_has_topic(bag_in, args.left_raw))  # 一般 raw 不会同时存在

    if need_decompress:
        bag_for_kalibr = os.path.join(args.tmp_dir, "bag_raw.bag")
        # 这里调用你已经验证过的解压脚本（路径按你容器内实际放置改）
        decompress_py = "/work/scripts/decompress_calibr_bag.py"
        cmd = f"python3 {decompress_py} --in {shlex.quote(bag_in)} --out {shlex.quote(bag_for_kalibr)}"
        run(cmd, log_path)
    else:
        print(">>> Decompress not needed.")

    # ---------- Stage 2: intrinsics (kalibr) ----------
    # 你只要把下面这条命令换成你实际要跑的 kalibr 命令
    intrinsics_cmd = (
        "rosrun kalibr kalibr_calibrate_cameras "
        f"--bag {shlex.quote(bag_for_kalibr)} "
        f"--topics {args.left_raw} {args.right_raw} "
        "--models pinhole-equi pinhole-equi "
        f"--target {shlex.quote(args.checkerboard)} "
        "--approx-sync 0.001"
    )
    run(intrinsics_cmd, log_path, cwd="/work")  # 如果你的 target 用相对路径，这里 cwd 很重要

    # ---------- Stage 3: extrinsics (lidar->cam) ----------
    # 把下面这条换成你“lidar 到 camera 外参标定包”的实际命令
    extrinsics_cmd = "echo TODO: run lidar->camera extrinsic calibration here"
    run(extrinsics_cmd, log_path, cwd="/work")

    # ---------- Stage 4: patch kilox_map.yaml ----------
    # 这里也先放占位：你给我“要写的 keys”和“你外参结果文件格式”，我再帮你补齐
    patch_cmd = "echo TODO: parse results and write into kilox_map.yaml"
    run(patch_cmd, log_path, cwd="/work")

    print("\n✅ Pipeline finished. Logs:", log_path)

if __name__ == "__main__":
    main()
