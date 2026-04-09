#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

from tqdm import tqdm   # 进度显示组件
import time

import yaml     # yaml 读写
import time
import subprocess
import shlex

import rosbag   # 负责剔除无关话题

ROOT = Path(__file__).resolve().parent
CALIBR_DATA = ROOT / "calibr_data"
KALIBR_WS = ROOT / "kalibr_ws"

DECOMPRESS_PY = ROOT / "decompress_calibr_bag.py"  # 你要把它改成支持 --in/--out
CHECKERBOARD_YAML = KALIBR_WS / "src" / "config" / "checkerboard.yaml"  # 按你的实际路径改

LEFT_BAG = CALIBR_DATA / "left_inner.bag"
RIGHT_BAG = CALIBR_DATA / "right_inner.bag"

LEFT_RAW_BAG = CALIBR_DATA / "left_inner_raw.bag"
RIGHT_RAW_BAG = CALIBR_DATA / "right_inner_raw.bag"

# kalibr 的输出文件（你描述的是 left1_raw-camchain.yam这里按你实际输出名来）
LEFT_CAMCHAIN = CALIBR_DATA / "left1_raw-camchain.yaml"
RIGHT_CAMCHAIN = CALIBR_DATA / "right1_raw-camchain.yaml"

# 外参相关
LIDAR2CAM_WS = ROOT / "lidar2cam_calib_ws"
LIDAR2CAM_CONFIG = LIDAR2CAM_WS / "src" / "lidar2cam_calib" / "config" / "config.yaml"

LEFT_EXT_BAG  = CALIBR_DATA / "left_extrinsic.bag"
RIGHT_EXT_BAG = CALIBR_DATA / "right_extrinsic.bag"

LEFT_EXT_RAW  = CALIBR_DATA / "left_extrinsic_raw.bag"
RIGHT_EXT_RAW = CALIBR_DATA / "right_extrinsic_raw.bag"

LEFT_CAMCHAIN_YAML  = CALIBR_DATA / "left_inner_raw-camchain.yaml"
RIGHT_CAMCHAIN_YAML = CALIBR_DATA / "right_inner_raw-camchain.yaml"

# 外参相关
def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml0(path: Path, data):       # 不加引号 lidar: /livox/lidar
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def save_yaml1(path, data):              # 加引号 lidar: "/livox/lidar"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True
        )

def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=FlowSeqInBlockDumper,
            default_flow_style=False,  # 外层用块
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )

def filter_extrinsic_bag(in_bag: Path,
                         out_bag: Path,
                         lidar_topic: str,
                         expected_token: str):
    """
    只保留：
      - lidar_topic（完全匹配）
      - topic 名中包含 expected_token 的相机话题
    """
    print(f"[filter] input bag : {in_bag}")
    print(f"[filter] output bag: {out_bag}")
    print(f"[filter] keep lidar : {lidar_topic}")
    print(f"[filter] keep camera topics containing: '{expected_token}'")

    with rosbag.Bag(str(in_bag), 'r') as inbag, \
         rosbag.Bag(str(out_bag), 'w') as outbag:

        kept_topics = set()

        for topic, msg, t in inbag.read_messages():
            keep = False

            if topic == lidar_topic:
                keep = True
            elif expected_token in topic:
                keep = True

            if keep:
                outbag.write(topic, msg, t)
                kept_topics.add(topic)

    print(f"[filter] kept topics:")
    for t in sorted(kept_topics):
        print(f"   - {t}")


class FlowSeqInBlockDumper(yaml.SafeDumper):
    pass

def _represent_sequence(dumper, tag, sequence, flow_style=None):
    # 若是“短的纯标量列表”，用行内 [a, b, c]
    if all(isinstance(x, (int, float, str, bool, type(None))) for x in sequence) and len(sequence) <= 6:
        flow_style = True
    else:
        flow_style = False
    return yaml.SafeDumper.represent_sequence(dumper, tag, sequence, flow_style=flow_style)

FlowSeqInBlockDumper.add_representer(list, lambda dumper, data:
    _represent_sequence(dumper, 'tag:yaml.org,2002:seq', data)
)

def camchain_to_K_D(camchain_path: Path):
    """
    从 camchain.yaml 读取:
      intrinsics: [fx, fy, cx, cy]
      distortion_coeffs: [k1,k2,k3,k4]
      rostopic: /xxx
    返回: (rostopic, K(3x3 list), D(list))
    """
    ensure_exists(camchain_path, "camchain yaml")

    y = load_yaml(camchain_path)
    if "cam0" not in y:
        raise RuntimeError(f"camchain missing cam0: {camchain_path}")

    cam0 = y["cam0"]
    rostopic = cam0.get("rostopic", "")     # 键名 默认值
    intr = cam0.get("intrinsics", None)
    dist = cam0.get("distortion_coeffs", None)
    if not rostopic or intr is None or dist is None:
        raise RuntimeError(f"camchain format unexpected: {camchain_path}")

    fx, fy, cx, cy = intr
    K = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]
    D = [float(x) for x in dist]
    return rostopic, K, D

def update_lidar2cam_config(config_path: Path, lidar_topic: str, cam_topic: str, K, D):
    """
    按你的 config.yaml 格式写入：
      topics.lidar / topics.camera
      cameraIntrinsic (3x3)
      cameraDistcoff (4x1 list)
    其它字段原样保留（尤其 T_LtoC）
    """
    ensure_exists(config_path, "lidar2cam config yaml")

    cfg = load_yaml(config_path)

    # topics
    cfg.setdefault("topics", {})
    cfg["topics"]["lidar"] = lidar_topic
    cfg["topics"]["camera"] = cam_topic

    # intrinsics / distortion
    cfg["cameraIntrinsic"] = K
    cfg["cameraDistcoff"] = D

    save_yaml(config_path, cfg)

# 继承终端形式
def popen_ros_cmd(cmd_inner: str, ws_setup: Path, log_prefix: str):
    """
    以 bash -lc 方式 source 环境，然后 Popen 常驻进程。
    """
    algo_cmd = (
        f"bash -lc "
        f"\"source /opt/ros/noetic/setup.bash && "
        f"source {shlex.quote(str(ws_setup))} && "
        f"{cmd_inner}\""
    )
    print(f"\n>>> POPEN: {algo_cmd}")
    # stdout/stderr 合并，方便你后续重定向到日志文件（现在先打印即可）
    # return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.Popen(algo_cmd, shell=True, stdin=None, stdout=None, stderr=None, text=True)

# 不继承终端形式
def popen_ros_cmd2(cmd_inner: str, ws_setup: Path, log_prefix: str):
    """
    以 bash -lc 方式 source 环境，然后 Popen 常驻进程。
    """
    algo_cmd = (
        f"bash -lc "
        f"\"source /opt/ros/noetic/setup.bash && "
        f"source {shlex.quote(str(ws_setup))} && "
        f"{cmd_inner}\""
    )
    print(f"\n>>> POPEN: {algo_cmd}")
    # stdout/stderr 合并，方便你后续重定向到日志文件（现在先打印即可）
    # return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.Popen(algo_cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,text=True,)

# 暂时不需要
def stream_process_output(proc, tag: str, max_lines: int = 200):
    """
    防止 RosNode 输出太多把终端刷爆：只在启动阶段取一部分行用于确认。
    """
    if proc.stdout is None:
        return
    count = 0
    while count < max_lines:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[{tag}] {line}", end="")
        count += 1

def calibrate_one_camera_extrinsics(which: str,
                                    ext_bag: Path,
                                    ext_raw_bag: Path,
                                    camchain_yaml: Path,
                                    expected_token: str,   # "left" or "right"
                                    lidar_topic: str = "/livox/lidar",
                                    pbar=None):
    """
    外参流程：
    (0) decompress前 保留必要项传感器话题
    (1) decompress ext bag -> ext_raw
    (2) 读 camchain，检查 rostopic 与 left/right 匹配，写入 lidar2cam config.yaml
    (3) rosrun lidar2cam_calib RosNode config.yaml (常驻)
    (4) rosbag play ext_raw_bag
    (5) 写回 kilox_map.yaml（占位 print）
    """

    print(f"\n============================================================")
    print(f"  进入 {which} 外参标定:")
    print(f"  Calibrating {which}")
    print(f"  RIGHT_CAMCHAIN_YAML: {camchain_yaml}")
    print(f"============================================================")

    ensure_exists(LIDAR2CAM_WS, "lidar2cam_calib_ws directory")
    ensure_exists(ext_bag, f"{which} extrinsic bag")
    ensure_exists(camchain_yaml, f"{which} camchain yaml")
    ensure_exists(LIDAR2CAM_CONFIG, "lidar2cam config.yaml")

    # (0) 过滤无关话题 只留 左/右目 和 雷达 
    filtered_bag = ext_bag.with_name(ext_bag.stem + "_filtered.bag")

    filter_extrinsic_bag(
        in_bag=ext_bag,
        out_bag=filtered_bag,
        lidar_topic=lidar_topic,
        expected_token=expected_token
    )

    # 后续流程统一用 filtered_bag
    ext_bag = filtered_bag

    # (1) decompress
    if pbar:
        pbar.set_description(f"{which} decompress")
    decompress_bag(ext_bag, ext_raw_bag)
    if pbar:
        pbar.update(1)

    # (2) read camchain + update config
    rostopic, K, D = camchain_to_K_D(camchain_yaml)     # 读取 rostopic, K, D没问题

    # token check：要求 rostopic 里能看出 left/right 校验一下
    if expected_token.lower() not in rostopic.lower():
        raise RuntimeError(
            f"[{which}] camchain rostopic '{rostopic}' does not contain token '{expected_token}'. "
            f"Please check camchain_yaml: {camchain_yaml}"
        )

    update_lidar2cam_config(
        config_path=LIDAR2CAM_CONFIG,
        lidar_topic=lidar_topic,
        cam_topic=rostopic,
        K=K,
        D=D
    )
    print("\n")
    print(f"[{which}] lidar2cam config updated: camera={rostopic}")

    # (3) start RosNode
    if pbar:
        pbar.set_description(f"{which} calibrating")

    ws_setup = LIDAR2CAM_WS / "devel" / "setup.bash"
    ensure_exists(ws_setup, "lidar2cam ws devel/setup.bash (please catkin_make first)")

    node_cmd_inner = f"rosrun lidar2cam_calib RosNode {shlex.quote(str(LIDAR2CAM_CONFIG))}"
    # cmd2 = (
    #     f"bash -lc "
    #     f"\"source /opt/ros/noetic/setup.bash && "
    #     f"source {shlex.quote(str(ws_setup))} && "
    #     f"{node_cmd_inner}\""
    # )
    # run_cmd(cmd2, cwd=ROOT)

    # 返回一个 Popen 子进程对象（RosNode 常驻进程）
    node_proc = popen_ros_cmd(node_cmd_inner, ws_setup=ws_setup, log_prefix=which)
    # node_proc.stdin.close()  # 关闭 stdin 避免占用
    # node_proc.wait()  # 给节点一点启动时间

    rc = node_proc.poll()
    print(f"[{which}] RosNode started with PID {node_proc.pid}, initial return code: {rc}")     # None 表示还在跑；非 None 表示已经退出（可能启动失败）
    time.sleep(2)

    # # TODO 封版 开了子线程挂起监听rosbag播包 12.19
    # # 可选：打印部分输出确认节点启动没问题 防止控制台打印撑爆
    # # stream_process_output(node_proc, tag=f"{which}-RosNode", max_lines=50)

    # (4) play bag（播包放在同一个 ROS 环境里更稳）
    play_cmd_inner = f"rosbag play {shlex.quote(str(ext_raw_bag))}"
    # play_cmd = (
    #     f"bash -lc "
    #     f"\"source /opt/ros/noetic/setup.bash && "
    #     f"source {shlex.quote(str(ws_setup))} && "
    #     f"{play_cmd_inner}\""
    # )
    # run_cmd(play_cmd, cwd=ROOT)
    play_cmd_inner = f"rosbag play {shlex.quote(str(ext_raw_bag))}"
    node_proc2 = popen_ros_cmd2(play_cmd_inner, ws_setup=ws_setup, log_prefix=which)
    # node_proc2.wait(15)  # 等待播包结束   # TODO 这里要不要加超时？12.20 (y/n)逻辑中n的话会卡住 还是要传命令字让包重播

    # 播完之后：结束节点
    try:
        node_proc.wait()
        node_proc.terminate()
    except Exception:
        node_proc.kill()
    
    # # TODO 封版 12.19 要考虑三次数据求超定优化的情况
    # # if pbar:
    # #     pbar.update(1)  # 这里把 “RosNode+播包” 合并当作一步

    # # # (5) 写入 kilox_map.yaml 占位
    # # print("write success")  # TODO: patch kilox_map.yaml



# 内参相关
def run_cmd(cmd: str, cwd=None):
    """运行命令：实时打印输出，失败则直接退出"""
    print(f"\n>>> {cmd}")
    p = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        return rc
    # assert rc != 0,
    #     raise RuntimeError(f"Command failed (rc={rc}): {cmd}")

def ensure_exists(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")

def decompress_bag(in_bag: Path, out_bag: Path):
    ensure_exists(DECOMPRESS_PY, "decompress script")
    ensure_exists(in_bag, "input bag")
    # TODO 这里是否检查包已经存在才执行
    # 你要求“每次传参 IN_BAG/OUT_BAG 路径适配”，所以这里走 CLI 参数
    cmd = f"python3 {DECOMPRESS_PY} --in {in_bag} --out {out_bag}"
    run_cmd(cmd, cwd=ROOT)

    ensure_exists(out_bag, "output raw bag")

def kalibr_intrinsics(raw_bag: Path, topic: str, models: str = "pinhole-equi"):
    ensure_exists(raw_bag, "raw bag")
    ensure_exists(CHECKERBOARD_YAML, "checkerboard.yaml")

    # 注意：你命令里是 target src/config/checkerboard.yaml
    # 我这里用绝对路径，减少 cwd/相对路径坑
    cmd_inner = (
        "rosrun kalibr kalibr_calibrate_cameras "
        f"--bag {raw_bag} "
        f"--topics {topic} "
        f"--models {models} "
        f"--target {CHECKERBOARD_YAML}"
        # "--dont-show-report"
    )

    cmd = (
        "bash -lc "
        f"\"source /opt/ros/noetic/setup.bash && "
        f"source {ROOT/'kalibr_ws'}/devel/setup.bash && "
        f"{cmd_inner}\""
    )

    run_cmd(cmd, cwd=ROOT)

def ask_write_or_rerun(which: str) -> bool:
    """返回 True 表示写入并继续，False 表示重跑该相机标定"""
    while True:
        ans = input(f"\n[{which}] ⚠️  请人工审核生成的标定文件数据，Write results into kilox_map.yaml? (y/n): y写入 n重标").strip().lower()
        if ans in ("y", "yes"):
            print("write success")  # TODO 你要求：先不写逻辑，用 print 占位
            return True
        if ans in ("n", "no"):
            print(f"[{which}] rerun calibration...")
            return False
        print("Please input y or n.")


# 单目相机总揽
def calibrate_one_camera(which: str, in_bag: Path, out_raw_bag: Path, topic: str, pbar: tqdm):
    print(f"\n============================================================")
    print(f"  进入 {which} 内参标定:")
    print(f"  Calibrating {which}")
    print(f"  IN BAG PATH: {in_bag}")
    print(f"  OUTPUT RAW BAG PATH: {out_raw_bag}")
    print(f"  TOPIC: {topic}")
    print(f"============================================================")

    while True:
        
        # (1) decompress
        pbar.set_description(f"{which} decompress")
        decompress_bag(in_bag, out_raw_bag)
        pbar.update(1)

        # (2) kalibr intrinsics
        pbar.set_description(f"{which} calibrating")
        kalibr_intrinsics(out_raw_bag, topic=topic)
        pbar.update(1)
        # (3) ask write or rerun
        if ask_write_or_rerun(which):
            break
        else:
            # 如果选择重跑，需要把进度回退 2 步，否则总进度会被“重复计数”
            pbar.update(-2)

def main():

    for i in range(2, 0, -1):
        if i == 2:
            print("\n")
        print(" 即将进入标定过程，按 Ctrl+C 可中断标定过程\n ")
        time.sleep(2)  # 让进度条先显示出来

    print("\n   标定进度 0/8 \n ")
    STEPS_PER_CAM = 2  # decompress + kalibr（ask 不算耗时可不计）
    TOTAL_STEPS = STEPS_PER_CAM * 4  # 左右两目(内外参各一次)
    pbar = tqdm(total=TOTAL_STEPS, ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]")
    # pbar.set_description("标定总进度\n")

    try:
        ensure_exists(CALIBR_DATA, "calibr_data directory")
        ensure_exists(KALIBR_WS, "kalibr_ws directory")

        # # 左目内参
        # calibrate_one_camera(
        #     which="LEFT_INTRINSICS",
        #     in_bag=LEFT_BAG,
        #     out_raw_bag=LEFT_RAW_BAG,
        #     topic="/usb_cam_left/image_raw",
        #     pbar = pbar
        # )

        # # 右目内参
        # calibrate_one_camera(
        #     which="RIGHT_INTRINSICS",
        #     in_bag=RIGHT_BAG,
        #     out_raw_bag=RIGHT_RAW_BAG,
        #     topic="/usb_cam_right/image_raw",
        #     pbar =  pbar
        # )

        # # 前目内参
        # calibrate_one_camera(
        #     which="LEFT_INTRINSICS",
        #     in_bag=LEFT_BAG,
        #     out_raw_bag=LEFT_RAW_BAG,
        #     topic="/usb_cam_front/image_raw",
        #     pbar = pbar
        # )

        # 左目外参
        calibrate_one_camera_extrinsics(
            which="LEFT_EXTRINSICS",
            ext_bag=LEFT_EXT_BAG,
            ext_raw_bag=LEFT_EXT_RAW,
            camchain_yaml=LEFT_CAMCHAIN_YAML,
            expected_token="left",
            pbar=pbar
        )

        # # 右目外参
        # calibrate_one_camera_extrinsics(
        #     which="RIGHT_EXTRINSICS",
        #     ext_bag=RIGHT_EXT_BAG,
        #     ext_raw_bag=RIGHT_EXT_RAW,
        #     camchain_yaml=RIGHT_CAMCHAIN_YAML,
        #     expected_token="right",
        #     pbar=pbar
        # )
        
        # calibrate_one_camera_extrinsics(
        # # TODO 把上一步YAML作为参数放入 extrinsics 标定
        # # TODO 拉起 extrinsics 标定gingnengbao
        # # TODO 播包
        # )

        # calibrate_one_camera_extrinsics(
        # # TODO 把上一步YAML作为参数放入 extrinsics 标定
        # # TODO 拉起 extrinsics 标定
        # # TODO 播包
        # )


        print("\n✅ Intrinsics stage finished.")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()






# yaml格式调整
# from pathlib import Path
# import yaml
# import re

# class QuotedStr(str):
#     """仅用于让 YAML value 强制双引号输出（不影响 key）"""
#     pass

# class PrettyDumper(yaml.SafeDumper):
#     pass

# def _repr_quoted_str(dumper: yaml.Dumper, data: QuotedStr):
#     return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')

# def _repr_list(dumper: yaml.Dumper, data: list):
#     """
#     规则：
#     - 如果是“短的纯标量 list”（比如 [1,0,0,0] 或 [fx,0,cx]），用 flow style： [ ... ]
#     - 否则用 block style：- ...
#     """
#     is_scalar_list = all(isinstance(x, (int, float, str, bool, type(None))) for x in data)
#     flow = bool(is_scalar_list and len(data) <= 8)
#     return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

# PrettyDumper.add_representer(QuotedStr, _repr_quoted_str)
# PrettyDumper.add_representer(list, _repr_list)

# def save_yaml_pretty(path: Path, data: dict):
#     txt = yaml.dump(
#         data,
#         Dumper=PrettyDumper,
#         sort_keys=False,
#         indent=2,
#         default_flow_style=False,  # 顶层 / 外层用块风格
#         allow_unicode=True,
#         width=10**9,               # 避免自动换行把一行 list 拆掉
#     )

#     # 可选：给顶层块之间加空行（纯美观，不影响读取）
#     # 在这些 key 前面插入一行空行：
#     for key in ["T_LtoC:", "cameraIntrinsic:", "cameraDistcoff:"]:
#         txt = re.sub(rf"\n({re.escape(key)})", r"\n\n\1", txt)

#     path.write_text(txt, encoding="utf-8")


# 说明文档：目前只做了上述功能
# 1 对识别不到特征的帧的重处理功能暂未实现 分为image 和pointcloud两种
# 2 对实时点云的可视化暂未实现
# 3 对kilox_map,yaml的写入暂未实现
# 4 对多次采集n组bag数据进行多包运行 超定优化暂未实现(现在是只做单包标定)
# 封版时间2025.12.22