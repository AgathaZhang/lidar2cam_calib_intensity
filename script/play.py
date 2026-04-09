#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def natural_key(s: str):
    # img2 < img10
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def list_images(img_dir: Path):
    files = [p for p in img_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def parse_resize(resize: str):
    if not resize:
        return None
    w, h = resize.lower().split("x")
    return (int(w), int(h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="image directory inside container, e.g. /data/images")
    ap.add_argument("--out", required=True, help="output mp4 path, e.g. /data/out.mp4")
    ap.add_argument("--fps", type=float, default=20.0, help="video fps (default: 20)")
    ap.add_argument("--resize", default="", help="resize to WxH, e.g. 1280x720 (optional)")
    ap.add_argument("--codec", default="mp4v", help="fourcc codec (default: mp4v). try 'avc1' if supported")
    args = ap.parse_args()

    img_dir = Path(args.dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir} (supported: {sorted(IMG_EXTS)})")

    resize_wh = parse_resize(args.resize)

    # Read first valid frame to decide size
    first = None
    first_path = None
    for p in images:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            first, first_path = im, p
            break
    if first is None:
        raise SystemExit("All images failed to read.")

    if resize_wh is not None:
        first = cv2.resize(first, resize_wh, interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (w, h))

    if not writer.isOpened():
        raise SystemExit(
            f"Failed to open VideoWriter. Try a different --codec (e.g. mp4v or avc1) "
            f"or ensure ffmpeg/gstreamer is available in the container."
        )

    # Write first frame, then the rest
    writer.write(first)

    written = 1
    skipped = 0

    start_index = images.index(first_path) + 1
    for p in images[start_index:]:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1
            continue

        if resize_wh is not None:
            im = cv2.resize(im, re size_wh, interpolation=cv2.INTER_AREA)
        else:
            # Enforce same size as first frame
            if im.shape[1] != w or im.shape[0] != h:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(im)
        written += 1

    writer.release()
    print(f"Saved: {out_path}")
    print(f"Frames written: {written}, skipped: {skipped}, fps: {args.fps}, size: {w}x{h}, codec: {args.codec}")

if __name__ == "__main__":
    main()
