#!/usr/bin/env python3
"""
detect_sub_region_light.py

Updated version:
- Improved thresholding and morphology to reduce false positives on skin highlights.
- Cluster boxes conservatively and select subtitle-like boxes only in lower portion of frame.
- Optionally write preview image and return JSON to stdout.
"""

import sys
import os
import json
import math
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

def log(msg: str):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

def bytes_to_tempfile(data: bytes, suffix=".mp4") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.flush()
    f.close()
    return f.name

def detect_text_boxes(frame_bgr: np.ndarray,
                      min_area: int = 400,
                      morph_w: int = 9,
                      morph_h: int = 3) -> List[Tuple[int,int,int,int]]:
    """
    Detect probable text-contours in a single frame (BGR).
    Returns list of bounding boxes (x,y,w,h) in frame coordinates.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: use MEAN to be a bit more robust on gradients/skin highlights
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 19, 11)

    # Slight erosion first to remove tiny speckles, then close/open to join small pieces of text
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    th = cv2.erode(th, small_kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3,morph_w//2), max(1,morph_h//2)))
    mor = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area:
            continue
        if w < 16 or h < 6:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def scale_boxes(boxes, sx, sy):
    out = []
    for (x,y,w,h) in boxes:
        out.append((int(round(x * sx)), int(round(y * sy)),
                    int(round(w * sx)), int(round(h * sy))))
    return out

def sample_and_detect(path: str,
                      sample_rate_sec: float = 1.0,
                      max_samples: int = 60,
                      search_bottom_ratio: float = 0.35,
                      downscale_width: int = 640,
                      min_area: int = 400,
                      morph_w: int = 9,
                      morph_h: int = 3) -> Tuple[List[Tuple[int,int,int,int]], int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W == 0 or H == 0:
        cap.release()
        raise RuntimeError("Failed to read video dimensions")
    step = max(1, int(round(sample_rate_sec * fps)))

    boxes_all = []
    frame_idx = 0
    samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # crop bottom region optionally
            bottom_h = int(round(H * search_bottom_ratio))
            crop_y1 = max(0, H - bottom_h) if search_bottom_ratio < 1.0 else 0
            roi = frame[crop_y1:H, :, :]

            # downscale for speed
            if downscale_width and W > downscale_width:
                scale = downscale_width / float(W)
                new_w = downscale_width
                new_h = max(1, int(round(roi.shape[0] * scale)))
                roi_small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                boxes_small = detect_text_boxes(roi_small,
                                                min_area=max(200, int(min_area * scale * scale)),
                                                morph_w=max(3, int(morph_w*scale)),
                                                morph_h=max(1, int(morph_h*scale)))
                sx = 1.0 / scale
                boxes_full = scale_boxes(boxes_small, sx, sx)
            else:
                boxes_full = detect_text_boxes(roi, min_area=min_area, morph_w=morph_w, morph_h=morph_h)

            # shift boxes to full frame coords
            boxes_full = [(x, y + crop_y1, w, h) for (x,y,w,h) in boxes_full]
            boxes_all.extend(boxes_full)

            samples += 1
            if samples >= max_samples:
                break
        frame_idx += 1

    cap.release()
    return boxes_all, W, H

# clustering helpers
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB - xA) * (yB - yA)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter/union

def center_dist(boxA, boxB):
    ax = boxA[0] + boxA[2]/2.0
    ay = boxA[1] + boxA[3]/2.0
    bx = boxB[0] + boxB[2]/2.0
    by = boxB[1] + boxB[3]/2.0
    return math.hypot(ax-bx, ay-by)

def cluster_boxes(boxes: List[Tuple[int,int,int,int]], img_w:int, img_h:int,
                  iou_thresh=0.12, dist_ratio=0.12) -> List[Tuple[int,int,int,int]]:
    if not boxes:
        return []
    diag = math.hypot(img_w, img_h)
    used = [False]*len(boxes)
    merged = []
    for i,b in enumerate(boxes):
        if used[i]: continue
        cluster = [b]
        used[i]=True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if iou(b, boxes[j]) > iou_thresh or center_dist(b, boxes[j]) < dist_ratio*diag:
                cluster.append(boxes[j]); used[j]=True
        x1 = min([c[0] for c in cluster])
        y1 = min([c[1] for c in cluster])
        x2 = max([c[0]+c[2] for c in cluster])
        y2 = max([c[1]+c[3] for c in cluster])
        merged.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
    return merged

def is_sub_like(box, W, H):
    x,y,w,h = box
    ar = w / (h + 1e-6)
    # Heuristic: subtitles are wide, in lower area, not too tall, and not hugging top/mid face
    return (ar > 2.2 or w > W*0.45) and (y > H*0.45) and (y + h < H * 0.95)

def clamp_box(box, W, H, pad=12):
    x,y,w,h = box
    x = max(0, x-pad)
    y = max(0, y-pad)
    x2 = min(W, x+w+pad)
    y2 = min(H, y+h+pad)
    return (int(x), int(y), int(x2-x), int(y2-y))

def write_preview_frame(path: str, merged, selected, out_preview: str = None):
    try:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        for (x,y,w,h) in merged:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 4)
        for (x,y,w,h) in selected:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 4)
        if not out_preview:
            p = Path(path).with_suffix(".preview.jpg")
            out_preview = str(p)
        cv2.imwrite(out_preview, frame)
        return out_preview
    except Exception as e:
        return None

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"found": False, "error": "usage: detect_sub_region_light.py <input.mp4 | ->"}))
        sys.exit(0)

    arg = sys.argv[1]
    temp_to_delete = None
    try:
        if arg == "-":
            data = sys.stdin.buffer.read()
            if not data or len(data) < 1024:
                print(json.dumps({"found": False, "error": "no stdin data"}))
                return
            temp_to_delete = bytes_to_tempfile(data, suffix=".mp4")
            path = temp_to_delete
        else:
            path = arg

        # params via env
        sample_rate_sec = float(os.environ.get("DTS_SAMPLE_RATE_SEC", "1.0"))
        max_samples = int(os.environ.get("DTS_MAX_SAMPLES", "60"))
        search_bottom_ratio = float(os.environ.get("DTS_BOTTOM_RATIO", "0.35"))
        downscale_width = int(os.environ.get("DTS_DOWNSCALE_WIDTH", "640"))
        min_area = int(os.environ.get("DTS_MIN_AREA", "400"))
        morph_w = int(os.environ.get("DTS_MORPH_W", "9"))
        morph_h = int(os.environ.get("DTS_MORPH_H", "3"))
        iou_thresh = float(os.environ.get("DTS_IOU_THRESH", "0.12"))
        dist_ratio = float(os.environ.get("DTS_DIST_RATIO", "0.12"))
        write_preview = os.environ.get("DTS_WRITE_PREVIEW", "0") == "1"

        boxes_all, W, H = sample_and_detect(path,
                                            sample_rate_sec=sample_rate_sec,
                                            max_samples=max_samples,
                                            search_bottom_ratio=search_bottom_ratio,
                                            downscale_width=downscale_width,
                                            min_area=min_area,
                                            morph_w=morph_w,
                                            morph_h=morph_h)

        # fallback if nothing found: expand search_bottom_ratio
        if not boxes_all:
            boxes_all, W, H = sample_and_detect(path,
                                                sample_rate_sec=sample_rate_sec,
                                                max_samples=max(10, max_samples//2),
                                                search_bottom_ratio=1.0,
                                                downscale_width=downscale_width,
                                                min_area=max(200, min_area//2),
                                                morph_w=morph_w,
                                                morph_h=morph_h)

        merged = cluster_boxes(boxes_all, W, H, iou_thresh=iou_thresh, dist_ratio=dist_ratio)
        selected = [clamp_box(b, W, H, pad=12) for b in merged if is_sub_like(b, W, H)]
        if not selected and merged:
            merged_sorted = sorted(merged, key=lambda b: b[2]*b[3], reverse=True)
            selected = [clamp_box(merged_sorted[0], W, H, pad=12)]

        result = {
            "found": bool(selected),
            "video_w": int(W), "video_h": int(H),
            "all_boxes": boxes_all,
            "merged_boxes": merged,
            "selected_boxes": selected,
            "params": {
                "sample_rate_sec": sample_rate_sec,
                "max_samples": max_samples,
                "bottom_ratio": search_bottom_ratio,
                "downscale_width": downscale_width,
                "min_area": min_area,
                "morph_w": morph_w,
                "morph_h": morph_h,
                "iou_thresh": iou_thresh,
                "dist_ratio": dist_ratio
            }
        }

        preview_path = None
        if write_preview:
            preview_path = write_preview_frame(path, merged, selected)
            if preview_path:
                result["preview_path"] = preview_path

        print(json.dumps(result, ensure_ascii=False))
    finally:
        if temp_to_delete and os.path.exists(temp_to_delete):
            try:
                os.unlink(temp_to_delete)
            except Exception:
                pass

if __name__ == "__main__":
    main()
