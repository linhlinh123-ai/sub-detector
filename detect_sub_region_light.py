#!/usr/bin/env python3
# detect_sub_region_light.py
import sys, os, json, tempfile
from pathlib import Path
import cv2
import numpy as np

def log(msg: str):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()

def bytes_to_tempfile(data: bytes, suffix=".mp4") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.flush()
    f.close()
    return f.name

def union_box(boxes):
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2-x1, y2-y1)

def detect_text_boxes(frame_bgr, min_area=400, morph_w=15, morph_h=3):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, morph_h))
    mor = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area: continue
        if w < 24 or h < 8: continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def sample_and_detect(path, sample_rate_sec=1.0, max_samples=60, search_bottom_ratio=0.35, downscale_width=640, min_area=400):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(sample_rate_sec * fps)))
    boxes_all = []
    frame_idx = 0
    samples = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            bottom_h = int(round(H * search_bottom_ratio))
            crop_y1 = max(0, H - bottom_h)
            roi = frame[crop_y1:H, :, :]
            if downscale_width and W > downscale_width:
                scale = downscale_width / float(W)
                new_w = downscale_width
                new_h = max(1, int(round(roi.shape[0] * scale)))
                roi_small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                boxes_small = detect_text_boxes(roi_small, min_area=max(200, int(min_area * scale * scale)))
                sx = 1.0 / scale
                sy = 1.0 / scale
                boxes_full = [(int(round(x*sx)), int(round(y*sy)), int(round(w*sx)), int(round(h*sy))) for (x,y,w,h) in boxes_small]
            else:
                boxes_full = detect_text_boxes(roi, min_area=min_area)
            boxes_full = [(x, y + crop_y1, w, h) for (x,y,w,h) in boxes_full]
            boxes_all.extend(boxes_full)
            samples += 1
            if samples >= max_samples:
                break
        frame_idx += 1
    cap.release()
    return boxes_all, W, H

def clamp_box(box, W, H, pad=12):
    x,y,w,h = box
    x2 = x+w
    y2 = y+h
    x = max(0, x-pad)
    y = max(0, y-pad)
    x2 = min(W, x2+pad)
    y2 = min(H, y2+pad)
    return (int(x), int(y), int(x2-x), int(y2-y))

def detect_from_path(path):
    boxes, W, H = sample_and_detect(path)
    if not boxes:
        # fallback: try full-frame sample
        boxes, W, H = sample_and_detect(path, search_bottom_ratio=1.0, max_samples=30)
    if not boxes:
        return {"found": False, "video_w": W, "video_h": H}
    uni = union_box(boxes)
    final = clamp_box(uni, W, H, pad=12)
    x,y,w,h = final
    return {"found": True, "x":x, "y":y, "w":w, "h":h, "video_w":W, "video_h":H}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"found": False, "error": "usage: detect_sub_region_light.py <path>"}))
        sys.exit(0)
    path = sys.argv[1]
    res = detect_from_path(path)
    print(json.dumps(res))
