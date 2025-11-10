# detect_sub_region_light.py
"""
Lightweight subtitle/text region detector with optional Tesseract OCR.
- sample_and_detect(video_path, use_ocr=False, ...)
Returns:
    boxes: list of [x,y,w,h]
    ocr_texts: list of corresponding OCR strings ("" if none)
    video_w, video_h
    preview_b64: optional base64 PNG preview showing boxes (for debugging)
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import pytesseract
import math

# -----------------------
# Helper utilities
# -----------------------
def read_video_size(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def frame_iterator(path, max_samples=80, stride_seconds=0.5):
    """Yield sampled frames (BGR) from video path."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for sampling.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / (fps if fps>0 else 25.0)
    # sample up to max_samples uniformly using stride_seconds where possible
    step = max(1, int(round((stride_seconds * fps))))
    idx = 0
    sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield frame
            sampled += 1
            if sampled >= max_samples:
                break
        idx += 1
    cap.release()

def boxes_from_mask(mask, min_area=300):
    """
    Convert binary mask to list of bounding boxes [x,y,w,h].
    mask: uint8 0/255
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area:
            continue
        boxes.append([x,y,w,h])
    # optionally merge overlapping boxes
    boxes = merge_boxes(boxes)
    return boxes

def merge_boxes(boxes, iou_thresh=0.2):
    """Simple greedy merge for overlapping boxes."""
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    merged = []
    used = [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1,y1,w1,h1 = boxes[i]
        rx1,ry1,rx2,ry2 = x1,y1,x1+w1,y1+h1
        changed = True
        while changed:
            changed = False
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                x2,y2,w2,h2 = boxes[j]
                sx1,sy1,sx2,sy2 = x2,y2,x2+w2,y2+h2
                iou = rect_iou((rx1,ry1,rx2,ry2), (sx1,sy1,sx2,sy2))
                if iou > iou_thresh or rect_overlap((rx1,ry1,rx2,ry2),(sx1,sy1,sx2,sy2)):
                    # merge
                    rx1 = min(rx1, sx1)
                    ry1 = min(ry1, sy1)
                    rx2 = max(rx2, sx2)
                    ry2 = max(ry2, sy2)
                    used[j] = True
                    changed = True
        merged.append([rx1, ry1, rx2-rx1, ry2-ry1])
    return merged

def rect_iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
    inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2-inter_x1)*(inter_y2-inter_y1)
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter_area / float(area_a + area_b - inter_area + 1e-9)

def rect_overlap(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def ensure_padding(box, W, H, pad=8):
    x,y,w,h = box
    x = max(0, x-pad)
    y = max(0, y-pad)
    w = min(W - x, w + 2*pad)
    h = min(H - y, h + 2*pad)
    return [x,y,w,h]

def pil_preview_with_boxes(bgr_frame, boxes):
    # convert to PIL and draw rectangles
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, 'RGBA')
    for b in boxes:
        x,y,w,h = b
        draw.rectangle([x, y, x+w, y+h], outline=(255,0,0,200), width=5)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return b64

# -----------------------
# OCR helper
# -----------------------
def ocr_confirms_text(bgr_image, box, min_chars=1, psm=7, lang=None):
    """
    Run Tesseract on crop; return (is_text, text)
    psm: page segmentation mode (7 usually single line)
    lang: optional tesseract language code
    """
    x,y,w,h = box
    H, W = bgr_image.shape[:2]
    if x<0 or y<0 or x>=W or y>=H:
        return False, ""
    crop = bgr_image[y:y+h, x:x+w]
    if crop.size == 0:
        return False, ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # upscale small crops
    if max(gray.shape) < 100:
        factor = int(math.ceil(100.0 / max(gray.shape)))
        gray = cv2.resize(gray, (gray.shape[1]*factor, gray.shape[0]*factor), interpolation=cv2.INTER_CUBIC)
    # basic threshold to help OCR
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ "
    try:
        txt = pytesseract.image_to_string(t, config=config, lang=lang or 'eng')
    except Exception:
        txt = ""
    txt = txt.strip()
    if len(txt) >= min_chars:
        return True, txt
    return False, ""

# -----------------------
# Main detection pipeline
# -----------------------
def detect_text_candidates_on_frame(bgr, downscale=1.0, detect_yellow=False, detect_blue=False):
    """
    Detect bright text-like regions on a single frame.
    Returns binary mask (uint8 0/255) same size as original bgr image.
    """
    H, W = bgr.shape[:2]
    # optional downscale to speed up
    if downscale != 1.0:
        small = cv2.resize(bgr, (int(W*downscale), int(H*downscale)), interpolation=cv2.INTER_AREA)
    else:
        small = bgr.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to pick bright text on variable background
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 25, -10)
    # morphological ops to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    # bring back to original size if scaled
    if downscale != 1.0:
        thr = cv2.resize(thr, (W, H), interpolation=cv2.INTER_NEAREST)
    # If color-specific detection needed, detect yellow/blue text
    color_mask = np.zeros_like(thr)
    if detect_yellow or detect_blue:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if detect_yellow:
            lower = np.array([15, 60, 150]); upper = np.array([40, 255, 255])
            color_mask |= cv2.inRange(hsv, lower, upper)
        if detect_blue:
            lower = np.array([90, 50, 50]); upper = np.array([140, 255, 255])
            color_mask |= cv2.inRange(hsv, lower, upper)
    combined = cv2.bitwise_or(thr, color_mask)
    # final morphology: dilate to merge letters into lines
    dil = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (9,3)), iterations=1)
    # ensure binary 0/255
    _, out = cv2.threshold(dil, 127, 255, cv2.THRESH_BINARY)
    return out

def sample_and_detect(video_path, max_samples=40, stride_seconds=0.5, downscale=0.5,
                      min_area=300, vote_ratio=0.25, use_ocr=False, ocr_min_chars=1, ocr_lang=None):
    """
    Main entry point for detection.
    - sample frames -> per-frame mask -> accumulate votes -> threshold votes -> get boxes
    Returns: boxes, ocr_texts, video_w, video_h, preview_b64
    """
    W, H = read_video_size(video_path)
    vote_map = np.zeros((H, W), dtype=np.uint16)
    sampled = 0
    last_frame = None

    for frame in frame_iterator(video_path, max_samples=max_samples, stride_seconds=stride_seconds):
        last_frame = frame.copy()
        mask = detect_text_candidates_on_frame(frame, downscale=downscale)
        # morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        # accumulate votes
        vote_map += (mask > 0).astype(np.uint16)
        sampled += 1

    if sampled == 0:
        return [], [], W, H, None

    # threshold votes
    thr_votes = int(math.ceil(sampled * vote_ratio))
    agg_mask = (vote_map >= thr_votes).astype('uint8') * 255
    # extra dilate to join text lines
    agg_mask = cv2.dilate(agg_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9,3)), iterations=1)

    # get boxes
    boxes = boxes_from_mask(agg_mask, min_area=min_area)
    # pad boxes a bit and clamp
    boxes = [ensure_padding(b, W, H, pad=8) for b in boxes]

    # optional OCR verification (run only on boxes to save time)
    ocr_texts = []
    valid_boxes = []
    if use_ocr and last_frame is not None:
        for b in boxes:
            is_text, txt = ocr_confirms_text(last_frame, b, min_chars=ocr_min_chars, lang=ocr_lang)
            if is_text:
                valid_boxes.append(b)
                ocr_texts.append(txt)
    else:
        valid_boxes = boxes
        ocr_texts = [""] * len(valid_boxes)

    # preview base64
    preview_b64 = None
    if last_frame is not None and len(valid_boxes) > 0:
        preview_b64 = pil_preview_with_boxes(last_frame, valid_boxes)

    return valid_boxes, ocr_texts, W, H, preview_b64

# If run directly, simple CLI
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--use-ocr", action="store_true")
    args = p.parse_args()
    boxes, texts, W, H, prev = sample_and_detect(args.video, use_ocr=args.use_ocr)
    out = {"boxes": boxes, "ocr_texts": texts, "video_w": W, "video_h": H}
    print(json.dumps(out, ensure_ascii=False, indent=2))
