# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import shutil
import tempfile
import base64

from detect_sub_region_light import sample_and_detect

app = FastAPI(title="Sub Detector API")

class DetectReq(BaseModel):
    video_path: Optional[str] = None  # path or URL already accessible by server
    use_ocr: Optional[bool] = False
    max_samples: Optional[int] = 40
    stride_seconds: Optional[float] = 0.5
    downscale: Optional[float] = 0.5
    min_area: Optional[int] = 300
    vote_ratio: Optional[float] = 0.25
    ocr_min_chars: Optional[int] = 1
    ocr_lang: Optional[str] = None

@app.post("/detect")
async def detect(req: DetectReq):
    # If no video_path, return error
    if not req.video_path:
        raise HTTPException(status_code=400, detail="video_path required in JSON body. Alternatively upload file to /detect_upload.")

    # If it's a URL remote, optionally download - but for safety we assume path accessible.
    path = req.video_path
    if path.startswith("http://") or path.startswith("https://"):
        # Download remote file to temp
        import requests
        try:
            r = requests.get(path, stream=True, timeout=60)
            r.raise_for_status()
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(tmpf.name, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            path = tmpf.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download video: {e}")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"video_path not found: {path}")

    try:
        boxes, ocr_texts, W, H, preview_b64 = sample_and_detect(
            path,
            max_samples=int(req.max_samples),
            stride_seconds=float(req.stride_seconds),
            downscale=float(req.downscale),
            min_area=int(req.min_area),
            vote_ratio=float(req.vote_ratio),
            use_ocr=bool(req.use_ocr),
            ocr_min_chars=int(req.ocr_min_chars),
            ocr_lang=req.ocr_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    # Construct selected_boxes as [x,y,w,h] and include text
    selected = []
    for i, b in enumerate(boxes):
        item = {"box": b, "ocr_text": ocr_texts[i] if i < len(ocr_texts) else ""}
        selected.append(item)

    resp = {
        "ok": True,
        "result": {
            "found": len(selected) > 0,
            "video_w": int(W),
            "video_h": int(H),
            "selected_boxes": selected
        }
    }
    if preview_b64:
        resp["result"]["preview_base64"] = preview_b64

    # Clean downloaded temp if used
    if req.video_path.startswith("http://") or req.video_path.startswith("https://"):
        try:
            os.remove(path)
        except Exception:
            pass

    return resp

@app.post("/detect_upload")
async def detect_upload(file: UploadFile = File(...), use_ocr: Optional[bool] = False):
    # save to temp and call detect
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(tmp_path, "wb") as fw:
        shutil.copyfileobj(file.file, fw)
    req = DetectReq(video_path=tmp_path, use_ocr=use_ocr)
    result = await detect(req)
    # cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return result

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
