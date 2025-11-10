# server.py
import os
import subprocess
import json
import tempfile
import requests
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Sub-Detector API")

class DetectReq(BaseModel):
    path: str  # public URL (recommended) or local path accessible in container
    sample_rate_sec: Optional[float] = None
    max_samples: Optional[int] = None
    bottom_ratio: Optional[float] = None
    downscale_width: Optional[int] = None
    min_area: Optional[int] = None
    morph_w: Optional[int] = None
    morph_h: Optional[int] = None
    iou_thresh: Optional[float] = None
    dist_ratio: Optional[float] = None
    write_preview: Optional[bool] = False

def download_to_temp(url: str) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tf.name
    tf.close()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    return tmp_path

@app.post("/detect")
def detect(req: DetectReq):
    p = req.path
    cleanup_download = False
    local_path = p
    try:
        if p.startswith("http://") or p.startswith("https://"):
            local_path = download_to_temp(p)
            cleanup_download = True

        # build command
        cmd = ["python", "detect_sub_region_light.py", local_path]

        # env overrides
        env = os.environ.copy()
        if req.sample_rate_sec is not None:
            env["DTS_SAMPLE_RATE_SEC"] = str(req.sample_rate_sec)
        if req.max_samples is not None:
            env["DTS_MAX_SAMPLES"] = str(req.max_samples)
        if req.bottom_ratio is not None:
            env["DTS_BOTTOM_RATIO"] = str(req.bottom_ratio)
        if req.downscale_width is not None:
            env["DTS_DOWNSCALE_WIDTH"] = str(req.downscale_width)
        if req.min_area is not None:
            env["DTS_MIN_AREA"] = str(req.min_area)
        if req.morph_w is not None:
            env["DTS_MORPH_W"] = str(req.morph_w)
        if req.morph_h is not None:
            env["DTS_MORPH_H"] = str(req.morph_h)
        if req.iou_thresh is not None:
            env["DTS_IOU_THRESH"] = str(req.iou_thresh)
        if req.dist_ratio is not None:
            env["DTS_DIST_RATIO"] = str(req.dist_ratio)
        if req.write_preview:
            env["DTS_WRITE_PREVIEW"] = "1"

        proc = subprocess.run(cmd, capture_output=True, timeout=300, env=env, text=True)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail={"stderr": proc.stderr, "stdout": proc.stdout})

        out = proc.stdout.strip()
        try:
            j = json.loads(out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"invalid json from detector: {e}. raw={out}")

        # attach preview as base64 if exists
        preview_b64 = None
        if req.write_preview and j.get("preview_path"):
            preview_path = j.get("preview_path")
            try:
                with open(preview_path, "rb") as f:
                    preview_b64 = base64.b64encode(f.read()).decode("ascii")
                # optional: remove preview file
                try:
                    Path(preview_path).unlink()
                except:
                    pass
            except Exception as e:
                preview_b64 = None

        result = {"ok": True, "result": j}
        if preview_b64:
            result["preview_b64"] = preview_b64

        return result

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="detector timeout")
    finally:
        if cleanup_download and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except:
                pass
