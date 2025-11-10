# server.py
import os, subprocess, json, tempfile, shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import requests

app = FastAPI(title="Sub Detector")

class DetectReq(BaseModel):
    path: str  # public URL (pref signed) or http(s)
    sample_rate_sec: float = None
    max_samples: int = None
    bottom_ratio: float = None

@app.post("/detect")
def detect(req: DetectReq):
    p = req.path
    if not (p.startswith("http://") or p.startswith("https://")):
        raise HTTPException(status_code=400, detail="path must be public http/https URL in Cloud Run test")
    # download to temp
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()
    try:
        with requests.get(p, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        # build command
        cmd = ["python", "detect_sub_region_light.py", tmp_path]
        env = os.environ.copy()
        if req.sample_rate_sec is not None:
            env["DTS_SAMPLE_RATE_SEC"] = str(req.sample_rate_sec)
        if req.max_samples is not None:
            env["DTS_MAX_SAMPLES"] = str(req.max_samples)
        if req.bottom_ratio is not None:
            env["DTS_BOTTOM_RATIO"] = str(req.bottom_ratio)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"detector error: {proc.stderr or proc.stdout}")
        out = proc.stdout.strip()
        try:
            j = json.loads(out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"invalid json from detector: {e}")
        return {"ok": True, "result": j}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass
