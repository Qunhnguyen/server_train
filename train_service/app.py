# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from threading import Thread

from training_runner import run_train, post_callback

app = FastAPI(title="YOLOv11 Train Service (Local)")

class TrainReq(BaseModel):
    data_yaml: str
    epochs: int = Field(..., ge=1, le=2000)
    batch:  int = Field(..., ge=1, le=1024)
    model_id: int
    dataset_version_id: int
    version_id: int
    callback_url: str
    base_weights: str | None = None      # "yolo11n.pt" | "C:\\..\\best.pt" | None
    imgsz: int | None = 640
    device: str | None = "cuda:0"           # "cpu" | "cuda:0"
    amp: bool | str | None = False       # <<< thêm: tắt AMP để tránh MemoryError
    workers: int | None = 0              # <<< thêm: hạn chế RAM

@app.post("/train")
def start_train(req: TrainReq):
    dy = Path(req.data_yaml)
    if not dy.exists():
        raise HTTPException(400, f"data_yaml not found: {dy}")

    bw = (req.base_weights or None)
    device  = req.device or "cuda:0"
    imgsz   = req.imgsz or 640
    amp     = req.amp if req.amp is not None else False
    workers = req.workers if req.workers is not None else 0

    def _bg():
        res = run_train(
            data_yaml=dy,
            epochs=req.epochs,
            batch=req.batch,
            model_id=req.model_id,
            version_id=req.version_id,
            callback_url=req.callback_url,
            base_weights=bw,
            imgsz=imgsz,
            device=device,
            amp=amp,
            workers=workers,
        )
        payload = {
            "versionId": req.version_id,
            "status":    res.status,                 # "READY" | "FAILED"
            "metrics":   res.metrics,                # {precision, recall, f1} (có thể None)
            "localPath": res.local_path or "",
            "logsUri":   res.logs_uri or "",
            "fileSha256": res.file_sha256 or "",
            "fileSizeBytes": res.file_size_bytes or 0,
            "error":     res.error or ""
        }
        print("[CB URL]", req.callback_url)
        print("[CB PAYLOAD]", payload)
        post_callback(req.callback_url, payload)

    Thread(target=_bg, daemon=True).start()
    return {"status": "STARTED", "version_id": req.version_id}

@app.get("/health")
def health():
    return {"ok": True}
