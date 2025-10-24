# app.py (trích)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from threading import Thread
from training_runner import run_train, post_callback

app = FastAPI(title="YOLOv11 Train Service (Local)")

class TrainReq(BaseModel):
    data_yaml: str
    epochs: int = Field(..., ge=1, le=2000)
    batch: int  = Field(..., ge=1, le=1024)
    model_id: int
    dataset_version_id: int
    version_id: int
    callback_url: str
    base_weights: str | None = None    # cho phép tên model (yolo11l.pt) hoặc đường dẫn
    imgsz: int | None = 640
    device: str 

@app.post("/train")
def start_train(req: TrainReq):
    dy = Path(req.data_yaml)
    if not dy.exists():
        raise HTTPException(400, f"data_yaml not found: {dy}")

    # KHÔNG ép base_weights phải tồn tại: cho phép tên model để Ultralytics tự tải
    bw = req.base_weights  # có thể là "yolo11l.pt" hoặc "C:\\path\\to\\file.pt" hoặc None

    def _bg():
        res = run_train(
            data_yaml=dy, epochs=req.epochs, batch=req.batch,
            model_id=req.model_id, version_id=req.version_id,
            callback_url=req.callback_url, base_weights=bw,   # truyền nguyên chuỗi xuống
            imgsz=req.imgsz or 640, device=req.device or "cpu"
        )
        payload = {
            "versionId": req.version_id,
            "status": res.status,
            "metrics": res.metrics,
            "localPath": res.local_path,
            "fileSha256": res.file_sha256,
            "fileSizeBytes": res.file_size_bytes,
            "logsUri": res.logs_uri,
            "error": res.error
        }
        post_callback(req.callback_url, payload)

    Thread(target=_bg, daemon=True).start()
    return {"status":"STARTED","version_id":req.version_id}
