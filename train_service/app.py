# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from threading import Thread
from typing import List, Dict, Optional, Any

import base64, io, uuid, os, traceback
from PIL import Image
from ultralytics import YOLO

from training_runner import run_train, post_callback

app = FastAPI(title="YOLOv11 Train Service (Local)")

# ========= ENV & HYPERPARAMS =========
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\dev\yolov11_best.pt")
DEVICE     = os.getenv("DEVICE", "cuda:0")
IMG_SIZE   = int(os.getenv("IMG_SIZE", "640"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.50"))
IOU_THRES  = float(os.getenv("IOU_THRES", "0.50"))

# ========= LOAD MODEL (SAFE) =========
print(f"[BOOT] MODEL_PATH={MODEL_PATH}")
print(f"[BOOT] DEVICE(ENV)={DEVICE}")
_yolo_model = YOLO(MODEL_PATH)

def _move_model_safe(model, device: str) -> str:
    try:
        model.to(device)
        print(f"[BOOT] moved model to {device}")
        return device
    except Exception as e:
        print(f"[BOOT] cannot use device '{device}': {e} -> fallback to CPU")
        model.to("cpu")
        return "cpu"

DEVICE = _move_model_safe(_yolo_model, DEVICE)

def _load_class_names(model) -> List[str]:
    names = getattr(model, "names", None)
    if names is None:
        return []
    if isinstance(names, dict):
        try:
            ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        except Exception:
            ordered = list(names.values())
        return [str(x) for x in ordered]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    try:
        return [str(names[i]) for i in range(len(names))]
    except Exception:
        return []

_CLASS_NAMES = _load_class_names(_yolo_model)
print(f"[BOOT] CLASS_NAMES={_CLASS_NAMES}")

# ========= SCHEMAS =========
class FrameImagePayload(BaseModel):
    id: Optional[str] = None
    movieId: Optional[str] = None        # để optional vì phía Java có thể map khác cấp
    frameIndex: int
    timeSec: float
    imageBase64: str                     # base64 thuần (không có 'data:...')
    contentType: Optional[str] = "image/jpeg"

class FrameViolationPayload(BaseModel):
    id: str
    movieId: Optional[str] = ""          # cho phép rỗng để tránh 500 khi thiếu
    frameIndex: int
    timeSec: float
    scores: Dict[str, float]
    imageBase64: Optional[str] = None
    contentType: Optional[str] = "image/jpeg"
    movie: Optional[Dict[str, Any]] = None

class TrainReq(BaseModel):
    data_yaml: str
    epochs: int = Field(..., ge=1, le=2000)
    batch:  int = Field(..., ge=1, le=1024)
    model_id: int
    dataset_version_id: int
    version_id: int
    callback_url: str
    base_weights: str                    # "yolo11n.pt" | "C:\\..\\best.pt" | None
    imgsz: Optional[int] = 640
    device: str                          # "cpu" | "cuda:0"
    amp: Optional[bool | str] = False    # tắt AMP để tránh OOM
    workers: Optional[int] = 0           # hạn chế RAM

# ========= UTILS =========
def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _detections_to_scores(result) -> Dict[str, float]:
    """
    Trả dict {class_name: max_conf} sau khi lọc theo CONF_THRES.
    Không có box cho class nào → 0.0
    """
    scores = {name: 0.0 for name in _CLASS_NAMES}
    boxes = getattr(result, "boxes", None)
    if not boxes:
        return scores

    cls_ids = getattr(boxes, "cls", None)
    confs   = getattr(boxes, "conf", None)
    if cls_ids is None or confs is None:
        return scores

    cls_list  = cls_ids.detach().cpu().numpy().astype(int).tolist()
    conf_list = confs.detach().cpu().numpy().tolist()

    for c, cf in zip(cls_list, conf_list):
        if cf < CONF_THRES:
            continue
        if 0 <= c < len(_CLASS_NAMES):
            name = _CLASS_NAMES[c]
            if cf > scores[name]:
                scores[name] = float(cf)
    return scores

# ========= ENDPOINTS =========
@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "classes": _CLASS_NAMES}

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
            "status":    getattr(res, "status", None),         # "READY" | "FAILED"
            "metrics":   getattr(res, "metrics", None),        # {precision, recall, f1}
            "localPath": getattr(res, "local_path", "") or "",
            "logsUri":   getattr(res, "logs_uri", "") or "",
            "fileSha256": getattr(res, "file_sha256", "") or "",
            "fileSizeBytes": getattr(res, "file_size_bytes", 0) or 0,
            "error":     getattr(res, "error", "") or ""
        }
        print("[CB URL]", req.callback_url)
        print("[CB PAYLOAD]", payload)
        post_callback(req.callback_url, payload)

    Thread(target=_bg, daemon=True).start()
    return {"status": "STARTED", "version_id": req.version_id}

@app.post("/images-classify", response_model=List[FrameViolationPayload])
def images_classify(req: List[FrameImagePayload]):
    if not req:
        raise HTTPException(status_code=400, detail="frames is empty")

    print(f"[REQ] received {len(req)} frames")
    # 1) decode
    try:
        pil_images = []
        for i, f in enumerate(req):
            try:
                pil_images.append(_b64_to_pil(f.imageBase64))
            except Exception as e:
                print(f"[DECODE][{i}] {e}")
                raise HTTPException(400, f"Invalid base64 at index {i}: {e}")
    except HTTPException:
        raise

    # 2) predict (KHÔNG truyền device ở đây; model đã .to() từ trước)
    try:
        results = _yolo_model.predict(
            pil_images,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=True
        )
    except Exception as e:
        print(f"[PREDICT] {e}")
        traceback.print_exc()
        raise HTTPException(500, f"YOLO predict failed: {e}")

    # 3) postprocess
    out: List[FrameViolationPayload] = []
    try:
        for meta, r in zip(req, results):
            scores = _detections_to_scores(r)
            out.append(FrameViolationPayload(
                id=str(uuid.uuid4()),
                movieId=meta.movieId or "",     # ép về "" nếu None
                frameIndex=meta.frameIndex,
                timeSec=meta.timeSec,
                scores=scores
            ))
    except Exception as e:
        print(f"[POSTPROCESS] {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Postprocess failed: {e}")

    return out
