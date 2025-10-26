# training_runner.py
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from config import RUNS_DIR, MODELS_DIR
from utils import parse_metrics, sha256_file
from yolo_adapter import build_train_cmd, weights_path


@dataclass
class TrainResult:
    status: str                      # "READY" | "FAILED"
    local_path: Optional[str]        # đ/c best.pt (đã copy về thư mục model_n/mv_m)
    file_sha256: Optional[str]
    file_size_bytes: Optional[int]
    logs_uri: Optional[str]          # file log
    metrics: Dict[str, Any]          # {"precision":..., "recall":..., "f1":...}
    error: Optional[str] = None


def _yolo_exe_in_venv() -> Optional[str]:
    scripts = Path(sys.executable).parent  # ...\.venv\Scripts
    yolo_exe = scripts / ("yolo.exe" if sys.platform.startswith("win") else "yolo")
    return str(yolo_exe) if yolo_exe.exists() else shutil.which("yolo")


def _train_via_cli(
    *,
    data_yaml: Path,
    epochs: int,
    batch: int,
    exp_name: str,
    project_dir: Path,
    base_weights: Optional[str],
    imgsz: int,
    device: str,
    amp: bool | str = False,          # <<< thêm
    workers: int = 0,                 # <<< thêm
) -> subprocess.CompletedProcess:
    cmd = build_train_cmd(
        data_yaml=data_yaml,
        epochs=epochs,
        batch=batch,
        exp_name=exp_name,
        project_dir=project_dir,
        base_weights=base_weights,
        imgsz=imgsz,
        device=device,
        amp=amp,                      # <<< truyền xuống adapter
        workers=workers,              # <<< truyền xuống adapter
    )

    yolo_path = _yolo_exe_in_venv()
    if cmd and cmd[0] == "yolo" and yolo_path:
        cmd[0] = yolo_path

    print("[TRAIN] CLI:", " ".join(str(x) for x in cmd))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )


def _train_via_api(
    *,
    data_yaml: Path,
    epochs: int,
    batch: int,
    exp_name: str,
    project_dir: Path,
    base_weights: Optional[str],
    imgsz: int,
    device: str,
    amp: bool | str = False,          # <<< thêm
    workers: int = 0,                 # <<< thêm
):
    from ultralytics import YOLO

    model_path = base_weights or "yolo11n.pt"
    print("[TRAIN] API: YOLO(...).train(...) model =", model_path)
    model = YOLO(model_path)
    _ = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(project_dir),
        name=exp_name,
        device=device,
        workers=workers,              # <<< quan trọng
        amp=amp,                      # <<< quan trọng: tắt auto-AMP check
    )

    class _CP:
        returncode = 0
        stdout = "ultralytics.YOLO API training finished"
        stderr = ""

    return _CP()


def _pick_logs_dir(project_dir: Path, exp_name: str) -> Path:
    """
    Chịu 2 layout:
      1) runs/exp_name/...
      2) runs/detect/exp_name/...
    """
    d1 = (project_dir / exp_name)
    return d1 


def run_train(
    *,
    data_yaml: Path,
    epochs: int,
    batch: int,
    model_id: int,
    version_id: int,
    callback_url: str,
    base_weights: Optional[str],
    imgsz: int = 640,
    device: str = "cuda:0",
    amp: bool | str = False,          # <<< thêm
    workers: int = 0,                 # <<< thêm
) -> TrainResult:
    exp_name = f"mv_{version_id}"
    project_dir = RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        if _yolo_exe_in_venv():
            proc = _train_via_cli(
                data_yaml=data_yaml,
                epochs=epochs,
                batch=batch,
                exp_name=exp_name,
                project_dir=project_dir,
                base_weights=base_weights,
                imgsz=imgsz,
                device=device,
                amp=amp,
                workers=workers,
            )
        else:
            proc = _train_via_api(
                data_yaml=data_yaml,
                epochs=epochs,
                batch=batch,
                exp_name=exp_name,
                project_dir=project_dir,
                base_weights=base_weights,
                imgsz=imgsz,
                device=device,
                amp=amp,
                workers=workers,
            )
    except Exception as e:
        logs_dir = _pick_logs_dir(project_dir, exp_name).resolve()
        return TrainResult(
            status="FAILED",
            local_path=None,
            file_sha256=None,
            file_size_bytes=None,
            logs_uri=str(logs_dir),
            metrics={"precision": None, "recall": None, "f1": None},
            error=f"Training launch error: {e}",
        )

    print("[TRAIN][RC]", proc.returncode)
    if getattr(proc, "stdout", ""):
        print("[TRAIN][STDOUT]\n", proc.stdout[-4000:])
    if getattr(proc, "stderr", ""):
        print("[TRAIN][STDERR]\n", proc.stderr[-4000:])

    # Ghi log ra file
    logs_dir = _pick_logs_dir(project_dir, exp_name).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "train.log"
    try:
        content = ""
        if getattr(proc, "stdout", None):
            content += proc.stdout
        if getattr(proc, "stderr", None):
            content += "\n--- STDERR ---\n" + proc.stderr
        log_file.write_text(content, encoding="utf-8", errors="replace")
    except Exception as e:
        print("[TRAIN] write log error:", e)

    if proc.returncode != 0:
        return TrainResult(
            status="FAILED",
            local_path=None,
            file_sha256=None,
            file_size_bytes=None,
            logs_uri=str(log_file),
            metrics={"precision": None, "recall": None, "f1": None},
            error=(proc.stderr[-4000:] if getattr(proc, "stderr", "") else "Train failed"),
        )

    # === Thành công: lấy best.pt & metrics ===
    src_best = weights_path(project_dir, exp_name)       # adapter sẽ dò đúng đường
    mets = parse_metrics(project_dir, exp_name)          # chịu nhiều format results.csv

    dst_dir = MODELS_DIR / f"model_{model_id}" / f"mv_{version_id}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_best = dst_dir / "best.pt"

    if src_best.exists():
        shutil.copy2(src_best, dst_best)
        sha = sha256_file(dst_best)
        size = dst_best.stat().st_size
        return TrainResult(
            status="READY",
            local_path=str(dst_best.resolve()),
            file_sha256=sha,
            file_size_bytes=size,
            logs_uri=str(log_file),
            metrics=mets,
        )
    else:
        return TrainResult(
            status="FAILED",
            local_path=None,
            file_sha256=None,
            file_size_bytes=None,
            logs_uri=str(log_file),
            metrics=mets,
            error=f"best.pt not found: {src_best}",
        )


def post_callback(url: str, payload: dict):
    try:
        r = requests.post(url, json=payload, timeout=15)
        print("[CALLBACK]", r.status_code, r.text)
    except Exception as e:
        print("[CALLBACK ERR]", e)
