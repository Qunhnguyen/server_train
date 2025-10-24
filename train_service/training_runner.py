# training_runner.py
import subprocess, shutil, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import requests

from config import RUNS_DIR, MODELS_DIR
from utils import parse_metrics, sha256_file

from yolo_adapter import build_train_cmd, weights_path
@dataclass
class TrainResult:
    status: str
    local_path: Optional[str]
    file_sha256: Optional[str]
    file_size_bytes: Optional[int]
    logs_uri: Optional[str]
    metrics: Dict[str, Any]
    error: Optional[str] = None


def _yolo_exe_in_venv() -> Optional[str]:
    # ví dụ: C:\ml\train_service\.venv\Scripts\yolo.exe
    scripts = Path(sys.executable).parent  # ...\.venv\Scripts
    yolo_exe = scripts / ("yolo.exe" if sys.platform.startswith("win") else "yolo")
    return str(yolo_exe) if yolo_exe.exists() else shutil.which("yolo")


def _train_via_cli(*, data_yaml: Path, epochs: int, batch: int,
                   exp_name: str, project_dir: Path, base_weights: Optional[Path],
                   imgsz: int, device: str) -> subprocess.CompletedProcess:
    # build_train_cmd cần hỗ trợ tham số device
    cmd = build_train_cmd(
        data_yaml=data_yaml,
        epochs=epochs,
        batch=batch,
        exp_name=exp_name,
        project_dir=project_dir,
        base_weights=base_weights,
        imgsz=imgsz,
        device=device
    )

    # ép dùng yolo.exe trong venv nếu có
    yolo_path = _yolo_exe_in_venv()
    if cmd and cmd[0] == "yolo" and yolo_path:
        cmd[0] = yolo_path

    print("[TRAIN] CLI:", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def _train_via_api(*, data_yaml: Path, epochs: int, batch: int,
                   exp_name: str, project_dir: Path, base_weights: Optional[Path],
                   imgsz: int, device: str):
    # Không dùng subprocess; gọi trực tiếp Ultralytics API
    from ultralytics import YOLO
    model_path = str(base_weights) if base_weights else "yolov10n.pt"
    print("[TRAIN] API: YOLO(...).train(...) model=", model_path)
    model = YOLO(model_path)
    # Ultralytics sẽ lưu logs/weights vào project/name
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(project_dir),
        name=exp_name,
        device=device,
    )
    # API không trả returncode; trả về object giả lập như CompletedProcess
    class _CP:  # dummy object
        returncode = 0
        stdout = "ultralytics.YOLO API training finished"
        stderr = ""
    return _CP()


def run_train(*, data_yaml: Path, epochs: int, batch: int,
              model_id: int, version_id: int,
              callback_url: str, base_weights: Optional[Path],
              imgsz: int = 640, device: str = "cpu") -> TrainResult:
    exp_name = f"mv_{version_id}"
    project_dir = RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    # 1) Thử CLI (yolo.exe). Nếu không có, 2) dùng Python API
    used_api = False
    try:
        if _yolo_exe_in_venv():
            proc = _train_via_cli(
                data_yaml=data_yaml, epochs=epochs, batch=batch,
                exp_name=exp_name, project_dir=project_dir,
                base_weights=base_weights, imgsz=imgsz, device=device
            )
        else:
            used_api = True
            proc = _train_via_api(
                data_yaml=data_yaml, epochs=epochs, batch=batch,
                exp_name=exp_name, project_dir=project_dir,
                base_weights=base_weights, imgsz=imgsz, device=device
            )
    except Exception as e:
        logs_path = (project_dir / "detect" / exp_name).resolve()
        return TrainResult(
            "FAILED", None, None, None, str(logs_path),
            {"precision": None, "recall": None, "f1": None},
            error=f"Training launch error: {e}"
        )

    print("[TRAIN][RC]", proc.returncode)
    if getattr(proc, "stdout", ""):
        print("[TRAIN][STDOUT]\n", proc.stdout[-4000:])
    if getattr(proc, "stderr", ""):
        print("[TRAIN][STDERR]\n", proc.stderr[-4000:])

    logs_path = (project_dir / "detect" / exp_name).resolve()

    if proc.returncode != 0:
        return TrainResult(
            "FAILED", None, None, None, str(logs_path),
            {"precision": None, "recall": None, "f1": None},
            error=(proc.stderr[-4000:] if getattr(proc, "stderr", "") else "Train failed")
        )

    # lấy best.pt + metrics như cũ
    src_best = weights_path(project_dir, exp_name)
    mets = parse_metrics(project_dir, exp_name)

    dst_dir = MODELS_DIR / f"model_{model_id}" / f"mv_{version_id}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_best = dst_dir / "best.pt"

    if src_best.exists():
        shutil.copy2(src_best, dst_best)
        sha = sha256_file(dst_best)
        size = dst_best.stat().st_size
        return TrainResult(
            "SUCCESS", str(dst_best.resolve()), sha, size,
            str(logs_path), mets
        )
    else:
        return TrainResult(
            "FAILED", None, None, None,
            str(logs_path), mets,
            error=f"best.pt not found: {src_best}"
        )


def post_callback(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("[CALLBACK ERR]", e)
