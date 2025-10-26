import os
from pathlib import Path

BASE_DIR   = Path(os.getenv("BASE_DIR", r"C:\ml"))
RUNS_DIR   = Path(os.getenv("RUNS_DIR", BASE_DIR / "runs"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
YOLO_REPO  = Path(os.getenv("YOLO_REPO", r"C:\repos\yolov11"))  # repo đã clone & pip install -e .

for p in (RUNS_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)
