import csv, hashlib, math
from pathlib import Path
from typing import Dict, Any

def _to_float(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None

def _pick(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] not in ("", None, "nan", "NaN"):
            v = _to_float(d[k])
            if v is not None:
                return v
    return None

def parse_metrics(project_dir: Path, exp_name: str) -> Dict[str, Any]:
    # results.csv có thể nằm ở project/exp_name/ hoặc project/detect/exp_name/
    rcsv = project_dir / exp_name / "results.csv"
    if not rcsv.exists():
        rcsv = project_dir / "detect" / exp_name / "results.csv"

    if not rcsv.exists():
        return {"precision": None, "recall": None, "f1": None}

    rows = list(csv.DictReader(rcsv.open("r", encoding="utf-8")))
    # Duyệt từ cuối lên để lấy dòng có số liệu hợp lệ
    for row in reversed(rows):
        # Các tên cột có thể gặp ở các bản Ultralytics khác nhau
        p = _pick(row, ["metrics/precision(B)", "precision", "P"])
        r = _pick(row, ["metrics/recall(B)",    "recall",    "R"])
        f = _pick(row, ["F1", "metrics/F1(B)"])  # hiếm khi có

        # Nếu không có F1, tự tính khi có P và R
        if f is None and p is not None and r is not None and (p + r) > 0:
            f = 2 * p * r / (p + r)

        if any(v is not None for v in (p, r, f)):
            return {"precision": p, "recall": r, "f1": f}

    return {"precision": None, "recall": None, "f1": None}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
