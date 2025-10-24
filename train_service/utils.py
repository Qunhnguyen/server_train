import csv, hashlib
from pathlib import Path
from typing import Dict, Any

def parse_metrics(project_dir: Path, exp_name: str) -> Dict[str, Any]:
    rcsv = project_dir / "detect" / exp_name / "results.csv"
    if rcsv.exists():
        rows = list(csv.DictReader(rcsv.open("r", encoding="utf-8")))
        if rows:
            last = rows[-1]
            prec = float(last.get("P") or last.get("precision") or 0)
            rec  = float(last.get("R") or last.get("recall") or 0)
            f1   = float(last.get("F1") or 0)
            return {"precision": prec, "recall": rec, "f1": f1}
    return {"precision": None, "recall": None, "f1": None}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
