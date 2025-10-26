from pathlib import Path

def build_train_cmd(*, data_yaml, epochs, batch, exp_name, project_dir,
                    base_weights, imgsz, device="cuda:0", amp=False, workers=0):
    cmd = [
        "yolo", "detect", "train",
        f"data={str(data_yaml)}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"project={str(project_dir.resolve())}",
        f"name={exp_name}",
        f"device={device}",
        f"workers={int(workers)}",
        f"amp={str(amp).lower() if isinstance(amp, bool) else amp}",
    ]
    if base_weights:
        cmd.append(f"model={str(base_weights)}")
    else:
        cmd.append("model=yolo11n.pt")
    return cmd

def weights_path(project_dir: Path, exp_name: str) -> Path:
    # chuẩn hoá: C:\ml\runs\mv_0\weights\best.pt
    return project_dir / exp_name / "weights" / "best.pt"