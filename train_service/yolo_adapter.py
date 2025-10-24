from pathlib import Path

def build_train_cmd(*, data_yaml, epochs, batch, exp_name, project_dir,
                    base_weights, imgsz, device="cpu"):
    cmd = [
        "yolo", "detect", "train",
        f"data={str(data_yaml)}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"project={str(project_dir.resolve())}",
        f"name={exp_name}",
        f"device={device}",
    ]
    # Nếu truyền base_weights (đường dẫn .pt/.yaml hoặc tên model), dùng cái đó
    if base_weights:
        cmd.append(f"model={str(base_weights)}")
    else:
        # Mặc định dùng YOLOv11 bản nhỏ
        cmd.append("model=yolo11n.pt")
    return cmd

def weights_path(project_dir: Path, exp_name: str) -> Path:
    # Ultralytics: runs/detect/<exp>/weights/best.pt
    return project_dir / "detect" / exp_name / "weights" / "best.pt"