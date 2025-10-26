cd C:\ml\train_service
.\.venv\Scripts\Activate.ps1


uvicorn app:app --host 0.0.0.0 --port 8093 --reload

$body = @{
  data_yaml = "C:\data\data.yaml"
  model_id  = 1
  dataset_version_id = 0
  version_id = 0
  job_id = [guid]::NewGuid().ToString()
  epochs = 150
  batch  = 16
  base_weights = "yolo11n.pt"   # <- nano, nhanh nhất
  imgsz  = 640
  device = "cuda:0"
  project = ".\runs"
  name = "mv_0"
  cache = $true
  workers = 8
  callback_url = "http://localhost:8099/api/train/callback"
} | ConvertTo-Json -Depth 6
