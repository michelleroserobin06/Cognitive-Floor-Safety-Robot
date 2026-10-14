
from pathlib import Path
from ultralytics import YOLO

# Resolve the dataset configuration from the project structure.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "dataset" / "yolo_dataset" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

# Start with a small pretrained YOLO model.
model = YOLO("yolo11n.pt")

# Train the first baseline model.
model.train(
    data=str(DATA_YAML),
    epochs=5,
    imgsz=640,
    batch=8,
    workers=2,
    project=str(PROJECT_DIR / "runs"),
    name="floor_hazard_baseline",
    patience=10,
    seed=42,
)
 