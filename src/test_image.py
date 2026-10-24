
from pathlib import Path
import sys
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"

# Find trained best.pt checkpoints
checkpoints = list(RUNS_DIR.rglob("best.pt"))
if not checkpoints:
    raise FileNotFoundError(
        f"No best.pt found under {RUNS_DIR}. "
        "Check that training completed successfully."
    )

# Use the most recently modified best.pt
MODEL_PATH = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"Loading model: {MODEL_PATH}")

if len(sys.argv) < 2:
    print('Usage: python src\\test_image.py "path\\to\\image.jpg"')
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(image_path),
    imgsz=640,
    conf=0.15,
    save=True,
)

for result in results:
    print("\nDetections:")
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"{model.names[class_id]}: {confidence:.1%} confidence; "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

print("\nAnnotated image saved in the runs/detect output folder.")
