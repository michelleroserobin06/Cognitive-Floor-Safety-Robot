
from pathlib import Path
import cv2
from ultralytics import YOLO

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"

# Find trained model checkpoints
checkpoints = list(RUNS_DIR.rglob("best.pt"))

if not checkpoints:
    raise FileNotFoundError(
        "No trained best.pt model found under runs/. "
        "Make sure training has completed successfully."
    )

# Use the most recently modified checkpoint
MODEL_PATH = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"Loading model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Open default webcam (usually camera 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera. Check camera permissions "
        "and close other apps using the camera."
    )

# Optional: request a camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Live detection started. Press Q to quit.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a camera frame.")
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        # Draw boxes, labels and confidence scores
        annotated_frame = results[0].plot()

        cv2.imshow("Floor Hazard Detection - Press Q to quit",
                   annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
