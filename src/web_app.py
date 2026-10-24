
from pathlib import Path
from threading import Thread, Lock
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"

# Find the newest trained checkpoint.
checkpoints = list(RUNS_DIR.rglob("best.pt"))

if not checkpoints:
    raise FileNotFoundError(
        "No best.pt found under runs/. Train your model first."
    )

MODEL_PATH = max(checkpoints, key=lambda p: p.stat().st_mtime)
print(f"Loading trained model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
app = Flask(__name__)

lock = Lock()
latest_frame = None
latest_detections = []
camera_status = "Starting camera"
last_update = None

# These are prototype triage rules, not medically validated risk ratings.
# Names must match the classes in your data.yaml.
RISK_RULES = {
    "safetypin": {
        "level": "HIGH",
        "reason": "Small, pointed object; potential puncture or choking hazard."
    },
    "earring": {
        "level": "HIGH",
        "reason": "Small object; potential choking or puncture hazard depending on design."
    },
    "hair clip": {
        "level": "MEDIUM",
        "reason": "Check for sharp edges and small detachable parts."
    },
    "hair slide": {
        "level": "MEDIUM",
        "reason": "Check for pointed edges and small detachable parts."
    },
    "hair bundle": {
        "level": "REVIEW",
        "reason": "Object needs manual inspection; class alone cannot establish risk."
    },
    "paper ball": {
        "level": "REVIEW",
        "reason": "Risk depends on size, material, and whether it can be mouthed."
    },
    "toy": {
        "level": "REVIEW",
        "reason": "Inspect for small detachable parts; not all toys have the same risk."
    }
}


def camera_loop():
    global latest_frame, latest_detections
    global camera_status, last_update

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        camera_status = "Camera unavailable"
        print("ERROR: Could not open camera.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_status = "Live"
    print("Camera opened. Detection is running.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                camera_status = "Camera frame error"
                time.sleep(0.1)
                continue

            # Higher resolution can help with small objects but may be slower.
            result = model.predict(
                source=frame,
                imgsz=960,
                conf=0.20,
                verbose=False
            )[0]

            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [
                    int(v) for v in box.xyxy[0].tolist()
                ]

                label = str(model.names[class_id]).strip().lower()

                rule = RISK_RULES.get(label, {
                    "level": "REVIEW",
                    "reason": "Unclassified object; inspect it manually."
                })

                detections.append({
                    "label": model.names[class_id],
                    "confidence": round(confidence * 100, 1),
                    "box": [x1, y1, x2, y2],
                    "box_pixels": [
                        max(0, x2 - x1),
                        max(0, y2 - y1)
                    ],
                    "risk": rule["level"],
                    "reason": rule["reason"]
                })

            # Draw YOLO's boxes and labels on the live image.
            annotated = result.plot()
            encoded_ok, buffer = cv2.imencode(".jpg", annotated)

            if encoded_ok:
                with lock:
                    latest_frame = buffer.tobytes()
                    latest_detections = detections
                    last_update = time.time()
                    camera_status = "Live"

    except Exception as exc:
        camera_status = "Detection error"
        print(f"Detection loop error: {exc}")

    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                frame = latest_frame

            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame + b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/detections")
def get_detections():
    with lock:
        return jsonify({
            "status": camera_status,
            "detections": list(latest_detections),
            "updated_at": last_update
        })


if __name__ == "__main__":
    Thread(target=camera_loop, daemon=True).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
