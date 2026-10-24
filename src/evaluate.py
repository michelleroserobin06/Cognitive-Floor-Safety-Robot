
from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "dataset" / "yolo_dataset" / "data.yaml"

# Change this if your training run has a different folder name.
MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "floor_hazard_baseline-3"
    / "weights"
    / "best.pt"
)

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Check the actual run folder under runs/."
        )

    model = YOLO(str(MODEL_PATH))

    # Evaluate against held-out test images
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # F1-score balances precision and recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print("\n===== FLOOR HAZARD DETECTOR RESULTS =====")
    print(f"Precision:   {precision * 100:.2f}%")
    print(f"Recall:      {recall * 100:.2f}%")
    print(f"F1-score:    {f1 * 100:.2f}%")
    print(f"mAP@50:      {map50 * 100:.2f}%")
    print(f"mAP@50-95:   {map50_95 * 100:.2f}%")

    # Per-class results
    print("\n===== PER-CLASS RESULTS =====")
    for class_id, class_name in model.names.items():
        if class_id < len(metrics.box.ap):
            ap50 = float(metrics.box.ap[class_id])
            print(f"{class_name}: AP@50 = {ap50 * 100:.2f}%")

if __name__ == "__main__":
    main()
