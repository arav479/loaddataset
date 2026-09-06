#!/usr/bin/env python3
"""
Simple & High-Accuracy Training Script for YOLO11s on Web PII Dataset.
Model weights: D:/models/yolo11s/yolo11s.pt
"""

from pathlib import Path
from ultralytics import YOLO

# 1. Paths configuration
MODEL_PATH = r"D:\models\yolo11s\yolo11s.pt"
DATA_CONFIG = r"D:\web-pii-detector\webpii_yolo\data.yaml"
OUTPUT_DIR = r"D:\web-pii-detector\runs\detect"
EXP_NAME = "webpii_yolo11s_high_acc"

def main():
    # Verify local model checkpoint exists
    model_file = Path(MODEL_PATH)
    if not model_file.is_file():
        print(f"[Notice] Local weight not found at {MODEL_PATH}, falling back to 'yolo11s.pt'")
        model = YOLO("yolo11s.pt")
    else:
        print(f"[Loading] Using local model checkpoint: {MODEL_PATH}")
        model = YOLO(str(model_file))

    # 2. High-Accuracy Training
    model.train(
        # Dataset & Model
        data=DATA_CONFIG,
        epochs=50,                  # Optimal convergence for 35.8k images
        batch=16,                   # Balanced batch size for stable gradients
        imgsz=1024,                 # 1024px preserves tiny text, numbers, and form boxes
        patience=10,                # Early stopping if mAP plateaus for 10 epochs
        device=0,                   # GPU 0 (or '0,1' for multi-GPU, 'cpu' for CPU)
        workers=8,
        project=OUTPUT_DIR,
        name=EXP_NAME,
        exist_ok=True,

        # Optimizer calibrated for 260 fine-grained classes
        optimizer="AdamW",          # Best optimizer for large class sets & sparse gradients
        lr0=0.001,                  # Initial learning rate
        lrf=0.01,                   # Final learning rate factor (cosine decay down to 1e-5)
        cos_lr=True,                # Smooth cosine annealing scheduler
        warmup_epochs=3.0,          # 3 epochs warmup for stable head convergence
        weight_decay=0.0005,

        # Loss weights tuned for exact bounding box precision
        box=7.5,                    # Higher box loss weight for tight text boundary localization
        cls=0.5,                    # Classification loss weight
        dfl=1.5,                    # Distribution Focal Loss for sub-pixel text corners

        # Augmentations calibrated for Web UI screenshots (Max Accuracy)
        degrees=0.0,                # 0 rotation: web text is strictly horizontal
        shear=0.0,                  # 0 shear: prevents text skewing
        perspective=0.0,            # 0 perspective distortion
        flipud=0.0,                 # 0 vertical flip
        fliplr=0.0,                 # 0 horizontal flip (preserves left-to-right reading order)
        scale=0.3,                  # Scale variation (+/- 30%) for different screen DPIs
        translate=0.1,              # Small viewport translations
        mosaic=1.0,                 # Multi-context learning
        close_mosaic=10,            # Disable mosaic in final 10 epochs for clean box refinement
        mixup=0.0,                  # Disabled to avoid overlapping text artifacts

        # Validation & Logging
        val=True,
        save=True,
        plots=True,
    )

    # 3. Final Validation on Test Set
    print("\nRunning Final Evaluation on Test Split...")
    metrics = model.val(data=DATA_CONFIG, split="test", imgsz=1024, batch=16)
    print(f"\nTest mAP50-95: {metrics.box.map:.4f}")
    print(f"Test mAP50   : {metrics.box.map50:.4f}")
    print(f"Test mAP75   : {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
