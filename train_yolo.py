#!/usr/bin/env python3
"""
Simple & High-Accuracy Training Script for YOLO11s on Web PII Dataset.
Configured for seamless execution on Linux (NVIDIA DGX) and local environments.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Project root directory anchored dynamically
ROOT_DIR = Path(__file__).resolve().parent

# Default Paths
DEFAULT_MODEL_PATH = ROOT_DIR / "yolo11s" / "yolo11s.pt"
DEFAULT_DATA_CONFIG = ROOT_DIR / "webpii_yolo" / "data.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "detect"
DEFAULT_EXP_NAME = "webpii_yolo11s_high_acc"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11s on Web PII Dataset (Linux / DGX Compatible)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to initial weights (.pt) checkpoint (default: ./yolo11s/yolo11s.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to dataset data.yaml (default: ./webpii_yolo/data.yaml)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save training runs (default: ./runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_EXP_NAME,
        help="Experiment name (default: webpii_yolo11s_high_acc)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index, e.g. 0 or '0,1,2,3' for multi-GPU, or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker processes (default: 8)",
    )
    return parser.parse_args()


def resolve_model(model_arg: str) -> str:
    """Resolve model path, preferring local checkpoint if available."""
    model_path = Path(model_arg)
    if model_path.is_file():
        print(f"[Loading] Using local model checkpoint: {model_path}")
        return str(model_path)

    # Secondary check in ./yolo11s/yolo11s.pt
    if DEFAULT_MODEL_PATH.is_file():
        print(f"[Loading] Using local model checkpoint: {DEFAULT_MODEL_PATH}")
        return str(DEFAULT_MODEL_PATH)

    # Secondary check in root ./yolo11s.pt
    root_model = ROOT_DIR / "yolo11s.pt"
    if root_model.is_file():
        print(f"[Loading] Using local model checkpoint: {root_model}")
        return str(root_model)

    print(f"[Notice] Local weight not found at {model_arg}, falling back to official 'yolo11s.pt'")
    return "yolo11s.pt"


def main():
    args = parse_args()

    # Verify dataset configuration exists
    data_file = Path(args.data).resolve()
    if not data_file.is_file():
        print(f"[Error] Dataset configuration not found at {data_file}", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print(" Web PII Fine-Tuning - YOLO11s")
    print(f" Root Directory   : {ROOT_DIR}")
    print(f" Data Config      : {data_file}")
    print(f" Output Directory : {args.project}/{args.name}")
    print(f" Target Device    : {args.device}")
    print(f" Epochs / Batch   : {args.epochs} / {args.batch} (imgsz={args.imgsz})")
    print("==================================================")

    # 1. Initialize YOLO model
    model_source = resolve_model(args.model)
    model = YOLO(model_source)

    # 2. High-Accuracy Training (Preserving all original hyperparameters)
    model.train(
        # Dataset & Model
        data=str(data_file),
        epochs=args.epochs,         # Optimal convergence for 35.8k images
        batch=32,           # Balanced batch size for stable gradients
        imgsz=args.imgsz,           # 1024px preserves tiny text, numbers, and form boxes
        patience=10,                # Early stopping if mAP plateaus for 10 epochs
        device=args.device,         # GPU 0 (or '0,1' for multi-GPU, 'cpu' for CPU)
        workers=args.workers,
        project=args.project,
        name=args.name,
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
    metrics = model.val(data=str(data_file), split="test", imgsz=args.imgsz, batch=args.batch)
    print(f"\nTest mAP50-95: {metrics.box.map:.4f}")
    print(f"Test mAP50   : {metrics.box.map50:.4f}")
    print(f"Test mAP75   : {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
