# Web PII Detector

A high-performance object detection system for identifying and localizing Personally Identifiable Information (PII) on web pages and web applications using fine-tuned YOLO11 models.

## 🚀 Quick Links
- **[Full Dataset & Conversion Guide](file:///d:/web-pii-detector/DATASET_PIPELINE.md)**: Detailed step-by-step instructions from downloading WebPII to YOLO training.
- **[Dataset Configuration](file:///d:/web-pii-detector/webpii_yolo/data.yaml)**: YOLOv8 / YOLO11 `data.yaml` specifying splits and 260 class names.
- **[Training Script](file:///d:/web-pii-detector/train_yolo.py)**: Calibrated training pipeline for web screenshots (imgsz=1024, AdamW, horizontal orientations).

## 🛠️ Pipeline Scripts
- [loaddataset.py](file:///d:/web-pii-detector/loaddataset.py): Download dataset from Hugging Face Hub.
- [inspect_webpii.py](file:///d:/web-pii-detector/inspect_webpii.py): Inspect schema and sample PII annotations.
- [preparedataset.py](file:///d:/web-pii-detector/preparedataset.py): Merge raw parquet splits into `webpii_selected.parquet`.
- [build_pii_classes.py](file:///d:/web-pii-detector/build_pii_classes.py): Stream parquet annotations to generate `pii_classes.json`.
- [convert_webpii_to_yolo.py](file:///d:/web-pii-detector/convert_webpii_to_yolo.py): Convert parquet dataset to YOLO images and normalized label bounding boxes.
- [train_yolo.py](file:///d:/web-pii-detector/train_yolo.py): Train YOLO11s on the generated dataset.
