#!/usr/bin/env python3
"""
Convert WebPII Parquet Dataset to YOLO Object Detection Format.

This script converts WebPII Parquet datasets containing:
- image (raw bytes or struct with 'bytes' and 'path')
- pii_elements_json (JSON string with bounding boxes)
- image_width, image_height
- pagetype / page_type
- variant

Outputs a complete YOLO dataset with:
- images/{train, val, test}
- labels/{train, val, test}
- data.yaml
- dataset_summary.json
- Full mathematical consistency verification
"""

import argparse
import io
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image


def load_class_mapping(mapping_path: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Load class mapping from JSON file.

    Supports formats:
    1. {"class_names": [...], "class_to_id": {...}}
    2. Direct dictionary: {"BACKGROUND": 0, "PII_ADDRESS": 1, ...}
    3. List of class names: ["BACKGROUND", "PII_ADDRESS", ...]
    """
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping file not found at: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "class_to_id" in data and "class_names" in data:
            class_names = data["class_names"]
            class_to_id = data["class_to_id"]
        elif "class_to_id" in data:
            class_to_id = data["class_to_id"]
            class_names = sorted(class_to_id.keys(), key=lambda k: class_to_id[k])
        elif "class_names" in data:
            class_names = data["class_names"]
            class_to_id = {name: idx for idx, name in enumerate(class_names)}
        else:
            # Direct key -> id dictionary
            class_to_id = data
            class_names = sorted(class_to_id.keys(), key=lambda k: class_to_id[k])
    elif isinstance(data, list):
        class_names = data
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
    else:
        raise ValueError(f"Unrecognized class mapping format in {mapping_path}")

    id_to_class = {int(idx): str(name) for name, idx in class_to_id.items()}

    # Validate mapping
    num_classes = len(class_to_id)
    expected_ids = set(range(num_classes))
    actual_ids = set(int(v) for v in class_to_id.values())
    if actual_ids != expected_ids:
        print(f"[WARNING] Class IDs are not contiguous 0..{num_classes - 1}. Found: {sorted(actual_ids)[:10]}...")

    return class_names, class_to_id, id_to_class


def parse_pii_elements(raw_json: Any) -> List[Dict[str, Any]]:
    """Recursively extracts PII elements from JSON string, list, or nested dict."""
    if raw_json is None:
        return []

    if isinstance(raw_json, str):
        raw_json = raw_json.strip()
        if not raw_json:
            return []
        try:
            parsed = json.loads(raw_json)
        except Exception:
            return []
    else:
        parsed = raw_json

    elements = []

    def _extract(obj):
        if isinstance(obj, dict):
            if "key" in obj and ("bbox_x" in obj or "bbox" in obj or "x" in obj):
                elements.append(obj)
            elif "pii_elements" in obj and isinstance(obj["pii_elements"], list):
                for elem in obj["pii_elements"]:
                    _extract(elem)
            else:
                for v in obj.values():
                    _extract(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    _extract(parsed)
    return elements


def extract_bbox(element: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Extracts (bbox_x, bbox_y, bbox_width, bbox_height) from element dict.
    Handles multiple bounding box representations safely.
    """
    try:
        if "bbox_x" in element and "bbox_y" in element and "bbox_width" in element and "bbox_height" in element:
            x = float(element["bbox_x"])
            y = float(element["bbox_y"])
            w = float(element["bbox_width"])
            h = float(element["bbox_height"])
            return x, y, w, h
        elif "bbox" in element and isinstance(element["bbox"], dict):
            b = element["bbox"]
            x = float(b.get("x", b.get("bbox_x", 0)))
            y = float(b.get("y", b.get("bbox_y", 0)))
            w = float(b.get("width", b.get("bbox_width", 0)))
            h = float(b.get("height", b.get("bbox_height", 0)))
            return x, y, w, h
        elif "bbox" in element and isinstance(element["bbox"], (list, tuple)) and len(element["bbox"]) == 4:
            b = element["bbox"]
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (ValueError, TypeError):
        return None
    return None


def convert_bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> Optional[Tuple[float, float, float, float, Tuple[float, float, float, float]]]:
    """
    Converts (bbox_x, bbox_y, bbox_width, bbox_height) to YOLO format:
    x_center = bbox_x + bbox_width / 2
    y_center = bbox_y + bbox_height / 2

    Normalized:
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = bbox_width / image_width
    height_normalized = bbox_height / image_height

    Clips to image boundaries and ensures values are in [0, 1].

    Returns:
        (x_center_norm, y_center_norm, w_norm, h_norm, clipped_raw_bbox) or None if invalid.
    """
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    # Check for non-positive or NaN values
    if bbox_w <= 0 or bbox_h <= 0 or image_width <= 0 or image_height <= 0:
        return None
    if any(math.isnan(v) or math.isinf(v) for v in (bbox_x, bbox_y, bbox_w, bbox_h, image_width, image_height)):
        return None

    # Step 1: Clip bounding box coordinates to image boundaries
    x1 = bbox_x
    y1 = bbox_y
    x2 = bbox_x + bbox_w
    y2 = bbox_y + bbox_h

    x1_clipped = max(0.0, min(float(x1), float(image_width)))
    y1_clipped = max(0.0, min(float(y1), float(image_height)))
    x2_clipped = max(0.0, min(float(x2), float(image_width)))
    y2_clipped = max(0.0, min(float(y2), float(image_height)))

    w_clipped = x2_clipped - x1_clipped
    h_clipped = y2_clipped - y1_clipped

    if w_clipped <= 0.0 or h_clipped <= 0.0:
        return None

    # Step 2: Compute center coordinates
    x_center = x1_clipped + w_clipped / 2.0
    y_center = y1_clipped + h_clipped / 2.0

    # Step 3: Normalize using image dimensions
    x_center_norm = max(0.0, min(1.0, x_center / float(image_width)))
    y_center_norm = max(0.0, min(1.0, y_center / float(image_height)))
    width_norm = max(0.0, min(1.0, w_clipped / float(image_width)))
    height_norm = max(0.0, min(1.0, h_clipped / float(image_height)))

    clipped_raw = (x1_clipped, y1_clipped, w_clipped, h_clipped)
    return x_center_norm, y_center_norm, width_norm, height_norm, clipped_raw


def save_image_bytes(
    image_data: Any,
    output_path: Path,
    image_format: str = "PNG",
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Decodes image binary from Parquet row and writes to destination.
    Optimized for speed: directly writes if format matches, otherwise re-encodes via PIL.
    Returns (width, height).
    """
    if isinstance(image_data, dict):
        raw_bytes = image_data.get("bytes")
    elif isinstance(image_data, (bytes, bytearray)):
        raw_bytes = image_data
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")

    if not raw_bytes:
        raise ValueError("Empty image bytes")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")

    # Fast path: direct binary write if formats match and dimensions are known
    if (image_format == "PNG" and is_png) or (image_format in ("JPEG", "JPG") and is_jpeg):
        with open(output_path, "wb") as f:
            f.write(raw_bytes)
        if expected_width and expected_height and expected_width > 0 and expected_height > 0:
            return expected_width, expected_height

    # Fallback / validation path: Open in PIL
    with Image.open(io.BytesIO(raw_bytes)) as img:
        width, height = img.size
        # Only re-save if not already written in fast path
        if not output_path.exists():
            img_rgb = img.convert("RGB")
            img_rgb.save(output_path, format=image_format)

    return width, height


def build_split_indices(
    total_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[int, str]:
    """
    Generates a reproducible mapping from sample index -> 'train' | 'val' | 'test'.
    """
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(round(total_samples * train_ratio))
    n_val = int(round(total_samples * val_ratio))
    # Test gets remaining samples to ensure exact partition
    n_test = total_samples - n_train - n_val

    split_map = {}
    for idx in indices[:n_train]:
        split_map[idx] = "train"
    for idx in indices[n_train : n_train + n_val]:
        split_map[idx] = "val"
    for idx in indices[n_train + n_val :]:
        split_map[idx] = "test"

    return split_map


def verify_mathematical_consistency(
    verification_samples: List[Dict[str, Any]],
    output_dir: Path,
    tolerance: float = 1e-4,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Verifies that YOLO normalized coordinates reconstruct original pixel bounding boxes accurately.
    """
    results = []
    all_passed = True

    print("\n" + "=" * 80)
    print("MATHEMATICAL CONSISTENCY VERIFICATION REPORT")
    print("=" * 80)

    for sample in verification_samples:
        split = sample["split"]
        filename_stem = sample["filename_stem"]
        orig_img_w = sample["image_width"]
        orig_img_h = sample["image_height"]
        orig_boxes = sample["original_boxes"]

        label_path = output_dir / "labels" / split / f"{filename_stem}.txt"
        if not label_path.is_file():
            all_passed = False
            results.append({
                "sample": filename_stem,
                "status": "FAILED",
                "reason": f"Label file missing at {label_path}",
            })
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) != len(orig_boxes):
            all_passed = False
            results.append({
                "sample": filename_stem,
                "status": "FAILED",
                "reason": f"Line count mismatch: expected {len(orig_boxes)}, found {len(lines)}",
            })
            continue

        sample_passed = True
        box_diffs = []

        for line_idx, (line, orig_info) in enumerate(zip(lines, orig_boxes)):
            parts = line.split()
            if len(parts) != 5:
                sample_passed = False
                continue

            class_id = int(parts[0])
            x_c_norm = float(parts[1])
            y_c_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Reconstruct pixel bbox
            reconstructed_w = w_norm * orig_img_w
            reconstructed_h = h_norm * orig_img_h
            reconstructed_x1 = (x_c_norm - w_norm / 2.0) * orig_img_w
            reconstructed_y1 = (y_c_norm - h_norm / 2.0) * orig_img_h

            target_x1, target_y1, target_w, target_h = orig_info["clipped_raw"]
            target_class_id = orig_info["class_id"]

            diff_class = class_id != target_class_id
            diff_x = abs(reconstructed_x1 - target_x1)
            diff_y = abs(reconstructed_y1 - target_y1)
            diff_w = abs(reconstructed_w - target_w)
            diff_h = abs(reconstructed_h - target_h)

            max_pixel_diff = max(diff_x, diff_y, diff_w, diff_h)
            # Normalized error relative to image dimensions
            normalized_max_diff = max(
                diff_x / orig_img_w,
                diff_y / orig_img_h,
                diff_w / orig_img_w,
                diff_h / orig_img_h,
            )

            box_diffs.append({
                "class_id": class_id,
                "class_name": orig_info["class_name"],
                "yolo_line": line,
                "reconstructed_pixel_box": (
                    round(reconstructed_x1, 3),
                    round(reconstructed_y1, 3),
                    round(reconstructed_w, 3),
                    round(reconstructed_h, 3),
                ),
                "original_pixel_box": (
                    round(target_x1, 3),
                    round(target_y1, 3),
                    round(target_w, 3),
                    round(target_h, 3),
                ),
                "max_pixel_error": max_pixel_diff,
                "max_norm_error": normalized_max_diff,
            })

            if diff_class or normalized_max_diff > tolerance:
                sample_passed = False
                all_passed = False

        status_str = "PASSED" if sample_passed else "FAILED"
        print(f"[*] Sample: {split}/{filename_stem} (Image: {orig_img_w}x{orig_img_h}) -> {status_str}")
        for b in box_diffs:
            print(
                f"    - Class {b['class_id']:3d} ({b['class_name']}): "
                f"Orig: {b['original_pixel_box']} -> YOLO: [{b['yolo_line']}] -> "
                f"Reconstructed: {b['reconstructed_pixel_box']} (Max pixel error: {b['max_pixel_error']:.4f} px)"
            )

        results.append({
            "sample": filename_stem,
            "split": split,
            "status": status_str,
            "boxes": box_diffs,
        })

    print("-" * 80)
    print(f"Consistency Verification Status: {'ALL CHECKS PASSED (Exact Match)' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 80 + "\n")
    return all_passed, results


def process_dataset(
    input_path: Path,
    output_dir: Path,
    mapping_path: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    batch_size: int = 256,
    image_format: str = "PNG",
    max_rows: Optional[int] = None,
    verify_sample_count: int = 5,
) -> Dict[str, Any]:
    """
    Main conversion routine.
    """
    print("=" * 80)
    print("WebPII to YOLO Object Detection Dataset Converter")
    print("=" * 80)
    print(f"Input Dataset:        {input_path}")
    print(f"Output Directory:     {output_dir}")
    print(f"Class Mapping File:   {mapping_path}")
    print(f"Split Ratios:         Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
    print(f"Random Seed:          {seed}")
    print(f"Batch Size:           {batch_size}")
    print(f"Image Format:         {image_format}")
    if max_rows:
        print(f"Max Rows Limit:       {max_rows}")
    print("-" * 80)

    # 1. Load class mapping
    class_names, class_to_id, id_to_class = load_class_mapping(mapping_path)
    print(f"[+] Loaded {len(class_names)} classes from {mapping_path.name}")
    print(f"    Sample classes: {list(class_to_id.items())[:6]}")

    # 2. Open Parquet File & Metadata
    if input_path.is_dir():
        parquet_files = sorted(list(input_path.glob("*.parquet")) + list(input_path.glob("**/*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in directory: {input_path}")
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
    elif input_path.is_file():
        parquet_files = [input_path]
        total_rows = pq.ParquetFile(input_path).metadata.num_rows
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if max_rows and max_rows < total_rows:
        total_rows = max_rows

    print(f"[+] Total images to process: {total_rows:,}")

    # 3. Setup Split Map
    split_map = build_split_indices(
        total_samples=total_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 4. Prepare Output Directories
    splits = ["train", "val", "test"]
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. Statistics Counters
    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_pii_annotations": 0,
        "annotations_per_class": Counter(),
        "skipped_malformed_annotations": 0,
        "images_without_pii": 0,
        "missing_image_label_pairs": 0,
    }

    verification_candidates = []
    global_index = 0
    img_extension = image_format.lower()
    if img_extension == "jpeg":
        img_extension = "jpg"

    # 6. Stream Parquet Batches
    columns_needed = ["image", "pii_elements_json", "image_width", "image_height"]

    print("\n[+] Converting batches...")
    for p_file in parquet_files:
        pq_reader = pq.ParquetFile(p_file)
        
        # Use schema_arrow to properly detect top-level struct columns like 'image'
        available_cols = set(pq_reader.schema_arrow.names)
        cols_to_read = [c for c in columns_needed if c in available_cols]

        for batch in pq_reader.iter_batches(columns=cols_to_read, batch_size=batch_size):
            batch_dict = batch.to_pydict()
            batch_len = len(batch_dict[cols_to_read[0]])

            for row_idx in range(batch_len):
                if global_index >= total_rows:
                    break

                split = split_map[global_index]
                filename_stem = f"image_{global_index + 1:06d}"
                image_dest = output_dir / "images" / split / f"{filename_stem}.{img_extension}"
                label_dest = output_dir / "labels" / split / f"{filename_stem}.txt"

                # Extract image
                raw_image = batch_dict["image"][row_idx] if "image" in batch_dict else None
                orig_width = batch_dict["image_width"][row_idx] if "image_width" in batch_dict else None
                orig_height = batch_dict["image_height"][row_idx] if "image_height" in batch_dict else None

                actual_width = orig_width
                actual_height = orig_height

                image_saved_successfully = False
                if raw_image is not None:
                    try:
                        decoded_w, decoded_h = save_image_bytes(
                            image_data=raw_image,
                            output_path=image_dest,
                            image_format=image_format,
                            expected_width=orig_width,
                            expected_height=orig_height,
                        )
                        if actual_width is None or actual_width <= 0:
                            actual_width = decoded_w
                        if actual_height is None or actual_height <= 0:
                            actual_height = decoded_h
                        image_saved_successfully = True
                    except Exception as e:
                        print(f"[ERROR] Failed to save image for sample {global_index}: {e}")
                        stats["missing_image_label_pairs"] += 1
                        global_index += 1
                        continue
                else:
                    print(f"[WARNING] No image data at row {global_index}")

                # Extract PII elements
                raw_pii = batch_dict["pii_elements_json"][row_idx] if "pii_elements_json" in batch_dict else None
                pii_elements = parse_pii_elements(raw_pii)

                yolo_lines = []
                sample_verification_boxes = []

                for element in pii_elements:
                    key = element.get("key")
                    if not key or key not in class_to_id:
                        stats["skipped_malformed_annotations"] += 1
                        continue

                    class_id = class_to_id[key]

                    bbox = extract_bbox(element)
                    if bbox is None:
                        stats["skipped_malformed_annotations"] += 1
                        continue

                    conversion = convert_bbox_to_yolo(
                        bbox=bbox,
                        image_width=float(actual_width),
                        image_height=float(actual_height),
                    )

                    if conversion is None:
                        stats["skipped_malformed_annotations"] += 1
                        continue

                    x_c_norm, y_c_norm, w_norm, h_norm, clipped_raw = conversion
                    line = f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_lines.append(line)

                    stats["total_pii_annotations"] += 1
                    stats["annotations_per_class"][key] += 1

                    sample_verification_boxes.append({
                        "class_id": class_id,
                        "class_name": key,
                        "clipped_raw": clipped_raw,
                    })

                # Write YOLO label file (one annotation per line, or empty file if no PII)
                with open(label_dest, "w", encoding="utf-8") as lf:
                    if yolo_lines:
                        lf.write("\n".join(yolo_lines) + "\n")

                # Verify file creation pair
                if not (image_dest.is_file() and label_dest.is_file()):
                    stats["missing_image_label_pairs"] += 1

                # Update counts
                stats["total_images"] += 1
                if split == "train":
                    stats["train_images"] += 1
                elif split == "val":
                    stats["val_images"] += 1
                elif split == "test":
                    stats["test_images"] += 1

                if not yolo_lines:
                    stats["images_without_pii"] += 1

                # Collect verification sample candidates
                if yolo_lines and len(verification_candidates) < verify_sample_count:
                    verification_candidates.append({
                        "split": split,
                        "filename_stem": filename_stem,
                        "image_width": actual_width,
                        "image_height": actual_height,
                        "original_boxes": sample_verification_boxes,
                    })

                global_index += 1
                if global_index % 500 == 0 or global_index == total_rows:
                    print(f"    Progress: {global_index:,} / {total_rows:,} images processed ({(global_index / total_rows) * 100:.1f}%)")

            if global_index >= total_rows:
                break

    # 7. Generate data.yaml
    yaml_path = output_dir / "data.yaml"
    # Format names mapping
    names_block = []
    for idx, name in sorted(id_to_class.items(), key=lambda x: x[0]):
        names_block.append(f"  {idx}: {name}")

    yaml_content = f"""# YOLOv8 / YOLOv5 Dataset Configuration for WebPII
path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(class_to_id)}

names:
""" + "\n".join(names_block) + "\n"

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)
    print(f"\n[+] Generated YOLO configuration: {yaml_path}")

    # 8. Run Mathematical Consistency Verification
    verification_passed, verification_details = verify_mathematical_consistency(
        verification_samples=verification_candidates,
        output_dir=output_dir,
    )

    # 9. Generate dataset_summary.json
    summary_data = {
        "dataset_name": "WebPII_YOLO",
        "total_images": stats["total_images"],
        "train_images": stats["train_images"],
        "validation_images": stats["val_images"],
        "test_images": stats["test_images"],
        "total_pii_annotations": stats["total_pii_annotations"],
        "annotations_per_class": dict(stats["annotations_per_class"]),
        "skipped_malformed_annotations": stats["skipped_malformed_annotations"],
        "images_without_pii": stats["images_without_pii"],
        "missing_image_label_pairs": stats["missing_image_label_pairs"],
        "num_classes": len(class_to_id),
        "split_distribution": {
            "train": f"{(stats['train_images'] / max(1, stats['total_images'])) * 100:.2f}%",
            "val": f"{(stats['val_images'] / max(1, stats['total_images'])) * 100:.2f}%",
            "test": f"{(stats['test_images'] / max(1, stats['total_images'])) * 100:.2f}%",
        },
        "random_seed": seed,
        "mathematical_consistency_verified": verification_passed,
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2, ensure_ascii=False)
    print(f"[+] Saved summary statistics to: {summary_path}")

    # 10. Print Final Comprehensive Report
    print_report(stats, len(class_to_id), summary_path, yaml_path)

    return summary_data


def print_report(stats: Dict[str, Any], num_classes: int, summary_path: Path, yaml_path: Path) -> None:
    """Prints formatted ASCII verification report to stdout."""
    total_imgs = stats["total_images"]
    train_pct = (stats["train_images"] / max(1, total_imgs)) * 100
    val_pct = (stats["val_images"] / max(1, total_imgs)) * 100
    test_pct = (stats["test_images"] / max(1, total_imgs)) * 100

    print("\n" + "=" * 80)
    print("                      FINAL DATASET VERIFICATION REPORT")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Value':<40}")
    print("-" * 80)
    print(f"{'Total Images':<35} | {stats['total_images']:,}")
    print(f"{'Train Images':<35} | {stats['train_images']:,} ({train_pct:.2f}%)")
    print(f"{'Validation Images':<35} | {stats['val_images']:,} ({val_pct:.2f}%)")
    print(f"{'Test Images':<35} | {stats['test_images']:,} ({test_pct:.2f}%)")
    print(f"{'Total Classes Defined':<35} | {num_classes:,}")
    print(f"{'Total PII Annotations':<35} | {stats['total_pii_annotations']:,}")
    print(f"{'Skipped / Malformed Annotations':<35} | {stats['skipped_malformed_annotations']:,}")
    print(f"{'Images Without PII':<35} | {stats['images_without_pii']:,}")
    print(f"{'Missing Image/Label Pairs':<35} | {stats['missing_image_label_pairs']:,}")
    print("-" * 80)
    print("TOP 15 PII ANNOTATION CLASSES BY FREQUENCY:")
    top_classes = stats["annotations_per_class"].most_common(15)
    if top_classes:
        for idx, (cls_name, count) in enumerate(top_classes, 1):
            pct = (count / max(1, stats["total_pii_annotations"])) * 100
            print(f"  {idx:2d}. {cls_name:<30} : {count:6,d} ({pct:5.2f}%)")
    else:
        print("  (No annotations found)")
    print("-" * 80)
    print(f"Configuration File: {yaml_path}")
    print(f"Summary File:       {summary_path}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WebPII Parquet Dataset to YOLO Object Detection Format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to WebPII Parquet dataset file or directory containing parquet files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output YOLO dataset directory",
    )
    parser.add_argument(
        "--classes-mapping", "-c",
        type=Path,
        default=Path("pii_classes.json"),
        help="Path to pii_classes.json file containing complete class mapping",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Proportion of images for training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Proportion of images for validation set",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Proportion of images for test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splitting",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of rows to stream per Parquet batch",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="PNG",
        choices=["PNG", "JPEG", "JPG"],
        help="Image format for saved images",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max limit on rows to process (for testing)",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="Number of samples to mathematically verify for coordinate consistency",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0. Current sum: {total_ratio} "
            f"(train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})"
        )

    process_dataset(
        input_path=args.input,
        output_dir=args.output,
        mapping_path=args.classes_mapping,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        image_format=args.image_format,
        max_rows=args.max_rows,
        verify_sample_count=args.verify_samples,
    )


if __name__ == "__main__":
    main()