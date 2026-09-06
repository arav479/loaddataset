
"""
Convert selected WebPII Parquet data into a PyTorch object-detection dataset.

Output structure:

webpii_pytorch_detection/
│
├── images/
│   ├── 00000000.png
│   ├── 00000001.png
│   └── ...
│
├── targets/
│   ├── 00000000.pt
│   ├── 00000001.pt
│   └── ...
│
└── manifest.jsonl

Images are stored as PNG files.
Only detection annotations are stored as .pt files.

This avoids storing large decoded image tensors inside .pt files.
"""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_PATH = Path(
    r"D:\web-pii-detector\webpii_selected.parquet"
)

CLASS_MAPPING_PATH = Path(
    r"D:\web-pii-detector\pii_classes.json"
)

OUTPUT_DIR = Path(
    r"D:\web-pii-detector\webpii_pytorch_detection"
)

IMAGES_DIR = OUTPUT_DIR / "images"
TARGETS_DIR = OUTPUT_DIR / "targets"

BATCH_SIZE = 64


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_to_id() -> dict[str, int]:
    """
    Load PII class mapping.

    ID 0 must be reserved for BACKGROUND.
    """

    if not CLASS_MAPPING_PATH.is_file():
        raise FileNotFoundError(
            f"Class mapping not found:\n{CLASS_MAPPING_PATH}"
        )

    mapping = json.loads(
        CLASS_MAPPING_PATH.read_text(encoding="utf-8")
    )["class_to_id"]

    if mapping.get("BACKGROUND") != 0:
        raise ValueError(
            "pii_classes.json must reserve ID 0 for BACKGROUND."
        )

    if len(mapping) != len(set(mapping.values())):
        raise ValueError(
            "pii_classes.json contains duplicate class IDs."
        )

    return mapping


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    value: Any,
    output_path: Path,
) -> tuple[int, int]:
    """
    Decode WebPII encoded image bytes and save as PNG.

    Returns:
        width, height
    """

    if isinstance(value, dict):
        value = value.get("bytes")

    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(
            "The WebPII image field must contain encoded image bytes."
        )

    with Image.open(io.BytesIO(value)) as image:
        image = image.convert("RGB")

        width, height = image.size

        image.save(
            output_path,
            format="PNG",
            optimize=False,
        )

    return width, height


# ============================================================
# FIND PII ELEMENTS
# ============================================================

def find_pii_elements(
    value: Any,
) -> Iterator[dict[str, Any]]:
    """
    Recursively find PII elements inside WebPII annotation JSON.
    """

    if isinstance(value, dict):

        elements = value.get("pii_elements")

        if isinstance(elements, list):
            for element in elements:
                if isinstance(element, dict):
                    yield element

        for child in value.values():
            yield from find_pii_elements(child)

    elif isinstance(value, list):

        for child in value:
            yield from find_pii_elements(child)


# ============================================================
# CREATE DETECTION TARGET
# ============================================================

def make_target(
    annotation: Any,
    image_width: int,
    image_height: int,
    class_to_id: dict[str, int],
    image_id: int,
) -> dict[str, torch.Tensor]:

    boxes: list[list[float]] = []
    labels: list[int] = []

    seen: set[
        tuple[str, float, float, float, float]
    ] = set()

    for element in find_pii_elements(annotation):

        key = element.get("key")

        label = class_to_id.get(key)

        bbox = element.get("bbox")

        # Ignore unknown classes
        if label is None:
            continue

        # Ignore BACKGROUND
        if label == 0:
            continue

        # Ignore invisible elements
        if not element.get("visible", True):
            continue

        if not isinstance(bbox, dict):
            continue

        try:

            x = float(bbox["x"])
            y = float(bbox["y"])

            width = float(bbox["width"])
            height = float(bbox["height"])

            x1 = max(
                0.0,
                min(x, float(image_width)),
            )

            y1 = max(
                0.0,
                min(y, float(image_height)),
            )

            x2 = max(
                0.0,
                min(
                    x + width,
                    float(image_width),
                ),
            )

            y2 = max(
                0.0,
                min(
                    y + height,
                    float(image_height),
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        # Invalid box
        if x2 <= x1 or y2 <= y1:
            continue

        signature = (
            str(key),
            x1,
            y1,
            x2,
            y2,
        )

        # Remove duplicates
        if signature in seen:
            continue

        seen.add(signature)

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2,
            ]
        )

        labels.append(label)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    box_tensor = torch.tensor(
        boxes,
        dtype=torch.float32,
    ).reshape(-1, 4)

    label_tensor = torch.tensor(
        labels,
        dtype=torch.int64,
    )

    if len(box_tensor) > 0:

        area = (
            (box_tensor[:, 2] - box_tensor[:, 0])
            *
            (box_tensor[:, 3] - box_tensor[:, 1])
        )

    else:

        area = torch.zeros(
            (0,),
            dtype=torch.float32,
        )

    target = {
        "boxes": box_tensor,

        "labels": label_tensor,

        "image_id": torch.tensor(
            [image_id],
            dtype=torch.int64,
        ),

        "area": area,

        "iscrowd": torch.zeros(
            len(labels),
            dtype=torch.int64,
        ),
    }

    return target


# ============================================================
# CONVERSION
# ============================================================

def convert() -> None:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not SOURCE_PATH.is_file():

        raise FileNotFoundError(
            f"WebPII dataset not found:\n{SOURCE_PATH}"
        )

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            f"""
Output directory already exists:

{OUTPUT_DIR}

Delete it manually before running conversion again.
"""
        )

    class_to_id = load_class_to_id()

    print("=" * 60)
    print("WebPII → PyTorch Detection Dataset Converter")
    print("=" * 60)

    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_DIR}")
    print()

    print("Classes:")

    for name, class_id in sorted(
        class_to_id.items(),
        key=lambda item: item[1],
    ):
        print(
            f"  {class_id:3d} -> {name}"
        )

    print()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    TARGETS_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_path = OUTPUT_DIR / "manifest.jsonl"

    written = 0

    skipped = 0

    total_boxes = 0

    # --------------------------------------------------------
    # Open Parquet
    # --------------------------------------------------------

    parquet = pq.ParquetFile(
        SOURCE_PATH
    )

    try:

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as manifest:

            for batch_number, batch in enumerate(
                parquet.iter_batches(
                    columns=[
                        "image",
                        "pii_elements_json",
                        "image_width",
                        "image_height",
                    ],
                    batch_size=BATCH_SIZE,
                )
            ):

                rows = batch.to_pylist()

                print(
                    f"Processing batch {batch_number + 1} "
                    f"({len(rows)} samples)..."
                )

                for row in rows:

                    image_id = written

                    filename = (
                        f"{image_id:08d}"
                    )

                    image_path = (
                        IMAGES_DIR
                        / f"{filename}.png"
                    )

                    target_path = (
                        TARGETS_DIR
                        / f"{filename}.pt"
                    )

                    try:

                        # ------------------------------------------------
                        # Decode and save image
                        # ------------------------------------------------

                        actual_width, actual_height = save_image(
                            row["image"],
                            image_path,
                        )

                        # ------------------------------------------------
                        # Parse annotation
                        # ------------------------------------------------

                        raw_annotation = (
                            row["pii_elements_json"]
                        )

                        if isinstance(
                            raw_annotation,
                            str,
                        ):

                            annotation = json.loads(
                                raw_annotation
                            )

                        else:

                            annotation = raw_annotation

                        # ------------------------------------------------
                        # Create target
                        # ------------------------------------------------

                        target = make_target(
                            annotation,
                            actual_width,
                            actual_height,
                            class_to_id,
                            image_id,
                        )

                        number_of_boxes = len(
                            target["labels"]
                        )

                        # ------------------------------------------------
                        # Save ONLY target
                        # ------------------------------------------------

                        torch.save(
                            target,
                            target_path,
                        )

                        # ------------------------------------------------
                        # Manifest entry
                        # ------------------------------------------------

                        manifest.write(
                            json.dumps(
                                {
                                    "image": (
                                        f"images/{filename}.png"
                                    ),
                                    "target": (
                                        f"targets/{filename}.pt"
                                    ),
                                    "image_id": image_id,
                                    "width": actual_width,
                                    "height": actual_height,
                                    "boxes": number_of_boxes,
                                }
                            )
                            + "\n"
                        )

                        written += 1

                        total_boxes += number_of_boxes

                    except Exception as error:

                        skipped += 1

                        # Remove partially-created files
                        if image_path.exists():

                            try:
                                image_path.unlink()

                            except OSError:
                                pass

                        if target_path.exists():

                            try:
                                target_path.unlink()

                            except OSError:
                                pass

                        print(
                            f"WARNING: skipped row "
                            f"{image_id}: {error}"
                        )

                print(
                    f"  Converted: {written}"
                )

                print(
                    f"  Skipped  : {skipped}"
                )

                print(
                    f"  Boxes    : {total_boxes}"
                )

                print()

    except Exception:

        # IMPORTANT:
        #
        # We intentionally do NOT call shutil.rmtree()
        # here.
        #
        # If Windows still has a file handle open,
        # rmtree() can produce WinError 32 and hide
        # the original error.

        print()
        print(
            "Conversion stopped because of an error."
        )
        print(
            f"Successfully converted: {written}"
        )
        print(
            f"Skipped: {skipped}"
        )

        raise

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(
        f"Images converted : {written}"
    )

    print(
        f"Images skipped   : {skipped}"
    )

    print(
        f"Total PII boxes  : {total_boxes}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print(
        f"Manifest         : {manifest_path}"
    )

    print("=" * 60)


# ============================================================
# PYTORCH DATASET
# ============================================================

class WebPIIDetectionDataset(Dataset):
    """
    PyTorch Dataset for the converted WebPII dataset.

    Images are loaded only when __getitem__ is called.
    """

    def __init__(
        self,
        root: Path = OUTPUT_DIR,
    ) -> None:

        self.root = Path(root)

        self.images_dir = (
            self.root / "images"
        )

        self.targets_dir = (
            self.root / "targets"
        )

        if not self.images_dir.exists():

            raise FileNotFoundError(
                f"Images directory not found:\n"
                f"{self.images_dir}"
            )

        if not self.targets_dir.exists():

            raise FileNotFoundError(
                f"Targets directory not found:\n"
                f"{self.targets_dir}"
            )

        self.image_files = sorted(
            self.images_dir.glob("*.png")
        )

        if not self.image_files:

            raise FileNotFoundError(
                f"No images found in:\n"
                f"{self.images_dir}"
            )

    def __len__(self) -> int:

        return len(self.image_files)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        image_path = self.image_files[index]

        target_path = (
            self.targets_dir
            / f"{image_path.stem}.pt"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            # PIL → Tensor
            image_tensor = torch.from_numpy(
                __import__("numpy").array(
                    image,
                    dtype="uint8",
                )
            )

        # H,W,C → C,H,W
        image_tensor = (
            image_tensor
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        # ----------------------------------------------------
        # Load target
        # ----------------------------------------------------

        target = torch.load(
            target_path,
            map_location="cpu",
            weights_only=True,
        )

        return image_tensor, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def detection_collate(batch):

    """
    Detection datasets have variable numbers of boxes,
    so the default DataLoader collate function cannot be used.
    """

    return tuple(
        zip(*batch)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    convert()
