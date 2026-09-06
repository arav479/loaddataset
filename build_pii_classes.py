"""Build the PII detector class mapping from WebPII annotation data."""

import json
from pathlib import Path

import pyarrow.parquet as pq


DATASET_PATH = Path(r"D:\web-pii-detector\webpii_selected.parquet")
OUTPUT_PATH = Path(r"D:\web-pii-detector\pii_classes.json")
BATCH_SIZE = 10_000
BACKGROUND_CLASS = "BACKGROUND"


def collect_keys(value, keys: set[str]) -> None:
    """Recursively collect annotation values stored under the literal field 'key'."""
    if isinstance(value, dict):
        key = value.get("key")
        if isinstance(key, str) and key.strip():
            keys.add(key.strip())
        for child in value.values():
            collect_keys(child, keys)
    elif isinstance(value, list):
        for child in value:
            collect_keys(child, keys)


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"WebPII parquet file was not found: {DATASET_PATH}")

    pii_keys: set[str] = set()
    parquet = pq.ParquetFile(DATASET_PATH)

    # Read only pii_elements_json, never the multi-gigabyte image column.
    for batch in parquet.iter_batches(
        columns=["pii_elements_json"], batch_size=BATCH_SIZE
    ):
        for raw_annotation in batch.column(0).to_pylist():
            if raw_annotation is None:
                continue
            annotation = (
                json.loads(raw_annotation)
                if isinstance(raw_annotation, str)
                else raw_annotation
            )
            collect_keys(annotation, pii_keys)

    if not pii_keys:
        raise RuntimeError("No non-empty annotation 'key' values were found in WebPII.")

    # ID 0 is reserved for pixels/cells with no PII. Sorting the remaining
    # labels gives stable PII IDs across future training runs.
    class_names = [BACKGROUND_CLASS, *sorted(pii_keys - {BACKGROUND_CLASS})]
    class_to_id = {name: index for index, name in enumerate(class_names)}
    OUTPUT_PATH.write_text(
        json.dumps(
            {"class_names": class_names, "class_to_id": class_to_id},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Found {len(class_names)} PII classes from WebPII annotations.")
    for name, index in class_to_id.items():
        print(f"{index}: {name}")
    print(f"Saved class mapping to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
