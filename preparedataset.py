import pandas as pd
from pathlib import Path

DATASET_DIR = Path(r"D:\web-pii-detector\webpii_dataset\data")
OUTPUT_FILE = Path(r"D:\web-pii-detector\webpii_selected.parquet")

columns = [
    "image",
    "pii_elements_json",
    "image_width",
    "image_height",
    "page_type",
    "variant"
]

parquet_files = sorted(DATASET_DIR.glob("*.parquet"))

print(f"Found {len(parquet_files)} files")

dataframes = []

for file in parquet_files:
    print(f"Reading: {file.name}")

    df = pd.read_parquet(
        file,
        columns=columns
    )

    dataframes.append(df)

dataset = pd.concat(dataframes, ignore_index=True)

print("\nDataset created!")
print("Rows:", len(dataset))
print("Columns:", dataset.columns.tolist())

dataset.to_parquet(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)