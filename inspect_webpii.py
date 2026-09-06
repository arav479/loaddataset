import pandas as pd
import json

file = r"D:\web-pii-detector\webpii_dataset\sample\schema_sample_100.parquet"

df = pd.read_parquet(file)

for i, row in df.iterrows():

    if row["num_pii_elements"] > 0:

        print("Row:", i)
        print("Image size:", row["image_width"], "x", row["image_height"])
        print("Number of PII elements:", row["num_pii_elements"])

        print("\nPII JSON:")
        print(row["pii_elements_json"])

        break