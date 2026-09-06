import os

# Disable Xet to avoid CAS reconstruction errors
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset
ds = load_dataset("WebPII/webpii")
print(ds)
