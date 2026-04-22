"""
ingest.py — Download and split the dataset from HuggingFace.
Saves train / validation / test splits as Parquet under data/raw/.
"""

import os
from datasets import load_dataset
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ingest(params: dict) -> None:
    raw_path = params["data"]["raw_path"]
    os.makedirs(raw_path, exist_ok=True)

    print("Downloading dataset from HuggingFace...")
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")

    # split original train into train + validation (80/20)
    split = ds["train"].train_test_split(test_size=0.2, seed=42)

    train_path = os.path.join(raw_path, "train.parquet")
    val_path   = os.path.join(raw_path, "validation.parquet")
    test_path  = os.path.join(raw_path, "test.parquet")

    split["train"].to_parquet(train_path)
    split["test"].to_parquet(val_path)
    ds["test"].to_parquet(test_path)

    print(f"Train      : {len(split['train'])} rows -> {train_path}")
    print(f"Validation : {len(split['test'])} rows -> {val_path}")
    print(f"Test       : {len(ds['test'])} rows -> {test_path}")
    print("Ingestion complete.")


if __name__ == "__main__":
    params = load_params()
    ingest(params) 