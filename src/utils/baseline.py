"""
baseline.py — Compute and store embedding baseline statistics.
Used later for drift detection during inference.
"""

import os
import json
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_baseline(params: dict) -> None:
    processed_path = params["data"]["processed_path"]

    print("Loading training data...")
    train = pd.read_parquet(os.path.join(processed_path, "train.parquet"))

    # combine resume + jd text for each row
    texts = (
        train["resume_text"] + " [SEP] " + train["job_description_text"]
    ).tolist()

    print("Loading encoder model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Encoding {len(texts)} training samples...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # compute mean and variance across all training embeddings
    mean = embeddings.mean(axis=0).tolist()
    var  = embeddings.var(axis=0).tolist()

    baseline = {
        "mean": mean,
        "variance": var,
        "num_samples": len(texts),
        "embedding_dim": embeddings.shape[1],
    }

    out_path = os.path.join(processed_path, "baseline_stats.json")
    with open(out_path, "w") as f:
        json.dump(baseline, f)

    print(f"Baseline saved -> {out_path}")
    print(f"  Samples       : {baseline['num_samples']}")
    print(f"  Embedding dim : {baseline['embedding_dim']}")


if __name__ == "__main__":
    params = load_params()
    compute_baseline(params)