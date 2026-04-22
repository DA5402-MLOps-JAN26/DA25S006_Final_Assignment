"""
preprocess.py — Clean and normalise resume and JD text.
Reads from data/raw/, writes to data/processed/.
"""

import os
import re
import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, emails, special chars, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_min_tokens(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Remove rows where either text has fewer than min_count words."""
    before = len(df)
    mask = (
        df["resume_text"].str.split().str.len() >= min_count
    ) & (
        df["job_description_text"].str.split().str.len() >= min_count
    )
    df = df[mask].reset_index(drop=True)
    print(f"  Filtered {before - len(df)} rows below min_token_count={min_count}")
    return df


def preprocess_split(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Apply cleaning and filtering to a dataframe split."""
    df["resume_text"]          = df["resume_text"].apply(clean_text)
    df["job_description_text"] = df["job_description_text"].apply(clean_text)
    df = filter_min_tokens(df, min_count)
    return df


def preprocess(params: dict) -> None:
    raw_path       = params["data"]["raw_path"]
    processed_path = params["data"]["processed_path"]
    min_count      = params["data"]["min_token_count"]

    os.makedirs(processed_path, exist_ok=True)

    for split in ["train", "validation", "test"]:
        print(f"Processing {split}...")
        df = pd.read_parquet(os.path.join(raw_path, f"{split}.parquet"))
        df = preprocess_split(df, min_count)
        out_path = os.path.join(processed_path, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    params = load_params()
    preprocess(params)