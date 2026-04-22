"""
validate.py - Validate processed data before model training.
Checks schema, labels, nulls and minimum row counts.
"""

import os
import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_split(df: pd.DataFrame, split: str, valid_labels: list) -> None:
    """Run all checks on a single split."""

    # check required columns exist
    required = {"resume_text", "job_description_text", "label"}
    missing = required - set(df.columns)
    assert not missing, f"[{split}] Missing columns: {missing}"

    # check no nulls
    nulls = df[list(required)].isnull().sum()
    assert nulls.sum() == 0, f"[{split}] Null values found:\n{nulls}"

    # check no empty strings
    empty_resume = (df["resume_text"].str.strip() == "").sum()
    empty_jd     = (df["job_description_text"].str.strip() == "").sum()
    assert empty_resume == 0, f"[{split}] {empty_resume} empty resume_text rows"
    assert empty_jd == 0,     f"[{split}] {empty_jd} empty job_description_text rows"

    # check label values
    actual_labels = set(df["label"].unique())
    unexpected    = actual_labels - set(valid_labels)
    assert not unexpected, f"[{split}] Unexpected labels: {unexpected}"

    # check minimum row count
    assert len(df) >= 100, f"[{split}] Too few rows: {len(df)}"

    print(f"  [{split}] OK - {len(df)} rows, labels: {df['label'].value_counts().to_dict()}")


def validate(params: dict) -> None:
    processed_path = params["data"]["processed_path"]
    valid_labels   = params["dataset"]["labels"]

    print("Validating processed data...")
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(os.path.join(processed_path, f"{split}.parquet"))
        validate_split(df, split, valid_labels)

    print("Validation complete.")


if __name__ == "__main__":
    params = load_params()
    validate(params)