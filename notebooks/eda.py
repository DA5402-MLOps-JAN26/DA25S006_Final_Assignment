import pandas as pd
import numpy as np

train = pd.read_parquet("data/raw/train.parquet")
val   = pd.read_parquet("data/raw/validation.parquet")
test  = pd.read_parquet("data/raw/test.parquet")

print("=" * 60)
print("SHAPE")
print(f"  Train      : {train.shape}")
print(f"  Validation : {val.shape}")
print(f"  Test       : {test.shape}")

print("\n" + "=" * 60)
print("COLUMNS & DTYPES")
print(train.dtypes)

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION - Train")
print(train['label'].value_counts())
print("\nLABEL DISTRIBUTION - Validation")
print(val['label'].value_counts())
print("\nLABEL DISTRIBUTION - Test")
print(test['label'].value_counts())

print("\n" + "=" * 60)
print("NULL VALUES - Train")
print(train.isnull().sum())

train['resume_len'] = train['resume_text'].str.split().str.len()
train['jd_len']     = train['job_description_text'].str.split().str.len()

print("\n" + "=" * 60)
print("RESUME TEXT LENGTH (words)")
print(train['resume_len'].describe().round(1))

print("\n" + "=" * 60)
print("JD TEXT LENGTH (words)")
print(train['jd_len'].describe().round(1))

print("\n" + "=" * 60)
print("ALL UNIQUE LABELS:", train['label'].unique().tolist())

print("\n" + "=" * 60)
for label in train['label'].unique():
    row = train[train['label'] == label].iloc[0]
    print(f"\nSAMPLE - {label}")
    print(f"  Resume (200 chars) : {row['resume_text'][:200]}")
    print(f"  JD     (200 chars) : {row['job_description_text'][:200]}")