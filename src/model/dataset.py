"""
dataset.py — PyTorch Dataset for (resume, JD, label) triples.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class ResumeFitDataset(Dataset):
    """
    Loads resume + JD text pairs and their labels.
    Tokenisation is handled inside the training loop
    so we can swap tokenisers per model easily.
    """

    LABEL2ID = {
        "Good Fit": 0,
        "Potential Fit": 1,
        "No Fit": 2,
    }

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.resumes = df["resume_text"].tolist()
        self.jds     = df["job_description_text"].tolist()
        self.labels  = [self.LABEL2ID[l] for l in df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "resume": self.resumes[idx],
            "jd":     self.jds[idx],
            "label":  torch.tensor(self.labels[idx], dtype=torch.long),
        }