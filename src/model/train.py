"""
train.py — Two-phase training for BiEncoderClassifier.
Phase 1: Train classifier head only (frozen encoder) for all 3 models.
Phase 2: Full fine-tuning of best model (encoder + head).
All runs logged to MLflow.
"""

import os
import time
import json
import subprocess
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch

from src.model.model import BiEncoderClassifier
from src.model.dataset import ResumeFitDataset


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def get_device(params: dict) -> torch.device:
    if params["training"]["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_fn(batch: list, tokenizer, max_length: int = 512) -> dict:
    """Tokenise a batch of (resume, jd, label) samples."""
    resumes = [b["resume"] for b in batch]
    jds     = [b["jd"]     for b in batch]
    labels  = torch.stack([b["label"] for b in batch])

    r_enc = tokenizer(
        resumes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    j_enc = tokenizer(
        jds,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "resume_input_ids":      r_enc["input_ids"],
        "resume_attention_mask": r_enc["attention_mask"],
        "jd_input_ids":          j_enc["input_ids"],
        "jd_attention_mask":     j_enc["attention_mask"],
        "labels":                labels,
    }


def run_epoch(
    model, loader, optimizer, loss_fn, device, training: bool
) -> tuple:
    """Run one epoch. Returns (loss, accuracy, macro_f1)."""
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            r_ids  = batch["resume_input_ids"].to(device)
            r_mask = batch["resume_attention_mask"].to(device)
            j_ids  = batch["jd_input_ids"].to(device)
            j_mask = batch["jd_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(r_ids, r_mask, j_ids, j_mask)
            loss   = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item()
            preds        = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def compute_class_weights(dataset: ResumeFitDataset) -> torch.Tensor:
    """Compute inverse frequency weights to handle class imbalance."""
    counts = np.bincount(dataset.labels)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)


def train_model(
    model_name: str,
    params: dict,
    phase: int,
    device: torch.device,
    freeze_encoder: bool,
) -> dict:
    """
    Train one model for one phase.
    Returns dict of best metrics.
    """
    batch_size  = params["training"]["batch_size"]
    epochs      = params["training"][f"phase{phase}_epochs"]
    head_lr     = params["training"]["head_lr"]
    encoder_lr  = params["training"]["encoder_lr"]
    dropout     = params["training"]["dropout"]
    patience    = params["training"]["early_stopping_patience"]
    processed   = params["data"]["processed_path"]

    # datasets
    train_ds = ResumeFitDataset(os.path.join(processed, "train.parquet"))
    val_ds   = ResumeFitDataset(os.path.join(processed, "validation.parquet"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # model
    model = BiEncoderClassifier(model_name, dropout=dropout).to(device)

    # freeze encoder for phase 1
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=head_lr
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(),   "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ])

    # weighted loss for class imbalance
    weights  = compute_class_weights(train_ds).to(device)
    loss_fn  = nn.CrossEntropyLoss(weight=weights)

    best_f1        = 0.0
    best_metrics   = {}
    no_improve     = 0
    model_name_short = model_name.split("/")[-1]

    # mlflow run
    run_name = f"{model_name_short}_phase{phase}"
    with mlflow.start_run(run_name=run_name):
        # log params
        mlflow.log_params({
            "model_name":     model_name,
            "phase":          phase,
            "epochs":         epochs,
            "batch_size":     batch_size,
            "head_lr":        head_lr,
            "encoder_lr":     encoder_lr if not freeze_encoder else "frozen",
            "dropout":        dropout,
            "freeze_encoder": freeze_encoder,
            "git_commit":     get_git_commit(),
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc, train_f1 = run_epoch(
                model, train_loader, optimizer, loss_fn, device, training=True
            )
            val_loss, val_acc, val_f1 = run_epoch(
                model, val_loader, optimizer, loss_fn, device, training=False
            )

            elapsed = time.time() - t0

            # log metrics per epoch
            mlflow.log_metrics({
                "train_loss":     round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "train_macro_f1": round(train_f1, 4),
                "val_loss":       round(val_loss, 4),
                "val_accuracy":   round(val_acc, 4),
                "val_macro_f1":   round(val_f1, 4),
            }, step=epoch)

            print(
                f"  Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_metrics = {
                    "val_loss":       round(val_loss, 4),
                    "val_accuracy":   round(val_acc, 4),
                    "val_macro_f1":   round(val_f1, 4),
                }
                # save best checkpoint
                os.makedirs("models", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"models/{model_name_short}_phase{phase}_best.pt"
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # log best metrics as summary
        mlflow.log_metrics({
            "best_val_macro_f1": round(best_f1, 4),
        })

        mlflow.set_tags({
            "phase":      str(phase),
            "model_name": model_name,
        })

        print(f"  Best val_macro_f1: {best_f1:.4f}")
        print(f"  Checkpoint saved: models/{model_name_short}_phase{phase}_best.pt")

    return best_metrics


def phase1(params: dict, device: torch.device) -> None:
    """Train all 3 candidate models with frozen encoder."""
    candidates = params["models"]["candidates"]
    experiment  = params["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment)

    print("\n" + "=" * 60)
    print("PHASE 1 — Comparing 3 models (frozen encoder)")
    print("=" * 60)

    results = {}
    for model_name in candidates:
        short = model_name.split("/")[-1]
        print(f"\nTraining: {short}")
        metrics = train_model(
            model_name=model_name,
            params=params,
            phase=1,
            device=device,
            freeze_encoder=True,
        )
        results[model_name] = metrics

    # pick best model
    best_model = max(results, key=lambda m: results[m]["val_macro_f1"])
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS:")
    for m, r in results.items():
        print(f"  {m.split('/')[-1]}: val_macro_f1={r['val_macro_f1']}")
    print(f"\nBest model: {best_model}")
    print("=" * 60)

    # save results
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/phase1_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_model}, f, indent=2)

    print("\nUpdate params.yaml → models.best_model with the best model name above.")
    print("Then run: python3 src/model/train.py --phase 2")


def phase2(params: dict, device: torch.device) -> None:
    """Full fine-tuning of the best model."""
    best_model = params["models"]["best_model"]
    if not best_model:
        raise ValueError(
            "params.yaml → models.best_model is null. "
            "Run phase 1 first and update params.yaml."
        )

    experiment = params["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment)

    print("\n" + "=" * 60)
    print(f"PHASE 2 — Full fine-tuning: {best_model}")
    print("=" * 60)

    metrics = train_model(
        model_name=best_model,
        params=params,
        phase=2,
        device=device,
        freeze_encoder=False,
    )

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/phase2_results.json", "w") as f:
        json.dump({"best_model": best_model, "metrics": metrics}, f, indent=2)

    print("\nPhase 2 complete. Check MLflow UI for run details.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="1 = compare all models, 2 = fine-tune best model"
    )
    args   = parser.parse_args()
    params = load_params()
    device = get_device(params)

    print(f"Device: {device}")

    if args.phase == 1:
        phase1(params, device)
    else:
        phase2(params, device)