"""
evaluate.py — Evaluate the best fine-tuned model on the test set.
Logs final metrics and artifacts to MLflow.
"""

import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.model import BiEncoderClassifier
from src.model.dataset import ResumeFitDataset
from src.model.train import collate_fn, get_device, load_params


def plot_confusion_matrix(cm, labels, out_path):
    """Save confusion matrix as PNG."""
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Confusion matrix saved -> {out_path}")


def evaluate(params: dict) -> None:
    best_model     = params["models"]["best_model"]
    processed_path = params["data"]["processed_path"]
    batch_size     = params["training"]["batch_size"]
    labels         = params["dataset"]["labels"]
    device         = get_device(params)

    model_name_short = best_model.split("/")[-1]
    checkpoint_path  = f"models/{model_name_short}_phase2_best.pt"

    print(f"Loading model: {best_model}")
    print(f"Checkpoint   : {checkpoint_path}")

    # load model
    model = BiEncoderClassifier(best_model).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.eval()

    # load test set
    tokenizer = AutoTokenizer.from_pretrained(best_model)
    test_ds   = ResumeFitDataset(os.path.join(processed_path, "test.parquet"))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # run inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            r_ids  = batch["resume_input_ids"].to(device)
            r_mask = batch["resume_attention_mask"].to(device)
            j_ids  = batch["jd_input_ids"].to(device)
            j_mask = batch["jd_attention_mask"].to(device)

            logits = model(r_ids, r_mask, j_ids, j_mask)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    # metrics
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm       = confusion_matrix(all_labels, all_preds)
    report   = classification_report(
        all_labels, all_preds, target_names=labels, output_dict=True
    )

    print(f"\nTest Accuracy  : {acc:.4f}")
    print(f"Test Macro F1  : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=labels))

    # save artifacts
    os.makedirs("metrics", exist_ok=True)
    cm_path     = "metrics/confusion_matrix.png"
    report_path = "metrics/classification_report.json"

    plot_confusion_matrix(cm, labels, cm_path)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # save metrics for dvc
    metrics_out = {
        "test_accuracy":  round(acc, 4),
        "test_macro_f1":  round(macro_f1, 4),
    }
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # log to mlflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{model_name_short}_test_evaluation"):
        mlflow.log_params({
            "model_name": best_model,
            "phase":      "evaluation",
            "checkpoint": checkpoint_path,
        })
        mlflow.log_metrics({
            "test_accuracy": round(acc, 4),
            "test_macro_f1": round(macro_f1, 4),
        })
        # log per class f1
        for label in labels:
            mlflow.log_metric(
                f"f1_{label.replace(' ', '_')}",
                round(report[label]["f1-score"], 4),
            )
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

    print("\nEvaluation complete. Results logged to MLflow.")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)