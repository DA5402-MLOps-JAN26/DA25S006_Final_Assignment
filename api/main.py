"""
main.py — FastAPI inference backend for Resume Fit Analyzer.
"""

import os
import time
import io
import yaml
import torch
import numpy as np
import pdfplumber
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoTokenizer
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import mlflow

from api.schemas import PredictResponse
from api.middleware import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    LABEL_COUNT,
    CONFIDENCE,
)
from src.model.model import BiEncoderClassifier
from src.model.train import collate_fn


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


params     = load_params()
LABELS     = params["dataset"]["labels"]
BEST_MODEL = params["models"]["best_model"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_short = BEST_MODEL.split("/")[-1]
checkpoint_path  = f"models/{model_name_short}_phase2_best.pt"

print(f"Loading model: {BEST_MODEL}")
print(f"Device       : {DEVICE}")

model     = BiEncoderClassifier(BEST_MODEL).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL)

print("Model loaded and ready.")

app = FastAPI(
    title="Resume Fit Analyzer",
    description="Predicts whether a resume is a Good Fit, Potential Fit, or No Fit for a job description.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


def predict_fit(resume_text: str, jd_text: str) -> dict:
    """Run model inference on a resume+JD pair."""
    batch = [{
        "resume": resume_text,
        "jd":     jd_text,
        "label":  torch.tensor(0),
    }]

    encoded = collate_fn(batch, tokenizer)

    r_ids  = encoded["resume_input_ids"].to(DEVICE)
    r_mask = encoded["resume_attention_mask"].to(DEVICE)
    j_ids  = encoded["jd_input_ids"].to(DEVICE)
    j_mask = encoded["jd_attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(r_ids, r_mask, j_ids, j_mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx    = int(np.argmax(probs))
    label       = LABELS[pred_idx]
    confidence  = float(probs[pred_idx])

    return {"label": label, "confidence": confidence}


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness check — confirms model is loaded."""
    return {"model_loaded": True, "model": BEST_MODEL}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
):
    """
    Accept a resume PDF and job description text.
    Returns label, confidence, and latency.
    """
    start = time.time()

    try:
        # validate file type
        if not resume.filename.endswith(".pdf"):
            ERROR_COUNT.labels(endpoint="/predict").inc()
            raise HTTPException(
                status_code=422,
                detail="Only PDF files are accepted.",
            )

        # validate job description
        if not job_description.strip():
            ERROR_COUNT.labels(endpoint="/predict").inc()
            raise HTTPException(
                status_code=422,
                detail="Job description cannot be empty.",
            )

        # extract text from PDF
        file_bytes  = await resume.read()
        resume_text = extract_text_from_pdf(file_bytes)

        if not resume_text.strip():
            ERROR_COUNT.labels(endpoint="/predict").inc()
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF.",
            )

        # run inference
        result     = predict_fit(resume_text, job_description)
        latency_ms = round((time.time() - start) * 1000, 2)

        # update prometheus metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
        LABEL_COUNT.labels(label=result["label"]).inc()
        CONFIDENCE.observe(result["confidence"])

        return PredictResponse(
            label=result["label"],
            confidence=round(result["confidence"], 4),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    REQUEST_COUNT.labels(endpoint="/metrics", status="200").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/confusion-matrix")
def get_confusion_matrix():
    path = "/app/metrics/confusion_matrix.png"
    if not os.path.exists(path):
        return {"error": "Confusion matrix not found"}
    return FileResponse(path, media_type="image/png")