"""
schemas.py — Pydantic request and response models for the FastAPI app.
"""

from pydantic import BaseModel


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float