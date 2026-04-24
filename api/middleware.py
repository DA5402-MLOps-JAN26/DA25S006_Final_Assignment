"""
middleware.py — Prometheus metrics instrumentation.
"""

from prometheus_client import Counter, Histogram, Gauge
import time

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total request count",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "http_request_exceptions_total",
    "Total error count",
    ["endpoint"],
)

LABEL_COUNT = Counter(
    "resume_fit_label_total",
    "Verdict counts per label",
    ["label"],
)

CONFIDENCE = Histogram(
    "resume_fit_confidence",
    "Model confidence distribution",
)