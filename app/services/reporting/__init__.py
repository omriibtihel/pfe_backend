"""LLM-backed prediction report generation.

This package converts a single-row prediction (with optional LIME
explanations) into a patient-readable Markdown/JSON report.

Public entry points:
- ``ReportService.generate_stream(context)`` → async generator of SSE chunks
- ``ReportContextBuilder.build(...)`` → typed report context
- ``TemplateClient`` → deterministic fallback used when no LLM is available
"""
from __future__ import annotations

from app.services.reporting.context_builder import (
    CounterfactualChange,
    FeatureContribution,
    ModelQualitySignal,
    PredictionClassContext,
    ReportContext,
    ReportContextBuilder,
)
from app.services.reporting.service import ReportService

__all__ = [
    "CounterfactualChange",
    "FeatureContribution",
    "ModelQualitySignal",
    "PredictionClassContext",
    "ReportContext",
    "ReportContextBuilder",
    "ReportService",
]
