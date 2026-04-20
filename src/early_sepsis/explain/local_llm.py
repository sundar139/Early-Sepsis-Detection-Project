from __future__ import annotations

from typing import Any, Mapping, Sequence

import httpx

from early_sepsis.logging_utils import get_logger
from early_sepsis.settings import AppSettings, get_settings

logger = get_logger(__name__)


def _fallback_explanation(risk_score: float) -> str:
    risk_label = "high" if risk_score >= 0.7 else "moderate" if risk_score >= 0.4 else "low"
    return (
        f"Predicted sepsis risk is {risk_label} ({risk_score:.2f}). "
        "Use this signal alongside temporal trends and clinician judgment."
    )


def _build_prompt(record: Mapping[str, Any], risk_score: float) -> str:
    return (
        "You are assisting with sepsis early-warning support. "
        "Write a concise, neutral explanation in 2 to 3 sentences.\n"
        f"Risk score: {risk_score:.4f}\n"
        f"Features: {dict(record)}\n"
        "Do not provide treatment instructions."
    )


def explain_prediction(
    record: Mapping[str, Any],
    risk_score: float,
    settings: AppSettings | None = None,
) -> str:
    """Generates one explanation using a local LLM endpoint when enabled."""

    resolved_settings = settings or get_settings()
    if not resolved_settings.enable_local_llm:
        return _fallback_explanation(risk_score)

    payload = {
        "model": resolved_settings.local_llm_model,
        "prompt": _build_prompt(record=record, risk_score=risk_score),
        "stream": False,
    }

    try:
        response = httpx.post(
            resolved_settings.local_llm_endpoint,
            json=payload,
            timeout=resolved_settings.local_llm_timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json()
        text = parsed.get("response") or parsed.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        logger.exception("Local LLM explanation request failed.")

    return _fallback_explanation(risk_score)


def explain_predictions(
    records: Sequence[Mapping[str, Any]],
    risk_scores: Sequence[float],
    settings: AppSettings | None = None,
) -> list[str]:
    """Generates explanations for a batch of predictions."""

    if len(records) != len(risk_scores):
        msg = "records and risk_scores must have matching lengths"
        raise ValueError(msg)

    resolved_settings = settings or get_settings()
    return [
        explain_prediction(record=record, risk_score=risk, settings=resolved_settings)
        for record, risk in zip(records, risk_scores, strict=True)
    ]
