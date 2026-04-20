from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from early_sepsis.demo.app import resolve_demo_threshold_for_mode
from early_sepsis.serving.sequence_service import SequenceServingError


def _manifest_payload() -> dict[str, dict[str, float]]:
    return {
        "thresholds": {
            "default": 0.5,
            "balanced": 0.7,
            "high_recall": 0.3,
        }
    }


def test_resolve_demo_threshold_for_mode_returns_expected_threshold() -> None:
    manifest = _manifest_payload()

    assert resolve_demo_threshold_for_mode(manifest, "default") == pytest.approx(0.5)
    assert resolve_demo_threshold_for_mode(manifest, "balanced") == pytest.approx(0.7)
    assert resolve_demo_threshold_for_mode(manifest, "high_recall") == pytest.approx(0.3)


def test_resolve_demo_threshold_for_mode_rejects_invalid_mode() -> None:
    manifest = _manifest_payload()

    with pytest.raises(SequenceServingError, match="operating_mode must be one of"):
        resolve_demo_threshold_for_mode(manifest, "invalid_mode")
