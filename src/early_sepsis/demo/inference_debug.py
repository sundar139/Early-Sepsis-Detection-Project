from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _as_float_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == object:
        array = np.asarray(array.tolist(), dtype=np.float32)
    else:
        array = np.asarray(value, dtype=np.float32)

    if array.ndim != 2:
        msg = "Expected a 2D matrix payload"
        raise ValueError(msg)
    return array


def _as_float_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        msg = "Expected a 1D vector payload"
        raise ValueError(msg)
    return array


def _extract_probability_scalar(value: Any) -> tuple[float, tuple[int, ...]]:
    array = np.asarray(value, dtype=np.float64)
    shape = tuple(int(item) for item in array.shape)

    if array.ndim == 0:
        return float(array), shape
    if array.size == 1:
        return float(array.reshape(())), shape

    msg = "predicted_probability payload must resolve to one scalar value"
    raise ValueError(msg)


def _hash_sample(sample: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()

    features = _as_float_matrix(sample.get("features"))
    hasher.update(features.shape[0].to_bytes(4, byteorder="little", signed=False))
    hasher.update(features.shape[1].to_bytes(4, byteorder="little", signed=False))
    hasher.update(np.ascontiguousarray(features).tobytes())

    missing_mask = sample.get("missing_mask")
    if missing_mask is not None:
        mask_matrix = _as_float_matrix(missing_mask)
        hasher.update(mask_matrix.shape[0].to_bytes(4, byteorder="little", signed=False))
        hasher.update(mask_matrix.shape[1].to_bytes(4, byteorder="little", signed=False))
        hasher.update(np.ascontiguousarray(mask_matrix).tobytes())

    static_features = sample.get("static_features")
    if static_features is not None:
        static_vector = _as_float_vector(static_features)
        hasher.update(static_vector.shape[0].to_bytes(4, byteorder="little", signed=False))
        hasher.update(np.ascontiguousarray(static_vector).tobytes())

    return hasher.hexdigest()


def count_unique_demo_windows(samples: Sequence[Mapping[str, Any]]) -> int:
    return len({_hash_sample(sample) for sample in samples})


def extract_probability_array(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    values: list[float] = []
    shapes: list[tuple[int, ...]] = []

    for prediction in predictions:
        probability_value, probability_shape = _extract_probability_scalar(
            prediction.get("predicted_probability")
        )
        values.append(probability_value)
        shapes.append(probability_shape)

    if not values:
        return np.zeros((0,), dtype=np.float64), shapes

    return np.asarray(values, dtype=np.float64), shapes


def build_inference_diagnostics(
    *,
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    displayed_scores: Sequence[float] | None = None,
    displayed_round_decimals: int | None = None,
) -> dict[str, Any]:
    probabilities, probability_shapes = extract_probability_array(predictions)

    displayed_matches: bool | None = None
    if displayed_scores is not None:
        displayed_array = np.asarray(list(displayed_scores), dtype=np.float64)
        if probabilities.shape[0] != displayed_array.shape[0]:
            displayed_matches = False
        elif displayed_round_decimals is None:
            displayed_matches = bool(np.allclose(displayed_array, probabilities, atol=1e-12))
        else:
            rounded = np.round(probabilities, displayed_round_decimals)
            displayed_matches = bool(np.allclose(displayed_array, rounded, atol=1e-12))

    unique_probability_count = int(np.unique(np.round(probabilities, 12)).shape[0])

    return {
        "sample_count": len(samples),
        "unique_window_count": int(count_unique_demo_windows(samples)),
        "probability_count": int(probabilities.shape[0]),
        "unique_probability_count": unique_probability_count,
        "probability_shapes": [list(shape) for shape in probability_shapes],
        "probability_min": float(probabilities.min()) if probabilities.size else None,
        "probability_max": float(probabilities.max()) if probabilities.size else None,
        "displayed_scores_match": displayed_matches,
    }
