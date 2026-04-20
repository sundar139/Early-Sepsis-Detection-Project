from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def _as_2d_float32(value: Any, column_name: str) -> np.ndarray:
    materialized = value
    if isinstance(materialized, np.ndarray) and materialized.dtype == object:
        materialized = [np.asarray(item, dtype=np.float32) for item in materialized.tolist()]

    array = np.asarray(materialized, dtype=np.float32)
    if array.ndim != 2 and isinstance(materialized, (list, tuple)):
        array = np.vstack([np.asarray(item, dtype=np.float32) for item in materialized])

    if array.ndim != 2:
        msg = f"Column '{column_name}' must contain 2D arrays per row."
        raise ValueError(msg)
    return array


def _as_1d_float32(value: Any, column_name: str) -> np.ndarray:
    materialized = value.tolist() if isinstance(value, np.ndarray) else value
    array = np.asarray(materialized, dtype=np.float32)
    if array.ndim != 1:
        msg = f"Column '{column_name}' must contain 1D arrays per row."
        raise ValueError(msg)
    return array


class SepsisWindowDataset(Dataset[dict[str, Tensor]]):
    """PyTorch dataset for pre-generated sepsis prediction windows."""

    def __init__(
        self,
        parquet_path: str | Path,
        include_mask: bool = True,
        include_static: bool = False,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            msg = f"Window parquet file not found: {self.parquet_path}"
            raise FileNotFoundError(msg)

        frame = pd.read_parquet(self.parquet_path)
        if frame.empty:
            msg = f"Window parquet file is empty: {self.parquet_path}"
            raise ValueError(msg)

        self._frame = frame.reset_index(drop=True)
        self._include_mask = include_mask
        self._include_static = include_static

        if self._include_mask and "missing_mask" not in self._frame.columns:
            msg = "missing_mask column not found but include_mask=True"
            raise KeyError(msg)
        if self._include_static and "static_features" not in self._frame.columns:
            msg = "static_features column not found but include_static=True"
            raise KeyError(msg)

        self._labels = torch.tensor(self._frame["label"].to_numpy(dtype=np.float32))

    @property
    def labels(self) -> Tensor:
        """Returns labels tensor aligned with dataset rows."""

        return self._labels

    @property
    def labels_numpy(self) -> np.ndarray:
        """Returns labels as a NumPy array for sampler/statistics logic."""

        return self._labels.detach().cpu().numpy().astype(np.int64)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self._frame.iloc[index]
        features = torch.from_numpy(_as_2d_float32(row["features"], "features"))
        item: dict[str, Tensor] = {
            "features": features,
            "label": self._labels[index],
        }

        if self._include_mask:
            mask = torch.from_numpy(_as_2d_float32(row["missing_mask"], "missing_mask"))
            item["mask"] = mask

        if self._include_static:
            static_values = torch.from_numpy(
                _as_1d_float32(row["static_features"], "static_features")
            )
            item["static"] = static_values

        return item


def create_window_dataloader(
    parquet_path: str | Path,
    batch_size: int,
    shuffle: bool,
    include_mask: bool = True,
    include_static: bool = False,
    num_workers: int = 0,
) -> DataLoader[dict[str, Tensor]]:
    """Creates a DataLoader for window-level training/evaluation."""

    dataset = SepsisWindowDataset(
        parquet_path=parquet_path,
        include_mask=include_mask,
        include_static=include_static,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
