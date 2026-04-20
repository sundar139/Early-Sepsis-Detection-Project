from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from early_sepsis.data.torch_dataset import SepsisWindowDataset, create_window_dataloader


def test_torch_window_dataset_shapes(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    parquet_path = tmp_path / "windows.parquet"
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "end_hour": [5.0, 6.0],
            "label": [0, 1],
            "features": [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
            ],
            "missing_mask": [
                [[0, 1], [0, 0], [1, 0]],
                [[0, 0], [0, 1], [0, 0]],
            ],
            "static_features": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    frame.to_parquet(parquet_path, index=False)

    dataset = SepsisWindowDataset(parquet_path, include_mask=True, include_static=True)
    sample = dataset[0]

    assert tuple(sample["features"].shape) == (3, 2)
    assert tuple(sample["mask"].shape) == (3, 2)
    assert tuple(sample["static"].shape) == (2,)

    dataloader = create_window_dataloader(
        parquet_path=parquet_path,
        batch_size=2,
        shuffle=False,
        include_mask=True,
        include_static=True,
    )
    batch = next(iter(dataloader))

    assert tuple(batch["features"].shape) == (2, 3, 2)
    assert tuple(batch["mask"].shape) == (2, 3, 2)
    assert tuple(batch["static"].shape) == (2, 2)
