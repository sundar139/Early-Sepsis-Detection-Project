from __future__ import annotations

from pathlib import Path

from early_sepsis.data.ingestion import load_dataset

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_load_csv_fixture() -> None:
    dataframe = load_dataset(FIXTURES_DIR / "synthetic_tabular.csv", dataset_format="csv")

    assert len(dataframe) == 12
    assert {"HR", "O2Sat", "Temp", "SepsisLabel"}.issubset(dataframe.columns)


def test_load_physionet_fixture() -> None:
    dataframe = load_dataset(FIXTURES_DIR, dataset_format="physionet")

    assert len(dataframe) >= 3
    assert "patient_id" in dataframe.columns


def test_load_csv_drops_invalid_time_target_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "kaggle_like.csv"
    csv_path.write_text(
        "ICULOS,HR,SepsisLabel,Patient_ID\n"
        "0,80,0,patient_1\n"
        "1,82,0,patient_1\n"
        "2,84,1,patient_1\n"
        "3,90,,patient_1\n",
        encoding="utf-8",
    )

    dataframe = load_dataset(csv_path, dataset_format="csv")

    assert len(dataframe) == 3
    assert dataframe["SepsisLabel"].isna().sum() == 0
    assert dataframe["ICULOS"].isna().sum() == 0


def test_load_csv_recognizes_patient_id_alias(tmp_path: Path) -> None:
    csv_path = tmp_path / "kaggle_like_ids.csv"
    csv_path.write_text(
        "ICULOS,HR,SepsisLabel,Patient_ID\n"
        "0,80,0,patient_1\n"
        "1,82,0,patient_1\n"
        "0,90,0,patient_2\n"
        "1,92,1,patient_2\n",
        encoding="utf-8",
    )

    dataframe = load_dataset(csv_path, dataset_format="csv")

    assert "patient_id" in dataframe.columns
    assert dataframe["patient_id"].nunique() == 2
