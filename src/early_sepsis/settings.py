from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - optional until config loading is used.
    OmegaConf = None

load_dotenv()


class AppSettings(BaseSettings):
    """Application-level settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SEPSIS_",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    json_logs: bool = Field(default=True)

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)

    model_artifact_path: Path = Field(default=Path("artifacts/models/model.pkl"))
    selected_sequence_manifest_path: Path = Field(
        default=Path("artifacts/models/registry/selected_model.json")
    )
    serving_default_operating_mode: str = Field(default="default")

    enable_mlflow: bool = Field(default=False)
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment_name: str = Field(default="early-sepsis-local")

    enable_prefect: bool = Field(default=False)

    enable_local_llm: bool = Field(default=False)
    local_llm_endpoint: str = Field(default="http://127.0.0.1:11434/api/generate")
    local_llm_model: str = Field(default="mistral")
    local_llm_timeout_seconds: float = Field(default=20.0, gt=0)

    default_target_column: str = Field(default="SepsisLabel")
    random_seed: int = Field(default=42)
    train_test_split_ratio: float = Field(default=0.2, gt=0, lt=0.5)

    raw_data_dir: Path = Field(default=Path("data/raw"))
    local_csv_data_dir: Path = Field(default=Path("data/local_csv"))
    processed_data_dir: Path = Field(default=Path("artifacts/processed"))
    window_data_dir: Path = Field(default=Path("artifacts/windows"))
    split_manifest_dir: Path = Field(default=Path("artifacts/processed/split_manifests"))

    train_ratio: float = Field(default=0.7, gt=0)
    validation_ratio: float = Field(default=0.15, gt=0)
    test_ratio: float = Field(default=0.15, gt=0)

    window_length: int = Field(default=8, ge=1)
    prediction_horizon: int = Field(default=6, ge=1)
    enable_padding_windows: bool = Field(default=False)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        normalized = value.upper()
        if normalized not in allowed_levels:
            msg = f"Unsupported log level: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("serving_default_operating_mode")
    @classmethod
    def validate_serving_default_operating_mode(cls, value: str) -> str:
        allowed_modes = {"default", "balanced", "high_recall"}
        normalized = value.strip().lower()
        if normalized not in allowed_modes:
            msg = f"Unsupported serving operating mode: {value}"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_split_ratios(self) -> AppSettings:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if total <= 0:
            msg = "Split ratios must sum to a positive value."
            raise ValueError(msg)
        return self

    def ensure_runtime_directories(self) -> None:
        """Ensures runtime output directories exist."""

        self.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.selected_sequence_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.window_data_dir.mkdir(parents=True, exist_ok=True)
        self.split_manifest_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns cached settings to keep runtime behavior consistent."""

    settings = AppSettings()
    settings.ensure_runtime_directories()
    return settings


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file into a native dictionary."""

    if OmegaConf is None:
        msg = "omegaconf is required to load YAML configs. Install hydra-core dependencies first."
        raise ModuleNotFoundError(msg)

    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    loaded = OmegaConf.load(path)
    materialized = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(materialized, dict):
        msg = f"Config file must contain a mapping at root: {path}"
        raise ValueError(msg)

    return cast(dict[str, Any], materialized)
