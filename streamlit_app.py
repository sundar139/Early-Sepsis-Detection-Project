from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _apply_streamlit_secrets_to_env() -> None:
    env_mapping = {
        "environment": "SEPSIS_ENVIRONMENT",
        "selected_sequence_manifest_path": "SEPSIS_SELECTED_SEQUENCE_MANIFEST_PATH",
        "public_artifacts_dir": "SEPSIS_PUBLIC_ARTIFACTS_DIR",
        "serving_default_operating_mode": "SEPSIS_SERVING_DEFAULT_OPERATING_MODE",
        "demo_public_mode": "SEPSIS_DEMO_PUBLIC_MODE",
        "demo_sample_parquet_path": "SEPSIS_DEMO_SAMPLE_PARQUET_PATH",
        "project_root": "SEPSIS_PROJECT_ROOT",
    }

    try:
        secrets_section = st.secrets.get("sepsis", st.secrets)
    except Exception:
        return

    if not isinstance(secrets_section, Mapping):
        return

    for secret_key, env_key in env_mapping.items():
        if env_key in os.environ:
            continue
        value = secrets_section.get(secret_key)
        if value is None:
            continue
        os.environ[env_key] = str(value)


_apply_streamlit_secrets_to_env()


def _run_app() -> None:
    from early_sepsis.demo.app import main

    main()


if __name__ == "__main__":
    _run_app()
