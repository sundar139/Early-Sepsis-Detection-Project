from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "SEPSIS_PROJECT_ROOT"


def get_project_root(explicit_root: str | Path | None = None) -> Path:
    """Resolves the repository root used for portable artifact paths."""

    if explicit_root is not None:
        return Path(explicit_root).resolve()

    env_value = os.getenv(PROJECT_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd

    return Path(__file__).resolve().parents[2]


def resolve_runtime_path(
    path_value: str | Path,
    *,
    project_root: str | Path | None = None,
    anchor: str | Path | None = None,
) -> Path:
    """Resolves relative paths against project root, with optional anchor fallback."""

    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    resolved_project_root = get_project_root(project_root)
    root_candidate = (resolved_project_root / path).resolve()
    if root_candidate.exists() or anchor is None:
        return root_candidate

    anchor_candidate = (Path(anchor).resolve() / path).resolve()
    if anchor_candidate.exists():
        return anchor_candidate

    return root_candidate


def make_portable_path(
    path_value: str | Path,
    *,
    project_root: str | Path | None = None,
) -> str:
    """Converts absolute in-repository paths to repo-relative portable strings."""

    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()

    resolved_path = path.resolve()
    resolved_project_root = get_project_root(project_root)
    try:
        relative_path = resolved_path.relative_to(resolved_project_root)
        return relative_path.as_posix()
    except ValueError:
        return resolved_path.as_posix()


def sanitize_public_path(
    path_value: str | Path,
    *,
    allow_raw_paths: bool,
    project_root: str | Path | None = None,
) -> str:
    """Sanitizes filesystem paths for public API payloads."""

    path = Path(path_value)
    if allow_raw_paths:
        return str(path)

    if not path.is_absolute():
        return path.as_posix()

    resolved_path = path.resolve()
    resolved_project_root = get_project_root(project_root)
    try:
        relative_path = resolved_path.relative_to(resolved_project_root)
        return relative_path.as_posix()
    except ValueError:
        return "<redacted>"
