from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from early_sepsis.modeling.model_manifest import (
    load_model_manifest,
    rewrite_manifest_paths_portable,
    save_model_manifest,
)
from early_sepsis.runtime_paths import (
    get_project_root,
    make_portable_path,
    resolve_runtime_path,
)


def _rewrite_path_fields(payload: Any, *, project_root: Path, key_name: str | None = None) -> Any:
    if isinstance(payload, dict):
        return {
            key: _rewrite_path_fields(value, project_root=project_root, key_name=key)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [
            _rewrite_path_fields(item, project_root=project_root, key_name=key_name)
            for item in payload
        ]

    if isinstance(payload, str) and key_name is not None:
        normalized_key = key_name.lower()
        if "path" in normalized_key or "dir" in normalized_key:
            return make_portable_path(payload, project_root=project_root)

    return payload


def _update_related_json_paths(file_path: Path, *, project_root: Path, dry_run: bool) -> bool:
    if not file_path.exists():
        return False

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    rewritten_payload = _rewrite_path_fields(payload, project_root=project_root)
    if rewritten_payload == payload:
        return False

    if not dry_run:
        file_path.write_text(json.dumps(rewritten_payload, indent=2) + "\n", encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrites selected model manifest and calibration metadata path fields to "
            "portable repository-relative paths when possible."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/models/registry/selected_model.json"),
        help="Path to selected_model.json.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root override. Defaults to SEPSIS_PROJECT_ROOT or repository root.",
    )
    parser.add_argument(
        "--calibration-summary-path",
        type=Path,
        default=None,
        help="Optional explicit calibration summary path to rewrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = get_project_root(args.project_root)
    manifest_path = resolve_runtime_path(args.manifest_path, project_root=project_root)
    manifest = load_model_manifest(manifest_path)

    rewritten_manifest = rewrite_manifest_paths_portable(manifest, project_root=project_root)
    manifest_changed = rewritten_manifest != manifest

    if manifest_changed and not args.dry_run:
        save_model_manifest(manifest_path, rewritten_manifest)

    related_paths: set[Path] = set()
    threshold_metadata = rewritten_manifest.get("threshold_metadata")
    if isinstance(threshold_metadata, dict):
        for field_name in ("recommendations_path", "calibration_summary_path"):
            field_value = threshold_metadata.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                related_paths.add(
                    resolve_runtime_path(
                        field_value,
                        project_root=project_root,
                        anchor=manifest_path.parent,
                    )
                )

    if args.calibration_summary_path is not None:
        related_paths.add(
            resolve_runtime_path(args.calibration_summary_path, project_root=project_root)
        )

    updated_related_files: list[str] = []
    for related_path in sorted(related_paths):
        if _update_related_json_paths(
            related_path,
            project_root=project_root,
            dry_run=args.dry_run,
        ):
            updated_related_files.append(
                make_portable_path(related_path, project_root=project_root)
            )

    mode = "dry-run" if args.dry_run else "write"
    print(
        json.dumps(
            {
                "mode": mode,
                "manifest": make_portable_path(manifest_path, project_root=project_root),
                "manifest_updated": manifest_changed,
                "related_files_updated": updated_related_files,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
