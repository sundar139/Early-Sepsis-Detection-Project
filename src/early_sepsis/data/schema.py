from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

PATIENT_ID_COLUMN: Final[str] = "patient_id"
TIME_COLUMN: Final[str] = "ICULOS"
TARGET_COLUMN: Final[str] = "SepsisLabel"
SOURCE_FILE_COLUMN: Final[str] = "_source_file"
SOURCE_ROW_COLUMN: Final[str] = "_source_row"

PATIENT_ID_ALIASES: Final[tuple[str, ...]] = (
    "patient_id",
    "PatientID",
    "patient",
    "pid",
    "subject_id",
)
TIME_ALIASES: Final[tuple[str, ...]] = (
    "ICULOS",
    "iculos",
    "Hour",
    "hour",
    "hours",
    "hours_from_admit",
)
TARGET_ALIASES: Final[tuple[str, ...]] = (
    "SepsisLabel",
    "sepsis_label",
    "label",
    "target",
    "sepsis",
)
ROW_SOURCE_PATIENT_ALIASES: Final[tuple[str, ...]] = (
    "source",
    "Source",
    "record_source",
    "patient_source",
)

INTERNAL_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        PATIENT_ID_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_COLUMN,
    }
)


@dataclass(slots=True)
class SchemaValidationIssue:
    """Represents one schema validation issue from an input file."""

    file_path: str
    reason: str


def first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Returns the first candidate present in a set of columns."""

    available = list(columns)

    normalized_to_original: dict[str, str] = {}
    for column in available:
        normalized = "".join(character for character in str(column).lower() if character.isalnum())
        if normalized not in normalized_to_original:
            normalized_to_original[normalized] = str(column)

    for candidate in candidates:
        if candidate in available:
            return str(candidate)

        normalized_candidate = "".join(
            character for character in str(candidate).lower() if character.isalnum()
        )
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    return None


def infer_feature_columns(columns: Iterable[str]) -> list[str]:
    """Returns model feature columns by excluding internal metadata columns."""

    return [column for column in columns if column not in INTERNAL_COLUMNS]
