from pathlib import Path
from typing import Optional

import pandas as pd

from src.movie_review_understanding.config.settings import (
    DEFAULT_DATASET_CANDIDATES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
)


def resolve_dataset_path(data_path: Optional[Path] = None) -> Path:
    """Resolve a dataset path from an explicit argument or common raw-data filenames."""
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    candidate_names = ", ".join(path.name for path in DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(
        "No dataset file found in data/raw. "
        f"Expected one of: {candidate_names}"
    )


def _read_dataframe(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".tsv":
        return pd.read_csv(data_path, sep="\t")
    if suffix == ".json":
        return pd.read_json(data_path)
    if suffix == ".jsonl":
        return pd.read_json(data_path, lines=True)

    raise ValueError(
        f"Unsupported dataset format: {suffix}. Use csv, tsv, json, or jsonl."
    )


def load_reviews(
    data_path: Optional[Path] = None,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Load a local review dataset and standardize the key columns."""
    resolved_path = resolve_dataset_path(data_path)
    dataframe = _read_dataframe(resolved_path)

    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        available = ", ".join(str(column) for column in dataframe.columns)
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {available}"
        )

    standardized = dataframe[[text_column, label_column]].copy()
    standardized = standardized.rename(
        columns={text_column: DEFAULT_TEXT_COLUMN, label_column: DEFAULT_LABEL_COLUMN}
    )
    standardized = standardized.dropna(
        subset=[DEFAULT_TEXT_COLUMN, DEFAULT_LABEL_COLUMN]
    ).reset_index(drop=True)
    standardized[DEFAULT_TEXT_COLUMN] = standardized[DEFAULT_TEXT_COLUMN].astype(str)

    return standardized
