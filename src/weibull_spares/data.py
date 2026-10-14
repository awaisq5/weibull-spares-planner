"""Reading field data from CSV, with errors that say which row to fix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

FAILED = {"failed", "failure", "f", "1", "true", "yes"}
SUSPENDED = {"suspended", "suspension", "running", "s", "0", "false", "no", "removed"}


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class History:
    unit_ids: np.ndarray
    hours: np.ndarray
    failed: np.ndarray


def _read(source: str | Path | IO[str] | IO[bytes], required: list[str], name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pandas raises several types for malformed files
        raise DataError(f"{name}: could not read the file as CSV ({exc}).") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name} needs the column{'s' if len(missing) > 1 else ''} {', '.join(missing)}.")
    if df.empty:
        raise DataError(f"{name} has no rows.")
    return df


def _hours(df: pd.DataFrame, column: str, name: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce")
    bad = values.isna() | (values <= 0)
    if bad.any():
        row = int(np.argmax(bad.to_numpy())) + 2  # +1 header, +1 one-based
        raise DataError(f"{name}, row {row}: {column} must be a positive number, got {df[column].iloc[row - 2]!r}.")
    return values.to_numpy(dtype=float)


def load_history(source: str | Path | IO[str] | IO[bytes]) -> History:
    """Life history: one row per unit life, ``unit_id, hours, status``.

    ``status`` is ``failed`` for a failure, ``suspended`` for a unit that is still
    running or was removed for another reason (a censored observation).
    """
    df = _read(source, ["unit_id", "hours", "status"], "History file")
    hours = _hours(df, "hours", "History file")
    status = df["status"].astype(str).str.strip().str.lower()
    unknown = ~status.isin(FAILED | SUSPENDED)
    if unknown.any():
        row = int(np.argmax(unknown.to_numpy())) + 2
        raise DataError(
            f"History file, row {row}: status should be 'failed' or 'suspended', got {df['status'].iloc[row - 2]!r}."
        )
    return History(df["unit_id"].astype(str).to_numpy(), hours, status.isin(FAILED).to_numpy())


def load_fleet(source: str | Path | IO[str] | IO[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Installed base: ``unit_id, age_hours`` for every unit currently in service."""
    df = _read(source, ["unit_id", "age_hours"], "Fleet file")
    return df["unit_id"].astype(str).to_numpy(), _hours(df, "age_hours", "Fleet file")
