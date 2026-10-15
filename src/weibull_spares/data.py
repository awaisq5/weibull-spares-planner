"""Reading a parts register, and turning scrap reasons into censoring.

One row per part: how many circulating hours it has accumulated, and whether
and why it was scrapped. Censoring follows the scrap reason:

* scrapped because it **wore out** (worn, washed out, cracked …): the life
  ended, so the observation is complete (MATLAB censoring flag 0);
* scrapped for a reason **unrelated to life** (lost in hole, handling damage,
  obsolete revision …): we only know it lasted at least this long, so it is
  right-censored (flag 1);
* **still in service**: right-censored too (flag 1), and part of the fleet
  that future forecasts are about.

Treating every scrapped part as a failure would bias life estimates downward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

LIFE_REASONS = {
    "worn",
    "wear",
    "worn out",
    "washout",
    "washed out",
    "eroded",
    "erosion",
    "cracked",
    "crack",
    "fatigue",
    "corroded",
    "corrosion",
    "out of tolerance",
    "failed",
}
NON_LIFE_REASONS = {
    "lost in hole",
    "lost",
    "handling damage",
    "damaged in handling",
    "transport damage",
    "obsolete",
    "obsolete revision",
    "design change",
    "customer retained",
    "sold",
}
SERVICE_WORDS = {"", "in service", "in-service", "active", "running", "none", "n/a"}


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class PartsRegister:
    part_id: np.ndarray
    part_number: np.ndarray
    location: np.ndarray
    hours: np.ndarray
    #: True where the life is fully observed (scrapped for a life reason).
    failed: np.ndarray
    #: True where the part is still in service.
    in_service: np.ndarray
    reason: np.ndarray
    unknown_reasons: list[str] = field(default_factory=list)

    @property
    def censoring(self) -> np.ndarray:
        """MATLAB-style censoring flags: 1 = right-censored, 0 = fully observed."""
        return (~self.failed).astype(int)

    def where(self, mask: np.ndarray) -> PartsRegister:
        return PartsRegister(
            self.part_id[mask],
            self.part_number[mask],
            self.location[mask],
            self.hours[mask],
            self.failed[mask],
            self.in_service[mask],
            self.reason[mask],
            self.unknown_reasons,
        )

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Outcome": np.select(
                    [self.failed, self.in_service],
                    ["Scrapped: worn out (failure)", "In service (censored)"],
                    "Scrapped: other reason (censored)",
                ),
                "reason": self.reason,
            }
        )
        return df.groupby("Outcome").size().rename("Parts").reset_index()


def _read(source: str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
    try:
        df = pd.read_csv(source, keep_default_na=False, dtype=str)
    except Exception as exc:  # pandas raises several types for malformed files
        raise DataError(f"Could not read the file as CSV ({exc}).") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        raise DataError("The parts register has no rows.")
    return df


def load_parts(
    source: str | Path | IO[str] | IO[bytes],
    life_reasons: set[str] | None = None,
    non_life_reasons: set[str] | None = None,
) -> PartsRegister:
    """Read ``part_id, part_number, location, circulating_hours, scrap_reason``.

    A ``censored`` column with MATLAB flags (1/0) may be given instead of
    ``scrap_reason``; it is used as is. Scrap reasons that are in neither list
    are reported, not guessed.
    """
    df = _read(source)
    required = ["part_id", "part_number", "location", "circulating_hours"]
    missing = [c for c in required if c not in df.columns]
    if "scrap_reason" not in df.columns and "censored" not in df.columns:
        missing.append("scrap_reason (or censored)")
    if missing:
        raise DataError(f"The parts register needs the column{'s' if len(missing) > 1 else ''} {', '.join(missing)}.")

    hours = pd.to_numeric(df["circulating_hours"].str.replace(",", "."), errors="coerce")
    bad = hours.isna() | (hours <= 0)
    if bad.any():
        row = int(np.argmax(bad.to_numpy())) + 2
        raise DataError(
            f"Row {row}: circulating_hours must be a positive number, got {df['circulating_hours'].iloc[row - 2]!r}."
        )

    life = {r.lower() for r in (life_reasons or LIFE_REASONS)}
    non_life = {r.lower() for r in (non_life_reasons or NON_LIFE_REASONS)}
    reason = df.get("scrap_reason", pd.Series([""] * len(df))).str.strip().str.lower()

    if "censored" in df.columns:
        flags = df["censored"].str.strip()
        bad = ~flags.isin(["0", "1"])
        if bad.any():
            row = int(np.argmax(bad.to_numpy())) + 2
            raise DataError(
                f"Row {row}: censored must be 1 (censored) or 0 (fully observed), got {flags.iloc[row - 2]!r}."
            )
        failed = (flags == "0").to_numpy()
        in_service = (reason.isin(SERVICE_WORDS) & (flags == "1")).to_numpy()
        unknown: list[str] = []
    else:
        in_service = reason.isin(SERVICE_WORDS).to_numpy()
        failed = reason.isin(life).to_numpy()
        known = in_service | failed | reason.isin(non_life).to_numpy()
        unknown = sorted(set(reason[~known]))
        if unknown:
            raise DataError(
                "Unknown scrap reason"
                + ("s" if len(unknown) > 1 else "")
                + ": "
                + ", ".join(repr(u) for u in unknown)
                + ". Say whether each one ends a part's life (failure) or not (censored)."
            )

    return PartsRegister(
        part_id=df["part_id"].to_numpy(),
        part_number=df["part_number"].to_numpy(),
        location=df["location"].to_numpy(),
        hours=hours.to_numpy(dtype=float),
        failed=failed,
        in_service=in_service,
        reason=reason.to_numpy(),
        unknown_reasons=unknown,
    )
