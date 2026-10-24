"""Dated membership snapshots with explicit coverage and security identity limits."""

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
import re

import pandas as pd


def _date(value, name: str) -> pd.Timestamp:
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{name} must be a date in YYYY-MM-DD format.")
    elif not isinstance(value, (date, datetime, pd.Timestamp)):
        raise ValueError(f"{name} must be a calendar date.")
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid calendar date.") from exc
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError(f"{name} must be a calendar date without a time or time zone.")
    return result


@dataclass(frozen=True)
class MembershipHistory:
    """Each complete roster takes effect on its date, through the next change.

    Symbols are exact price-column identifiers. No ticker renames or share-class
    conversions occur here. Custom data must supply stable security identifiers.
    """

    snapshots: pd.Series
    coverage_end: pd.Timestamp
    source: str = "User-supplied membership CSV"
    limitations: tuple[str, ...] = ()
    _provenance: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not isinstance(self.snapshots, pd.Series) or self.snapshots.empty:
            raise ValueError("Membership snapshots must contain at least one full roster.")
        dates = pd.DatetimeIndex([_date(d, "Snapshot date") for d in self.snapshots.index])
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Snapshot dates must be unique and in increasing order.")
        rosters = []
        for roster in self.snapshots:
            if not isinstance(roster, (list, tuple)) or not roster:
                raise ValueError("Each membership snapshot must contain a nonempty roster.")
            if any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", s) for s in roster):
                raise ValueError("Membership symbols must be nonempty price-column identifiers without spaces.")
            if len(roster) != len(set(roster)):
                raise ValueError("A membership snapshot contains duplicate symbols.")
            rosters.append(tuple(roster))
        coverage_end = _date(self.coverage_end, "Membership coverage end")
        if coverage_end < dates[-1]:
            raise ValueError("Membership coverage end must be on or after the last snapshot.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Membership source must be a nonempty description.")
        object.__setattr__(self, "snapshots", pd.Series(rosters, index=dates, name="tickers"))
        object.__setattr__(self, "coverage_end", coverage_end)

    @property
    def coverage_start(self) -> pd.Timestamp:
        return self.snapshots.index[0]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return all raw identifiers in their first observed order."""
        return tuple(dict.fromkeys(s for roster in self.snapshots for s in roster))

    @property
    def metadata(self) -> dict:
        return {
            **self._provenance,
            "source": self.source,
            "coverage_start": self.coverage_start.date().isoformat(),
            "coverage_end": self.coverage_end.date().isoformat(),
            "snapshot_count": len(self.snapshots),
            "symbol_count": len(self.symbols),
            "limitations": list(self.limitations),
        }

    def at(self, date) -> tuple[str, ...]:
        """Return the last full roster effective on or before the requested date."""
        date = _date(date, "Membership lookup date")
        if not self.coverage_start <= date <= self.coverage_end:
            raise ValueError(
                f"Membership date {date.date()} is outside coverage "
                f"{self.coverage_start.date()} through {self.coverage_end.date()}."
            )
        return self.snapshots.iloc[self.snapshots.index.searchsorted(date, side="right") - 1]

    def identity_exclusions(self, date, *, through=None) -> dict[str, str]:
        """Find known unsafe raw labels anywhere in an inclusive price interval.

        Price history can predate membership coverage. Exclusion end dates are
        exclusive. This does not certify identities absent from the rule list.
        """
        start = _date(date, "Identity interval start")
        end = start if through is None else _date(through, "Identity interval end")
        if end < start:
            raise ValueError("Identity interval end must be on or after its start.")
        return {
            rule["symbol"]: rule["reason"]
            for rule in self._provenance.get("identity_exclusions", [])
            if (rule.get("start") is None or end >= pd.Timestamp(rule["start"]))
            and (rule.get("end") is None or start < pd.Timestamp(rule["end"]))
        }


def load_membership_csv(path_or_buffer, coverage_end, source=None) -> MembershipHistory:
    """Load a CSV with date,tickers columns and quoted comma-separated full rosters.

    Each row replaces the previous roster. The caller must declare the last date
    covered by the source. A last observed change alone does not establish it.
    """
    try:
        table = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, engine="python")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError("Membership CSV must contain date,tickers columns and full rosters.") from exc
    if (list(table.columns) != ["date", "tickers"] or table.empty
            or not isinstance(table.index, pd.RangeIndex) or table.isna().any().any()):
        raise ValueError("Membership CSV must contain date,tickers columns and at least one full roster.")
    snapshots = pd.Series(
        [tuple(s.strip() for s in roster.split(",")) for roster in table["tickers"]],
        index=table["date"], name="tickers",
    )
    return MembershipHistory(
        snapshots, coverage_end, source=source if source is not None else "User-supplied membership CSV",
        limitations=(
            "User-supplied full rosters and coverage dates are not independently verified.",
            "Price-column identifiers must identify the same security throughout the input history.",
        ),
    )


def bundled_membership() -> MembershipHistory:
    """Load the pinned community Nasdaq-100 reconstruction without network access."""
    directory = Path(__file__).with_name("membership_data")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    history = load_membership_csv(
        directory / "nasdaq100_membership_snapshots.csv", provenance["coverage_end"],
        source=provenance["source"],
    )
    return MembershipHistory(
        history.snapshots, history.coverage_end, history.source,
        tuple(provenance["limitations"]), provenance,
    )
