"""TTL-aware local cache. Every network pull in ff-edge goes through here.

The point is not speed for its own sake — it's that draft research is iterative.
You re-run the same join twenty times while figuring out what you're looking at,
and none of those runs should hit nflverse or Sleeper again. A cold bootstrap
takes minutes; every run after it takes seconds.

Two stores because there are two shapes of data: tabular (parquet) and Sleeper's
nested JSON (blob).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import polars as pl

from config import DATA_DIR, TTL


def _resolve_ttl(ttl: str | float) -> float:
    """Accept either a TTL key ('weekly') or raw hours (12.0)."""
    if isinstance(ttl, str):
        if ttl not in TTL:
            raise KeyError(f"unknown TTL key {ttl!r}; known: {sorted(TTL)}")
        return float(TTL[ttl])
    return float(ttl)


def _age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600.0


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    return path.exists() and _age_hours(path) < ttl_hours


def _to_polars(obj: Any) -> pl.DataFrame:
    """Normalize whatever a loader returned into a polars DataFrame."""
    if isinstance(obj, pl.DataFrame):
        return obj
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    # nflreadpy returns polars, but pfr/ngs helpers and ad-hoc loaders may hand
    # back pandas. Converting here keeps that detail out of every call site.
    return pl.from_pandas(obj)


def _parquet_safe(df: pl.DataFrame) -> pl.DataFrame:
    """Cast pl.Object columns to Utf8 so parquet round-tripping doesn't fail.

    Ragged JSON (Sleeper metadata, nested nflverse fields) infers as Object,
    which has no parquet representation. Stringify rather than drop — the
    content is still readable, just not queryable as structure.
    """
    casts = [
        pl.col(name).cast(pl.Utf8, strict=False)
        for name, dtype in df.schema.items()
        if dtype == pl.Object
    ]
    return df.with_columns(casts) if casts else df


def frame(
    name: str,
    ttl: str | float,
    loader: Callable[[], Any],
    force: bool = False,
) -> pl.DataFrame:
    """Return a cached DataFrame, calling `loader` only when stale or missing."""
    path = DATA_DIR / f"{name}.parquet"
    if not force and _is_fresh(path, _resolve_ttl(ttl)):
        return pl.read_parquet(path)

    df = _parquet_safe(_to_polars(loader()))
    df.write_parquet(path)
    return df


def blob(
    name: str,
    ttl: str | float,
    loader: Callable[[], Any],
    force: bool = False,
) -> Any:
    """Return a cached JSON-serializable object (Sleeper's nested responses)."""
    path = DATA_DIR / f"{name}.json"
    if not force and _is_fresh(path, _resolve_ttl(ttl)):
        with path.open() as fh:
            return json.load(fh)

    obj = loader()
    with path.open("w") as fh:
        json.dump(obj, fh)
    return obj


_INVENTORY_SCHEMA = {
    "name": pl.Utf8,
    "kind": pl.Utf8,
    "mb": pl.Float64,
    "age_hours": pl.Float64,
}


def inventory() -> pl.DataFrame:
    """What's on disk, how big, how old — sorted by size descending."""
    rows = []
    for path in DATA_DIR.iterdir():
        if path.suffix not in (".parquet", ".json"):
            continue
        rows.append(
            {
                "name": path.stem,
                "kind": path.suffix.lstrip("."),
                "mb": round(path.stat().st_size / 1_048_576, 2),
                "age_hours": round(_age_hours(path), 1),
            }
        )
    if not rows:
        # An empty frame with the right schema, so callers can .filter()/.sort()
        # on a cold cache without special-casing.
        return pl.DataFrame(schema=_INVENTORY_SCHEMA)
    return pl.DataFrame(rows, schema=_INVENTORY_SCHEMA).sort("mb", descending=True)


def clear(pattern: str = "*") -> int:
    """Delete matching cache files. Returns the count removed."""
    removed = 0
    for suffix in ("parquet", "json"):
        for path in DATA_DIR.glob(f"{pattern}.{suffix}"):
            path.unlink()
            removed += 1
    return removed
