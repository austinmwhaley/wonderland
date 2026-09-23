"""Read/write the customer event stream (the deliverable of rabbit_hole).

The stream is one row per event and MUST carry the canonical five fields:

    customer_key, event_ts, brand, event_type, event_attributes

Extra columns are allowed and preserved, but the five are always present.

Storage backends (chosen by file extension):
  * ``.arrow`` / ``.feather`` / ``.ipc`` -> Arrow IPC (primary data layer;
    memory-mappable, zero-copy into Polars/DuckDB)
  * ``.duckdb`` / ``.ddb``  -> DuckDB (query engine over the Arrow stream)
  * ``.parquet`` / ``.pq``  -> Parquet (compressed archival/portability)

No SQLite, no pandas.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .schema import (CANONICAL_FIELDS, EVENT_STREAM_TABLE, canonicalize_row,
                     parse_attributes)

# Non-canonical columns we carry through when present (contract extras).
EXTRA_FIELDS = ("event_id", "entity_type", "entity_id", "source_table", "value")


# ---------------------------------------------------------------------------
# engine detection
# ---------------------------------------------------------------------------
def _ext(path) -> str:
    return "".join(Path(str(path)).suffixes).lower()


def engine_of(path) -> str:
    e = _ext(path)
    if e.endswith(".arrow") or e.endswith(".feather") or e.endswith(".ipc"):
        return "arrow"
    if e.endswith(".parquet") or e.endswith(".pq"):
        return "parquet"
    return "duckdb"


# ---------------------------------------------------------------------------
# row helpers
# ---------------------------------------------------------------------------
def _canon_all(rows: Iterable[Mapping]) -> list[dict]:
    """Canonicalize each row and keep any extra columns."""
    out = []
    for r in rows:
        c = canonicalize_row(r)
        for k, v in dict(r).items():
            if k not in c:
                c[k] = v
        # ensure all five are present (missing -> None/{}).
        for f in CANONICAL_FIELDS:
            c.setdefault(f, "{}" if f == "event_attributes" else None)
        out.append(c)
    return out


def _order(rows: list[dict]) -> list[dict]:
    rows.sort(key=lambda r: (str(r.get("customer_key")), str(r.get("event_ts"))))
    return rows


def _columns(rows: list[dict]) -> list[str]:
    cols = list(CANONICAL_FIELDS)
    for e in EXTRA_FIELDS:
        if any(e in r for r in rows):
            cols.append(e)
    return cols


class CustomerEventStream:
    """Iterable view over a customer event stream (canonical rows)."""

    def __init__(self, rows: Iterable[Mapping]):
        self._rows = _order(_canon_all(rows))

    @classmethod
    def load(cls, path: str, table: str = EVENT_STREAM_TABLE):
        return cls(read_events(path, table=table))

    def __iter__(self) -> Iterator[dict]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def payloads(self) -> Iterator[dict]:
        for r in self._rows:
            yield parse_attributes(r.get("event_attributes"))


# ---------------------------------------------------------------------------
# read / write, dispatched by engine
# ---------------------------------------------------------------------------
def read_events(path: str, table: str = EVENT_STREAM_TABLE,
                where: str | None = None, limit: int | None = None) -> list[dict]:
    eng = engine_of(path)
    if eng == "arrow":
        raw = _read_arrow(path)
    elif eng == "parquet":
        raw = _read_parquet(path)
    else:
        raw = _read_duckdb(path, table, where)
    rows = _order(_canon_all(raw))
    return rows[:limit] if limit else rows


def write_events(path: str, rows: Iterable[Mapping],
                 table: str = EVENT_STREAM_TABLE) -> int:
    rows = _canon_all(rows)
    eng = engine_of(path)
    if eng == "arrow":
        return _write_arrow(path, rows)
    if eng == "parquet":
        return _write_parquet(path, rows)
    return _write_duckdb(path, rows, table)


def to_duckdb(src: str, dst: str, table: str = EVENT_STREAM_TABLE) -> int:
    """Materialize the canonical stream from any source into DuckDB."""
    rows = read_events(src, table=table)
    return write_events(dst, rows, table=table)


def to_arrow(src: str, dst: str, table: str = EVENT_STREAM_TABLE) -> int:
    """Materialize the canonical stream from any source into Arrow IPC.
    Zero-copy frame path (no row-wise dicts) so it scales to tens of millions."""
    df = read_frame(src, table=table)
    return write_frame(dst, df, table=table)


def to_parquet(src: str, dst: str, table: str = EVENT_STREAM_TABLE) -> int:
    rows = read_events(src, table=table)
    return write_events(dst, rows, table=table)


# ---------------------------------------------------------------------------
# zero-copy dataframe path (preferred for the arrow data layer)
# ---------------------------------------------------------------------------
def read_frame(path: str, table: str = EVENT_STREAM_TABLE):
    """Return the stream as a Polars DataFrame with zero dict materialization.

    Arrow IPC is memory-mapped (zero-copy); DuckDB/Parquet results are handed
    back via Arrow without a row-wise round trip.
    """
    import polars as pl
    eng = engine_of(path)
    if eng == "arrow":
        return pl.read_ipc(str(path), memory_map=True)
    if eng == "parquet":
        return pl.read_parquet(str(path))
    import duckdb
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table}").pl()
    finally:
        con.close()


def write_frame(path: str, df, table: str = EVENT_STREAM_TABLE) -> int:
    """Write a Polars DataFrame as the stream (uncompressed IPC for zero-copy)."""
    eng = engine_of(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if eng == "arrow":
        df.write_ipc(str(path))
    elif eng == "parquet":
        df.write_parquet(str(path))
    else:
        import duckdb
        con = duckdb.connect(str(path))
        try:
            con.register("rh_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM rh_df")
        finally:
            con.close()
    return df.height


def _as_text(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return v
    return json.dumps(v, separators=(",", ":"))


# ---------------------------------------------------------------------------
# duckdb (primary) + parquet, via polars/arrow (no pandas)
# ---------------------------------------------------------------------------
def _to_polars(rows):
    import polars as pl
    cols = _columns(rows)
    data = {c: [_as_text(r.get(c)) for r in rows] for c in cols}
    return pl.DataFrame(data)


def _from_polars(df):
    return [dict(zip(df.columns, row)) for row in df.iter_rows()]


def _write_duckdb(path, rows, table):
    import duckdb
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = _to_polars(rows)
    con = duckdb.connect(str(path))
    try:
        con.register("rh_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM rh_df")
    finally:
        con.close()
    return len(rows)


def _read_duckdb(path, table, where):
    import duckdb
    con = duckdb.connect(str(path), read_only=True)
    try:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        df = con.execute(sql).pl()
        return _from_polars(df)
    finally:
        con.close()


def _write_parquet(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _to_polars(rows).write_parquet(str(path))
    return len(rows)


def _read_parquet(path):
    import polars as pl
    return _from_polars(pl.read_parquet(str(path)))


# ---------------------------------------------------------------------------
# arrow ipc (primary data layer): memory-mappable, zero-copy into polars/duckdb
# ---------------------------------------------------------------------------
def _write_arrow(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed IPC so readers can memory-map (zero-copy) the file.
    _to_polars(rows).write_ipc(str(path))
    return len(rows)


def _read_arrow(path):
    import polars as pl
    return _from_polars(pl.read_ipc(str(path), memory_map=True))
