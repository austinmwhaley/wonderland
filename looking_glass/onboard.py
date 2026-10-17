"""Layer-A onboarding: map an arbitrary customer-event table onto the canonical stream.

``onboard`` reads a parquet/duckdb/csv path or a polars DataFrame, detects the
customer/time/event/value columns, folds everything else into
``event_attributes`` (JSON), discovers integer-valued action keys, derives the
reward event, and writes a canonical Parquet stream plus a metadata receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import polars as pl

CANONICAL_COLUMNS = (
    "customer_key",
    "event_ts",
    "event_type",
    "brand",
    "entity_type",
    "entity_id",
    "source_table",
    "value",
    "event_attributes",
)

_MAPPABLE = (
    "customer_key",
    "event_ts",
    "event_type",
    "brand",
    "entity_type",
    "entity_id",
    "source_table",
    "value",
)

_ALIASES = {
    "customer_key": (
        "customer_key",
        "customer_id",
        "cust_id",
        "customer",
        "user_id",
        "user",
        "uid",
        "account_id",
        "client_id",
        "member_id",
        "subscriber_id",
        "shopper_id",
        "buyer_id",
        "anonymous_id",
        "device_id",
        "visitor_id",
    ),
    "event_ts": (
        "event_ts",
        "timestamp",
        "ts",
        "event_time",
        "event_timestamp",
        "time",
        "datetime",
        "event_datetime",
        "event_date",
        "date",
        "created_at",
        "occurred_at",
        "when",
        "created",
        "updated",
        "order_time",
        "txn_time",
        "transaction_time",
    ),
    "event_type": (
        "event_type",
        "type",
        "event_name",
        "evt",
        "event",
        "action",
        "kind",
        "event_kind",
        "activity",
        "activity_type",
    ),
    "value": (
        "value",
        "amount",
        "revenue",
        "price",
        "total",
        "order_value",
        "monetary",
        "monetary_value",
        "gmv",
        "spend",
        "purchase_amount",
        "total_value",
        "order_total",
        "sales",
        "cost",
    ),
    "brand": ("brand", "brand_id", "brand_name"),
    "entity_type": (
        "entity_type",
        "entity_kind",
        "item_type",
        "object_type",
        "content_type",
        "product_type",
    ),
    "entity_id": (
        "entity_id",
        "item_id",
        "object_id",
        "content_id",
        "product_id",
        "sku",
        "sku_id",
        "product",
        "item",
    ),
    "source_table": (
        "source_table",
        "source",
        "stream",
        "stream_name",
        "table_name",
        "table",
    ),
}

_ATTR_ALIASES = (
    "event_attributes",
    "event_payload_json",
    "payload_json",
    "payload",
    "attributes",
    "attrs",
    "event_payload",
    "event_properties",
    "properties",
    "metadata",
    "extra",
    "extras",
)

_ACTION_META = (
    "idx",
    "index",
    "week",
    "campaign",
    "time",
    "date",
    "day",
    "send",
    "id",
    "customer",
    "user",
    "event",
    "ts",
    "timestamp",
    "value",
    "amount",
    "reward",
    "order",
    "price",
)

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _canonical_name(name: Any) -> str | None:
    n = _norm(name)
    for field, aliases in _ALIASES.items():
        if n == _norm(field) or any(n == _norm(a) for a in aliases):
            return field
    return None


def _normalize_columns(columns: Mapping[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (columns or {}).items():
        if not isinstance(v, str):
            raise ValueError(f"columns[{k!r}] must name a source column")
        kn, vn = str(k), v
        if kn in _MAPPABLE or kn == "event_attributes":
            out[kn] = vn
        elif vn in _MAPPABLE or vn == "event_attributes":
            out[vn] = kn
        else:
            field = _canonical_name(kn)
            if field is None:
                field = _canonical_name(vn)
                if field is not None:
                    out[field] = kn
                    continue
                raise ValueError(f"cannot map column override {k!r} -> {v!r}")
            out[field] = vn
    return out


def _is_numeric_col(df: pl.DataFrame, col: str) -> bool:
    dtype = df.schema[col]
    if dtype.is_numeric():
        return True
    sample = df[col].drop_nulls().head(32).to_list()
    if not sample:
        return False
    for v in sample:
        if isinstance(v, bool):
            return False
        if not isinstance(v, Real):
            try:
                float(v)
            except (TypeError, ValueError):
                return False
    return True


def _hint(field: str, n: str) -> bool:
    if field == "customer_key":
        tokens = (
            "customer",
            "user",
            "account",
            "client",
            "member",
            "subscriber",
            "shopper",
            "buyer",
            "visitor",
        )
        return any(t in n for t in tokens) and any(t in n for t in ("id", "key", "uid"))
    if field == "event_ts":
        return (
            "timestamp" in n
            or n.endswith("ts")
            or n.endswith("time")
            or n.endswith("date")
            or "datetime" in n
        ) and not any(t in n for t in ("update", "duration", "count"))
    if field == "event_type":
        return n == "evt" or ("event" in n and any(t in n for t in ("type", "name", "kind")))
    if field == "value":
        return any(
            t in n
            for t in (
                "amount",
                "price",
                "revenue",
                "spend",
                "gmv",
                "sales",
                "monetary",
                "value",
                "total",
                "cost",
            )
        ) and not any(t in n for t in ("id", "count", "qty", "quantity", "num"))
    if field == "brand":
        return "brand" in n
    if field == "entity_type":
        return any(t in n for t in ("entity", "item", "product", "object", "content")) and any(
            t in n for t in ("type", "kind")
        )
    if field == "entity_id":
        return (
            any(t in n for t in ("entity", "item", "product", "object", "content"))
            and any(t in n for t in ("id", "key", "sku"))
        ) or n in ("sku",)
    if field == "source_table":
        return n in ("sourcetable", "source", "stream", "streamname", "tablename")
    return False


def _pick_column(
    field: str,
    df: pl.DataFrame,
    consumed: set[str],
    overrides: Mapping[str, str],
) -> str | None:
    if field in overrides:
        col = overrides[field]
        if col not in df.columns:
            raise ValueError(f"override for {field!r} names missing column {col!r}")
        consumed.add(col)
        return col
    by_norm: dict[str, str] = {}
    for c in df.columns:
        if c not in consumed:
            by_norm.setdefault(_norm(c), c)
    for alias in _ALIASES[field]:
        col = by_norm.get(_norm(alias))
        if col is None:
            continue
        if field == "value" and not _is_numeric_col(df, col):
            continue
        consumed.add(col)
        return col
    for c in df.columns:
        if c in consumed:
            continue
        if _hint(field, _norm(c)):
            if field == "value" and not _is_numeric_col(df, c):
                continue
            consumed.add(c)
            return c
    return None


def _parse_attrs(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except (TypeError, ValueError):
            return None
        if isinstance(obj, Mapping):
            return dict(obj)
    return None


def _detect_attr_columns(df: pl.DataFrame, consumed: set[str]) -> list[str]:
    out: list[str] = []
    alias_norms = {_norm(a) for a in _ATTR_ALIASES}
    for c in df.columns:
        if c in consumed:
            continue
        if _norm(c) in alias_norms:
            out.append(c)
            continue
        vals = [v for v in df[c].drop_nulls().head(64).to_list()]
        if not vals:
            continue
        parsed = 0
        for v in vals:
            if isinstance(v, Mapping):
                parsed += 1
            elif isinstance(v, str):
                s = v.strip()
                if s.startswith("{"):
                    try:
                        obj = json.loads(s)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(obj, Mapping):
                        parsed += 1
        if parsed >= max(1, int(0.9 * len(vals))):
            out.append(c)
    return out


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _sanitize(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _attrs_json(attrs: Mapping[str, Any]) -> str:
    return json.dumps(
        _sanitize(dict(attrs)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _from_epoch(x: float) -> datetime | None:
    if not math.isfinite(x):
        return None
    ax = abs(x)
    if ax >= 1e17:
        x /= 1e9
    elif ax >= 1e14:
        x /= 1e6
    elif ax >= 1e11:
        x /= 1e3
    try:
        return datetime.fromtimestamp(x, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, Real):
        return _from_epoch(float(value))
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if _NUM_RE.fullmatch(s):
        if len(s) == 8 and s[:2] in ("19", "20"):
            try:
                return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        if len(s) == 14 and s[:2] in ("19", "20"):
            try:
                return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return _from_epoch(float(s))
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        return _as_utc(datetime.fromisoformat(iso))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else 0.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _key_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    s = str(value).strip()
    return s or None


def _read_source(source: Any) -> tuple[pl.DataFrame, str]:
    if isinstance(source, pl.DataFrame):
        return source.clone(), "dataframe"
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    suffix = path.suffix.lower()
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError(f"no parquet files in directory: {path}")
        return pl.read_parquet([str(f) for f in files]), path.name
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path), path.stem
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(path, separator=sep, infer_schema_length=10000), path.stem
    if suffix in (".ipc", ".feather", ".arrow"):
        return pl.read_ipc(path), path.stem
    if suffix in (".json", ".jsonl", ".ndjson"):
        if suffix == ".json":
            return pl.read_json(path), path.stem
        return pl.read_ndjson(path), path.stem
    if suffix in (".duckdb", ".ddb"):
        import duckdb

        con = duckdb.connect(str(path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if not tables:
                raise ValueError(f"no tables in duckdb source: {path}")
            preferred = [t for t in tables if t in ("customer_events", "customer_event_stream")]
            table = preferred[0] if preferred else sorted(tables)[0]
            df = con.execute(f'SELECT * FROM "{table}"').pl()
        finally:
            con.close()
        return df, path.stem
    raise ValueError(f"unsupported source type: {path.suffix or path}")


def _discover_action_keys(df: pl.DataFrame, max_distinct: int = 24) -> list[str]:
    send = df.filter(pl.col("event_type").str.to_lowercase().str.contains("send", literal=True))
    if send.height == 0:
        return []
    values: dict[str, list[Any]] = {}
    for raw in send["event_attributes"].to_list():
        attrs = _parse_attrs(raw) or {}
        for k, v in attrs.items():
            if v is not None:
                values.setdefault(str(k), []).append(v)
    out: list[str] = []
    for k in sorted(values):
        if any(t in k.lower() for t in _ACTION_META):
            continue
        vals = values[k]
        if len(vals) < 2:
            continue
        ints = [_int_value(v) for v in vals]
        ok = [v for v in ints if v is not None]
        if len(ok) / len(vals) < 0.99:
            continue
        distinct = set(ok)
        if 1 < len(distinct) <= max_distinct:
            out.append(k)
    return out


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        try:
            x = float(value.strip())
        except (TypeError, ValueError):
            return None
        return int(x) if math.isfinite(x) and x.is_integer() else None
    return None


def _derive_reward_event(df: pl.DataFrame, value_col: str | None) -> str | None:
    if value_col is None or df.height == 0:
        return None
    agg = (
        df.group_by("event_type")
        .agg(pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0).sum().alias("mv"))
        .sort(["mv", "event_type"])
    )
    if agg.height == 0:
        return None
    return str(agg["event_type"][-1])


def onboard(
    source: Any,
    *,
    out_dir: str | Path,
    columns: Mapping[str, str] | None = None,
    customer_col: str | None = None,
    time_col: str | None = None,
    event_col: str | None = None,
    value_col: str | None = None,
    action_keys: list[str] | str | None = None,
    cadence: str = "week",
    reward_event: str | None = None,
) -> dict:
    df, source_name = _read_source(source)
    if df.height == 0:
        raise ValueError("source table is empty")

    overrides = _normalize_columns(columns)
    for field, col in (
        ("customer_key", customer_col),
        ("event_ts", time_col),
        ("event_type", event_col),
        ("value", value_col),
    ):
        if col is not None:
            overrides[field] = col

    consumed: set[str] = set()
    mapped: dict[str, str | None] = {}
    for field in _MAPPABLE:
        mapped[field] = _pick_column(field, df, consumed, overrides)

    attr_cols = _detect_attr_columns(df, consumed)
    folded_cols = [c for c in df.columns if c not in consumed]

    src_customer = mapped["customer_key"]
    src_ts = mapped["event_ts"]
    src_event = mapped["event_type"]
    src_value = mapped["value"]
    src_brand = mapped["brand"]
    src_entity_type = mapped["entity_type"]
    src_entity_id = mapped["entity_id"]
    src_source = mapped["source_table"]

    warnings: list[str] = []
    if src_customer is None:
        raise ValueError("could not detect a customer key column; pass customer_col")
    if src_ts is None:
        raise ValueError("could not detect an event timestamp column; pass time_col")
    if src_event is None:
        warnings.append("no event_type column detected; filled with 'unknown'")
    if src_value is None:
        warnings.append("no monetary value column detected; value filled with 0.0")

    records: list[dict[str, Any]] = []
    dropped = 0
    missing_event = 0
    value_from_attrs = False
    for row in df.to_dicts():
        customer = _key_str(row.get(src_customer))
        ts = _parse_timestamp(row.get(src_ts))
        if customer is None or ts is None:
            dropped += 1
            continue

        event = row.get(src_event) if src_event is not None else None
        event = str(event).strip() if event is not None else ""
        if not event:
            event = "unknown"
            missing_event += 1

        attrs: dict[str, Any] = {}
        for c in attr_cols:
            parsed = _parse_attrs(row.get(c))
            if parsed is not None:
                attrs.update(parsed)
            elif row.get(c) is not None:
                attrs[c] = row.get(c)
        for c in folded_cols:
            if c in attr_cols or row.get(c) is None:
                continue
            attrs[c] = row.get(c)

        brand = row.get(src_brand) if src_brand is not None else None
        entity_type = row.get(src_entity_type) if src_entity_type is not None else None
        entity_id = row.get(src_entity_id) if src_entity_id is not None else None
        if brand is None or str(brand).strip() == "":
            brand = attrs.pop("brand", None)
        if entity_type is None or str(entity_type).strip() == "":
            entity_type = attrs.pop("entity_type", None)
        if entity_id is None or str(entity_id).strip() == "":
            entity_id = attrs.pop("entity_id", None)

        value = _to_float(row.get(src_value)) if src_value is not None else 0.0
        if src_value is None and _is_number(attrs.get("value")):
            value = _to_float(attrs.pop("value"))
            value_from_attrs = True

        source_table = row.get(src_source) if src_source is not None else None
        if source_table is None or str(source_table).strip() == "":
            source_table = source_name

        records.append(
            {
                "customer_key": customer,
                "event_ts": ts.isoformat(),
                "_ts": ts,
                "event_type": event,
                "brand": "" if brand is None else str(brand),
                "entity_type": "" if entity_type is None else str(entity_type),
                "entity_id": "" if entity_id is None else str(entity_id),
                "source_table": str(source_table),
                "value": value,
                "event_attributes": _attrs_json(attrs),
            }
        )

    if not records:
        raise ValueError("no valid rows remain after dropping null customer/time rows")

    records.sort(
        key=lambda r: (
            r["customer_key"],
            r["_ts"],
            r["event_type"],
            r["source_table"],
            r["value"],
            r["event_attributes"],
        )
    )

    monotone = True
    prev_customer = None
    prev_ts = None
    for r in records:
        if r["customer_key"] == prev_customer and prev_ts is not None and r["_ts"] < prev_ts:
            monotone = False
            break
        prev_customer = r["customer_key"]
        prev_ts = r["_ts"]

    if dropped:
        warnings.append(f"dropped {dropped} rows with null/invalid customer_key or event_ts")
    if missing_event:
        warnings.append(f"{missing_event} rows had missing event_type; filled 'unknown'")
    if not monotone:
        warnings.append("event_ts is not monotone per customer after sorting")

    data = {c: [r[c] for r in records] for c in CANONICAL_COLUMNS}
    canonical = pl.DataFrame(
        data,
        schema={
            "customer_key": pl.String,
            "event_ts": pl.String,
            "event_type": pl.String,
            "brand": pl.String,
            "entity_type": pl.String,
            "entity_id": pl.String,
            "source_table": pl.String,
            "value": pl.Float64,
            "event_attributes": pl.String,
        },
    )

    if action_keys is None:
        discovered = _discover_action_keys(canonical)
        if not discovered:
            warnings.append("no integer-valued action keys discovered on send-like events")
    elif isinstance(action_keys, str):
        discovered = [k.strip() for k in action_keys.split(",") if k.strip()]
    else:
        discovered = [str(k) for k in action_keys]

    derived_reward = (
        reward_event
        if reward_event is not None
        else _derive_reward_event(
            canonical,
            src_value
            if src_value is not None
            else ("event_attributes.value" if value_from_attrs else None),
        )
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    canonical_path = out_path / "customer_event_stream.parquet"
    canonical.write_parquet(canonical_path)

    columns_mapped: dict[str, Any] = {field: mapped[field] for field in _MAPPABLE}
    columns_mapped["event_attributes"] = folded_cols

    receipt = {
        "canonical_path": str(canonical_path),
        "n_rows": canonical.height,
        "n_customers": canonical["customer_key"].n_unique(),
        "columns_mapped": columns_mapped,
        "action_keys": discovered,
        "reward_event": derived_reward,
        "cadence": cadence,
        "value_col": (
            src_value
            if src_value is not None
            else ("event_attributes.value" if value_from_attrs else None)
        ),
        "warnings": warnings,
    }
    (out_path / "onboard_meta.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map an arbitrary customer-event table onto the canonical stream."
    )
    parser.add_argument("source", help="parquet/duckdb/csv path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--columns",
        action="append",
        default=[],
        metavar="FIELD=COLUMN",
        help="canonical field -> source column (repeatable; JSON object also accepted)",
    )
    parser.add_argument("--customer-col")
    parser.add_argument("--time-col")
    parser.add_argument("--event-col")
    parser.add_argument("--value-col")
    parser.add_argument("--action-keys", help="comma-separated action keys")
    parser.add_argument("--cadence", default="week")
    parser.add_argument("--reward-event")
    args = parser.parse_args(argv)

    columns: dict[str, str] = {}
    for item in args.columns:
        text = item.strip()
        if text.startswith("{"):
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise SystemExit("--columns JSON must be an object")
            columns.update({str(k): str(v) for k, v in obj.items()})
            continue
        key, sep, value = text.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise SystemExit(f"--columns expects FIELD=COLUMN, got {item!r}")
        columns[key.strip()] = value.strip()

    receipt = onboard(
        args.source,
        out_dir=args.out_dir,
        columns=columns or None,
        customer_col=args.customer_col,
        time_col=args.time_col,
        event_col=args.event_col,
        value_col=args.value_col,
        action_keys=args.action_keys,
        cadence=args.cadence,
        reward_event=args.reward_event,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
