"""Benchmark Polars CPU vs cudf-polars GPU on representative event-stream group-bys.

The pipeline's heavy data path is DuckDB SQL + PyTorch; this receipt answers
whether the GPU dataframe engine is worth adding for the Polars side.

Input: a FLAT event parquet with columns customer_key, event_type, t, v
(use DuckDB to flatten the canonical JSON stream first). Run with the RAPIDS
venv (cudf-polars-cu12 26.10 + polars 1.44), e.g.:

  uv venv /tmp/cudf --python 3.12
  uv pip install --python /tmp/cudf/bin/python cudf-polars-cu12==26.10.0 duckdb
  /tmp/cudf/bin/python scripts/bench_polars_gpu.py --parquet /tmp/flat.parquet --out /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import polars as pl


def _queries():
    def full(lf):
        return lf.group_by("customer_key").agg(
            pl.len().alias("n"),
            pl.col("v").sum().alias("sv"),
            pl.col("v").mean().alias("mv"),
            pl.col("t").min().alias("tmin"),
            pl.col("t").max().alias("tmax"),
        ).sort("customer_key")

    def purchase(lf):
        return (
            lf.filter(pl.col("event_type") == "purchase")
            .group_by("customer_key")
            .agg(pl.len().alias("n"), pl.col("v").sum().alias("sv"), pl.col("v").mean().alias("mv"))
            .sort("customer_key")
        )

    def etype(lf):
        return (
            lf.group_by("event_type")
            .agg(pl.len().alias("n"), pl.col("v").sum().alias("sv"))
            .sort("event_type")
        )

    def joinseg(lf):
        cust = pl.DataFrame(
            {"customer_key": [f"ap{i:07d}" for i in range(12000)], "seg": [i % 5 for i in range(12000)]}
        ).lazy()
        return (
            lf.join(cust, on="customer_key", how="left")
            .group_by("seg")
            .agg(pl.len().alias("n"), pl.col("v").sum().alias("sv"))
            .sort("seg")
        )

    def strings(lf):
        return (
            lf.with_columns(pl.col("customer_key").str.slice(0, 2).alias("p"))
            .group_by("p")
            .agg(pl.len().alias("n"))
            .sort("p")
        )

    return {
        "full_groupby": full,
        "purchase_groupby": purchase,
        "eventtype_groupby": etype,
        "join_groupby": joinseg,
        "string_groupby": strings,
    }


def _median(fn, lf, engine, reps=3):
    times, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(lf).collect() if engine is None else fn(lf).collect(engine=engine)
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="flat parquet (customer_key,event_type,t,v)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)

    lf = pl.scan_parquet(a.parquet)
    out = {"polars": pl.__version__, "gpu_engine": str(pl.GPUEngine), "rows": None}
    res = {}
    for name, fn in _queries().items():
        try:
            tc, c = _median(fn, lf, None, a.reps)
        except Exception as e:  # noqa: BLE001
            res[name] = {"cpu_err": str(e)[:200]}
            continue
        row = {"cpu_s": round(tc, 3), "rows": c.height}
        try:
            tg, g = _median(fn, lf, pl.GPUEngine(raise_on_fail=True), a.reps)
            row["gpu_s"] = round(tg, 3)
            row["speedup"] = round(tc / tg, 2)
            # exact for counts/min/max; tolerance for float sums (GPU reduction order)
            eq = c.height == g.height
            if eq:
                j = c.join(g, on=c.columns[0], suffix="_g")
                for col in c.columns:
                    if col == c.columns[0]:
                        continue
                    av = j[col].to_numpy()
                    bv = j[col + "_g"].to_numpy()
                    if av.dtype.kind in "iu":
                        eq = eq and bool(np.array_equal(av, bv))
                    else:
                        eq = eq and bool(np.allclose(av, bv, rtol=1e-9, atol=1e-9))
            row["equal_within_tolerance"] = bool(eq)
        except Exception as e:  # noqa: BLE001
            row["gpu_err"] = f"{type(e).__name__}: {str(e)[:240]}"
        res[name] = row
        print(name, row, flush=True)

    try:
        import duckdb

        con = duckdb.connect()
        for name, sql in {
            "full_groupby": "SELECT customer_key, count(*) n, sum(v) sv, avg(v) mv, min(t) tmin, max(t) tmax FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            "purchase_groupby": "SELECT customer_key, count(*) n, sum(v) sv, avg(v) mv FROM read_parquet(?) WHERE event_type='purchase' GROUP BY 1 ORDER BY 1",
        }.items():
            ts = []
            for _ in range(a.reps):
                t = time.perf_counter()
                con.execute(sql, [a.parquet]).fetchall()
                ts.append(time.perf_counter() - t)
            res[name]["duckdb_s"] = round(statistics.median(ts), 3)
    except Exception as e:  # noqa: BLE001
        out["duckdb_err"] = str(e)[:200]

    out["queries"] = res
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "queries"}, indent=1))
    return out


if __name__ == "__main__":
    main()
