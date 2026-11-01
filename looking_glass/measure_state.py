"""Lawful, schema-derived measure state for the frozen foundation model.

The learned encoder represents shape; this module computes the extensive measure a
neural network should not have to rediscover. All fields and vocabulary are
discovered by the stream loader. No labels and no business field names are used.

Law: accumulate raw quantities exactly, normalize only at readout. Categorical
distributions are normalized after counting; numeric statistics are emitted in raw
units so a consumer can choose its own scaling.

`seqs` is the loader tuple per customer:
(event_type, categorical, numeric, time, gaps, timestamps, raw).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasureSpec:
    """Resolved, data-derived state specification (receipted by the caller)."""

    keep: tuple[np.ndarray, ...]
    coverage: float
    n_numeric: int
    n_event_types: int
    dim: int


class MeasureBuilder:
    """Fixed-width exact measure vector per customer history."""

    @staticmethod
    def derive_keep(counts, coverage: float):
        """THE mechanical vocabulary recipe: coverage-capped, mass-budgeted, from counts alone.

        Extracted from `__init__` so the G1 guardrail can recompute a published vocabulary from
        the data and compare — a persisted or hand-edited vocabulary that disagrees with what the
        formula produces is exactly the "PATH A smuggled back in" failure G1 exists to catch.
        One definition, not a second copy in the audit.
        """
        counts = np.asarray(counts, dtype=np.int64)
        nonzero = np.flatnonzero(counts > 0)
        if len(nonzero) == 0:
            return np.array([], dtype=np.int64)
        order = nonzero[np.argsort(counts[nonzero])[::-1]]
        cumulative = np.cumsum(counts[order]) / max(float(counts.sum()), 1.0)
        n_keep = int(np.searchsorted(cumulative, coverage, side="left") + 1)
        # Bounded vocabulary: a high-cardinality identifier must not dominate the
        # state. Budget derived from the field's observed mass.
        budget = max(8, int(np.sqrt(max(int(counts.sum()), 1))))
        n_keep = min(n_keep, budget)
        return np.sort(order[:n_keep]).astype(np.int64)

    def __init__(
        self,
        seqs,
        train_keys,
        vs,
        coverage: float = 0.99,
        n_recency: int = 4,
        amount_field: int | None = None,
    ):
        self.vs = vs
        # DEC-473: recency block — exact, history-only facts the table's own churn uses
        # (silence/tenure/span/count). The ledger's job is the companion of counts; measured
        # on the judge population, ledger churn rises 0.056 -> 0.650 with these four.
        self.n_recency = int(n_recency)
        self.coverage = float(coverage)
        self.n_numeric = int(vs["gnum"])
        self.n_et = int(vs["et"])
        # The amount field is the stream's dominant extensive numeric field, DERIVED
        # (cfm_data._primary_numeric_field: largest total absolute mass, ties broken by name).
        # It is stored by index only — no field name ever appears in the recipe — so the
        # return/purchase split below is mechanical, target-blind (G3), and schema-agnostic.
        self.amount_field = (
            amount_field
            if amount_field is not None and 0 <= amount_field < self.n_numeric
            else None
        )
        keep = []
        for j, size in enumerate(vs["gcat"]):
            size = int(size)
            counts = np.zeros(size, dtype=np.int64)
            for k in train_keys:
                col = np.asarray(seqs[k][1][:, j])
                col = col[(col > 0) & (col < size)]
                if len(col):
                    counts += np.bincount(col, minlength=size)[:size]
            keep.append(self.derive_keep(counts, self.coverage))
        self.keep = tuple(keep)
        self.dim = int(
            self.n_et + 1 + sum(len(k) + 3 for k in self.keep) + 6 * self.n_numeric + self.n_recency
        )
        # return_ratio: n_ret, n_pur, sum_ret, sum_pur, ret_count_ratio, ret_value_ratio.
        # Value sign is the mechanical discriminator (returns reduce the balance, purchases
        # increase it); zero/null events are neither. Exact additive counts/sums, ratios at
        # readout — accumulate exactly, normalize only at readout.
        if self.amount_field is not None:
            self.dim += 6
        self.spec = MeasureSpec(
            keep=self.keep,
            coverage=self.coverage,
            n_numeric=self.n_numeric,
            n_event_types=self.n_et,
            dim=self.dim,
        )

    def part_slices(self) -> dict:
        """{part name: [(lo, hi), ...]} — THE definition of the column layout `transform` makes.

        Owned by the builder rather than by the publisher because the builder IS the recipe: a
        second list of widths somewhere else is a second definition, and the two drift (the
        ragged-`shape` bug of DEC-425 was exactly that class of defect). The per-field blocks
        are INTERLEAVED in `transform` (field j's share, missing, richness, then field j+1), so
        a part is a LIST of runs, not one span — flattening them into a single range would
        silently publish the wrong numbers.
        """
        off = self.n_et + 1
        runs = {"category_share": [], "missing_rate": [], "richness": []}
        for keep in self.keep:
            width = len(keep) + 1
            runs["category_share"].append((off, off + width))
            off += width
            runs["missing_rate"].append((off, off + 1))
            off += 1
            runs["richness"].append((off, off + 1))
            off += 1
        runs["event_mass"] = [(0, self.n_et)]
        runs["total_events"] = [(self.n_et, self.n_et + 1)]
        runs["numeric_stats"] = [(off, off + 6 * self.n_numeric)]
        off += 6 * self.n_numeric
        if self.n_recency:
            runs["recency"] = [(off, off + self.n_recency)]
            off += self.n_recency
        if self.amount_field is not None:
            runs["return_ratio"] = [(off, off + 6)]
            off += 6
        if off != self.dim:
            raise RuntimeError(
                f"ledger layout {off} != derived dim {self.dim} — the recipe and its published "
                f"column widths disagree; fix the recipe, do not restamp the artifact"
            )
        return runs

    def transform(self, seqs, keys):
        """Return (len(keys), dim) float32; one row per requested key."""
        keys = list(keys)
        n = len(keys)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        data = [seqs[k] for k in keys]
        lens = np.array([len(d[0]) for d in data], dtype=np.int64)
        if (lens <= 0).any():
            raise ValueError("measure state requires a non-empty history per entity")
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        group = np.repeat(np.arange(n, dtype=np.int64), lens)
        et = np.concatenate([np.asarray(d[0], dtype=np.int64) for d in data])
        gcat = np.concatenate([np.asarray(d[1], dtype=np.int64) for d in data], axis=0)
        graw = np.concatenate([np.asarray(d[6], dtype=np.float64) for d in data], axis=0)
        total = lens.astype(np.float64)
        blocks = []

        # Event-type counts (exact additive mass).
        ec = np.bincount(group * self.n_et + et, minlength=n * self.n_et).reshape(n, self.n_et)
        blocks.append(ec.astype(np.float64))
        blocks.append(total[:, None])

        # Per-field categorical distributions over the coverage-capped vocabulary.
        for j, keep in enumerate(self.keep):
            size = int(self.vs["gcat"][j])
            nk = len(keep)
            lut = np.full(size, nk, dtype=np.int64)  # unseen/uncovered -> other bucket
            if nk:
                lut[keep] = np.arange(nk, dtype=np.int64)
            col = np.clip(gcat[:, j], 0, size - 1)
            mapped = lut[col]
            counts = np.bincount(group * (nk + 1) + mapped, minlength=n * (nk + 1)).reshape(
                n, nk + 1
            )
            blocks.append(counts / total[:, None])
            missing = np.bincount(group, weights=(col == 0).astype(np.float64), minlength=n)
            blocks.append((missing / total)[:, None])
            # Per-entity richness: how many distinct values of this field were seen.
            # Exact, additive per entity, and bounded regardless of field cardinality.
            pairs = np.unique(group * (size + 1) + col)
            seen = np.bincount(pairs // (size + 1), minlength=n).astype(np.float64)
            blocks.append(seen[:, None])

        # Numeric sufficient statistics over observed (nonzero) values.
        for j in range(self.n_numeric):
            x = graw[:, j]
            observed = x != 0
            xo = np.where(observed, x, 0.0)
            cnt = np.add.reduceat(observed.astype(np.float64), starts)
            sm = np.add.reduceat(xo, starts)
            ss = np.add.reduceat(xo * xo, starts)
            safe = np.maximum(cnt, 1.0)
            mean = sm / safe
            var = np.maximum(ss / safe - mean * mean, 0.0)
            mn = np.minimum.reduceat(np.where(observed, x, np.inf), starts)
            mx = np.maximum.reduceat(np.where(observed, x, -np.inf), starts)
            mn = np.where(np.isfinite(mn), mn, 0.0)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            mean = np.where(cnt > 0, mean, 0.0)
            blocks.append(np.column_stack([cnt, sm, mean, np.sqrt(var), mn, mx]))

        if self.n_recency:
            # DEC-473 recency block: exact facts from the history's own clock, cut at the same
            # as_of the encoder used (history-only; no target, no brain input).
            ts_all = np.concatenate([np.asarray(d[3], dtype=np.float64) for d in data])
            tmax = np.maximum.reduceat(ts_all, starts)
            tmin = np.minimum.reduceat(ts_all, starts)
            ref = getattr(self, "as_of", None)
            ref = float(ref) if ref is not None else float(ts_all.max())
            silence = np.log1p(np.maximum(ref - tmax, 0.0) / 86400.0)
            tenure = np.log1p(np.maximum(ref - tmin, 0.0) / 86400.0)
            span = np.log1p(np.maximum(tmax - tmin, 0.0) / 86400.0)
            n_ev = np.log1p(total)
            blocks.append(np.column_stack([silence, tenure, span, n_ev]))

        # return_ratio: exact counts and sums of the amount field split by value sign.
        # Returns are negative-value events, purchases positive-value events; zero/null
        # events are neither. Counts and sums are accumulated exactly (additive across
        # customers and across split boundaries); the ratios are formed at readout.
        af = self.amount_field
        if af is not None:
            x = graw[:, af]
            neg = x < 0.0
            pos = x > 0.0
            n_ret = np.add.reduceat(neg.astype(np.float64), starts)
            n_pur = np.add.reduceat(pos.astype(np.float64), starts)
            sum_ret = np.add.reduceat(np.where(neg, x, 0.0), starts)
            sum_pur = np.add.reduceat(np.where(pos, x, 0.0), starts)
            safe_pur = np.maximum(n_pur, 1.0)
            safe_spend = np.maximum(np.abs(sum_pur), 1.0)
            ret_count_ratio = n_ret / safe_pur
            ret_value_ratio = sum_ret / safe_spend  # negative / positive -> negative
            blocks.append(
                np.column_stack([n_ret, n_pur, sum_ret, sum_pur, ret_count_ratio, ret_value_ratio])
            )

        out = np.concatenate(blocks, axis=1)
        if out.shape[1] != self.dim:
            raise RuntimeError(f"measure dim {out.shape[1]} != derived {self.dim}")
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite measure state")
        return out.astype(np.float32)


def _builder_from_keep(vs, keep, coverage, n_recency=0, amount_field=None):
    """Reconstruct a builder from a persisted vocabulary (no re-scan)."""
    obj = MeasureBuilder.__new__(MeasureBuilder)
    obj.vs = vs
    obj.coverage = float(coverage)
    obj.n_numeric = int(vs["gnum"])
    obj.n_et = int(vs["et"])
    obj.n_recency = int(n_recency)
    obj.amount_field = (
        amount_field if amount_field is not None and 0 <= amount_field < obj.n_numeric else None
    )
    obj.keep = tuple(np.asarray(k, dtype=np.int64) for k in keep)
    obj.dim = int(
        obj.n_et + 1 + sum(len(k) + 3 for k in obj.keep) + 6 * obj.n_numeric + obj.n_recency
    )
    if obj.amount_field is not None:
        obj.dim += 6
    obj.spec = MeasureSpec(
        keep=obj.keep,
        coverage=obj.coverage,
        n_numeric=obj.n_numeric,
        n_event_types=obj.n_et,
        dim=obj.dim,
    )
    return obj


def measure_to_dict(builder):
    """Serialise the resolved vocabulary so a frozen artifact is self-contained."""
    return {
        "coverage": float(builder.coverage),
        "dim": int(builder.dim),
        "keep": [np.asarray(k, dtype=np.int64).tolist() for k in builder.keep],
        "n_recency": int(getattr(builder, "n_recency", 0)),
        "amount_field": builder.amount_field,
    }


def measure_from_dict(vs, d):
    return _builder_from_keep(
        vs,
        d.get("keep", []),
        d.get("coverage", 0.99),
        d.get("n_recency", 0),
        d.get("amount_field"),
    )
