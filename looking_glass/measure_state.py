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

    def __init__(self, seqs, train_keys, vs, coverage: float = 0.99):
        self.vs = vs
        self.coverage = float(coverage)
        self.n_numeric = int(vs["gnum"])
        self.n_et = int(vs["et"])
        keep = []
        for j, size in enumerate(vs["gcat"]):
            size = int(size)
            counts = np.zeros(size, dtype=np.int64)
            for k in train_keys:
                col = np.asarray(seqs[k][1][:, j])
                col = col[(col > 0) & (col < size)]
                if len(col):
                    counts += np.bincount(col, minlength=size)[:size]
            nonzero = np.flatnonzero(counts > 0)
            if len(nonzero) == 0:
                keep.append(np.array([], dtype=np.int64))
                continue
            order = nonzero[np.argsort(counts[nonzero])[::-1]]
            cumulative = np.cumsum(counts[order]) / max(float(counts.sum()), 1.0)
            n_keep = int(np.searchsorted(cumulative, self.coverage, side="left") + 1)
            # Bounded vocabulary: a high-cardinality identifier must not dominate the
            # state. Budget derived from the field's observed mass.
            budget = max(8, int(np.sqrt(max(int(counts.sum()), 1))))
            n_keep = min(n_keep, budget)
            keep.append(np.sort(order[:n_keep]).astype(np.int64))
        self.keep = tuple(keep)
        self.dim = int(self.n_et + 1 + sum(len(k) + 3 for k in self.keep) + 6 * self.n_numeric)
        self.spec = MeasureSpec(
            keep=self.keep,
            coverage=self.coverage,
            n_numeric=self.n_numeric,
            n_event_types=self.n_et,
            dim=self.dim,
        )

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

        out = np.concatenate(blocks, axis=1)
        if out.shape[1] != self.dim:
            raise RuntimeError(f"measure dim {out.shape[1]} != derived {self.dim}")
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite measure state")
        return out.astype(np.float32)
