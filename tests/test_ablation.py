"""E-vs-R ablation (ROADMAP P1-4): raw rows must align to the donor rows —
a paired comparison on different row sets compares nothing, so misalignment
fails safe instead of shrinking."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest


def _stream():
    return pl.DataFrame(
        {
            "customer_key": ["a", "a", "a", "b", "b", "b"],
            "event_ts": [
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T06:00:00+00:00",
                "2025-01-02T00:00:00+00:00",
                "2025-01-01T00:00:00+00:00",
                "2025-01-01T12:00:00+00:00",
                "2025-01-03T00:00:00+00:00",
            ],
            "event_type": ["view", "view", "order_placed", "view", "order_placed", "view"],
            "value": [1.0, 2.0, 3.0, 1.0, None, 3.0],  # NULL must not crash (Instacart)
        }
    )


def _orders():
    return pl.DataFrame({"customer_id": ["a", "b"], "t": [1.0, 2.0], "gm": [10.0, 20.0]})


_ANCHOR = datetime(2025, 1, 10, tzinfo=timezone.utc).timestamp()


def test_raw_matrix_aligns_row_for_row():
    from plugins.ablation import raw_matrix

    ds = SimpleNamespace(keys=np.array(["a", "b"]), anchor_epoch=np.array([_ANCHOR, _ANCHOR]))
    R = raw_matrix(ds, _stream(), _orders())
    assert R.shape[0] == 2
    assert np.isfinite(R).all()
    # battery-convention width: 7 RFM fields + per-type counts
    assert R.shape[1] == 7 + _stream()["event_type"].n_unique()


def test_raw_matrix_misalignment_fails_safe():
    from plugins.ablation import raw_matrix

    ds = SimpleNamespace(
        keys=np.array(["a", "missing-customer"]),
        anchor_epoch=np.array([_ANCHOR, _ANCHOR]),
    )
    with pytest.raises(SystemExit, match="alignment failed"):
        raw_matrix(ds, _stream(), _orders())


def test_raw_matrix_anchor_needs_three_events():
    from plugins.ablation import raw_matrix

    # anchor BEFORE the third event -> only 2 events of history -> fail safe
    early = datetime(2025, 1, 1, 6, tzinfo=timezone.utc).timestamp()
    ds = SimpleNamespace(keys=np.array(["a"]), anchor_epoch=np.array([early]))
    with pytest.raises(SystemExit, match="alignment failed"):
        raw_matrix(ds, _stream(), _orders())
