"""Fail-safe identifiability guards for causal claims.

AGENTS principles 7/11: never silently substitute, never fabricate. If the data
cannot identify a causal quantity (no randomized holdout, no logged propensity),
REJECT with a clear diagnosis and say what still works.

On observational logs (no holdout, no A/B) — the realistic production case:

  * WORKS offline:
      - white_queen OPE: `estimate_propensity=True` (provenance receipt
        "estimated") + sensitivity analysis for hidden confounding;
      - prediction heads (CLV / churn / segmentation) — pure prediction;
      - certification-gated controllers that read frozen artifacts.
  * REJECTS with `NotIdentifiableError` (never NaN, never a raw crash):
      - red_queen incrementality (needs the randomized control);
      - red_queen uplift/response targeting (needs the control);
      - IPW arm effects without logged propensity.
"""

from __future__ import annotations

import duckdb

OBSERVATIONAL_HINT = (
    "causal lift is not identifiable from these logs (no randomized control / "
    "logged propensity). What still works offline: white_queen OPE with "
    "estimate_propensity=True + sensitivity receipts, prediction heads, and "
    "certification-gated controllers. For lift claims, provide holdout or "
    "propensity-logged logs (rabbit_hole lab data)."
)


class NotIdentifiableError(ValueError):
    """A causal quantity the data cannot identify — reject, do not fabricate."""


def require_stream_view(con, name, purpose):
    """Resolve `name` or raise NotIdentifiableError with the observational hint."""
    try:
        con.execute(f'SELECT * FROM "{name}" LIMIT 1')
    except duckdb.CatalogException as e:
        raise NotIdentifiableError(
            f"stream view '{name}' not found ({purpose}). {OBSERVATIONAL_HINT}"
        ) from e


def require_propensity(con, table, column="propensity"):
    """The logging propensity column must exist for IPW-family estimates."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    cols = {r[0] for r in rows}
    if column not in cols:
        raise NotIdentifiableError(
            f"'{table}.{column}' is not logged ({len(cols)} columns present). {OBSERVATIONAL_HINT}"
        )


def require_holdout(frame, name="email_holdout"):
    """Both arms must exist: treated periods AND a randomized held-out control."""
    if "holdout" not in frame.columns:
        raise NotIdentifiableError(f"{name} has no 'holdout' column. {OBSERVATIONAL_HINT}")
    n = len(frame)
    n_hold = int(frame["holdout"].sum())
    if n == 0 or n_hold == 0 or n_hold == n:
        raise NotIdentifiableError(
            f"no usable randomized holdout in {name} (rows={n}, held={n_hold}); "
            f"both treated and held-out periods are required. {OBSERVATIONAL_HINT}"
        )
