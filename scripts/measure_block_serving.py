#!/usr/bin/env python3
"""B3.4 block-aware serving probe: can the exact counter survive beside the shape?

B3.2 measured the fine structure at M0 exact counter (persona 0.189 / future 0.173) and showed the
`w_counter` fold destroys it (0.066 / 0.093). This probe asks the immediate serving question:
if the exact counter is served as its own block, beside the learned shape, does the geometry
recover? It is a CPU probe on the current artifacts; no retraining.

Pre-registered rule: z-score each block on split A, scale each block by 1/sqrt(dim) so each block
contributes equal total variance, concatenate, score on split B. No fitting to B and no tuning to
the scored metric. The exact counter is a served statistic, not a learned-shape claim; the card
must report it as such.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_battery_law import _ridge_head  # noqa: E402
from measure_judge_law import TARGETS, _arms, _z1  # noqa: E402
from measure_stage_decomposition import (  # noqa: E402
    _geometry,
    _load_stage_arrays,
    _round,
    _segment_ari,
)
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_block_serving.json"


def _block(
    shape: dict[str, np.ndarray],
    counter: dict[str, np.ndarray],
    akeys,
    bkeys,
    counter_weight: float = 1.0,
) -> dict[str, np.ndarray]:
    """Concatenate the two blocks with equal total variance on A (the pre-registered rule)."""
    a_shape = np.vstack([shape[k] for k in akeys])
    a_counter = np.vstack([counter[k] for k in akeys])
    mu_s, sd_s = a_shape.mean(0), a_shape.std(0)
    mu_c, sd_c = a_counter.mean(0), a_counter.std(0)
    sd_s[sd_s < 1e-6] = 1.0
    sd_c[sd_c < 1e-6] = 1.0
    # divide by sqrt(dim) so each block has unit TOTAL variance on A, not unit per-dim variance
    s_shape = 1.0 / np.sqrt(a_shape.shape[1])
    s_counter = counter_weight / np.sqrt(a_counter.shape[1])
    out: dict[str, np.ndarray] = {}
    for k in bkeys:
        zs = _z1(shape[k], mu_s, sd_s) * s_shape
        zc = _z1(counter[k], mu_c, sd_c) * s_counter
        out[k] = np.concatenate([zs, zc])
    for k in akeys:  # A is needed for the geometry's own A-fit; same transform
        zs = _z1(shape[k], mu_s, sd_s) * s_shape
        zc = _z1(counter[k], mu_c, sd_c) * s_counter
        out[k] = np.concatenate([zs, zc])
    return out


def _linear_access(
    arm: dict[str, np.ndarray],
    truth: dict,
    akeys,
    bkeys,
    fields,
) -> tuple[float | None, int]:
    """Mean held-out R2 of a standardized ridge head fit on A and scored on B."""
    ka = [k for k in akeys if k in truth and k in arm]
    kb = [k for k in bkeys if k in truth and k in arm]
    if len(ka) < 30 or len(kb) < 30:
        return None, 0
    xa = np.vstack([arm[k] for k in ka])
    xb = np.vstack([arm[k] for k in kb])
    scores = []
    for f in fields:
        ya = np.asarray([truth[k][f] for k in ka], float)
        yb = np.asarray([truth[k][f] for k in kb], float)
        if np.std(ya) < 1e-12 or np.std(yb) < 1e-12:
            continue
        head = _ridge_head().fit(xa, ya)
        scores.append(float(r2_score(yb, head.predict(xb))))
    return (float(np.mean(scores)) if scores else None), len(scores)


def _equal_dimension(
    shape: dict[str, np.ndarray],
    counter: dict[str, np.ndarray],
    akeys,
    bkeys,
    n: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Equal-dimension blocks: shape PCA(n) fit on A + counter z-scored on A and tiled to n.

    The judge z-scores every dimension, so dimension COUNT is the only block-weight lever.
    Tiling the counter is exact (no new information); PCA on the shape is fit on A only.
    """
    a_shape = np.vstack([shape[k] for k in akeys])
    pca = PCA(n_components=n, random_state=0).fit(a_shape)
    s_a = {k: pca.transform(shape[k][None, :])[0] for k in akeys}
    s_b = {k: pca.transform(shape[k][None, :])[0] for k in bkeys}
    a_counter = np.vstack([counter[k] for k in akeys])
    mu_c, sd_c = a_counter.mean(0), a_counter.std(0)
    sd_c[sd_c < 1e-6] = 1.0

    def c_block(k: str) -> np.ndarray:
        z = _z1(counter[k], mu_c, sd_c)
        return np.tile(z, int(np.ceil(n / len(z))))[:n]

    c_a = {k: c_block(k) for k in akeys}
    c_b = {k: c_block(k) for k in bkeys}
    equal = {k: np.concatenate([s_b[k], c_b[k]]) for k in bkeys}
    equal.update({k: np.concatenate([s_a[k], c_a[k]]) for k in akeys})
    shape_only = dict(s_a)
    shape_only.update(s_b)
    counter_only = dict(c_a)
    counter_only.update(c_b)
    return equal, shape_only, counter_only


def main() -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k",
    )
    ap.add_argument(
        "--demo",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2",
    )
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.perf_counter()
    products = Path(a.products)
    stages = _load_stage_arrays(products)
    arms = _arms(products / "cfm_products.duckdb", Path(a.demo))
    if arms["problems"]:
        raise SystemExit("block serving: arms not aligned: " + "; ".join(arms["problems"][:3]))
    truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]

    sample = next(iter(truth.values()))
    numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
    future_fields = [f for f in TARGETS if f in numeric]
    persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]
    seg = {k: truth[k].get("segment") for k in truth if truth[k].get("segment") is not None}
    klab = len({seg[k] for k in bkeys if k in seg})

    shape = stages["M4_served"]
    counter = stages["M0_exact_counter"]
    block = _block(shape, counter, akeys, bkeys)
    equal_dim: dict[str, dict[str, np.ndarray]] = {}
    for n in (39, 128):
        eq, sp, ct = _equal_dimension(shape, counter, akeys, bkeys, n)
        equal_dim[f"equal_{n}"] = eq
        equal_dim[f"shape_pca_{n}"] = sp
        equal_dim[f"counter_tile_{n}"] = ct
    # A diagnostic sensitivity grid: the accepted arm is counter_weight=1 (pre-registered);
    # the grid is reported, never used to pick the arm.
    sensitivity = {}
    for w in (0.5, 1.0, 2.0, 4.0):
        b = _block(shape, counter, akeys, bkeys, counter_weight=w)
        s, _, _ = _geometry(
            b,
            b,
            [k for k in akeys if k in truth],
            [k for k in bkeys if k in truth],
            truth,
            persona_fields,
        )
        sensitivity[str(w)] = _round(s)

    arm_mats = {
        "served": shape,
        "exact_counter": counter,
        "block_equal_variance": block,
        "path_a": arms["table"],
        **equal_dim,
    }
    table = {}
    for name, mat in arm_mats.items():
        ka = [k for k in akeys if k in truth and k in mat]
        kb = [k for k in bkeys if k in truth and k in mat]
        row = {"dim": int(next(iter(mat.values())).shape[0]), "n": len(kb)}
        for metric, fields in (
            ("persona_geometry", persona_fields),
            ("future_geometry", future_fields),
        ):
            if not fields or len(ka) < 30 or len(kb) < 30:
                row[metric] = None
                continue
            score, n, pairs = _geometry(mat, mat, ka, kb, truth, fields)
            row[metric] = _round(score)
            row[f"{metric}_n"] = n
            row[f"{metric}_pairs"] = pairs
        ka_s = [k for k in ka if k in seg]
        kb_s = [k for k in kb if k in seg]
        row["segment_ari"] = (
            _round(_segment_ari(mat, mat, ka_s, kb_s, seg, klab))
            if klab >= 2 and len(ka_s) >= 30 and len(kb_s) >= 30
            else None
        )
        table[name] = row

    access = {}
    for name in ("served", "exact_counter", "equal_39", "equal_128", "path_a"):
        mean_r2, n_fields = _linear_access(
            arm_mats[name], truth, akeys, bkeys, persona_fields + future_fields
        )
        access[name] = {"mean_r2": _round(mean_r2), "n_fields": int(n_fields)}
    access_complete = bool(
        access["served"]["mean_r2"] is not None and access["path_a"]["mean_r2"] is not None
    )
    equal_complete = bool(
        table["equal_128"].get("persona_geometry") is not None
        and table["equal_39"].get("persona_geometry") is not None
    )
    complete = bool(
        table["block_equal_variance"].get("persona_geometry") is not None
        and table["served"].get("persona_geometry") is not None
        and table["path_a"].get("persona_geometry") is not None
        and equal_complete
    )
    body = {
        "kind": "block_serving",
        "complete": complete,
        "products": products_identity(products),
        "n_fit_A": len(akeys),
        "n_score_B": len(bkeys),
        "serving_rule": "per-block z-score on A; each block scaled by 1/sqrt(dim) so its TOTAL variance "
        "is 1; concatenated; scored on B; no B fit, no metric tuning",
        "counter_weight": 1.0,
        "arms": table,
        "sensitivity_counter_weight": sensitivity,
        "comparator": "PATH A wide table",
        "honesty": "the exact counter is a SERVED statistic, not a learned-shape claim; "
        "a gain from the block arm is served-statistic sufficiency, not brain-only progress",
        "linear_access": {
            "complete": access_complete,
            "protocol": "standardized RidgeCV fit on A, R2 scored on B, mean over the lab's "
            "persona+future numeric fields; served within 0.01 R2 of PATH A is non-inferiority",
            "arms": access,
            "served_minus_path_a": _round(
                None
                if access["served"]["mean_r2"] is None or access["path_a"]["mean_r2"] is None
                else access["served"]["mean_r2"] - access["path_a"]["mean_r2"]
            ),
        },
        "equal_dimension": {
            "complete": equal_complete,
            "protocol": "shape PCA(n) fit on A; counter z-scored on A and tiled to n; "
            "concatenated and scored on B; n=128 is the operator's 128/128 idea, n=39 is "
            "shape-PCA to the counter's own width",
            "arms": {
                **{k: table[k] for k in equal_dim},
                "served": table["served"],
                "path_a": table["path_a"],
            },
        },
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out),
        body,
        " ".join([Path(__file__).name] + sys.argv[1:]),
        stamp_blocks=False,
    )
    print(f"block serving -> {a.out} ({body['seconds']}s)")
    for name, row in table.items():
        print(
            f"  {name:22s} dim={row['dim']:3d} persona={row['persona_geometry']} "
            f"future={row['future_geometry']} segment_ari={row['segment_ari']}"
        )
    print(f"  sensitivity (counter_weight -> persona): {sensitivity}")
    for name, row in access.items():
        print(f"  linear_access {name:22s} mean_r2={row['mean_r2']} n_fields={row['n_fields']}")
    for name in equal_dim:
        row = table[name]
        print(
            f"  {name:22s} dim={row['dim']:3d} persona={row['persona_geometry']} "
            f"future={row['future_geometry']} segment_ari={row['segment_ari']}"
        )
    print(f"  complete={complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
