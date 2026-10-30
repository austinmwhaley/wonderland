"""Accessibility-vs-absence axis probe for the supervised leg (DEC-456, CPU-only receipt).

The supervised deficit decomposes into three axes (DEC-456 analysis of the judge receipt):
rate (`x0`), time-structure (`churn_day`), and segmentation (the regime traits). Each failing
trait must first be ATTRIBUTED — the loop's own Diagnosable rule — and the discriminating test
is the consumer contract itself (A3): shipping consumers are <= {linear, kNN, small MLP}, so:

  * kNN or small-MLP beats the linear head on the SAME arm by a material margin  -> the
    information is PRESENT but ENTANGLED: an accessibility failure of the served geometry.
  * no consumer beats linear                                    -> the information is ABSENT
    from the arm: an encoder-side (dynamics/objective) question.

Both readings are encoder duties under A3 (the encoder must pre-adapt; accessibility is not a
consumer problem) — but they name different levers, and DEC-431's lesson says the probe must
exist before any protocol is proposed on top of it.

Arms mirror the judge exactly (`_arms`: the same split, the same table encoding, the same
columns), so a number here is comparable with the card. This is a diagnostic: it gates nothing.

Usage:
  scripts/wonder_container.sh -- .venv/bin/python scripts/measure_access_axis.py \
      --products /work/cfm_d_qte --demo /work/demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_judge_law import SEED, _arms, _ridge_oos, _target_matrix  # noqa: E402

# principle-ok: evaluation-contract shapes for a DIAGNOSTIC consumer class (A3 explicitly
# permits {linear, kNN, small MLP}); a 15-NN and a 64-unit one-hidden-layer MLP are the plain,
# un-tuned members of those classes, fixed so the receipt is deterministic.
KNN_K = 15
MLP_UNITS = 64


def _consumers():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "linear": None,  # _ridge_oos (the shipping head)
        "knn": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=KNN_K, weights="uniform")
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(MLP_UNITS,),
                max_iter=2000,
                early_stopping=True,
                random_state=SEED,
            ),
        ),
    }


def _r2(y, p) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def _score(arm_a, arm_b, ya, yb, kind):
    if kind == "linear":
        p, r2 = _ridge_oos(arm_a, ya, arm_b, yb)
        return None if r2 is None else round(float(r2), 4)
    m = _consumers()[kind]
    m.fit(arm_a, ya)
    return round(_r2(yb, m.predict(arm_b)), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument(
        "--rewhiten",
        action="store_true",
        help="also score a `brain_rewhiten` arm: E re-served through the frozen post-train "
        "whitening stored in the checkpoint instead of the isometric boundary (a MEASUREMENT "
        "of what that serving ruling would do — it changes no artifact)",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/report_cards/experiments/access_axis_supervised.json"),
    )
    a = ap.parse_args()
    t0 = time.time()
    products = Path(a.products)
    db = products / "cfm_products.duckdb"
    if not db.exists():
        db = products
    rep: dict = {"working": False, "products": str(products)}
    try:
        arms = _arms(db, Path(a.demo))
        if arms["problems"] or arms["table"] is None:
            rep["note"] = "arms/table not aligned: " + "; ".join(
                (arms["problems"] or [arms["path_a"].get("error", "?")])[:3]
            )
            _write(rep, a.out, t0)
            return 0
        truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]
        usable, excluded = _target_matrix(truth, akeys, bkeys)
        rep["n_fit_A"], rep["n_score_B"] = len(akeys), len(bkeys)

        def rows(arm, side):
            # brain/ledger arms are key->vector dicts (stack in split order); the table arm
            # arrives pre-stacked by _arms over A ∪ B — one reader for both shapes.
            src = arm[side]
            if isinstance(src, dict):
                return np.vstack([src[k] for k in (akeys if side == "A" else bkeys)])
            return src

        cols = arms["columns"]
        table = {
            s: np.vstack([arms["table"][k] for k in (akeys if s == "A" else bkeys)]) for s in "AB"
        }
        ledger_ok = arms["ledger_served"]
        matrix = {
            "brain": cols["brain"],
            "table": {"A": table["A"], "B": table["B"]},
        }
        if ledger_ok:
            matrix["ledger"] = arms["ledger"]
            matrix["brain_ledger"] = {
                s: np.hstack([rows(cols["brain"], s), rows(arms["ledger"], s)]) for s in "AB"
            }
        if a.rewhiten:
            # E' = (v - mu) @ Wh with v = (E/scale) @ R the exact inverse of the isometric
            # boundary the artifact was served through. Both maps come from the checkpoint;
            # the whole round-trip is verified on the scored population before any use.
            from looking_glass.cfm_state import load_frozen_encoder

            import torch

            ckpt = sorted(products.glob("cfm_*.pt"))
            if not ckpt:
                rep["note_rewhiten"] = "no checkpoint for --rewhiten"
            else:
                tag = ckpt[0].stem[len("cfm_") :].replace("_", ".", 2)
                mdl, _cfg = load_frozen_encoder(tag, products)
                if not getattr(mdl, "isometric_boundary", False) or mdl.whiten_W is None:
                    rep["note_rewhiten"] = (
                        "boundary is not isometric or no frozen whitening stored — rewhiten N/A"
                    )
                else:
                    with torch.no_grad():
                        R = mdl.iso_matrix().double()
                        sc = float(mdl.boundary_scale)
                        Wh = mdl.whiten_W.double()
                        mu = mdl.whiten_mean.double()

                    def rewh(arm_dict, keys):
                        E = np.vstack([arm_dict[k] for k in keys])
                        v = (torch.from_numpy(E).double() / sc) @ R
                        return ((v - mu) @ Wh).numpy().astype(np.float64)

                    E_A = rewh(cols["brain"]["A"], akeys)
                    E_B = rewh(cols["brain"]["B"], bkeys)
                    # verify the re-whitened table goes back: E' -> v -> E must reproduce E
                    inv_Wh = torch.linalg.inv(Wh)
                    v2 = torch.from_numpy(E_B).double() @ inv_Wh + mu
                    fwd = (v2 @ R.T * sc).numpy().astype(np.float64)
                    rep["rewhiten_roundtrip_max_err"] = float(np.max(np.abs(fwd - E_B)))
                    matrix["brain_rewhiten"] = {"A": E_A, "B": E_B}
                    if ledger_ok:
                        matrix["brain_ledger_rewhiten"] = {
                            s: np.hstack(
                                [
                                    (E_A if s == "A" else E_B),
                                    rows(arms["ledger"], s),
                                ]
                            )
                            for s in "AB"
                        }

        per_trait: dict = {}
        for name, (ya, yb) in usable.items():
            per_trait[name] = {}
            for arm_name, src in matrix.items():
                try:
                    A, B = rows(src, "A"), rows(src, "B")
                except KeyError:
                    continue
                scores = {}
                for kind in _consumers():
                    try:
                        scores[kind] = _score(A, B, ya, yb, kind)
                    except Exception as exc:  # noqa: BLE001 - one consumer failing is reported
                        scores[kind] = f"error:{type(exc).__name__}"
                per_trait[name][arm_name] = scores
        rep["traits"] = per_trait
        rep["excluded"] = excluded

        # attribution guide (diagnostic strings, not a gate): compare consumer classes ON THE BRAIN
        guide: list = []
        for name, byarm in per_trait.items():
            b = byarm.get("brain", {})
            lin, knn, mlp = b.get("linear"), b.get("knn"), b.get("mlp")
            nums = [v for v in (lin, knn, mlp) if isinstance(v, (int, float))]
            if len(nums) == 3 and max(knn, mlp) - lin > 0.05:
                guide.append(
                    f"{name}: a permitted consumer beats the linear head on brain by "
                    f">{0.05} (lin {lin}, knn {knn}, mlp {mlp}) — PRESENT BUT ENTANGLED "
                    "(served-geometry accessibility axis)"
                )
            tab = byarm.get("table", {}).get("linear")
            if len(nums) == 3 and isinstance(tab, (int, float)) and max(nums) < tab - 0.05:
                guide.append(
                    f"{name}: no permitted consumer on brain reaches within 0.05 of the table "
                    f"(best brain {max(nums):.4f} vs table {tab:.4f}) — ABSENT from the state "
                    "(dynamics/objective axis)"
                )
        rep["interpretation"] = guide
        rep["working"] = True
    except Exception as exc:  # noqa: BLE001 - a diagnostic receipts its own failure, loudly
        rep["note"] = f"{type(exc).__name__}: {exc}"
    _write(rep, a.out, t0)
    return 0


def _write(rep: dict, out: str, t0: float) -> None:
    rep["seconds"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(f"access_axis working={rep['working']} -> {out} ({rep['seconds']}s)", flush=True)
    for name, byarm in rep.get("traits", {}).items():
        line = " ".join(
            f"{arm}:{v.get('linear')}/{v.get('knn')}/{v.get('mlp')}" for arm, v in byarm.items()
        )
        print(f"  {name:20s} (lin/knn/mlp) {line}", flush=True)
    for g in rep.get("interpretation", []):
        print(f"  GUIDE {g}", flush=True)
    if "note" in rep:
        print(f"  note: {rep['note']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
