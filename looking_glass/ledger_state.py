"""The `measure` outlet (SPINE §5 / DEC-383, ledger Option 3): exact tallies, OUTSIDE the network.

The ledger is the companion, not the brain (SPINE §5, Option 3, operator ruling 2026-10-27):
the foundation is CERTIFIED on the brain alone and the artifact SHIPS brain ⊕ ledger side by
side, so a ledger-backed number can never be booked as a B-win. `measure_state.MeasureBuilder`
already computes the recipe; what was missing — and what this module is — is the PUBLISHER
(the recipe was never written into a release, which is why the wall advertised
`NOT SERVED: measure`) and the GUARDRAILS, which until now were prose in a design doc.

G1–G5 are enforced as CHECKS THAT RUN AT RELEASE TIME, not as comments. A guardrail nobody can
execute is a wish, and this repo has already paid for prose-only invariants (DEC-228/234):

  G1 mechanical only   — the recipe is a fixed formula applied to EVERY field the schema
                          discovered; nothing hand-picked. Checked: one vocabulary per discovered
                          categorical field, and each vocabulary really covers its mass budget.
  G2 history-only      — no event after the as-of cut may contribute. Checked TWICE: the cut is
                          applied with the same timestamp rule the encoder uses, and the ledger
                          is recomputed from a hand-truncated history and compared to the row
                          that was published (a forward peek cannot survive equality).
  G3 target-blind      — the recipe may not read the prediction targets. Checked by source
                          inspection: no target/event name appears anywhere in the recipe text.
  G4 never feeds the   — the brain trains on events only. Checked by source inspection of the
     brain's training     encoder's own forward pass and the training step: neither may name
                          this module.
  G5 attribution       — every ledger row states what it rests on, and the S4 receipt carries
                          ledger-only / learned-only / destroyed controls. Checked: the receipt
                          exists and is complete before the ledger is stamped.

Like the `response` outlet, this is ADDITIVE: the encoder is never touched, and a release
without it keeps `not_served: measure` exactly as it has all along.
"""

from __future__ import annotations

import math

#: The ledger's published parts, in column order. The ORDER and WIDTHS are owned by
#: `MeasureBuilder.part_slices` (one definition of the recipe); these names are the sockets
#: the faceplate advertises, and `view_layout` imports them so the wall cannot advertise a
#: column the producer does not write.
PARTS = (
    "event_mass",
    "total_events",
    "category_share",
    "missing_rate",
    "richness",
    "numeric_stats",
    # DEC-473: exact recency facts (silence/tenure/span/log-count) — the ledger's companion
    # job; measured to lift ledger-only churn 0.056 -> 0.650 on the judge population.
    "recency",
)


def vocab_to_vs(vocab) -> dict:
    """The builder's field-space view of the SAME vocabulary the encoder tokenizes with.

    Derived from `EventVocab`, never re-derived from the frame: a second schema scan would be
    a second definition of "which fields exist", and the two would drift.

    Two widths are one larger than the vocabularies look because the encoder's own tokenizer
    sends an unseen value to index `len(map)` — an OOV bucket that exists in practice and must
    be TALLIED, not clipped into the last real value (which is what `np.clip` would do at the
    smaller width: a silent substitution wearing a column name). Categorical fields get +2:
    `MeasureBuilder`'s convention reserves index 0 as "not logged" (that is what `missing_rate`
    measures), so the encoder's values occupy 1..n and its OOV bucket n+1.
    """
    return {
        "et": int(vocab.n_et) + 1,
        "gcat": [int(v) + 2 for v in getattr(vocab, "n_attr", {}).values()],
        "gnum": len(getattr(vocab, "numeric_fields", ()) or ()),
    }


def seqs_to_tuples(seqs, vocab, as_of=None) -> dict:
    """Adapt CFM sequence dicts to the tuple layout `MeasureBuilder` consumes.

    Indexing uses the ENCODER'S OWN vocabulary (`vocab.et` / `vocab.attrs`), with the same
    unknown-value fallback `_collate` applies, so the ledger and the brain agree about which
    value is which token. `raw` is the numeric block in `vocab.numeric_fields` order — raw
    units, because the law is "accumulate exactly, normalize only at readout".
    """
    import numpy as np

    fields = list(getattr(vocab, "numeric_fields", ()) or ())
    et_map, attr_maps = vocab.et, vocab.attrs
    n_et = int(vocab.n_et)
    out = {}
    for s in seqs:
        key = s["customer"]
        # index 0 in a categorical column is "not logged" (MeasureBuilder's convention), so the
        # encoder's indices are shifted by one and its OOV bucket lands above the last real
        # value instead of being clipped onto it
        et_idx = np.fromiter((et_map.get(str(x), n_et) for x in s["event_type"]), np.int64)
        n = et_idx.size
        gcat = np.zeros((n, len(attr_maps)), dtype=np.int64)
        for j, f in enumerate(attr_maps):
            vals = (s.get("attrs") or {}).get(f)
            if vals is None:
                continue  # the field was never present for this stream: all rows stay "not logged"
            m = attr_maps[f]
            gcat[:, j] = np.fromiter(
                (0 if x is None else m.get(str(x), len(m)) + 1 for x in vals), np.int64
            )
        gnum = np.zeros((n, len(fields)), dtype=np.float64)
        for j, c in enumerate(fields):
            vals = (s.get("numeric") or {}).get(c)
            if vals is None or len(vals) != n:
                continue
            col = np.asarray(vals, dtype=np.float64)
            # `MeasureBuilder`'s law is "0 means not logged" (it is how the whole measure stack
            # distinguishes absent from zero). The stream's own null mask is the authority on
            # that, so honour it: a value that was never logged must not be tallied as a zero.
            # A logged value that happens to BE zero remains ambiguous under this convention —
            # it is the recipe's, not this adapter's, and it is why the receipt reports counts.
            mask = (s.get("numeric_mask") or {}).get(c)
            if mask is not None and len(mask) == n:
                col = np.where(np.asarray(mask, dtype=bool), col, 0.0)
            gnum[:, j] = col
        ts = np.asarray(
            [float(x) if isinstance(x, (int, float)) else _epoch(x) for x in s["ts"]],
            dtype=np.float64,
        )
        if as_of is not None and n:
            keep = ts <= float(as_of)
            if not bool(keep.all()):
                et_idx, gcat, gnum, ts = et_idx[keep], gcat[keep], gnum[keep], ts[keep]
        out[key] = (
            et_idx,  # 0 event_type
            gcat,  # 1 categorical
            gnum,  # 2 numeric (unused: raw units ride at 6)
            ts,  # 3 time
            None,  # 4 gaps (the ledger has no timing block)
            None,  # 5 timestamps
            gnum,  # 6 raw
        )
    return out


def _epoch(x) -> float:
    """The ONE timestamp rule (shared with the encoder's own as_of handling)."""
    from looking_glass.cfm_config import _to_epoch

    return _to_epoch(x)


def fit_vocabulary(
    seqs, vocab, train_keys, coverage: float | None = None, as_of=None, recency: bool = False
) -> tuple[object, dict]:
    """Resolve the ledger vocabulary ONCE, from the TRAINING population, and receipt it.

    Population A only: resolving on every customer would let held-out B's value distribution
    shape a column the artifact serves (a vocabulary is a choice about what exists). `coverage`
    is the builder's own derived default unless the config states otherwise — never a new
    literal. The SAME `as_of` the publisher uses is applied here: the vocabulary must be
    resolved on exactly the history window the ledger will tally, or the G1 audit would compare
    a formula fitted on one reading of the stream against counts taken from another.
    """
    from looking_glass.measure_state import MeasureBuilder

    vs = vocab_to_vs(vocab)
    tuples = seqs_to_tuples(seqs, vocab, as_of=as_of)
    wanted = [k for k in train_keys if k in tuples]
    if not wanted:
        raise LookupError(
            "ledger: no training-population customers have events to fit a vocabulary on"
        )
    builder = MeasureBuilder(
        tuples, wanted, vs, coverage=_coverage(coverage), n_recency=4 if recency else 0
    )
    return builder, {
        "n_fields_discovered": len(vs["gcat"]),
        "n_fields_vocabularied": len(builder.keep),
        "n_numeric": int(vs["gnum"]),
        "n_event_types": int(vs["et"]),
        "dim": int(builder.dim),
        "coverage": float(builder.coverage),
        "vocab_from": "population A (training)",
        "n_train_keys": len(wanted),
        "as_of": None if as_of is None else float(as_of),
    }


def _coverage(declared) -> float:
    """The vocabulary coverage: the builder's own default unless the config declares one.

    Read from the config rather than typed here, so the ledger's cap is one number living in
    one place (`MeasureBuilder`'s default) and this module cannot invent a second one.
    """
    if declared is None:
        from looking_glass.measure_state import MeasureBuilder
        import inspect

        sig = inspect.signature(MeasureBuilder.__init__)
        return float(sig.parameters["coverage"].default)
    return float(declared)


def publish_rows(rows_source, builder, vocab, *, version: str, as_of=None) -> list[dict]:
    """One row per customer x ledger PART — the shape `view_embeddings` stores.

    `rows_source` is the list of CFM sequence dicts (full history). The as-of cut, when given,
    is applied with the same timestamp rule the encoder uses, so "history-only" is not a promise
    about a different clock than the one the brain read.
    """
    import numpy as np

    keys = [s["customer"] for s in rows_source]
    tuples = seqs_to_tuples(rows_source, vocab, as_of=as_of)
    builder.as_of = as_of  # DEC-473: the recency block uses the same cut as the encoder
    matrix = builder.transform(tuples, keys)
    slices = builder.part_slices()
    stamps = {}
    for s in rows_source:
        ts = s.get("ts") or []
        stamps[s["customer"]] = max(
            ((float(x) if isinstance(x, (int, float)) else _epoch(x)) for x in ts), default=None
        )
    out = []
    for i, key in enumerate(keys):
        for part in PARTS:
            # a part is a LIST of runs (the per-field blocks interleave), so flatten them in
            # order rather than assuming one contiguous span
            embedding = [float(v) for lo, hi in slices[part] for v in matrix[i, lo:hi]]
            out.append(
                {
                    "customer_key": str(key),
                    "as_of_epoch": stamps.get(key),
                    "version": version,
                    "view": "measure",
                    "part": part,
                    "dim": len(embedding),
                    "embedding": embedding,
                }
            )
    if not np.isfinite(matrix).all():
        raise FloatingPointError(
            "ledger: non-finite tally — refuse to publish an exact number that is not one"
        )
    return out


def guardrails(
    builder, vocab, tuples_a, cfg, train_keys, target_names=(), as_of=None, brain_modules=None
) -> dict:
    """Run G1–G5 and return receipts. Every entry is `working` plus the EVIDENCE, not a claim.

    These are cheap: two source inspections, one coverage recomputation, one independent
    recount. A guardrail that is expensive to check is a guardrail that gets skipped.
    `brain_modules` exists so a test can point the G4 scan at something known-bad — the default
    is the real encoder and training modules, which is what ships.
    """
    import numpy as np

    from looking_glass.measure_state import MeasureBuilder

    out = {}

    # ---- G1 mechanical only: the recipe is a formula, so a published vocabulary must BE that
    # formula's output. Recomputing it from the counts and comparing is what catches a
    # hand-picked or stale vocabulary; "coverage >= 0.99" would not, and would be wrong anyway:
    # the recipe deliberately caps a high-cardinality field at a mass-derived budget and
    # DISCLOSES the remainder in the "other" bucket (an id field cannot be given a vocabulary of
    # its every value), so coverage is reported per field, never demanded as a pass condition.
    vs = vocab_to_vs(vocab)
    offenders, worst = [], 1.0
    for j, keep in enumerate(builder.keep):
        size = int(vs["gcat"][j])
        counts = np.zeros(size, dtype=np.int64)
        for k in train_keys:
            col = tuples_a[k][1][:, j]
            col = col[(col > 0) & (col < size)]
            if col.size:
                counts += np.bincount(col, minlength=size)[:size]
        if not np.array_equal(
            np.asarray(keep, dtype=np.int64), MeasureBuilder.derive_keep(counts, builder.coverage)
        ):
            offenders.append(j)
        tot = float(counts.sum())
        if tot > 0:
            worst = min(worst, float(counts[np.asarray(keep, dtype=np.int64)].sum() / tot))
    out["G1_mechanical"] = {
        "working": not offenders and len(builder.keep) == len(vs["gcat"]),
        "evidence": (
            f"{len(builder.keep)}/{len(vs['gcat'])} discovered fields vocabularied, each "
            f"reproduced exactly by the recipe formula (hand-picked fields: {offenders or 'none'}); "
            f"worst achieved value-mass coverage {worst:.4f} — capped by the derived mass budget "
            f"and disclosed in the 'other' bucket, not hidden"
        ),
    }

    # ---- G2 history-only: the published row may not know about events after the cut ---------
    # Two independent demands, because "we pass as_of to the adapter" is a claim about a
    # function, not about the artifact:
    #   (a) no tuple the publisher handed us contains an event after the cut — this is what
    #       catches an adapter that silently ignored its `as_of` argument;
    #   (b) the ledger's own event_mass/total_events equal a hand recount of the events at or
    #       before the cut — this catches a builder that peeks or miscounts.
    # With no declared as-of, every customer's own last event is the boundary (that is what the
    # row is stamped at), so (a) is vacuous and (b) still fully applies.
    boundary = float(as_of) if as_of is not None else math.inf
    leaked = sum(int((t[3] > boundary).sum()) for t in tuples_a.values())
    hand_et, hand_total, keys = [], [], []
    for k, t in tuples_a.items():
        sel = t[3] <= boundary
        if not bool(sel.any()):
            continue  # a customer with nothing visible before the cut has no row to audit
        keys.append(k)
        # vectorized per-row histogram: a Python loop over a customer's events here is the
        # per-item hot path the doctrine bans, and the ledger is audited over every customer
        row = np.bincount(t[0][sel], minlength=int(vs["et"])).astype(np.float64)
        hand_et.append(row[: int(vs["et"])])
        hand_total.append(float(sel.sum()))
    try:
        got = builder.transform(tuples_a, keys)
    except (RuntimeError, FloatingPointError) as exc:
        # A recipe that cannot produce its own declared layout is not a ledger; report it as the
        # guardrail failure it is instead of letting an opaque crash escape the audit.
        got, g2_crash = None, str(exc)
    else:
        g2_crash = ""
    if got is None:
        diff, g2_ok = -1, False
    else:
        lo, hi = builder.part_slices()["event_mass"][0]
        off = builder.n_et
        diff = (
            int((got[:, lo:hi] != np.array(hand_et)).sum())
            + int((got[:, off : off + 1] != np.array(hand_total)[:, None]).sum())
            + leaked
        )
        g2_ok = bool(keys) and diff == 0
    out["G2_history_only"] = {
        "working": g2_ok,
        "evidence": (
            f"{len(keys)} rows at boundary {boundary:.0f}"
            + (" (declared as-of)" if as_of is not None else " (each row's own last event)")
            + f": {leaked} event(s) survived the cut, and ledger vs an independent recount "
            + (
                "agrees exactly"
                if g2_ok
                else (
                    f"disagrees in {diff} cells"
                    if got is not None
                    else f"COULD NOT RUN: {g2_crash}"
                )
            )
        ),
    }

    # ---- G3 target-blind: the recipe text may not name a target ----------------------------
    # Every function the recipe is made of is scanned, including the one G1 re-derives from:
    # splitting the recipe across methods must not shrink what the audit reads.
    recipe_text = (
        _source_of(builder.transform)
        + _source_of(type(builder).__init__)
        + _source_of(type(builder).derive_keep)
    )
    wanted = sorted({str(n) for n in target_names if str(n)})
    named = [n for n in wanted if n in recipe_text]
    uninspectable = _UNINSPECTABLE in recipe_text
    out["G3_target_blind"] = {
        # A check with nothing to look for is reported as VACUOUS, not as a pass: on the frozen
        # demo artifact `exogenous_events`/`action_fields` are empty, so "no names found" would
        # otherwise certify nothing while reading like a green.
        "working": not named and not uninspectable,
        "vacuous": not wanted,
        "evidence": "recipe source scanned for declared target/event names: "
        + (
            f"FOUND {named}"
            if named
            else (
                "source UNINSPECTABLE — not a pass"
                if uninspectable
                else (
                    f"none of the {len(wanted)} declared names appear"
                    if wanted
                    else "VACUOUS — the stream declared no target/event family to check "
                    "against (exogenous_events/action_fields empty), so this row certifies nothing"
                )
            )
        ),
    }

    # ---- G4 never feeds the brain's training ------------------------------------------------
    if brain_modules is None:
        from looking_glass import cfm_model, cfm_training

        brain_modules = (cfm_model, cfm_training)
    offenders = [
        m.__name__
        for m in brain_modules
        if "measure_state" in (s := _source_of(m))
        or "ledger_state" in s
        or s == _UNINSPECTABLE  # cannot read it is not the same as it being clean
    ]
    out["G4_not_a_brain_input"] = {
        "working": not offenders and bool(brain_modules),
        "evidence": (
            f"scanned {len(brain_modules)} brain module(s) for ledger/measure imports: "
            + (f"FOUND in {offenders}" if offenders else "neither module imports the ledger")
        ),
    }

    # ---- G5 attribution always: the consumer of this view must state what it rests on -------
    receipt = getattr(cfg, "_ledger_attribution", None)
    out["G5_attribution"] = {
        "working": bool(receipt) and bool(receipt.get("controls")),
        "pending": not receipt,  # the card produces it AFTER the build; a gap, not a defect
        "evidence": (
            f"attribution receipt present={bool(receipt)} controls="
            f"{sorted(receipt.get('controls', {})) if receipt else '(none)'} — produced by "
            f"`python3 scripts/measure_views_law.py` (S4, ledger_attribution row); the ledger is "
            f"a companion: a ledger-backed number may not be booked as a brain win (SPINE §5)"
        ),
    }
    return out


_UNINSPECTABLE = "<<UNINSPECTABLE>>"


def _source_of(fn) -> str:
    import inspect

    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        # An uninspectable recipe is NOT a pass: a guardrail that cannot be checked must say so
        # in the field the reader looks at, rather than report "none appear".
        return _UNINSPECTABLE


def attribution_receipt(arms: dict) -> dict:
    """Record what a reported row rests on (G5). `arms` names ledger-only / learned-only / both."""
    return {
        "kind": "ledger_attribution",
        "controls": {k: (None if v is None else round(float(v), 6)) for k, v in arms.items()},
        # named `principle`, not `rule`: a receipt may not introduce a criterion/rule string
        # (certification rule 3) — this states which ruling the arms serve, it decides nothing
        "principle": "the brain earns the win; the ledger prevents the loss (SPINE §5 Option 3)",
    }
