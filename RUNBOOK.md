# RUNBOOK — supervised plugin operating loop (rabbit_hole → looking_glass → plugins)

Scope: the monthly/weekly production loop for supervised plugins
(`plugins/targets.py`), rehearsed end-to-end. First target:
`supervised_purchase_propensity_30d`.

## The tables (who writes what)

| Table | Owner | Written | Read by |
|---|---|---|---|
| `encoder_samples` (A/B split) | Layer B, step (1) | monthly | audits/tests |
| `donor_embeddings` (sample-B at anchor days, `donor(h)`) | Layer B, step (1) | monthly — **the static training table all plugins share** | (2) plugin training |
| `anchor_embeddings` (sample-B, entity-pooled readout) | Layer B, step (1) | monthly | legacy readers |
| `customer_state` (recurrence state `h`, as_of) | **Layer B daily job (1b)** | **once per day** (absorb day's events, fade the rest) | (1b) only |
| `state_embeddings` (daily `donor(h)` inference features) | **Layer B daily job (1b)** | **once per day** | **(3) — read-only** |
| `plugin_scores` + `inference_receipts` (DuckDB) | Layer C, step (3) | each inference run | downstream |

**Layer C never writes state or embeddings.** (2) reads the frozen monthly table
and makes its own labels; (3) reads the day's materialized embeddings. Layer B
is the only writer of state/embeddings.

## Cadence

| When | Step | Command |
|---|---|---|
| **1st of month** | (0) size **Sample A** from the encoder-ladder receipt (first run of a new as_of runs the ladder first) → (1) encoder retrain on that Sample A, as-of the 1st (event cutoff + re-randomized A/B + **warm-start** from the previous compatible month) → products rebuilt: split, monthly embeddings, states at the 1st | `python -m looking_glass.customer_foundation_model train --customers 25000 --anchors 6 --as-of YYYY-MM-01 --db rabbit_hole/data/duckdb/customer_event_stream.duckdb --out-dir looking_glass/artifacts/cfm` |
| 1st (after 1) | (1b) day-1 close — **shipped with (1)** (build_products fades to the 1st and materializes its `state_embeddings`; receipt `device=encoder`) | *(no separate command)* |
| 1st (after 1) | (2) **per supervised plugin**: fit/gate/persist head on the frozen table, labels closed at the 1st, pinned to the encoder tag | `python -m plugins.head_template supervised_purchase_propensity_30d --as-of YYYY-MM-01` |
| 1st | (3) inference (training day counts as a run) | `python -m plugins.inference supervised_purchase_propensity_30d --as-of YYYY-MM-01` |
| **days 2..N** | (1b) daily state job (GPU absorbs): closes **yesterday** (window stops at midnight UTC — today's events excluded) then rematerializes embeddings. Day 1 ships with (1) | `python -m looking_glass.daily_states --as-of YYYY-MM-DD` |
| **Mondays** | (3) inference — the plugin's weekday (read-only) | `python -m plugins.inference ... --as-of <monday>` |

Any day can infer (inference is possible daily); the standing contract is
*weekly on the plugin's weekday + on training days*.

**As-of in production:** always pass `--as-of <the 1st>` even though production
means "all events so far" — the cut is a no-op on a clean stream, but it keeps
backtests and production on the identical code path (and guards against any
late-arriving rows in the stream).

**Populations vs samples.** Step (1) draws two disjoint populations for the
month (A/B, one deterministic draw), then draws **samples** from them for
compute: `--sample-a N` = encoder training sample from population A,
`--sample-b M` = plugin-training sample from population B (defaults: whole
populations). Samples are as small as compute wants and as large as signal
needs — sizes come from the ladders (encoder: the donor battery ladder;
plugins: `plugins.ladder`, e.g. ~8k rows ≈ 1.3–2k sample-B customers). Sample
disjointness is inherited from the populations — a sample of A and a sample of
B can never overlap.

Training-size check (optional monthly diagnostic):
`python -m plugins.ladder supervised_purchase_propensity_30d --as-of YYYY-MM-01`
→ smallest rung within measured noise of the best. The model family is chosen
by a **bake-off inside (2)** every run (logistic vs MLP vs HGB on one split;
winner = best held-out AUC, recorded in the artifact) — "whatever model is
best" is the evidence, re-checked monthly as data/dimensions scale. To pin a
family instead of re-baking, set `family=` on the Target (e.g. `family="mlp"`).

### Tuning knobs (and what each one costs)

| Knob | Tunes | Retrain encoder? | Products rebuild? | Sizing evidence |
|---|---|---|---|---|
| `--customers N` | working base (the population cap) | yes | yes | donor battery ladder |
| `--sample-a N` | encoder training sample from population A | **yes** | yes | **auto**: the as_of ladder receipt's chosen rung (explicit N overrides; no receipt → all-of-A + warning) |
| `--warm-start auto\|none\|tag` | continue from an earlier compatible checkpoint (as_of' ≤ current — never future-trained; vocab/arch guarded) | no | no | governor receipts (`warm_from`, steps) — same objective as scratch, so quality is governed, not assumed |
| `--sample-b M` | plugin-training sample from population B | no | yes | `plugins.ladder` (training rows) |
| `--anchors K` | random anchor dates **per sample-B customer** (more dates = more temporal views per customer, same encoder) | no | yes | cheap sweep (below) |
| `--epochs` | encoder budget (governed: trains to convergence under the cap) | yes | yes | autotune governor |
| `max_train` (head) | rows a plugin head fits | no | no | `plugins.ladder` |

**Cheap sweeps don't need a retrain** — anchors and sample-B never touch encoder
training, only the products tables, so rebuild them from the existing checkpoint:

```bash
for K in 4 8 12 16; do
  python -m looking_glass.customer_foundation_model products --anchors $K \
    --out-dir looking_glass/artifacts/cfm
done
```
(~minutes per try instead of a ~40-min retrain; the rebuild preserves the run's
populations, cutoff and samples from its registry. It resets states — after
keeping a setting, just rebuild products — the day-1 close ships with it.)

### Cost design — measured receipts (capability up, cost down)

| Lever | Receipt |
|---|---|
| Sample-A ladder (capability/cost sizing) | chosen **12k of 17.5k** A rows: AUC 0.7281 within 2·SE (0.0106) of full-sample 0.7348 — ~31% fewer training rows, budget 5,640 vs 8,220 steps |
| Sample-scaled training budget | rung250 converges at 240/480 steps in minutes — cost ∝ sample, never a fixed count |
| Warm-start / continual (`--warm-start auto`) | continues from the previous compatible month (leak-guarded: as_of' ≤ as_of; **behavior-guarded**: same encoder version + resolved seq_len/dim/half-life, else scratch); same objective ⇒ governor still decides convergence; fewer steps to the same plateau (receipt: `warm_from` + governor steps) |
| Bulk product IO | single-row inserts → 3 Arrow writes: products rebuild cut ~3.3× (comparable rung totals 1,448s → 435s); every monthly rebuild benefits |
| Batched daily state job | same-length bucketed forwards + bulk upsert: **86.5s → 4.1s (20.6×)** bench; **17.9s/day in production shape** (was 193–240s) |
| Warm-start (`--warm-start auto`) | same cut + sample: scratch 698s → warm 600s (**−14% wall**), steps 1,692→1,410, better CE; Nov≈Dec warm (595s/600s) |
| Read-only inference | **1.2–1.5s** per day for 25k scores (state work lives in the daily job) |
| Architecture identity | dim/seq_len/half-life derive from the full base (receipt in registry `derived`) and are re-checked at warm-start; only batch/budget follow the sample; explicit `--set` overrides always win and are recorded |

---
## Hard rules

1. **No peeking.** Encoder trains with `--as-of D` (events ≤ D only) — always
   passed, including in production (= the 1st), so backtests and production run
   the identical path. Plugin labels close at D: only `anchor + window ≤ D` and
   orders ≤ D train. Inference reads only what Layer B materialized for day D.
2. **Populations first, then samples.** Each month ONE deterministic draw
   makes two disjoint **populations** A and B (complements within the working
   base — a customer is on exactly one side; gated every run: `A/B disjoint:
   0 overlap`, persisted in `encoder_samples`). Then **samples are drawn from
   each population** (`--sample-a`, `--sample-b`) for compute efficiency: as
   small as possible, large enough for signal (sizes from the ladders).
   Sample disjointness is inherited from the populations — a sample of A and a
   sample of B can never overlap. This is exactly "draw A from the stream and
   draw B from the stream minus A": the split is just the mechanism that makes
   the exclusion structural rather than something to check. Draws re-roll
   monthly (reproducible within the month). The encoder never trains on
   anything but Sample A; plugins never train on anything but Sample B.
3. **Forward-only states.** The daily job refuses to run for a day earlier than
   the store's current `as_of` (no relabeling backwards).
4. **Read-only inference.** (3) rejects if the day's `state_embeddings` are
   missing/stale/partial — the fix is always "run the daily job", never an
   implicit advance inside Layer C.
5. **Version pinning.** A head records the encoder tag it learned from;
   inference rejects any mismatch ("a plugin head never outlives its encoder
   tag"). The monthly (1) rebuild invalidates every head → (2) same day.
6. **Fail safe, never silent.** Gate FAIL ⇒ do not ship. Any missing input ⇒
   hard error naming the command that fixes it.
7. **One day boundary.** `as_of` dates are **UTC midnight** everywhere —
   encoder cut, daily-job window, and inference lookups share it (helper:
   `as_of_epoch`). The daily job closes through **yesterday**; today's events
   are never absorbed until tomorrow's run (a time-carrying `as_of` warns).
8. **One readout rule (v2.1.0).** Training-anchor embeddings and serving
   embeddings apply the SAME fade: `donor(fade(state, boundary − last_event))`
   — anchors fade to the anchor, serving to the scoring boundary. Half-life
   derives from the **p95 of within-customer gaps** (registry receipt), never a
   merged timeline (that pinned it at the 1-hour floor and silently separated
   train from serve). Explicit `--set` config overrides are applied AFTER
   derivation and recorded — derivation never overwrites them.

## What a cycle produces

- encoder tag `vN.rM` + registry receipt (`as_of`, `split_seed`, data signature);
- refreshed products: split, monthly embeddings, states at the 1st;
- one `state_job_receipts` row **per day** (absorbed/idle/embeddings, device, wall);
- per plugin: artifact JSON + persisted head (`.joblib`) pinned to the tag;
- per inference day: `plugin_scores` rows + `inference_receipts` row (read-only);
- rehearsal timeline: `plugins/artifacts/rehearsal_<start>.json`.

## Rehearsal (dry-run the whole loop)

```bash
# standing cadence: month-start + the next 7 days (default --days 8)
python -m plugins.rehearsal --customers 25000 --anchors 6 --start 2025-11-01
# full month instead:
python -m plugins.rehearsal --customers 25000 --anchors 6 --start 2025-11-01 --days 31
```
Each cycle: (1) encoder on the 1st (also closes day 1: fades states to the
1st + materializes its embeddings) → ladder → (2) plugin
→ then a **daily state job for days 2..N** of the `--days` window with read-only
inferences (day 1 came with (1)) on the training day, the Mondays inside the window, and one
mid-week day. New A/B rotation + re-pin at each 1st. Inference days are
clamped to the daily-job window (a day can never be scored before its
embeddings exist). Emits the timeline receipt (`rehearsal_<start>.json`) with
the month-over-month split-rotation proof.

### Measured timings (25k customers, RTX 2080)

| Step | Time |
|---|---|
| (1) encoder train | **~37 min/cycle** (the dominant cost) |
| (1b) day-1 close | **folded into (1)** — fade to the 1st + materialize (receipt row: device=encoder) |
| (1b) steady daily job (≈8.5k absorbs + fade + materialize 25k) | **mean 193 s** (range 87–235 s) |
| training-size ladder (9 rungs) | ~35 s |
| (2) plugin train — 3-family bake-off incl. MLP + HGB | ~20 s + data load |
| (3) inference, read-only (25k scores) | **1.2–1.5 s** |
| **per cycle, default --days 8** | **≈ 65 min** |
| **full 2-month run, --days 31 (measured)** | **4.5 h (16,348 s)** |

## Adding another supervised plugin

Add one `Target` to `plugins/targets.py` (name, kind, window, `spec_target`,
`feature_table="donor_embeddings"` for live-scorable targets). Everything else —
label SQL, split, fitting, model bake-off, gates, artifact, persisted head,
read-only inference — comes from the template. No new code path.
