# looking_glass v7.0 — Architecture RFC: Centered Covariance SSM State Dynamics

Status: **DRAFT** · Supersedes the v6 trunk state-aggregation mechanism · Owner: Layer B

## 1. Problem (proven, DEC-040)

The v6 trunk is a continuous selective SSM:
`h_t = decay_t ⊙ h_{t-1} + (1−decay_t) ⊙ (W_B x_t)`.

**Linear State Collapsibility (empirical theorem).** When such an SSM aggregates
asynchronous event sequences through **linear** input projections, the
**cross-sequence** state distribution `{h(c) : customers c}` is dominated by a
single common temporal-density component, so its covariance is **rank-1** (or
near). Concretely, measured across three streams:

| stream | cross-customer PR/dim |
|---|---|
| rabbit_hole (14 types, 77% alternation) | ≈ 0.017–0.19 |
| Instacart | ≈ 0.08–0.17 |
| ecommerce_2019 (3 types, 7% alternation) | ≈ 0.004–0.02 |

**Why orthogonalization cannot fix it:** `Q ∈ O(D)` preserves subspace rank, so
orthogonal `W_B`/`W_C`/`proj` only *rotate* the low-rank attractor. And a hard
log-det barrier can only *report* `+∞` at a collapsed init; with a monotone
(Armijo) line search it **deadlocks** (escape needs a transient loss increase).

**Four-mechanism wall:** soft DWA penalties (rank traded away) · Armijo
(converges to the collapsed min) · barrier + Armijo (deadlock) · Cayley init
(rank preserved, curvature exploded). ⇒ **This is a state-aggregation property,
not a loss/step/init defect.**

## 2. What v6 keeps (proven, frozen)

Per-sample Cayley isometric boundary (κ=1, no rank-faking) · conservation ratio
`Tr(Σ_readout)/P_in ≈ 1` · Markov sufficiency gap `Δ_suff` vs shuffled null ·
hard stability law (divergence = failure, no false green) · scale-free gates.

## 3. v7 objective

Raise the **cross-sample state rank** structurally, so the donor encodes
*per-customer variation* disentangled from the *common stream trend*, while the
Armijo line search stays monotone and the log-det barrier is satisfied at `t=0`.

Success criteria (all must hold, on ≥2 diverse streams):
- cross-customer `PR/dim ≥ τ` (τ derived from the stream's achievable capacity),
- Armijo monotone descent with **no** deadlock (barrier finite at `t=0`),
- intrinsic 6/6 and the portfolio objective skills PASS,
- reproducible; no faked capacity.

## 4. Candidate mechanisms (to be spiked, then measured)

1. **Centered-covariance recurrence.** Maintain a population running mean state
   `h̄` (EMA over customers). *Open question:* plain subtraction is a translation
   and is covariance-invariant — so the spike must test whether centering the
   **input** inside the recurrence (`W_B(x_t − x̄_t)`) changes the *dynamics* of
   the per-customer deviation rather than merely translating `h`. Report the
   measured cross-sample PR, not the intuition.
2. **Cross-sample Gramian rescaling.** At each step, rescale/rotate the state by
   the inverse square root of the **cross-batch** covariance (a per-batch
   decorrelation inside the recurrence). Must be per-sample at inference (frozen
   running transform), else it is batch-coupled and violates functoriality.
3. **Non-linear local manifold expansion.** Replace the linear `W_B x_t` with a
   local non-linear update (e.g., a small MLP/gated interaction) so orthogonal
   event-space channels are not forced onto a single latent direction.
4. **Cross-sample contrastive monoid action.** Define the objective over
   *sequence pairs* so the transition operator is pushed to map distinct
   customers to orthogonal sub-manifolds (a cross-sample orthogonality law),
   rather than only single-sequence temporal prediction.

## 5. Risks / open questions

- **Translation invariance** may make naive centering a no-op for rank — spike it.
- **Monotone descent vs escaping a low-rank basin**: if the high-rank solution is
  a saddle requiring loss increase, Armijo forbids it — a trust-region that
  *enlarges* only when the barrier is finite may be required.
- **Functoriality**: any cross-sample operation must be frozen (serving-time
  constant) to keep `F` a per-sample morphism (A1).
- **Data intrinsic rank**: on genuinely low-diversity streams (ecommerce),
  high rank may be unattainable; the gate must remain per-stream honest.

## 6. Non-goals

No further tuning of lr schedules, loss weights, or parameter initializers to
mask the collapse. No green tag on the current trunk.

## 7. Next steps

1. Spike (24h): centered-input recurrence (4.1) — measure cross-sample PR/dim on
   rabbit_hole before/after; keep iff it raises rank with Armijo stable.
2. If 4.1 fails, spike 4.2 (cross-sample Gramian, frozen transform).
3. Only after a mechanism demonstrably raises rank: integrate, gate, re-run the
   triad.

## 8. Spike log

### 8.1 Centered-covariance (input) recurrence — REJECTED (measured)
- Implementation: `h = SSM(x − E_batch[x])`, measured cross-sample PR/dim on
  rabbit_hole (dim=64, 512 seqs, random init).
- Result: baseline 0.0227 vs centered-input **0.0225** vs population-centered
  consumed state 0.0227 — **no rank change**. The centered input alters the state
  (`max|Δh−const|≈165`) but not its cross-sample covariance spectrum.
- Conclusion: for a **linear** SSM this is an affine reparameterization →
  translation-invariant spectrum. **Rejected** by §7.1's gate; do not integrate.
- Next candidate: §4.2 cross-sample Gramian (frozen, per-sample at inference) or
  §4.3 non-linear local expansion — the only ones that can change the *non-linear*
  geometry. Both must be spiked with the same measured gate.

### 8.2 Cross-sample Gramian inside the recurrence (§4.2) — REJECTED (measured)
- Implementation: per-step decorrelation of the state by the batch covariance
  `h ← μ + (h−μ)Σ^{-1/2}` (Newton–Schulz), measured on held-out rabbit_hole.
- Result: baseline held-out PR/dim **0.023**; **frozen (train-fit) transform
  0.0209** — No genuine gain. Batch-fit (transform fit on the eval batch)
  0.0414 — an artifact of batch-coupling (manufactures rank, same failure as the
  removed ZCA boundary).
- Conclusion: the per-step Gramian is a no-op when honestly frozen and dishonest
  (batch-coupled) otherwise. **Rejected**.
- **Both §4.1 (linear centering) and §4.2 (gramian normalization) are rejected**:
  neither changes the *genuine, transferable* cross-sample rank. The only
  structurally-distinct remaining candidate is **§4.3 — a NON-LINEAR local
  expansion** (an actual change of the update's function class, not a linear
  reparameterization or a batch normalization). Spike it with the same gate.

### 8.3 Non-linear local expansion (§4.3) — POSITIVE (first rank-changing mechanism)
- Variants measured on held-out rabbit_hole (dim=64, random init, no batch-fit):
  base **0.023** · sigmoid-gate 0.0196 · additive-tanh MLP 0.0234 ·
  **multiplicative `h ⊗ x` 0.0398 (1.73x, transfers)**.
- The **bilinear** term `h_t = decay·h + (1−decay)·bx + 0.3·(tanh(hA) ⊙ bx)` is
  the only candidate that raises the *transferable* cross-sample rank (linear
  centering and batch normalization were no-ops/manufacturing). It is
  **per-sample** (no batch coupling) → keeps `F` a morphism; rank-changing because
  the image of a bilinear map can exceed the linear aggregate's span.
- **Verdict: ADVANCE.** Integrate the multiplicative interaction into the
  SelectiveSSM update, then train + grade: keep iff held-out cross-sample PR/dim
  reaches the stream capacity floor with Armijo monotone and barrier finite at t=0.

## 9. v7 RFC item — Stiff-Update Stability Reconciliation (OPEN, DEC-042)

**Problem:** the bilinear update `h⊗x` (correct: +1.93× transferable
cross-sample rank) spikes the init curvature to `L≈1e7`, and any `lr∝1/L`
derivation collapses to `~4.6e-8` → frozen training. Curvature preconditioning
over-reacts to the stiff high-frequency directions of a non-linear manifold.

**Target:** a **schedule-free, curvature-free** stability guarantee compatible
with stiff non-linear updates. Candidate directions:
1. **Normalized / diffeomorphic update** — constrain the state map to a
   norm-preserving (or bounded-Jacobian) form so stability is structural, not a
   step-size estimate.
2. **Step acceptance on the un-preconditioned direction** — evaluate the actual
   training-loss change for a candidate step (Armijo on the true Adam direction)
   rather than imposing a global `2/L` bound; reject on increase. This is
   optimizer-agnostic and does not estimate `L`.
3. **Spectral-normalized bilinear term** — bound the Lipschitz constant of
   `tanh(W_nl ·)` (e.g., spectral norm ≤ 1) so the term cannot stiffen the
   landscape.

**Gate:** bilinear ON → held-out cross-sample PR/dim clears the stream capacity
floor **with monotone training-loss descent and no step-collapse** (no frozen
run), on ≥2 diverse streams. Only then integrate + run the triad.

### 9.1 Path-1 result (DEC-043) — necessary, not sufficient
Path 1 (unit-norm gate) + removing the rejected Cayley init **unfroze** training
(monotone, no deadlock), but the curvature-seeded base lr (`0.5/L ≈ 2.4e-7` on
the stiff composite loss) leaves the run undertrained (4/13 fast-run skills
fail). **Path 2 is the designated next change**: remove the curvature-seeded lr;
let Armijo accept/reject the true Adam step with no `2/L` bound.
