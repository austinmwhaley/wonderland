# Layer-B objective catalog — looking_glass

**What this is:** the reference for what the encoder can be trained on, what
each objective forces into the state, what we run today, and the full menu of
common self-supervised alternatives — organized so an operator can decide
which to turn on/off for any event stream. Maintained under DEC-008/DEC-009;
the standing grade for whatever is ON lives in `looking_glass/portfolio.py`.

**How to read status marks**

| mark | meaning |
|---|---|
| ✅ | active in the default portfolio, graded by the portfolio receipt |
| 🔧 | active but known-defective (fix scheduled, see notes) |
| ⏸ | candidate — documented, not enabled; enters only via the protocol below |
| ⛔ | excluded by doctrine or missing capability (reason stated) |

**Entry protocol for any objective (DEC-010):** capability probe must pass →
`cfg.objectives` / `--set objectives=(...)` turns it on (config, never code) →
it lands with seeds + a portfolio receipt + version bump → battery/ablation
guards unchanged → only then default-on. An objective that cannot be graded
by the destroyed-data structure-skill null gets an explicit geometry/report
row instead — never silence.

---

## 0. The three axes (frame)

Every objective sits on three independent axes; the set is only "complete"
when coverage is deliberate across all three:

1. **WHAT is predicted:** raw observable token · continuous attribute ·
   latent vector · aggregate statistic
2. **WHEN:** immediate next step · fixed horizon · whole future suffix ·
   continuous horizon sweep
3. **HOW the loss shapes space:** discriminative · generative · geometric
   (align/uniform/decorrelate) · structural (order/invariance)

Our default set covers all three values of WHAT and all four HOW families;
WHEN has next-step, suffix (jepa), sampled-moment (query) and horizon-sweep
(sf) plus exact finite windows (agg). The gaps are mostly in *structure of the
entity vocabulary*, *distributions instead of points*, and *non-stationarity*
(see §4).

---

## 1. Objectives we train today (13) — inductive bias & guarded failure

| # | objective | what it forces the state to encode | failure mode it guards | status |
|---|---|---|---|---|
| 1 | `next` | conditional distribution over the event alphabet given history — the Markov abstraction that transfers | a state that memorizes the last event instead of integrating history | ✅ |
| 2 | `entity` | fine-grained identity inside a type (which member, not just category), long-tail resolution | category-only representations that cannot separate SKUs/customers | ✅ (weakness: flat softmax over big vocab — see semantic-ID ⏸) |
| 3 | `dt` | timing/urgency — the intensity of the underlying process (log param = multiplicative gaps, heavy tails) | a "bag of events" representation blind to rhythm | ✅ |
| 4 | `value` | magnitude/severity orthogonal to type — money sensitivity | latents identical for identical events of wildly different size | ✅ (requires a value field — capability-gated in the plan) |
| 5 | `occur` | discrete-time hazard: "does anything happen soon?", class-balanced by construction (median-gap threshold) | zero-inflation swamping the timing gradient (optimization-stability role; subsumed informationally by `dt`) | ✅ |
| 6 | `mask` | interpolatability from **left context with the position redacted** — denoising robustness | fragile dependence on exact neighbors; sparse/missing events (a real serving condition) | ✅ **fixed this cycle**: see note A |
| 7 | `order` | sequential structure itself — "time flows" | permutation-invariant encoders acing content heads while ignoring order | ✅ |
| 8 | `contrast` | instance dispersion (repel other batch states on the sphere) | dimensional crowding | 🔧 **see note B** — intended alignment has no second view |
| 9 | `redundancy` | channel independence (decorrelation = Barlow/VICReg covariance term) | low-rank collapse of every other head's geometry; **load-bearing for jepa** | ✅ graded in geometry section too |
| 10 | `jepa` | sufficiency for the future's *representation* (EMA teacher over the suffix) — self-distillation, negative-free alignment | point-prediction overfitting to stochastic detail; teacher-smoothed long-horizon structure | ✅ |
| 11 | `sf` | linearly-decomposable model of future occupancy: discounted visitation to each φ-feature at sampled γ (horizon-free sweep) | representations that answer "what does the future look like" but not "what it's worth in feature space" | ✅ event-agnostic φ (value + per-event-type components; legacy purchase-mode kept for old checkpoints) |
| 12 | `query` | the readout at an *arbitrary moment*: state faded across a sampled gap must predict next type/time | the serving path (fade last-event → anchor) never receiving a gradient; recency hidden as magnitude shrinkage | ✅ new (DEC-006 S1 as self-supervision) |
| 13 | `agg` | exact window integration: log1p count and log1p value-sum in (t, t+h] at horizons derived from gap quantiles | long-horizon activity structure never demanded of the state (the statistics raw RFM hands downstream for free) | ✅ new (DEC-006 S2 as self-supervision) |

**Unifying observation (§1 cluster 1–5):** `{next, entity, dt, value}` factor
by construction = the likelihood of a **marked temporal point process**
(type × mark × time). What's missing for TPP completeness is an explicit
continuous-time intensity λ(t|h) — listed as candidate §5a.

**Note A — `mask` is causal in *our* implementation.** The backbone is a
prefix-scan recurrence: masked positions are predicted from left context
only; right context never participates (tested:
`tests/test_objectives_portfolio.py::test_masked_forward_causal_and_redacted`).
All content channels at the masked position are redacted (type/brand/entity/
value); arrival time (`dt`) and covariates (`co`) remain — the task is "an
event of unknown kind arrives after this gap", which matches serving on
sparse streams. If the backbone ever becomes bidirectional this note must be
re-derived. Mask rate is `cfg.mask_frac` (config).

**Note B — `contrast` currently has no positive pair.** The implemented loss
is `CE(normalize(proj(h)) @ normalize(proj(h)).T / τ, identity)` over one view
per sequence: the diagonal is trivially 1.0, so the loss reduces to *repelling
other batch members* — uniformity only, no alignment. `cfg.gamma_contrast` is
never referenced (dead config). Fix candidates: real augmentation views
(value-jitter, time-jitter, entity-dropout, subsequence crop — the view design
IS the inductive bias), or reclassify it explicitly as a uniformity term
alongside `redundancy`. Until fixed: 🔧.

**Tensions when running all 13 jointly (and our mitigations)**

| tension | status |
|---|---|
| gradient conflict (contrast/mask sensitivity vs invariance) | mitigated partially by learned uncertainty weights (`_combine`, no hand λs); watch when contrast gets real views |
| redundant supervision (`occur ⊂ dt`, sf-count ⊂ agg) | accepted — ensembling helps; weights learned, portfolio grades each separately |
| asymmetric collapse (jepa/contrast collapse without uniformity+variance) | `redundancy` is load-bearing; portfolio geometry gate (eff-rank vs permuted null) catches collapse |
| horizon mismatch (1-step heads dominate a shallow trunk) | `sf`/`agg`/`jepa`/`query` are long/moment horizons now; portfolio structure-skill per objective is the probe (a 1-step-only model scores ~0 on the long ones) |

---

## 2. The operating menu — seven families (operator taxonomy)

Status column uses §0 marks; **requires** = capability the stream probe
(§5) must confirm before the plan can enable it.

### 2.1 Next-step predictive (local autoregressive dynamics)
| item | candidate objective | status | requires |
|---|---|---|---|
| a | categorical transition (`next`) | ✅ | ≥2 event types |
| b | target/attribute regression (`entity`) | ✅ | ≥1 entity field, enough counts |
| c | temporal cadence (`dt`, `occur`) | ✅ | timestamps |
| d | magnitude/intensity (`value`, `agg` value dim) | ✅ | nonzero value variance |
| e | binary survival horizon (`occur` + `agg` counts) | ✅ | timestamps |

### 2.2 Masked reconstruction & order
| item | objective | status | requires |
|---|---|---|---|
| a | BERT-style token reconstruction (`mask`) — **left-context + redaction in our causal backbone, not bidirectional** | ✅ | seq_len ≥ ~4 |
| a2 | **field-level masking** (mask value only / type only / time only — fill each modality from the others) | ⏸ S-effort | typed records (we have them) |
| a3 | **span masking** (T5-style contiguous spans — trajectory completion vs single-token interpolation) | ⏸ S-effort | seq_len ≥ ~8 |
| b | temporal coherence (`order`; candidates: pairwise temporal-ranking, shuffle-learn variant, **future-state distance regression** (predict Δt between two states — cheap, ⏸ S)) | ✅ for `order`; ⏸ for the rest | timestamps |

### 2.3 Representation-level & joint-embedding
| item | objective | status | requires |
|---|---|---|---|
| a | contrastive similarity (`contrast`) | 🔧 needs views first | — |
| a2 | **augmentation-policy design** for the views (value-jitter, time-warp, entity-dropout, crop, type-preserving swap) | ⏸ ★ highest-ROI fix in this family | — |
| a3 | CPC / InfoNCE(future latent vs context), TS2Vec/TCC multi-scale variants, MoCo queues + hard negatives | ⏸ (variants of jepa+contrast; add only if the contrast fix needs negatives at scale) | batch ≥ 4, queue if vocab big |
| b | information-max / non-collapse (`redundancy`; candidates: VICReg **variance term** ⏸ S, SimSiam stop-grad ⏸, NNCLR/prototypical ⏸, SwAV/DeepCluster ⏸ L — §2.6) | ✅ redundancy | batch ≥ 2 |
| c | JEPA (suffix latent, EMA teacher) | ✅ | seq_len ≥ ~6 |
| c2 | **probabilistic JEPA** (predict distribution over target latents) / **horizon-sweep JEPA** (predict latent at h ∈ {1,4,16,64}) | ⏸ M-effort both | — |

### 2.4 Long-horizon aggregate
| item | objective | status | requires |
|---|---|---|---|
| a | successor features (`sf`, discounted sweep over φ) | ✅ | timestamps (+ value if money present) |
| b | **quantile / distributional future head** (predict FV of count/value, CRPS-style proper scoring rule; enables CVaR/LTV tail decisions) | ⏸ ★ top business-aligned candidate | agg horizons (already derived) |
| c | explicit Poisson/count head for window volume | ⏸ (mostly subsumed by `agg`; add only if a downstream volume forecast needs a proper count likelihood) | — |

### 2.5 Temporal & rate-based
| item | objective | status | requires |
|---|---|---|---|
| a | **continuous-time intensity / hazard λ(t\|h)** (neural Hawkes / marked-TPP likelihood — unifies dt+occur+value into one principled likelihood) | ⏸ ★ medium effort, high principle | dense timestamps (we have them) |
| b | **Fourier / periodicity modeling** (predict hour/day-of-week structure; periodic positional objectives) | ⏸ — auto-enable only when a **seasonality probe** fires (hour/dow histogram divergence from uniform); rabbit_hole currently scores ~flat, so the probe would correctly say OFF today | significant seasonality score |

### 2.6 Latent structure & clustering
| item | objective | status | requires |
|---|---|---|---|
| a | DeepCluster / SwAV online prototypical assignment (discrete codes for free — useful as segments) | ⏸ L-effort | stable batch ≥ ~1k |
| b | VICReg-style **variance penalty** (per-dim variance floor — closes what pure decorrelation misses) | ⏸ S-effort, pairs with `redundancy` | batch ≥ 2 |

### 2.7 Higher-order structural
| item | objective | status | requires |
|---|---|---|---|
| a | **subsequence / phrase prediction** (encode a span, predict it from outside — harder than token mask) | ⏸ S-effort (family 2.2a3 neighbor) | seq_len ≥ ~16 |
| b | future-state distance regression (TDM-style: how far apart are two states in time) | ⏸ S | timestamps |
| c | **seq2seq autoregressive** (encode prefix → decode multi-step future t+1..t+k jointly; closes 1-step vs whole-future gap) | ⏸ M | decoder head |

---

## 3. Wider landscape — other common objectives (not yet in the menu)

One-liners; each marked with what it would require. These complete the map
so nothing common is invisible when choosing.

| family | objective | note / requires |
|---|---|---|
| AR | multi-step joint prediction (t+1..t+k) | overlaps §2.7c |
| AR | inverse dynamics (state_t, state_{t+1}) → what action/caused it | ⏸ **capability: side-action labels** — rabbit_hole has `co` sends → plan-eligible; Instacart has none → correctly OFF. Turns encoder into attribution machinery for interventions |
| AR | forward dynamics (state + action → next latent) | pairs with inverse dynamics |
| AR | neural Hawkes / intensity matching | = §5a |
| Masked | substitution infilling (BART: corrupt → regenerate sequence) | ⏸ medium |
| Masked | partial-attribute reconstruction (subset of fields) | = field-level masking §2.2a2 |
| Contrastive | time-contrastive learning (TCN): discriminate time-window identity | ⏸ needs period labels (probe: spans_periods) |
| Contrastive | temporal neighborhood coding (TNC) | ⏸ — same family |
| Contrastive | cross-modal CLIP-style (pair sequence with text/campaign metadata) | ⛔ unless the stream carries an external modality |
| Clustering | SeLa / online Sinkhorn assignment, prototypical networks | = §2.6a |
| Generative | **VAE on sequences** (sample synthetic customers/trajectories — enables what-if rollout + privacy) | ⏸ L; needs KL balancing; only if downstream wants simulation |
| Generative | diffusion / score-matching over (value, dt) futures | ⏸ L; multimodal futures a Gaussian head can't express |
| Generative | sequence GAN | ⛔ dated; no unique gain over the above |
| Structural | graph objectives (link prediction / node masking on the co-purchase or customer-similarity graph) | ⏸ M — capability: co-occurrence graph derivable from stream (probe: distinct co-occurrence edges); transactional data usually has this and we ignore it today |
| Structural | set prediction / DeepSets over baskets (the "next" is really a set of simultaneous items) | ⏸ M; requires basket/session grouping (rabbit_hole sessions, Instacart orders — both present) |
| Latent | data2vec-style multi-position teacher prediction | = horizon-sweep JEPA §2.3c2 |
| Latent | forward+inverse dynamics pair (model-based RL world model) | downstream of inverse dynamics |
| Quantized | **semantic IDs via RQ-VAE/VQ-VAE for entity head** (short residual codes shared across similar items; fixes the long-tail flat softmax, unlocks cold-start) | ⏸ ★★ L-effort; capability: entity_vocab large + sparse counts (probe quantiles) — rabbit_hole's small vocab says OFF, real retail says ON |
| Invariance | **domain-adversarial / IRM across periods** (predict period, then adversarially remove it → transferable across quarters/markets) | ⏸ M; capability: spans_periods ≥ K (probe) |
| Invariance | temporal-consistency regularization across noisy views | ties to contrast fix |
| Info-theoretic | **surprise / info-gain weighting** (weight losses by conditional entropy → capacity on rare events: fraud/churn triggers) | ⏸ S; needs a surprise estimator (ppl of `next` head is a free one) |
| Info-theoretic | MINE / generic InfoNCE | = contrast family |
| Info-theoretic | auxiliary side-attribute prediction (channel, device, campaign tag from state) | ⏸ S; capability: side fields present (`co`/contacts) |
| Distributional | quantile heads / CVaR / CRPS proper scoring rules | = §2.4b |
| Periodicity | predict time-of-day/day-of-week of the next event | ⏸ S; capability: seasonality probe (see §2.5b) — strongest cheap seasonality objective when the data actually has seasonality |

---

## 4. Where the next quality comes from (ranking for our case)

The through-line: our WHAT/WHEN/HOW coverage is already broad; the remaining
gaps cluster exactly as the operator identified:

1. **Entity-vocabulary structure** → semantic IDs (§3 Quantized, ★★).
   Fixes the statistically weakest head (`entity` flat softmax) + cold-start.
2. **Distributions instead of points** → quantile/probabilistic future head
   (§2.4b) + probabilistic JEPA (§2.3c2). The business decision usually wants
   the tail (CVaR/LTV), not the mean.
3. **Non-stationarity** → drift invariance (§3 Invariance) + seasonality
   objectives that *activate only when the probe sees seasonality* (§2.5b).
4. **Immediate cheap wins**: contrast views (🔧), field-level + span masking
   (§2.2a2/a3), future-state distance (§2.2b), variance term (§2.6b),
   hazard head (§2.5a).
5. **Capability-conditional**: inverse dynamics + graph + set/basket
   objectives — all three are *available on rabbit_hole-class streams today*
   and correctly unavailable on Instacart — which is the whole point of §5.

---

## 5. Universal-encoder plan: any stream in → objective plan → train → grade

Goal (operator): *pass any event stream, and the encoder trains the optimal
universal objective set for it* — objectives selected by what the stream can
support, not by hardcoding.

```
stream ──► capability probe (receipt) ──► objective plan (receipt)
                                            │  config --set overrides, recorded
                                            ▼
                              train (held-out combined objective selects)
                                            │
                                            ▼
                       portfolio grade (structure-skill + geometry) ── receipt
                                            │
                     battery / ablation / plugin gate (veto-only seams)
```

**Probe fields (all data-derived, no literals beyond documented floors):**

| capability | derived how | unlocks |
|---|---|---|
| `has_money` | nonzero variance of value field | `value`, sf money dim, agg value dim |
| `entity_profile` | vocab size + per-entity count quantiles | `entity`; semantic-ID upgrade when counts are sparse & vocab large |
| `n_event_types` | distinct count | `next` (≥2), type-aggregate views |
| `seq_len_profile` | p50/p90 of events per customer | mask/span/phrase thresholds |
| `timestamp_resolution` | gcd/quantiles of positive gaps | `dt`, `occur`, hazard head, `query` |
| `span_days` / `periods` | data_end − data_start; distinct buckets | drift invariance, period contrast |
| `side_actions` | configured `company_actions` present in stream | inverse dynamics, covariates |
| `seasonality_score` | hour/dow histogram divergence from uniform | Fourier/periodicity objectives (§2.5b, §3 Periodicity) |
| `graph_density` | distinct co-occurrence pairs / customers | graph objectives |
| `basket_grouping` | sessions/orders with ≥2 line items | set/basket objectives |

**Plan rules:** every objective carries `requires: [capabilities]`; plan =
enabled iff all requirements pass; disabled entries recorded with their
missing capability (receipt, not silence). Core set that needs nothing but a
sequence: `next, dt, occur, mask, order, jepa, redundancy, query, agg, sf`.
`contrast` stays OFF in the plan until its views exist (🔧 note B) — today's
default tuple still includes it for continuity; the plan receipt will say so
explicitly.

**Profiles (starting defaults for the conversation):**

| profile | enables beyond core | disables beyond core |
|---|---|---|
| rabbit_hole-class (money, side actions, sessions, 25k) | `value`, inverse-dynamics candidate, graph candidate, set candidate | — |
| Instacart-class (no money fields beyond counts, no sends, orders as baskets) | set/basket candidate | `value`-driven dims degrade to counts; no inverse dynamics |
| big-retail (huge sparse entity vocab) | entity + **semantic-ID** upgrade | — |
| strongly seasonal stream | periodicity objectives (§2.5b) | — |
| multi-quarter / multi-market | drift invariance | — |

**Open choices for the operator (turn on/off conversation):**
1. contrast: implement views now (🔧 fix) vs turn it OFF until then?
2. hazard head (§2.5a) — I recommend next-medium: principled unification of
   dt/occur/value we already train.
3. quantile/distributional head (§2.4b) — I recommend next-medium: the
   downstream decisions want tails.
4. semantic IDs (§3, ★★) — largest win for the weakest head, but L-effort;
   needs a real big-vocab stream to justify (rabbit_hole cannot).
5. field-level + span masking + future-state distance (all S) — cheap, safe;
   I recommend batching these three into one cycle.
6. seasonal/graph/set/inverse-dynamics — leave to the probe; no discussion
   needed until a stream trips the capability.

---

## 6. References (canonical per family)

- TPP/marked point processes: Du et al. 2016 (neural Hawkes); Mehta et al.
  2022 (torchTPP survey).
- Masked: Devlin et al. 2019 (BERT); Raffel et al. 2020 (T5 spans).
- Contrastive/uniformity: Oord et al. 2018 (CPC/InfoNCE); Chen et al. 2020
  (SimCLR — view design); Wang & Isola 2020 (alignment/uniformity); Tian et al.
  2020 (information contrast); Bjorck et al. 2021 (Barlow/VICReg family).
- JEPA/latent: Grill et al. 2020 (BYOL); Chen & He 2021 (SimSiam); BALENTOVIC
  et al. 2022 (I-JEPA); data2vec (Baevski et al. 2022).
- Sequence SSL: T-Loss (Richards/... 2021); TS2Vec (Yue et al. 2022); TS-TCC
  (Eldele et al. 2021); TCC (Siam et al. 2021); Time-Contrastive (Hadsell
  et al. 2006).
- Order/structural: O3N shuffle-and-learn (Fernando et al. 2017); TDM
  (Weihs... / Varma et al. 2017).
- RL import: Sutton et al. successor features; Barreto et al. 2018.
- Quantized IDs: RQ-VAE (VQ tokenizers); TIGER (Rajput et al. 2023) for
  semantic IDs in recsys.
- Distributional: CRPS (Gneiting & Raftery 2007); quantile loss (Koenker);
  probabilistic forecasting heads.
- Non-stationarity: IRM (Arjovsky et al. 2019); domain-adversarial (Ganin et
  al. 2016).
