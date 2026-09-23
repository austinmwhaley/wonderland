"""Central configuration. QUICK_LOOK runs in minutes; PUBLICATION scales seeds,
episodes, and families without code changes. Adding an agent = one roster row
in colony/roster.py. Adding an OPE estimator = implement it + ESTIMATORS row
in tribunal/ope/estimators.py + one panel call + one gate row + slope ORDER
entry + judge rank note + tests (test_registry_keys_match_panel enforces it).

All artifacts live under white_queen/ (data/, checkpoints/, verdicts/) —
nothing is written outside this package. Paths are anchored to this file,
so runners work from any cwd.

Adaptive contract (v3): gate=None and ope_meta=None mean "derive from the
diet at runtime" (recommended). Pass partial dicts to pin individual keys;
missing keys are backfilled by tribunal.ope.autotune. risk_aversion picks the
judge operating point: 0.0 aggressive / 0.5 standard / 1.0 conservative.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
VERDICTS = ROOT / "verdicts"
OPE_CACHE = CKPT / "ope_cache"  # estimator weights cache (exact on hit).

QUICK_LOOK = {
    "env": "cartpole",
    "gamma": 0.99,
    "seeds": [0, 1],
    "db_path": str(DATA / "white_queen_quick.db"),
    "ckpt_dir": str(CKPT / "quick"),
    "verdict_dir": str(VERDICTS / "quick"),
    # Colony: {roster_key: episodes_per_seed} — roster rows live in colony/roster.py.
    "collect_episodes": 40,
    "collect_eps": 0.15,          # colony-level epsilon mixture (exact analytic probs)
    "dqn_train_steps": 30_000,    # birth->mastery compressed for quick-look
    "pg_train_steps": 30_000,     # PPO/A2C budget
    "checkpoint_fracs": (0.05, 0.5, 1.0),  # birth / mid / expert from ONE run each
    # Tribunal
    "offline_steps": 10_000,
    "ope_fqe_steps": 3_000,
    "ground_truth_episodes": 20,
    # Coverage diets (SQLite views over the same rows): headline + two stresses.
    "diets": ("mixed", "novice_only", "expert_only"),
    # Gate / OPE: None = autotune from data at runtime (recommended).
    # Legacy literals kept as comment for reproducibility:
    #   gate {"rel_edge_std": 0.5, "min_ess_frac": 0.05}
    "gate": None,
    "ope_meta": None,
    # Estimator weights cache (exact on hit, fail-open). Re-judges after
    # gate/judge changes cost seconds, not refits. Keys include budgets, so
    # screen-tier fits never collide with certify-tier fits.
    "ope_cache_dir": str(OPE_CACHE),
}

PUBLICATION = dict(
    QUICK_LOOK,
    seeds=[0, 1, 2, 3, 4],
    collect_episodes=150,
    dqn_train_steps=100_000,
    pg_train_steps=100_000,
    offline_steps=50_000,
    ope_fqe_steps=10_000,
    ground_truth_episodes=100,
    diets=("mixed", "novice_only", "expert_only", "low_coverage"),
)

# SMOKE: core-functionality test in ~1-2 min (PyTorch exercised, no 30-min wait).
# One diet, tiny budgets, 2 live episodes. Catches crashes/NaNs/schema breaks.
SMOKE = dict(
    QUICK_LOOK,
    seeds=[0],
    collect_episodes=5,
    dqn_train_steps=1_000,
    pg_train_steps=1_000,
    offline_steps=200,
    ope_fqe_steps=200,
    ground_truth_episodes=2,
    diets=("mixed",),
)

PRESETS = {"quick": QUICK_LOOK, "publication": PUBLICATION, "smoke": SMOKE}
