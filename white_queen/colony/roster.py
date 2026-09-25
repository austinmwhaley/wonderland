"""Behavior Colony roster. One row = one behavior family member. Adding an agent
= appending a dict; no other code changes. 'fracs' are used as independent
training budgets (fraction of the family budget): a 0.05-frac policy is a
genuinely undertrained agent — birth, honestly, with no checkpoint surgery.
"""

# (key, family, lab_module, lab_class, budget_key, fracs, extra_config)
ROSTER = [
    # Floor: uniform random. checkpoint_frac None -> novice_only diet by algo.
    {
        "key": "random",
        "family": "random",
        "module": None,
        "class": None,
        "budget": None,
        "fracs": (None,),
        "extra": {},
    },
    # Value-based DQN family: Dueling and Double variants ride DQN flags.
    {
        "key": "dqn",
        "family": "value-based",
        "module": "algorithms.deep.dqn",
        "class": "DQN",
        "budget": "dqn_train_steps",
        "fracs": None,  # filled from cfg["checkpoint_fracs"] at build time
        "extra": {},
    },
    {
        "key": "dueling",
        "family": "value-based",
        "module": "algorithms.deep.dqn",
        "class": "DQN",
        "budget": "dqn_train_steps",
        "fracs": None,
        "extra": {"dueling": True},
    },
    {
        "key": "double",
        "family": "value-based",
        "module": "algorithms.deep.dqn",
        "class": "DQN",
        "budget": "dqn_train_steps",
        "fracs": None,
        "extra": {"double": True},
    },
    # Policy-gradient family: different coverage footprint from value methods.
    {
        "key": "ppo",
        "family": "policy-gradient",
        "module": "algorithms.approx.ppo",
        "class": "PPO",
        "budget": "pg_train_steps",
        "fracs": None,
        "extra": {},
    },
    {
        "key": "a2c",
        "family": "policy-gradient",
        "module": "algorithms.approx.a2c",
        "class": "A2C",
        "budget": "pg_train_steps",
        "fracs": None,
        "extra": {},
    },
]


def build_roster(cfg):
    rows = []
    for row in ROSTER:
        fracs = row["fracs"] or cfg["checkpoint_fracs"]
        for frac in fracs:
            r = dict(row)
            r["frac"] = frac
            suffix = "rand" if frac is None else f"f{frac:g}".replace(".", "p")
            r["policy_id"] = f"{row['key']}_{suffix}"
            rows.append(r)
    return rows
