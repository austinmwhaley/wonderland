"""The `response` outlet must earn its name (DEC-384 / DEC-426).

Three claims are asserted rather than described: the target really is the discounted future
(the same scan the SF objective trains on), the conditioning really is strictly pre-action
(no outcome leak into its own prediction), and the published uncertainty really covers at the
nominal rate (an interval that does not is a false green with extra characters).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from looking_glass import response_outlet as R  # noqa: E402


def test_discounted_future_is_the_definition_it_claims():
    """The refactor out of `train_step` is only safe if the helper is the double sum it says it
    is. Checked against a literal O(T^3) triple loop, not against itself."""
    from looking_glass.cfm_training import discounted_future

    torch.manual_seed(0)
    B, T, D = 2, 6, 3
    phi = torch.randn(B, T, D)
    decay = 0.4 + 0.5 * torch.rand(B, T, 1)
    got = discounted_future(phi, decay)
    for b in range(B):
        for i in range(T):
            acc = torch.zeros(D)
            for j in range(i + 1, T):
                prod = 1.0
                for k in range(i + 1, j + 1):
                    prod *= float(decay[b, k, 0])
                acc = acc + prod * phi[b, j]
            assert torch.allclose(got[b, i], acc, atol=1e-6), (b, i, got[b, i], acc)


def test_the_last_step_has_no_future():
    """R_T must be exactly zero: a 'future' that includes the present would let the outlet
    predict an outcome from the outcome."""
    from looking_glass.cfm_training import discounted_future

    torch.manual_seed(1)
    phi, decay = torch.randn(2, 5, 4), 0.5 * torch.ones(2, 5, 1)
    assert torch.count_nonzero(discounted_future(phi, decay)[:, -1]) == 0


def test_gamma_is_derived_from_the_stream_cadence_not_typed():
    """A stream whose events are 10× further apart must get a proportionately flatter discount,
    and the derivation must be printed (no reader guesses what future was predicted)."""
    g1, r1 = R.derive_gamma([1.0] * 50)
    g10, r10 = R.derive_gamma([10.0] * 50)
    assert r1["tau_days"] == pytest.approx(1.0) and r10["tau_days"] == pytest.approx(10.0)
    assert g1 == pytest.approx(float(__import__("math").exp(-1.0)))
    assert g10 == pytest.approx(float(__import__("math").exp(-0.1)))
    assert g1 < g10 < 1.0  # a slower stream discounts each step less
    with pytest.raises(ValueError, match="no positive inter-event gaps"):
        R.derive_gamma([0.0, -1.0])


def _synthetic_rows(n=400, dim=6, seed=0):
    """Customers whose future is a known function of BOTH their state and their action, so a
    head that ignores either side must lose the attribution control."""

    import numpy as np

    rng = np.random.default_rng(seed)
    pre = rng.normal(size=(n, dim))
    delta = rng.normal(size=(n, dim))
    target = pre[:, 0] * 2.0 - delta[:, 1] * 1.5 + 0.3 * rng.normal(size=n)
    return {
        "pre": pre,
        "delta": delta,
        "target": target,
        "customer": [f"c{i}" for i in range(n)],
        "randomized": np.ones(n),
    }


def test_the_head_carries_both_sides_of_action_conditioned():
    """State-blind and action-blind controls are fitted on the same rows: each must lose to the
    full head on held-out squared error, or the 'response' is a main effect wearing a name."""
    import numpy as np

    a, b = _synthetic_rows(400, seed=1), _synthetic_rows(200, seed=2)
    head = R.fit(a, b)
    y = np.asarray(head["_pred_holdout"]["truth"])

    def mse(p):
        p = np.asarray(p)
        return float(np.mean((y - p) ** 2))

    assert mse(head["_pred_holdout"]["state_action"]) < mse(head["_pred_holdout"]["action_blind"])
    assert mse(head["_pred_holdout"]["state_action"]) < mse(head["_pred_holdout"]["state_blind"])
    assert head["target_r2_holdout"] > 0.8, head


def test_the_published_interval_covers_at_the_rate_it_states():
    """Uncertainty first (P8): σ is scaled until the empirical nominal coverage is met, and the
    achieved coverage is reported next to the claim — never asserted."""
    head = R.fit(*[_synthetic_rows(n, seed=s) for n, s in ((600, 3), (300, 4))])
    assert head["coverage_empirical"] >= head["coverage_nominal"] - 0.01, head
    # both widths are positive and finite, the wider-than-claiming floor is respected, and the
    # width is receipted so a reader can see the sharpness that calibration cost
    for tag in ("sigma_randomized", "sigma_unrandomized", "mean_interval_width"):
        assert head[tag] > 0 and math.isfinite(head[tag]), (tag, head)
    assert head["sigma_unrandomized"] >= head["sigma_randomized"], head  # DEC-384b


def test_the_action_masks_come_from_the_declared_design():
    """Which events are actions, and which were randomized, is decided by the stream's own
    declaration (`meta.action_keys` / `meta.arm_key`) — not by an event name in source, and not
    by `is_action`, which is empty whenever the run declared nothing (the frozen demo artifact
    trained that way). A row counts as an action iff it carries a declared key; it counts as
    RANDOMIZED only where the key it varied is the declared arm (DEC-384b)."""
    import numpy as np

    design = {
        "action_keys": ("freq", "offer"),
        "arm_key": "channel",
        "source": "truth.json",
    }
    group = [
        {
            "customer": "a",
            "event_type": np.array(["customer_signup", "email_send", "purchase"]),
            "numeric_mask": {
                "freq": np.array([False, True, False]),
                "offer": np.array([False, True, False]),
                "value": np.array([False, False, True]),
            },
            "attrs": {"target_key": np.array([None, "channel", None], dtype=object)},
            "is_action": np.array([False, False, False]),
        },
        {
            "customer": "b",
            "event_type": np.array(["view", "email_send"]),
            "numeric_mask": {"freq": np.array([False, True]), "offer": np.array([False, False])},
            "attrs": {"target_key": np.array([None, "offer"], dtype=object)},
            "is_action": np.array([False, False]),
        },
    ]
    act, rnd = R._action_masks(group, design)
    assert act.tolist() == [[0, 1, 0], [0, 1, 0]]  # purchase's `value` is not an action key
    # only the send that varied the ARM key is causal-grade; the other carried a deterministic
    # key, so its interval must stay wide
    assert rnd.tolist() == [[0, 1, 0], [0, 0, 0]]
    # a run that DID declare actions keeps its own flags (union, never replacement)
    group[0]["is_action"] = np.array([True, False, False])
    act2, _ = R._action_masks(group, design)
    assert act2[0].tolist() == [1, 1, 0]


def test_an_undeclared_stream_refuses_the_outlet_instead_of_guessing(tmp_path):
    """No truth file, no action design: guessing an event name would publish a causal-looking
    column with no randomization behind it (Principle 7 — reject, do not clamp)."""
    import pytest

    with pytest.raises(LookupError, match="no action declaration found"):
        R.declare_action_design(tmp_path / "stream.parquet")


def test_the_measured_interval_covers_at_the_rate_it_states():
    """P8 for this outlet, stated as a property of `_measured_sigma` itself rather than of a
    head that happens to be well-specified: for ANY residuals, the returned half-width covers
    the nominal fraction of them. That is the whole reason the width is measured."""
    import numpy as np

    from looking_glass.response_outlet import _measured_sigma, _z

    rng = np.random.default_rng(5)
    for n, nominal in ((200, 0.8), (4000, 0.9), (500, 0.5)):
        resid = rng.standard_t(2.0, n) * rng.choice([0.01, 1.0, 40.0], n)
        sigma, cover = _measured_sigma(resid, nominal=nominal)
        assert sigma > 0 and np.isfinite(sigma)
        assert cover >= nominal - 0.01, (n, nominal, sigma, cover)
        assert np.mean(np.abs(resid) <= _z(nominal) * sigma) == cover


def test_a_wide_group_is_measured_wider_not_multiplied():
    """DEC-384b as measurement: the over-confident rows (huge residuals) must set the width
    themselves. The fitted heteroscedastic head this replaced did the opposite — it put
    sigma ~ 1e-6 on exactly those rows, so reaching nominal took a 22932x global multiplier
    and the well-predicted customers were published +-3000."""
    import numpy as np

    from looking_glass.response_outlet import _measured_sigma

    rng = np.random.default_rng(11)
    n = 3000
    quiet = np.where(rng.random(n) < 0.5, rng.normal(0, 1e-3, n), 0.0)
    loud = rng.normal(0, 12.0, n)
    s_quiet, c_quiet = _measured_sigma(quiet)
    s_loud, c_loud = _measured_sigma(loud)
    assert s_loud > 100 * s_quiet  # the noisy group is MEASURED wider
    assert c_quiet >= 0.79 and c_loud >= 0.79


def test_an_unmeasurable_interval_is_refused_not_published():
    """Fail safe (P5/P7): a sigma that is not a positive number is not a wide interval, it is
    no interval — and a consumer reading it cannot tell the difference."""
    import numpy as np
    import pytest

    from looking_glass.response_outlet import _measured_sigma

    with pytest.raises(ValueError, match="zero rows"):
        _measured_sigma(np.array([]))
    with pytest.raises(RuntimeError, match="not a positive number"):
        _measured_sigma(np.array([float("nan"), 1.0, 2.0]))
    with pytest.raises(RuntimeError, match="not a positive number"):
        _measured_sigma(np.array([0.0, 0.0]))


def test_a_published_customer_sigma_is_never_zero_or_nan():
    """The per-customer blend (randomized fraction x randomized width + rest x unrandomized
    width) is the last place a σ can degenerate on its way to the table, and the first real
    build published NaN intervals from a floor in the deleted scale head without complaining."""
    import numpy as np
    import pytest

    from looking_glass.response_outlet import publish_rows

    head = {
        "mean_coef": [0.0, 0.0, 0.0],
        "mean_intercept": 0.5,
        "mu": [0.0, 0.0, 0.0],
        "sd": [1.0, 1.0, 1.0],
        "sigma_randomized": 1.0,
        "sigma_unrandomized": 2.0,
    }
    rows = {
        "pre": np.zeros((3, 1)),
        "delta": np.zeros((3, 1)),
        "customer": np.array(["c1", "c1", "c2"], dtype=object),
        "randomized": np.array([1.0, 0.0, 0.0]),
    }
    out = publish_rows(rows, head, version="test")
    by = {(r["customer_key"], r["part"]): r["embedding"][0] for r in out}
    assert by[("c1", "outcome_sigma")] == pytest.approx(1.5)  # half its own actions were logged
    assert by[("c2", "outcome_sigma")] == pytest.approx(2.0)  # never randomized -> the wide one
    assert by[("c1", "randomized")] == pytest.approx(0.5)
    assert by[("c1", "support")] == 2.0
    bad = dict(head, sigma_unrandomized=float("nan"))
    with pytest.raises(RuntimeError, match="not a positive number"):
        publish_rows(rows, bad, version="test")


def test_the_increment_diagnostic_separates_an_action_that_matters_from_one_that_does_not():
    """The receipt field that attributed the real build's state-blind failure must itself be
    able to tell the two cases apart, or it is decoration. Here the target really does depend on
    delta's own direction, so the part of delta the state cannot explain must still predict it."""
    head = R.fit(*[_synthetic_rows(600, seed=3), _synthetic_rows(300, seed=4)])
    assert head["delta_incremental_r2"] > 0.05, head
    # and the degenerate case: a target that is pure state -> no incremental action signal
    rows_a, rows_b = (_synthetic_rows(400, seed=s) for s in (8, 9))
    for rows in (rows_a, rows_b):
        rows["target"] = rows["pre"][:, 0] * 2.0
    flat = R.fit(rows_a, rows_b)
    assert flat["delta_incremental_r2"] < 0.01, flat
    assert flat["delta_effective_rank"] > 1.0  # rank is a real statistic, not a constant
