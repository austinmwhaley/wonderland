"""The judge's leg producers (DEC-434): the statistics must measure what they claim to.

Two of these tests exist because the FIRST run of a leg produced a number that was arithmetically
fine and informationally worthless — a distance task scored -649 because it compared two distance
vectors in different UNITS, and a negative control scored -4.6e15 because it was drawn at one
global sigma across a matrix mixing raw counts with fractions. A statistic whose value is
dominated by units or by a scale mismatch cannot support a verdict, and the card would have
published it anyway. These are the falsifiable forms of "the number means what the label says".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_battery_law as B  # noqa: E402
import measure_judge_law as ML  # noqa: E402


# ---------------------------------------------------------------- the distance statistic


def test_distance_agreement_is_independent_of_the_arms_units():
    """A representation rescaled by a million must score the SAME, not collapse toward zero.

    This is the whole reason the leg scores squared correlation instead of MSE-R2 of raw
    distances: the donor, the wide table and the lab's truth live on different scales, and a loss
    that cares about scale measures the scale.
    """
    rng = np.random.default_rng(7)
    truth = rng.normal(size=400)
    arm = 2.0 + 0.5 * truth + 0.05 * rng.normal(size=400)
    plain = ML._alignment(arm, truth)
    rescaled = ML._alignment(arm * 1e6 + 3.0, truth * 1e-4 - 12.0)
    assert plain == pytest.approx(rescaled)
    assert 0.9 < plain <= 1.0


def test_a_representation_that_knows_nothing_scores_about_zero_not_minus_millions():
    """The correct shape of a negative control: near zero, and readable as a control."""
    rng = np.random.default_rng(11)
    truth = rng.normal(size=600)
    noise = rng.normal(size=600)
    assert abs(ML._alignment(noise, truth)) < 0.01


def test_a_constant_geometry_is_no_number_rather_than_a_zero():
    """Degenerate is not the same as bad: a matrix with no spread cannot be scored."""
    assert ML._alignment(np.zeros(50), np.arange(50, dtype=float)) is None
    assert ML._round(None) is None


# ---------------------------------------------------------------- the control generator


def test_controls_keep_each_column_own_scale_not_one_global_sigma():
    """The defect this generator exists to fix, kept fixed.

    The view-law `_controls` draws gaussian noise at a single scalar sigma for the whole matrix.
    On the `brain + ledger` column the measure view puts raw event counts beside fractions, so one
    sigma is right for neither: the control's predictions exploded and it scored R2 = -4.6e15.
    DEC-433 lets an unreadable control veto a PASS, so an artifact of that kind has the power to
    turn a green row red for a reason that has nothing to do with the model.
    """
    rng = np.random.default_rng(3)
    fit = np.column_stack(
        [
            rng.normal(loc=1000.0, scale=250.0, size=400),  # a count-like column
            rng.normal(loc=0.02, scale=0.003, size=400),  # a fraction-like column
        ]
    )
    score = rng.normal(size=(400, 2))
    rand, destroyed = B.controls_matched(rng, fit, score)
    for j in range(2):
        assert rand[:, j].std() == pytest.approx(fit[:, j].std(), rel=0.15)
        assert rand[:, j].mean() == pytest.approx(fit[:, j].mean(), abs=0.15 * fit[:, j].std())
        # destroyed keeps the marginals it was built from
        assert np.array_equal(np.sort(destroyed[:, j]), np.sort(score[:, j]))


def test_a_control_never_keeps_a_customer_intact():
    """`destroyed` must break the CUSTOMER, not merely reorder them.

    Each column is permuted independently, so no feature in a control row still belongs to the
    customer whose cell it is scored on — the marginals survive and the identity does not. A
    permutation that moved whole rows instead would leave every within-row relationship intact and
    the control would quietly measure the thing it exists to disprove.
    """
    rng = np.random.default_rng(5)
    width = 24
    fit = rng.normal(size=(200, width))
    _rand, destroyed = B.controls_matched(rng, fit, fit)
    intact = int(np.all(np.isclose(destroyed, fit), axis=1).sum())
    assert intact == 0  # (1/200)^24 per row: not one customer survives the destruction
    for j in range(width):
        assert np.array_equal(np.sort(destroyed[:, j]), np.sort(fit[:, j]))


# ---------------------------------------------------------------- uplift identifiability


def _cells(rows):
    return pl.DataFrame(
        rows,
        orient="row",
        schema={
            "customer_key": pl.Utf8,
            "off": pl.Int64,
            "opt": pl.Int64,
            "days": pl.Int64,
            "buys": pl.Int64,
        },
    )


def test_an_under_sampled_contrast_is_unidentified_not_noisy():
    """A customer with a few arm-days on the comparison side gets no uplift at all.

    Reporting those days as a noisy estimate would let a handful of observations carry a causal
    claim; the floor is the difference between "we measured it" and "we are guessing".
    """
    floor = B.MIN_ARM_DAYS
    cells = _cells(
        [
            ("ok", 2, 2, 12, 6),  # identified on both sides of the contrast
            ("ok", 0, 2, 12, 1),
            ("thin", 2, 2, 12, 6),  # short on the comparison side only
            ("thin", 0, 2, 3, 0),
        ]
    )
    up = B.identified_uplift(cells)
    assert set(up) == {"ok"}
    assert up["ok"] == pytest.approx(6 / 12 - 1 / 12)


def test_dose_response_reports_levels_and_omits_the_unidentified_ones():
    """An absent level is unidentified, which is not the same as a response rate of zero."""
    floor = B.MIN_ARM_DAYS
    cells = _cells(
        [
            ("a", 0, 1, floor, 3),
            ("a", 1, 1, floor, 8),
            ("a", 2, 1, 4, 1),  # below the floor -> must be absent, not 0.0
        ]
    )
    dr = B.identified_dose_response(cells)
    assert sorted(dr["a"]) == [0, 1]
    assert dr["a"][1][0] == pytest.approx(8 / floor)
    assert dr["a"][1][1] == floor


# ---------------------------------------------------------------- shared geometry


def test_every_leg_scores_the_same_population_it_claims_to():
    """The receipt's N is the arm intersection, in split order — not a zip that can slip a row.

    A customer missing from one arm used to be able to shift a matrix by one row and show up as a
    model difference; the intersection is the only honest way to pair the arms.
    """
    arms = {
        "akeys": ["c1", "c2", "c3"],
        "bkeys": ["c4", "c5", "c6"],
        "columns": {
            "brain": {
                "A": {"c1": np.zeros(2), "c2": np.zeros(2)},  # c3 is not served
                "B": {"c4": np.zeros(2), "c5": np.zeros(2), "c6": np.zeros(2)},
            }
        },
        "table": {"c1": np.zeros(2), "c2": np.zeros(2), "c4": np.zeros(2), "c6": np.zeros(2)},
    }
    ak, bk = ML._common_populations(arms, "brain", set())
    assert ak == ["c1", "c2"]
    assert bk == ["c4", "c6"]


def test_a_ragged_served_column_is_a_problem_not_a_measurement():
    """A reader that silently pads what the producer published is reading something else."""
    problems: list = []
    src = {"c1": np.zeros(4), "c2": np.ones(3), "c3": None}
    out = ML._rows_by_key(src, ["c1", "c2", "c3"], "state_embeddings(B)", problems)
    assert set(out) == {"c1", "c2"}
    assert any("is not served" in p for p in problems)
    assert any("ragged" in p for p in problems)
