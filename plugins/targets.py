"""Supervised plugin TARGETS — the only thing a supervised plugin owns.

A Target declares WHAT to predict and the contract for it:
  * kind        binary | continuous  (drives label SQL, head family, metrics, gates)
  * window_days label horizon in days
  * spec_target the PluginSpec.target string recorded in the artifact
  * primary_head the head persisted for inference ("" -> kind default)

Everything else (dataset assembly, split, fitting, gating, persistence) is the
template's job (`plugins/head_template.py`). Adding a supervised plugin = add a
Target here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import PluginSpec


@dataclass(frozen=True)
class Target:
    name: str  # plugin / spec name (owns the artifact tag)
    kind: str  # "binary" | "continuous"
    window_days: int
    spec_target: str
    version: str = "v1.0.0"
    revision: int = 1
    primary_head: str = ""  # "" -> HeadTemplate default for the kind
    # binary model family to use. "" = run the bake-off (logistic/MLP/HGB) and
    # pick the winner by held-out evidence each training run; set a family
    # (e.g. "mlp") to pin the production model between bake-offs.
    family: str = ""
    # plugin-training feature source in the products db. Only a table that is a
    # PURE FUNCTION OF THE STATE (donor_embeddings = donor(h)) can also be
    # reproduced at inference from a live advanced state.
    feature_table: str = "anchor_embeddings"

    @property
    def tag(self) -> str:
        return f"{self.name}_{self.version}r{self.revision}"

    def spec(self) -> PluginSpec:
        return PluginSpec(
            name=self.name,
            kind="supervised",
            target=self.spec_target,
            window_days=self.window_days,
            version=self.version,
            revision=self.revision,
        )


def clv_target(window_days: int) -> Target:
    """The original supervised CLV plugin target (continuous gross margin)."""
    return Target(
        name=f"clv_supervised_{window_days}d",
        kind="continuous",
        window_days=window_days,
        spec_target="gross_margin",
        primary_head="two_part",
    )


# ---------------------------------------------------------------------------
# the production target: 30-day purchase propensity
# ---------------------------------------------------------------------------
PURCHASE_PROPENSITY_30D = Target(
    name="supervised_purchase_propensity_30d",
    kind="binary",
    window_days=30,
    spec_target="purchase_in_30d",
    primary_head="",  # auto: the bake-off winner picked by HeadTemplate.fit
    feature_table="donor_embeddings",
)

REGISTRY = {PURCHASE_PROPENSITY_30D.name: PURCHASE_PROPENSITY_30D}
