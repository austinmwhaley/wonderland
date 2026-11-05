"""red_king's per-customer effect estimator — re-export of the neutral panel.

The implementation moved to `red_queen.offer_panel` so red_queen's decision path
can use the same randomized-offer estimator without importing red_king (the project
rule: red_king is analyst-tool only; no red_queen module may import it).
"""

from red_queen.offer_panel import fit, known_effect, panel, policy

__all__ = ["fit", "known_effect", "panel", "policy"]
