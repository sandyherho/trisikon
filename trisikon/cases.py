"""The four case studies and the shared configuration.

Each case shares the seed, the ensemble sizes, the breathing period and
the integration step, so every difference in the diagnostics comes from
the two parameters that define a case: the dissipation range of the
breathing schedule and the per-axis dissipation ratio that either keeps
or breaks the cyclic symmetry.

    A  Classical Order   high dissipation, shallow breath   (ordered)
    B  Dialectic Breath  full ordered-to-chaotic breath     (balanced)
    C  Deconstruction    low dissipation, mostly chaotic    (unresolved)
    D  Mode Dominance    breath of B with broken symmetry   (one mode)
"""

from dataclasses import dataclass, field

__all__ = ["Config", "Case", "CASES", "case_list"]


@dataclass(frozen=True)
class Config:
    """Shared integration, ensemble and rendering parameters."""

    seed: int = 7
    dt: float = 0.025
    breath_period_s: float = 12.0
    # Ensemble for the differential-entropy breathing and the animation.
    traced_per_mode: int = 2
    cloud_per_mode: int = 40
    # Long single trajectory for the stationary diagnostics.
    realize_steps: int = 200000
    realize_dt: float = 0.02
    # Animation.
    anim_duration_s: float = 15.0
    anim_fps: int = 24
    phosphor_len: int = 430
    active_len: int = 120
    # Permutation complexity embedding.  The delay sets the embedding
    # window to about one characteristic oscillation period (2 pi) at
    # the realized sampling step, so the ordinal patterns resolve the
    # dynamics rather than the smooth local trend of an oversampled flow.
    embed_dx: int = 5
    embed_tau: int = 79


@dataclass(frozen=True)
class Case:
    """A single case: its label and the two parameters that define it."""

    key: str
    label: str
    b_harmony: float
    b_chaos: float
    damping_ratio: tuple = (1.0, 1.0, 1.0)

    @property
    def is_symmetric(self):
        """Return True when the three dissipations scale equally."""
        r = self.damping_ratio
        return r[0] == r[1] == r[2]


CASES = {
    "A": Case("A", "Classical Order", 0.400, 0.310),
    "B": Case("B", "Dialectic Breath", 0.315, 0.160),
    "C": Case("C", "Deconstruction", 0.220, 0.115),
    "D": Case("D", "Mode Dominance", 0.315, 0.160,
              damping_ratio=(0.70, 1.18, 1.18)),
}


def case_list():
    """Return the cases in canonical order A, B, C, D."""
    return [CASES[k] for k in ("A", "B", "C", "D")]
