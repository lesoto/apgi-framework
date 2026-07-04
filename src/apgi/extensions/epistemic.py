"""Three-tier epistemic architecture — Paper 4.

Implements the paper's two operational deliverables:

1.  The Tier 2→1 Landauer bridge (§3.2.1): the thermodynamic minimum cost
    of erasing information, E ≥ kT·ln2 per bit, and the resulting
    event-level inefficiency ratio against measured neural energy cost.
2.  The seven-criterion compact theory-evaluation scoring rubric (§5,
    Table 4): C1 Tier Transparency, C2 Bridge Principles, C3 Quantitative
    Benchmarks, C4 Falsification Conditions, C5 Alternative Comparison,
    C6 Evolutionary Plausibility, C7 Causal Roadmap. Each criterion is
    scored on a raw 0/1/2 scale and mapped to 0-100 by *50; the composite
    is the equal-weighted mean. A foundational gate forces provisional
    rejection if any of C1-C4 scores 0, regardless of the aggregate.

Not implemented (explicitly out of scope for this module): the T3→T2
approximate-inference bridge and the T3→T1 double bridge are qualitative,
unconverged/untraced per the paper's own text (§3.2.2-3.2.3) and have no
closed-form calculation to encode; the twenty-standard supplementary
rubric (Table S1) is a superset of the seven-criterion instrument and is
not separately implemented here. The Φ-boundary (phenomenal claims) is
explicitly out of scope by design (§3.1) and is not modelled.
"""

from __future__ import annotations

import math

BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K
BODY_TEMPERATURE_K: float = 310.0  # K, canonical APGI reference temperature

# §3.2.1 — per-bit metabolic conversion cost (unit cost, not the event-level
# inefficiency ratio); status: unmeasured per the paper's own text.
KAPPA_ATP_PER_BIT_DEFAULT: float = 100.0

CRITERIA: tuple[str, ...] = (
    "tier_transparency",  # C1
    "bridge_principles",  # C2
    "quantitative_benchmarks",  # C3
    "falsification_conditions",  # C4
    "alternative_comparison",  # C5
    "evolutionary_plausibility",  # C6
    "causal_roadmap",  # C7
)

# Standards 1-4 are foundational (§Table 4): a 0 on any of these forces
# provisional rejection regardless of the aggregate composite.
FOUNDATIONAL_CRITERIA: tuple[str, ...] = CRITERIA[:4]

ACCEPTANCE_THRESHOLD: float = 55.0  # composite ≥ 55/100 for provisional acceptance


def landauer_minimum_energy(
    n_bits: float,
    temperature_k: float = BODY_TEMPERATURE_K,
) -> float:
    """Thermodynamic minimum energy to erase n_bits, per Landauer's principle.

    E = n_bits · k·T·ln(2)

    Math Spec / Paper 4 §3.2.1. At 310K, kT·ln2 ≈ 3e-21 J per bit.
    """
    return float(n_bits * BOLTZMANN_CONSTANT * temperature_k * math.log(2))


def inefficiency_ratio(
    actual_energy_j: float,
    n_bits: float,
    temperature_k: float = BODY_TEMPERATURE_K,
) -> float:
    """Ratio of actual measured energy to the Landauer minimum for n_bits.

    Paper 4 §3.2.1: the primary Tier-1 falsification target is the
    bandwidth-derived ratio (~1.7e18 for ~20 bits/event against ~1e-1 J).
    This is a whole-event inefficiency ratio, distinct from kappa (the
    per-bit unit cost).
    """
    minimum = landauer_minimum_energy(n_bits, temperature_k)
    if minimum <= 0:
        raise ValueError("landauer minimum must be positive; check n_bits/temperature")
    return float(actual_energy_j / minimum)


def double_bridge_energy_estimate(
    n_bits: float,
    synaptic_overhead_factor: float,
    temperature_k: float = BODY_TEMPERATURE_K,
) -> float:
    """Four-step Tier 3->Tier 1 double-bridge energy estimate (§3.2.3).

    (1) n_bits processed, (2) x Landauer minimum, (3) x synaptic
    neural-inefficiency factor (~1e13-1e14, Attwell & Laughlin 2001). Step
    (4) — comparison against PET/fMRI-derived expenditure — is left to the
    caller, since it requires an external empirical measurement.
    """
    return float(landauer_minimum_energy(n_bits, temperature_k) * synaptic_overhead_factor)


def _raw_to_scaled(raw_score: int) -> float:
    """Map a raw 0/1/2 criterion score to the canonical 0-100 scale (x50)."""
    if raw_score not in (0, 1, 2):
        raise ValueError(f"raw_score must be 0, 1, or 2; got {raw_score!r}")
    return float(raw_score * 50)


def evaluate_theory(raw_scores: dict[str, int]) -> dict:
    """Score a theory against the seven-criterion rubric (Paper 4 §5, Table 4).

    Args:
        raw_scores: Mapping from criterion name (see :data:`CRITERIA`) to a
            raw score in {0, 1, 2}. All seven criteria must be present.

    Returns:
        dict with keys:

        - ``scaled_scores`` (dict[str, float]): each criterion mapped to 0-100.
        - ``composite`` (float): equal-weighted mean of the scaled scores.
        - ``foundational_gate_triggered`` (bool): True if any of C1-C4 scored 0.
        - ``verdict`` (str): "provisional_acceptance" or "provisional_rejection".
    """
    missing = set(CRITERIA) - set(raw_scores)
    if missing:
        raise ValueError(f"missing scores for criteria: {sorted(missing)}")

    scaled_scores = {c: _raw_to_scaled(raw_scores[c]) for c in CRITERIA}
    composite = sum(scaled_scores.values()) / len(CRITERIA)

    foundational_gate_triggered = any(
        raw_scores[c] == 0 for c in FOUNDATIONAL_CRITERIA
    )
    if foundational_gate_triggered or composite < ACCEPTANCE_THRESHOLD:
        verdict = "provisional_rejection"
    else:
        verdict = "provisional_acceptance"

    return {
        "scaled_scores": scaled_scores,
        "composite": composite,
        "foundational_gate_triggered": foundational_gate_triggered,
        "verdict": verdict,
    }
