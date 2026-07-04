"""Regression tests for the three-tier epistemic architecture — Paper 4."""

import math

import pytest

from apgi.extensions.epistemic import (
    ACCEPTANCE_THRESHOLD,
    CRITERIA,
    FOUNDATIONAL_CRITERIA,
    double_bridge_energy_estimate,
    evaluate_theory,
    inefficiency_ratio,
    landauer_minimum_energy,
)


class TestLandauerMinimumEnergy:
    def test_matches_paper_order_of_magnitude(self):
        # ~3e-21 J per bit at 310K (Paper 4 §3.1)
        result = landauer_minimum_energy(n_bits=1, temperature_k=310.0)
        assert result == pytest.approx(3e-21, rel=0.1)

    def test_scales_linearly_with_bits(self):
        one_bit = landauer_minimum_energy(n_bits=1)
        ten_bits = landauer_minimum_energy(n_bits=10)
        assert ten_bits == pytest.approx(one_bit * 10, rel=1e-9)


class TestInefficiencyRatio:
    def test_bandwidth_derived_ratio_matches_paper_order_of_magnitude(self):
        # ~20 bits/event, ~5.9e-20 J minimum, ~1e-1 J actual -> ratio ~1.7e18 (§3.2.1)
        ratio = inefficiency_ratio(actual_energy_j=1e-1, n_bits=20)
        assert ratio == pytest.approx(1.7e18, rel=0.2)

    def test_raises_for_zero_bits(self):
        with pytest.raises(ValueError):
            inefficiency_ratio(actual_energy_j=1.0, n_bits=0)


class TestDoubleBridgeEnergyEstimate:
    def test_scales_by_overhead_factor(self):
        base = landauer_minimum_energy(n_bits=1e6)
        result = double_bridge_energy_estimate(n_bits=1e6, synaptic_overhead_factor=1e13)
        assert result == pytest.approx(base * 1e13, rel=1e-9)


class TestEvaluateTheory:
    def _all_scores(self, value: int) -> dict:
        return {c: value for c in CRITERIA}

    def test_all_twos_gives_composite_100_and_acceptance(self):
        result = evaluate_theory(self._all_scores(2))
        assert result["composite"] == pytest.approx(100.0)
        assert result["verdict"] == "provisional_acceptance"
        assert result["foundational_gate_triggered"] is False

    def test_all_zeros_gives_composite_0_and_rejection(self):
        result = evaluate_theory(self._all_scores(0))
        assert result["composite"] == pytest.approx(0.0)
        assert result["verdict"] == "provisional_rejection"
        assert result["foundational_gate_triggered"] is True

    def test_foundational_gate_overrides_high_composite(self):
        # C1 = 0 but everything else maxed: composite would clear 55 on its
        # own (6*100/7 ≈ 85.7), but the foundational gate must still reject.
        scores = self._all_scores(2)
        scores[FOUNDATIONAL_CRITERIA[0]] = 0
        result = evaluate_theory(scores)
        assert result["composite"] > ACCEPTANCE_THRESHOLD
        assert result["foundational_gate_triggered"] is True
        assert result["verdict"] == "provisional_rejection"

    def test_missing_criterion_raises(self):
        scores = self._all_scores(1)
        del scores[CRITERIA[0]]
        with pytest.raises(ValueError):
            evaluate_theory(scores)

    def test_invalid_raw_score_raises(self):
        scores = self._all_scores(1)
        scores[CRITERIA[0]] = 3
        with pytest.raises(ValueError):
            evaluate_theory(scores)

    def test_below_threshold_without_gate_is_rejection(self):
        # Raw score 1 everywhere -> scaled 50 everywhere -> composite 50 < 55
        result = evaluate_theory(self._all_scores(1))
        assert result["composite"] == pytest.approx(50.0)
        assert result["foundational_gate_triggered"] is False
        assert result["verdict"] == "provisional_rejection"
