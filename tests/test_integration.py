"""Tests for APGICoreIntegration and somatic modulation."""

import numpy as np
import pytest

from apgi.integration import APGICoreIntegration


class TestAPGICoreIntegrationStep:
    def test_step_returns_trial_record(self):
        integ = APGICoreIntegration()
        rec = integ.step(1.2, 0.8, 1.0, 0.5, 1.0, 0.5)
        assert rec.S_t > 0
        assert rec.theta_t > 0
        assert isinstance(rec.ignition, bool)

    def test_trial_index_increments(self):
        integ = APGICoreIntegration()
        r0 = integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        r1 = integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        assert r0.t == 0
        assert r1.t == 1

    def test_theta_adapts_across_trials(self):
        integ = APGICoreIntegration(alpha=0.3, beta=0.7, gamma=0.9)
        thetas = [integ.step(1.0, 1.0, 1.0, 0.5, float(c), 0.5).theta_t for c in np.linspace(0.5, 2.0, 20)]
        # Threshold should vary as metabolic cost changes
        assert not all(t == thetas[0] for t in thetas)

    def test_records_accumulate(self):
        integ = APGICoreIntegration()
        for _ in range(10):
            integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        assert len(integ.records) == 10

    def test_reset_clears_records(self):
        integ = APGICoreIntegration()
        integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        integ.reset()
        assert len(integ.records) == 0

    def test_ignition_rate_empty_is_nan(self):
        integ = APGICoreIntegration()
        assert np.isnan(integ.ignition_rate())

    def test_mean_S_t_empty_is_nan(self):
        integ = APGICoreIntegration()
        assert np.isnan(integ.mean_S_t())

    def test_mean_theta_empty_is_nan(self):
        integ = APGICoreIntegration()
        assert np.isnan(integ.mean_theta())

    def test_mean_metrics_non_empty(self):
        integ = APGICoreIntegration()
        integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        integ.step(1.0, 1.0, 1.0, 0.5, 1.0, 0.5)
        assert integ.mean_S_t() > 0
        assert integ.mean_theta() > 0




class TestAPGICoreIntegrationRunSequence:
    def _make_arrays(self, n: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        return {
            "pi_e": rng.uniform(0.8, 1.5, n),
            "z_e": rng.uniform(0.2, 1.0, n),
            "pi_i": rng.uniform(0.5, 1.5, n),
            "z_i": rng.uniform(0.1, 0.8, n),
            "C_metabolic": rng.uniform(0.5, 2.0, n),
            "V_information": rng.uniform(0.1, 1.0, n),
        }

    def test_run_sequence_length(self):
        integ = APGICoreIntegration()
        arrs = self._make_arrays(30)
        records = integ.run_sequence(**arrs)
        assert len(records) == 30

    def test_ignition_rate_in_unit_interval(self):
        integ = APGICoreIntegration()
        integ.run_sequence(**self._make_arrays(100))
        rate = integ.ignition_rate()
        assert 0.0 <= rate <= 1.0

    def test_mean_S_t_positive(self):
        integ = APGICoreIntegration()
        integ.run_sequence(**self._make_arrays(50))
        assert integ.mean_S_t() > 0

    def test_mismatched_lengths_raise(self):
        integ = APGICoreIntegration()
        arrs = self._make_arrays(20)
        arrs["pi_e"] = arrs["pi_e"][:10]
        with pytest.raises(ValueError, match="same length"):
            integ.run_sequence(**arrs)


class TestSomaticModulation:
    """Verify that elevated C_metabolic suppresses Πⁱ_eff and reduces Sₜ."""

    def test_high_metabolic_cost_reduces_pi_i_eff(self):
        integ_low = APGICoreIntegration(alpha=0.3, beta=0.7)
        integ_high = APGICoreIntegration(alpha=0.3, beta=0.7)

        r_low = integ_low.step(1.0, 1.0, 1.0, 0.5, C_metabolic=0.5, V_information=0.5)
        r_high = integ_high.step(1.0, 1.0, 1.0, 0.5, C_metabolic=200.0, V_information=0.5)

        # Higher metabolic cost → lower Πⁱ_eff → lower Sₜ
        assert r_high.pi_i_eff < r_low.pi_i_eff
        assert r_high.S_t < r_low.S_t
