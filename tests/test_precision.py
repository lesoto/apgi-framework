"""Regression tests for the online precision/z-score pipeline — Math Spec §1-2."""

import pytest

from apgi.core import PI_MAX, PI_MIN
from apgi.precision import (
    OnlinePrecisionChannel,
    apply_neuromodulatory_gain,
    precision_from_variance,
    update_running_mean,
    update_running_variance,
    z_score,
)


class TestUpdateRunningMean:
    def test_hand_computed(self):
        result = update_running_mean(mu=0.0, eps_t=2.0, alpha_ema=0.1)
        assert result == pytest.approx(0.2)

    def test_converges_to_constant_input(self):
        mu = 0.0
        for _ in range(500):
            mu = update_running_mean(mu, eps_t=3.0, alpha_ema=0.1)
        assert mu == pytest.approx(3.0, abs=1e-3)


class TestUpdateRunningVariance:
    def test_two_step_form_uses_pre_update_mean(self):
        # sigma2(t+1) = 0.9*1.0 + 0.1*(eps - mu_t)^2, using mu_t (not mu_t+1)
        result = update_running_variance(sigma2=1.0, eps_t=2.0, mu_t=0.0, alpha_ema=0.1)
        assert result == pytest.approx(0.9 * 1.0 + 0.1 * (2.0 - 0.0) ** 2)

    def test_zero_variance_input_converges_toward_zero(self):
        mu, sigma2 = 5.0, 1.0
        for _ in range(500):
            sigma2 = update_running_variance(sigma2, eps_t=5.0, mu_t=mu, alpha_ema=0.1)
            mu = update_running_mean(mu, eps_t=5.0, alpha_ema=0.1)
        assert sigma2 == pytest.approx(0.0, abs=1e-3)


class TestPrecisionFromVariance:
    def test_hand_computed(self):
        result = precision_from_variance(sigma2=0.99, eps_stab=0.01)
        assert result == pytest.approx(1.0)

    def test_clamped_to_pi_max_for_near_zero_variance(self):
        result = precision_from_variance(sigma2=0.0, eps_stab=1e-6)
        assert result == pytest.approx(PI_MAX)

    def test_clamped_to_pi_min_for_huge_variance(self):
        result = precision_from_variance(sigma2=1e6)
        assert result == pytest.approx(PI_MIN)


class TestZScore:
    def test_hand_computed(self):
        result = z_score(eps_t=2.0, mu_t=1.0, sigma2_t=4.0, eps_stab=0.0)
        assert result == pytest.approx(0.5)

    def test_zero_at_mean(self):
        assert z_score(eps_t=1.0, mu_t=1.0, sigma2_t=1.0) == pytest.approx(0.0)


class TestApplyNeuromodulatoryGain:
    def test_hand_computed(self):
        assert apply_neuromodulatory_gain(pi=2.0, gain=1.5) == pytest.approx(3.0)


class TestOnlinePrecisionChannel:
    def test_initial_state_matches_spec(self):
        ch = OnlinePrecisionChannel()
        assert ch.mu == 0.0
        assert ch.sigma2 == 1.0

    def test_update_returns_expected_keys(self):
        ch = OnlinePrecisionChannel()
        result = ch.update(eps_t=1.0)
        assert set(result.keys()) == {"mu", "sigma2", "pi", "z"}

    def test_state_persists_across_updates(self):
        ch = OnlinePrecisionChannel(alpha_ema=0.2)
        ch.update(1.0)
        mu_after_one = ch.mu
        ch.update(1.0)
        assert ch.mu != mu_after_one  # state evolves, not reset each call

    def test_precision_stays_within_clamp_bounds(self):
        ch = OnlinePrecisionChannel()
        for eps in [0.0, 100.0, -100.0, 0.001, 50.0]:
            result = ch.update(eps)
            assert PI_MIN <= result["pi"] <= PI_MAX
