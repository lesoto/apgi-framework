"""Regression tests for APGI core equations."""

import math

import pytest

from apgi.core import (
    compute_pi_i_eff,
    compute_S_t,
    compute_theta_t,
    ignition_criterion,
    run_trial,
    update_theta,
)


class TestComputePiIEff:
    def test_zero_metabolic_cost_returns_pi_i(self):
        assert compute_pi_i_eff(1.5, 0.0) == pytest.approx(1.5)

    def test_hand_computed_value(self):
        result = compute_pi_i_eff(1.0, 50.0, kappa=100.0)
        assert result == pytest.approx(math.exp(-0.5), rel=1e-6)

    def test_large_metabolic_cost_approaches_zero(self):
        result = compute_pi_i_eff(2.0, 10_000.0, kappa=100.0)
        assert result < 1e-10

    def test_negative_pi_i_raises(self):
        with pytest.raises(ValueError, match="pi_i"):
            compute_pi_i_eff(-1.0, 0.0)

    def test_negative_metabolic_cost_raises(self):
        with pytest.raises(ValueError, match="C_metabolic"):
            compute_pi_i_eff(1.0, -0.1)


class TestComputeSt:
    def test_hand_computed_value(self):
        result = compute_S_t(pi_e=1.2, z_e=0.8, pi_i_eff=0.9, z_i=0.5)
        assert result == pytest.approx(1.41, rel=1e-9)

    def test_absolute_value_of_negative_z(self):
        positive = compute_S_t(1.0, 0.5, 1.0, 0.5)
        negative = compute_S_t(1.0, -0.5, 1.0, -0.5)
        assert positive == pytest.approx(negative)

    def test_zero_inputs_return_zero(self):
        assert compute_S_t(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_excitatory_only(self):
        result = compute_S_t(pi_e=2.0, z_e=3.0, pi_i_eff=0.0, z_i=99.0)
        assert result == pytest.approx(6.0)

    def test_inhibitory_only(self):
        result = compute_S_t(pi_e=0.0, z_e=99.0, pi_i_eff=1.5, z_i=4.0)
        assert result == pytest.approx(6.0)


class TestComputeThetaT:
    def test_hand_computed_value(self):
        result = compute_theta_t(
            C_metabolic=1.0, V_information=0.5, alpha=0.3, beta=0.7
        )
        assert result == pytest.approx(0.65, rel=1e-9)

    def test_zero_coefficients(self):
        assert compute_theta_t(10.0, 10.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_linearity_in_alpha(self):
        t1 = compute_theta_t(1.0, 0.0, 0.3, 0.0)
        t2 = compute_theta_t(1.0, 0.0, 0.6, 0.0)
        assert t2 == pytest.approx(2.0 * t1)


class TestUpdateTheta:
    def test_hand_computed_update(self):
        result = update_theta(
            theta_t=1.0,
            C_metabolic=1.0,
            V_information=0.5,
            alpha=0.3,
            beta=0.7,
            gamma=0.9,
        )
        assert result == pytest.approx(0.965, rel=1e-9)

    def test_gamma_zero_jumps_to_target(self):
        target = compute_theta_t(1.0, 0.5, 0.3, 0.7)
        result = update_theta(999.0, 1.0, 0.5, 0.3, 0.7, gamma=0.0)
        assert result == pytest.approx(target)

    def test_gamma_one_stays_at_current(self):
        result = update_theta(42.0, 1.0, 0.5, 0.3, 0.7, gamma=1.0)
        assert result == pytest.approx(42.0)

    def test_converges_after_many_steps(self):
        theta = 10.0
        target = compute_theta_t(1.0, 0.5, 0.3, 0.7)
        for _ in range(500):
            theta = update_theta(theta, 1.0, 0.5, 0.3, 0.7, gamma=0.9)
        assert abs(theta - target) < 1e-4


class TestIgnitionCriterion:
    EPS = 1e-9

    def test_below_threshold_no_ignition(self):
        assert ignition_criterion(S_t=0.9, theta_t=1.0) is False

    def test_above_threshold_ignition(self):
        assert ignition_criterion(S_t=1.1, theta_t=1.0) is True

    def test_exactly_at_threshold_fires(self):
        assert ignition_criterion(S_t=1.0, theta_t=1.0) is True

    def test_just_below_threshold(self):
        assert ignition_criterion(1.0 - self.EPS, 1.0) is False

    def test_just_above_threshold(self):
        assert ignition_criterion(1.0 + self.EPS, 1.0) is True


class TestRunTrial:
    def test_returns_all_keys(self):
        result = run_trial(
            pi_e=1.2,
            z_e=0.8,
            pi_i=1.0,
            z_i=0.5,
            C_metabolic=50.0,
            V_information=0.5,
            alpha=0.3,
            beta=0.7,
            kappa=100.0,
        )
        assert set(result.keys()) == {"pi_i_eff", "S_t", "theta_t", "ignition"}

    def test_pi_i_eff_consistent(self):
        result = run_trial(
            pi_e=1.0,
            z_e=1.0,
            pi_i=1.0,
            z_i=0.0,
            C_metabolic=0.0,
            V_information=0.0,
            alpha=0.0,
            beta=0.0,
        )
        assert result["pi_i_eff"] == pytest.approx(1.0)

    def test_s_t_consistent(self):
        result = run_trial(
            pi_e=1.2,
            z_e=0.8,
            pi_i=0.9,
            z_i=0.5,
            C_metabolic=0.0,
            V_information=0.5,
            alpha=0.3,
            beta=0.7,
        )
        expected_S = 1.2 * 0.8 + 0.9 * 0.5
        assert result["S_t"] == pytest.approx(expected_S, rel=1e-6)
