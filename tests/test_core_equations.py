"""Regression tests for APGI core equations — paper §4.1, Table 2."""

import math

import numpy as np
import pytest

from apgi.core import (
    DELTA_INFO_DEFAULT,
    KAPPA_META_DEFAULT,
    LAMBDA_THETA_DEFAULT,
    TAU_S_DEFAULT,
    THETA_0_DEFAULT,
    accumulate_S_t,
    compute_pi_i_eff,
    compute_S_t,
    ignition_criterion,
    ignition_probability,
    run_trial,
    step_theta,
    theta_equilibrium,
)

# ---------------------------------------------------------------------------
# compute_pi_i_eff  —  §4.2: Πⁱ_eff = Πⁱ_baseline · exp(β_SM · M̂)
# ---------------------------------------------------------------------------


class TestComputePiIEff:
    def test_zero_M_hat_returns_baseline(self):
        assert compute_pi_i_eff(1.5, beta_sm=0.6, M_hat=0.0) == pytest.approx(1.5)

    def test_positive_M_hat_increases_precision(self):
        result = compute_pi_i_eff(1.0, beta_sm=1.0, M_hat=1.0)
        assert result == pytest.approx(math.e, rel=1e-6)

    def test_negative_M_hat_decreases_precision(self):
        result = compute_pi_i_eff(1.0, beta_sm=1.0, M_hat=-1.0)
        assert result == pytest.approx(1.0 / math.e, rel=1e-6)

    def test_zero_beta_sm_returns_baseline(self):
        assert compute_pi_i_eff(2.0, beta_sm=0.0, M_hat=5.0) == pytest.approx(2.0)

    def test_negative_baseline_raises(self):
        with pytest.raises(ValueError, match="pi_i_baseline"):
            compute_pi_i_eff(-1.0, beta_sm=0.5, M_hat=0.0)

    def test_hand_computed_value(self):
        result = compute_pi_i_eff(1.0, beta_sm=0.6, M_hat=0.5)
        assert result == pytest.approx(math.exp(0.3), rel=1e-6)


# ---------------------------------------------------------------------------
# compute_S_t  —  Eq. 1: Sₜ = Πᵉ·|zᵉ| + Πⁱ_eff·|zⁱ|
# ---------------------------------------------------------------------------


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
        assert compute_S_t(pi_e=2.0, z_e=3.0, pi_i_eff=0.0, z_i=99.0) == pytest.approx(
            6.0
        )

    def test_inhibitory_only(self):
        assert compute_S_t(pi_e=0.0, z_e=99.0, pi_i_eff=1.5, z_i=4.0) == pytest.approx(
            6.0
        )


# ---------------------------------------------------------------------------
# accumulate_S_t  —  App. A.1 leaky integration
# ---------------------------------------------------------------------------


class TestAccumulateSt:
    def test_zero_prev_returns_fraction_of_input(self):
        tau_S = TAU_S_DEFAULT
        lam = 1.0 - math.exp(-1.0 / tau_S)
        result = accumulate_S_t(S_prev=0.0, S_input=1.0, tau_S=tau_S)
        assert result == pytest.approx(lam, rel=1e-6)

    def test_steady_state_converges_to_input(self):
        S = 0.0
        for _ in range(200):
            S = accumulate_S_t(S, S_input=2.0, tau_S=TAU_S_DEFAULT)
        assert abs(S - 2.0) < 1e-3

    def test_large_tau_S_slow_rise(self):
        fast = accumulate_S_t(0.0, 1.0, tau_S=1.0)
        slow = accumulate_S_t(0.0, 1.0, tau_S=20.0)
        assert fast > slow

    def test_small_tau_S_approaches_input_immediately(self):
        result = accumulate_S_t(0.0, 1.0, tau_S=0.01)
        assert result == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# ignition_probability  —  Eq. 2: σ(γ_sig·(Sₜ−θₜ))
# ---------------------------------------------------------------------------


class TestIgnitionProbability:
    def test_equal_St_theta_gives_half(self):
        p = ignition_probability(1.0, 1.0, gamma_sig=5.0)
        assert p == pytest.approx(0.5, abs=1e-9)

    def test_above_threshold_above_half(self):
        assert ignition_probability(1.2, 1.0, gamma_sig=5.0) > 0.5

    def test_below_threshold_below_half(self):
        assert ignition_probability(0.8, 1.0, gamma_sig=5.0) < 0.5

    def test_large_margin_approaches_one(self):
        assert ignition_probability(5.0, 0.0, gamma_sig=5.0) > 0.999

    def test_large_deficit_approaches_zero(self):
        assert ignition_probability(0.0, 5.0, gamma_sig=5.0) < 0.001

    def test_steeper_gamma_sharpens_transition(self):
        p_soft = ignition_probability(1.1, 1.0, gamma_sig=2.0)
        p_sharp = ignition_probability(1.1, 1.0, gamma_sig=7.5)
        assert p_sharp > p_soft


# ---------------------------------------------------------------------------
# ignition_criterion  —  stochastic and deterministic modes
# ---------------------------------------------------------------------------


class TestIgnitionCriterion:
    def test_deterministic_above_half_fires(self):
        assert ignition_criterion(1.1, 1.0) is True

    def test_deterministic_below_half_does_not_fire(self):
        assert ignition_criterion(0.9, 1.0) is False

    def test_deterministic_at_threshold_fires(self):
        # P = 0.5 ≥ 0.5 → True
        assert ignition_criterion(1.0, 1.0) is True

    def test_stochastic_mode_returns_bool(self):
        rng = np.random.default_rng(42)
        result = ignition_criterion(1.0, 1.0, rng=rng)
        assert isinstance(result, bool)

    def test_stochastic_high_probability_mostly_fires(self):
        rng = np.random.default_rng(0)
        fires = sum(ignition_criterion(2.0, 0.0, rng=rng) for _ in range(200))
        assert fires > 190

    def test_stochastic_low_probability_rarely_fires(self):
        rng = np.random.default_rng(0)
        fires = sum(ignition_criterion(0.0, 2.0, rng=rng) for _ in range(200))
        assert fires < 10


# ---------------------------------------------------------------------------
# step_theta  —  Eq. 3: dθ/dt = −λθ(θ−θ₀) + κ_meta·C − Δ_info·I + η_NE·NE
# ---------------------------------------------------------------------------


class TestStepTheta:
    def test_hand_computed_one_step(self):
        # dθ = -0.3*(1.0-0.6) + 0.2*1.0 - 0.15*0.5 + 0.0 = -0.12+0.2-0.075 = 0.005
        result = step_theta(
            theta_t=1.0,
            C_t=1.0,
            I_t=0.5,
            theta_0=0.6,
            lambda_theta=0.3,
            kappa_meta=0.2,
            delta_info=0.15,
            eta_NE=0.0,
        )
        assert result == pytest.approx(1.0 + 0.005, rel=1e-6)

    def test_mean_reversion_toward_theta_0(self):
        # Starting far above θ₀ with no metabolic/info drive → converges down
        theta = 5.0
        for _ in range(500):
            theta = step_theta(
                theta,
                C_t=0.0,
                I_t=0.0,
                theta_0=1.0,
                lambda_theta=0.3,
                kappa_meta=0.0,
                delta_info=0.0,
            )
        assert abs(theta - 1.0) < 0.01

    def test_metabolic_cost_raises_threshold(self):
        t1 = step_theta(1.0, C_t=0.0, I_t=0.0)
        t2 = step_theta(1.0, C_t=2.0, I_t=0.0)
        assert t2 > t1

    def test_information_value_lowers_threshold(self):
        t1 = step_theta(1.0, C_t=0.0, I_t=0.0)
        t2 = step_theta(1.0, C_t=0.0, I_t=2.0)
        assert t2 < t1

    def test_NE_raises_threshold(self):
        t1 = step_theta(1.0, C_t=0.0, I_t=0.0, NE_t=0.0)
        t2 = step_theta(1.0, C_t=0.0, I_t=0.0, NE_t=1.0)
        assert t2 > t1


# ---------------------------------------------------------------------------
# theta_equilibrium  —  steady-state helper
# ---------------------------------------------------------------------------


class TestThetaEquilibrium:
    def test_hand_computed(self):
        # θ* = 0.6 + (0.2*1.0 - 0.15*0.5)/0.3 = 0.6 + 0.125/0.3 ≈ 1.017
        result = theta_equilibrium(
            C=1.0,
            I=0.5,
            theta_0=0.6,
            lambda_theta=0.3,
            kappa_meta=0.2,
            delta_info=0.15,
        )
        assert result == pytest.approx(0.6 + (0.2 - 0.075) / 0.3, rel=1e-6)

    def test_step_theta_converges_to_equilibrium(self):
        theta_eq = theta_equilibrium(
            C=1.0,
            I=0.5,
            theta_0=THETA_0_DEFAULT,
            lambda_theta=LAMBDA_THETA_DEFAULT,
            kappa_meta=KAPPA_META_DEFAULT,
            delta_info=DELTA_INFO_DEFAULT,
        )
        theta = 0.0
        for _ in range(500):
            theta = step_theta(theta, C_t=1.0, I_t=0.5)
        assert abs(theta - theta_eq) < 0.01


# ---------------------------------------------------------------------------
# run_trial  —  full single-trial pipeline
# ---------------------------------------------------------------------------


class TestRunTrial:
    def test_returns_all_keys(self):
        result = run_trial(
            pi_e=1.2,
            z_e=0.8,
            pi_i_baseline=1.0,
            z_i=0.5,
            M_hat=0.5,
            C_t=1.0,
            I_t=0.5,
            theta_t=1.0,
        )
        expected = {
            "pi_i_eff",
            "S_input",
            "S_t",
            "theta_t",
            "theta_next",
            "ignition_prob",
            "ignition",
        }
        assert set(result.keys()) == expected

    def test_pi_i_eff_matches_formula(self):
        result = run_trial(
            pi_e=1.0,
            z_e=1.0,
            pi_i_baseline=1.0,
            z_i=0.0,
            M_hat=0.0,
            C_t=0.0,
            I_t=0.0,
            theta_t=10.0,
        )
        assert result["pi_i_eff"] == pytest.approx(1.0)

    def test_S_input_matches_formula(self):
        result = run_trial(
            pi_e=1.2,
            z_e=0.8,
            pi_i_baseline=1.0,
            z_i=0.5,
            M_hat=0.0,
            C_t=0.0,
            I_t=0.0,
            theta_t=10.0,
            beta_sm=0.0,
        )
        assert result["S_input"] == pytest.approx(1.2 * 0.8 + 1.0 * 0.5, rel=1e-6)

    def test_stochastic_mode(self):
        rng = np.random.default_rng(7)
        result = run_trial(
            pi_e=1.0,
            z_e=1.0,
            pi_i_baseline=1.0,
            z_i=0.5,
            M_hat=0.0,
            C_t=1.0,
            I_t=0.5,
            theta_t=1.0,
            rng=rng,
        )
        assert isinstance(result["ignition"], bool)
