"""Tests for LiquidNeuralNetwork and APGIHierarchy."""

import numpy as np
import pytest

from apgi.extensions.hierarchical import APGIHierarchy, HierarchicalLevel
from apgi.extensions.liquid_network import LiquidNeuralNetwork


class TestLiquidNeuralNetwork:
    def test_output_shape_single_step(self):
        lnn = LiquidNeuralNetwork(n_inputs=4, n_hidden=16, n_outputs=2)
        u = np.zeros(4)
        y = lnn.step(u)
        assert y.shape == (2,)

    def test_output_shape_run(self):
        lnn = LiquidNeuralNetwork(n_inputs=3, n_hidden=8, n_outputs=1)
        inputs = np.random.default_rng(0).standard_normal((20, 3))
        outputs = lnn.run(inputs)
        assert outputs.shape == (20, 1)

    def test_reset_zeros_state(self):
        lnn = LiquidNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
        lnn.step(np.ones(2))
        lnn.reset()
        assert np.allclose(lnn.x, 0.0)

    def test_deterministic_with_same_seed(self):
        inputs = np.ones((5, 2))
        out1 = LiquidNeuralNetwork(2, 8, 1, seed=7).run(inputs)
        out2 = LiquidNeuralNetwork(2, 8, 1, seed=7).run(inputs)
        assert np.allclose(out1, out2)

    def test_wrong_input_shape_raises(self):
        lnn = LiquidNeuralNetwork(n_inputs=4, n_hidden=8, n_outputs=2)
        with pytest.raises(ValueError):
            lnn.step(np.zeros(3))

    def test_invalid_tau_shape_raises(self):
        with pytest.raises(ValueError, match="tau must be scalar or shape"):
            LiquidNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1, tau=np.ones(3))

    def test_spectral_radius_constrained(self):
        lnn = LiquidNeuralNetwork(4, 32, 2, spectral_radius=0.9, seed=0)
        rho = np.max(np.abs(np.linalg.eigvals(lnn.W_rec)))
        assert rho <= 0.9 + 1e-9

    def test_state_changes_after_nonzero_input(self):
        lnn = LiquidNeuralNetwork(2, 8, 1, seed=0)
        before = lnn.x.copy()
        lnn.step(np.array([1.0, -1.0]))
        assert not np.allclose(lnn.x, before)

    def test_pi_t_none_matches_original_behaviour(self):
        # Passing no precision should be identical to Pi(t) == 1 (tau_eff == tau_baseline)
        lnn1 = LiquidNeuralNetwork(2, 8, 1, seed=0)
        lnn2 = LiquidNeuralNetwork(2, 8, 1, seed=0)
        u = np.array([0.5, -0.2])
        y1 = lnn1.step(u, pi_t=None)
        y2 = lnn2.step(u, pi_t=1.0)
        assert np.allclose(y1, y2)

    def test_high_precision_compresses_tau_eff_and_changes_dynamics(self):
        # tau_eff = tau_baseline / Pi(t): higher Pi should produce a
        # different (faster-decaying) trajectory than Pi=1.
        lnn_hi = LiquidNeuralNetwork(2, 8, 1, seed=0)
        lnn_lo = LiquidNeuralNetwork(2, 8, 1, seed=0)
        u = np.array([1.0, 1.0])
        y_hi = lnn_hi.step(u, pi_t=5.0)
        y_lo = lnn_lo.step(u, pi_t=0.5)
        assert not np.allclose(y_hi, y_lo)

    def test_ignite_fires_and_resets_when_above_threshold(self):
        lnn = LiquidNeuralNetwork(2, 8, 1, seed=0)
        u = np.array([5.0, 5.0])
        # Drive hard with a near-zero threshold so ignition is essentially guaranteed.
        _, fired = lnn.ignite(u, theta_t=1e-6, rho_S=0.3)
        assert fired is True

    def test_ignite_does_not_reset_when_below_threshold(self):
        lnn = LiquidNeuralNetwork(2, 8, 1, seed=0)
        u = np.array([0.001, 0.001])
        _, fired = lnn.ignite(u, theta_t=1e6, rho_S=0.3)
        assert fired is False


class TestHierarchicalLevel:
    def test_returns_tuple(self):
        level = HierarchicalLevel(level_id=1, n_units=8)
        bottom = np.ones(8)
        top = np.zeros(8)
        result = level.update(bottom, top, C_metabolic=1.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_s_t_contribution_non_negative(self):
        level = HierarchicalLevel(1, 8)
        _, s = level.update(
            np.random.default_rng(0).standard_normal(8), np.zeros(8), 1.0
        )
        assert s >= 0.0

    def test_theta_initialised_at_theta_star(self):
        level = HierarchicalLevel(1, 8, theta_star=0.55)
        assert level.theta == pytest.approx(0.55)

    def test_step_theta_hand_computed(self):
        # dtheta = -(1.0-0.6)/5 + 0.3*(0.8-0.6) + 0.1*2.0 + 0.05*0.0
        level = HierarchicalLevel(
            1, 8, theta_star=0.6, tau_theta=5.0, kappa_coupling=0.3, beta_k=0.1, eta_k=0.05
        )
        level.theta = 1.0
        result = level.step_theta(theta_next=0.8, theta_star_next=0.6, S_k=2.0)
        expected = 1.0 + (-(1.0 - 0.6) / 5.0 + 0.3 * (0.8 - 0.6) + 0.1 * 2.0)
        assert result == pytest.approx(expected)

    def test_higher_level_deviation_raises_lower_theta_coupling(self):
        level = HierarchicalLevel(1, 8, theta_star=0.6, kappa_coupling=0.3)
        level.theta = 0.6
        no_coupling = level.step_theta(theta_next=0.6, theta_star_next=0.6, S_k=0.0)
        level.theta = 0.6
        with_coupling = level.step_theta(theta_next=1.0, theta_star_next=0.6, S_k=0.0)
        assert with_coupling > no_coupling


class TestAPGIHierarchy:
    def test_forward_output_keys(self):
        hier = APGIHierarchy(n_sensory=32)
        result = hier.forward(np.zeros(32), C_metabolic=1.0)
        assert {"S_t_total", "level_S_t", "level_errors", "predictions"} <= set(
            result.keys()
        )

    def test_five_levels(self):
        hier = APGIHierarchy(n_sensory=32)
        result = hier.forward(np.zeros(32), C_metabolic=1.0)
        assert len(result["level_S_t"]) == 5

    def test_s_t_total_non_negative(self):
        hier = APGIHierarchy(n_sensory=16)
        result = hier.forward(
            np.random.default_rng(1).standard_normal(16), C_metabolic=0.5
        )
        assert result["S_t_total"] >= 0.0

    def test_s_t_total_equals_sum_of_levels(self):
        hier = APGIHierarchy(n_sensory=32)
        result = hier.forward(np.ones(32) * 0.3, C_metabolic=1.0)
        assert result["S_t_total"] == pytest.approx(sum(result["level_S_t"]))

    def test_forward_with_resizing_input(self):
        hier = APGIHierarchy(n_sensory=16)
        # sensory_input size 32, but n_sensory is 16
        result = hier.forward(np.ones(32), C_metabolic=1.0)
        assert len(result["level_errors"][0]) == 16

    def test_forward_returns_level_theta_and_ignition(self):
        hier = APGIHierarchy(n_sensory=32)
        result = hier.forward(np.ones(32) * 0.5, C_metabolic=1.0)
        assert len(result["level_theta"]) == 5
        assert len(result["level_ignition"]) == 5
        assert all(isinstance(b, (bool, np.bool_)) for b in result["level_ignition"])

    def test_top_level_theta_uncoupled_from_nonexistent_level_above(self):
        hier = APGIHierarchy(n_sensory=16)
        # Should not raise despite L5 having no level above.
        result = hier.forward(np.zeros(16), C_metabolic=0.0)
        assert np.isfinite(result["level_theta"][-1])

    def test_reset_restores_theta_to_theta_star(self):
        hier = APGIHierarchy(n_sensory=16)
        hier.forward(np.ones(16) * 5.0, C_metabolic=2.0)
        hier.reset()
        for level in hier.levels:
            assert level.theta == pytest.approx(level.theta_star)

    def test_reset_zeros_all_states(self):
        hier = APGIHierarchy(n_sensory=16)
        hier.forward(np.ones(16), 1.0)
        hier.reset()
        for level in hier.levels:
            assert np.allclose(level.state, 0.0)
            assert np.allclose(level.prediction, 0.0)
