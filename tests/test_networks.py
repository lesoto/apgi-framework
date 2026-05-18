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

    def test_reset_zeros_all_states(self):
        hier = APGIHierarchy(n_sensory=16)
        hier.forward(np.ones(16), 1.0)
        hier.reset()
        for level in hier.levels:
            assert np.allclose(level.state, 0.0)
            assert np.allclose(level.prediction, 0.0)
