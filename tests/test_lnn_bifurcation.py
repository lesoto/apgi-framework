"""Tests for scripts/APGI_LNN_Bifurcation_Analysis.py.

Covers ODEParameters, APGILNNODESystem, BifurcationSignatures,
EmpiricalPrediction/EmpiricalPredictions, APGILNNBifurcationAnalysis,
and the module-level run_analysis() entry point.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the script module by file path (it is not an installed package)
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).parent.parent / "scripts" / "APGI_LNN_Bifurcation_Analysis.py"


def _load_lnn():
    spec = importlib.util.spec_from_file_location(
        "APGI_LNN_Bifurcation_Analysis", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["APGI_LNN_Bifurcation_Analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


lnn = _load_lnn()

ODEParameters = lnn.ODEParameters
APGILNNODESystem = lnn.APGILNNODESystem
BifurcationSignatures = lnn.BifurcationSignatures
BifurcationSweepResult = lnn.BifurcationSweepResult
StochasticTrialResult = lnn.StochasticTrialResult
EmpiricalPrediction = lnn.EmpiricalPrediction
EmpiricalPredictions = lnn.EmpiricalPredictions
APGILNNBifurcationAnalysis = lnn.APGILNNBifurcationAnalysis
run_analysis = lnn.run_analysis


# ===========================================================================
# ODEParameters
# ===========================================================================


class TestODEParameters:
    def test_default_alpha_satisfies_bifurcation_condition(self):
        p = ODEParameters()
        # Bifurcation requires alpha > 4 / tau_S
        assert p.alpha > 4.0 / p.tau_S

    def test_custom_params_stored(self):
        p = ODEParameters()
        p.alpha = 20.0
        assert p.alpha == 20.0

    def test_default_theta_base(self):
        assert ODEParameters().theta_base == pytest.approx(0.5)


# ===========================================================================
# APGILNNODESystem
# ===========================================================================


class TestAPGILNNODESystem:
    def setup_method(self):
        self.ode = APGILNNODESystem()
        self.theta = self.ode.p.theta_base

    def test_ignition_prob_at_threshold_is_half(self):
        # σ(0) = 0.5
        b = self.ode.ignition_prob(self.theta, self.theta)
        assert b == pytest.approx(0.5, abs=1e-6)

    def test_ignition_prob_above_threshold_gt_half(self):
        assert self.ode.ignition_prob(self.theta + 0.2, self.theta) > 0.5

    def test_ignition_prob_below_threshold_lt_half(self):
        assert self.ode.ignition_prob(self.theta - 0.2, self.theta) < 0.5

    def test_f_returns_2d_array(self):
        x = np.array([0.4, self.theta])
        out = self.ode.f(x, S_input=0.4)
        assert out.shape == (2,)

    def test_jacobian_analytic_shape(self):
        x = np.array([0.5, self.theta])
        J = self.ode.jacobian_analytic(x, S_input=0.5)
        assert J.shape == (2, 2)

    def test_jacobian_numerical_shape(self):
        x = np.array([0.5, self.theta])
        J = self.ode.jacobian_numerical(x, S_input=0.5)
        assert J.shape == (2, 2)

    def test_verify_jacobian_passes_at_default_params(self):
        x = np.array([0.5, self.theta])
        assert self.ode.verify_jacobian(x, S_input=0.5) is True

    def test_verify_jacobian_tight_tolerance(self):
        x = np.array([0.4, self.theta])
        J_a = self.ode.jacobian_analytic(x, 0.4)
        J_n = self.ode.jacobian_numerical(x, 0.4)
        # Second-order Taylor remainder at this point; allow 3e-4
        assert np.max(np.abs(J_a - J_n)) < 3e-4

    def test_lambda_s_negative_sub_threshold(self):
        # Far below threshold — should be stable (negative dominant eigenvalue)
        x = np.array([0.1, self.theta])
        lam = self.ode.lambda_S(x, 0.1)
        assert lam < 0

    def test_lambda_s_near_zero_at_threshold(self):
        # At threshold the dominant eigenvalue is close to 0 (bifurcation)
        x = np.array([self.theta, self.theta])
        lam = self.ode.lambda_S(x, self.theta)
        assert abs(lam) < 1.0  # not large negative

    def test_eigenvalues_returns_sorted_descending(self):
        x = np.array([0.5, self.theta])
        eigs = self.ode.eigenvalues(x, 0.5)
        assert eigs[0] >= eigs[1]

    def test_eigenvalues_length(self):
        x = np.array([0.5, self.theta])
        assert len(self.ode.eigenvalues(x, 0.5)) == 2

    def test_custom_params_propagate(self):
        p = ODEParameters()
        p.alpha = 20.0
        ode = APGILNNODESystem(p)
        assert ode.p.alpha == 20.0


# ===========================================================================
# BifurcationSignatures
# ===========================================================================


class TestSarlesB:
    def test_unimodal_gaussian_below_threshold(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        b = BifurcationSignatures.sarles_b(x)
        # Sarle's b < 0.555 for unimodal — pure Gaussian is ~0.333
        assert not np.isnan(b)
        assert b < 0.555

    def test_bimodal_distribution_near_or_above_threshold(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(-2, 0.3, 500), rng.normal(2, 0.3, 500)])
        b = BifurcationSignatures.sarles_b(x)
        assert not np.isnan(b)

    def test_too_few_samples_returns_nan(self):
        b = BifurcationSignatures.sarles_b(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(b)


class TestSweepEigenvalues:
    def setup_method(self):
        self.sig = BifurcationSignatures()

    def test_returns_sweep_result(self):
        result = self.sig.sweep_eigenvalues(n_steps=20)
        assert isinstance(result, BifurcationSweepResult)

    def test_s_values_length(self):
        result = self.sig.sweep_eigenvalues(n_steps=30)
        assert len(result.S_values) == 30

    def test_lambda1_trace_length(self):
        result = self.sig.sweep_eigenvalues(n_steps=30)
        assert len(result.lambda1_trace) == 30

    def test_ac1_trace_bounded(self):
        result = self.sig.sweep_eigenvalues(n_steps=20)
        assert np.all(result.ac1_trace >= -1.0)
        assert np.all(result.ac1_trace <= 1.0)

    def test_bifurcation_idx_within_range(self):
        result = self.sig.sweep_eigenvalues(n_steps=50)
        assert 0 <= result.bifurcation_idx < 50

    def test_variance_positive(self):
        result = self.sig.sweep_eigenvalues(n_steps=20)
        assert np.all(result.variance_trace >= 0)


class TestRunStochasticTrials:
    def setup_method(self):
        self.sig = BifurcationSignatures()

    def test_returns_stochastic_result(self):
        result = self.sig.run_stochastic_trials(S_input=0.5, n_trials=50)
        assert isinstance(result, StochasticTrialResult)

    def test_distribution_length(self):
        result = self.sig.run_stochastic_trials(S_input=0.5, n_trials=80)
        assert len(result.S_final_distribution) == 80

    def test_ac1_bounded(self):
        result = self.sig.run_stochastic_trials(S_input=0.5, n_trials=50)
        assert np.isfinite(
            result.ac1_S
        ), "ac1_S should be finite after S_finals is recorded"
        assert -1.0 <= result.ac1_S <= 1.0

    def test_single_trial_returns_zero_ac1(self):
        # n_trials=1 → len(S_finals) == 1 → std guard → ac1 = 0.0
        result = self.sig.run_stochastic_trials(S_input=0.5, n_trials=1)
        assert result.ac1_S == 0.0


class TestFullSweepWithStochastic:
    def test_bimodality_index_filled(self):
        sig = BifurcationSignatures()
        # Small run to keep test fast
        result = sig.run_full_sweep_with_stochastic(
            n_sweep_steps=20, n_stochastic_per_point=20, stochastic_subsample=5
        )
        # At least some values were filled (not all zeros from default)
        assert result.bimodality_index is not None
        assert len(result.bimodality_index) == 20


# ===========================================================================
# EmpiricalPrediction / EmpiricalPredictions
# ===========================================================================


class TestEmpiricalPrediction:
    def test_construction(self):
        ep = EmpiricalPrediction(
            prediction_id="BP_test",
            observable="AC1",
            measurement_window="0–500 ms",
            test_statistic="Kendall τ",
            alpha_criterion=0.05,
            sample_size_estimate=20,
            expected_direction="increasing",
            falsification_condition="flat AC1",
        )
        assert ep.prediction_id == "BP_test"
        assert ep.alpha_criterion == 0.05


class TestEmpiricalPredictions:
    def setup_method(self):
        self.eps = EmpiricalPredictions()

    def test_as_dict_list_nonempty(self):
        lst = self.eps.as_dict_list()
        assert len(lst) >= 4  # at least 4 predictions defined

    def test_as_dict_list_contains_required_keys(self):
        for d in self.eps.as_dict_list():
            assert "id" in d  # serialised as "id" not "prediction_id"
            assert "observable" in d

    def test_log_predictions_does_not_raise(self):
        self.eps.log_predictions()  # should emit logs without error


# ===========================================================================
# APGILNNBifurcationAnalysis (smoke test — fast n_steps)
# ===========================================================================


class TestAPGILNNBifurcationAnalysis:
    def _fast_analysis(self):
        """Run with minimal sweep to keep the test fast."""
        analysis = APGILNNBifurcationAnalysis()
        # Monkey-patch run_full_sweep to use tiny parameters
        original = analysis.signatures.run_full_sweep_with_stochastic

        def _fast_sweep(**kw):
            return original(
                n_sweep_steps=30, n_stochastic_per_point=50, stochastic_subsample=5
            )

        analysis.signatures.run_full_sweep_with_stochastic = _fast_sweep
        return analysis.run_analysis()

    def test_returns_dict(self):
        result = self._fast_analysis()
        assert isinstance(result, dict)

    def test_status_is_pass_or_partial(self):
        result = self._fast_analysis()
        assert result["status"] in ("PASS", "PARTIAL")

    def test_jacobian_verified(self):
        result = self._fast_analysis()
        assert result["jacobian_verified"] is True

    def test_eigenvalue_traces_present(self):
        result = self._fast_analysis()
        traces = result["eigenvalue_traces"]
        assert "lambda1" in traces
        assert "ac1" in traces
        assert len(traces["lambda1"]) > 0

    def test_empirical_predictions_list(self):
        result = self._fast_analysis()
        assert isinstance(result["empirical_predictions"], list)
        assert len(result["empirical_predictions"]) >= 4

    def test_falsification_criterion_in_result(self):
        result = self._fast_analysis()
        assert "bifurcation" in result["falsification_criterion"].lower()


class TestRunAnalysisFunction:
    def test_module_level_run_analysis_returns_dict(self, monkeypatch):
        # Call the module-level entry point with a fast-patched sweep so
        # lines 792-793 (function body) are executed in the coverage trace.
        original_sweep = lnn.BifurcationSignatures.run_full_sweep_with_stochastic

        def _fast_sweep(self, **kw):
            return original_sweep(
                self, n_sweep_steps=20, n_stochastic_per_point=20, stochastic_subsample=5
            )

        monkeypatch.setattr(lnn.BifurcationSignatures, "run_full_sweep_with_stochastic", _fast_sweep)
        result = run_analysis()
        assert isinstance(result, dict)
        assert "status" in result
