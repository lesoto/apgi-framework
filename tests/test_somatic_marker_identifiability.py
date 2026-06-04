"""Tests for scripts/APGI_Somatic_Marker_Identifiability.py.

Covers _sigmoid, _draw_subject_params, generative models (_generate_session1,
_generate_session2), likelihood functions (_ll_session1, _ll_session2,
_ll_free_beta, _ll_joint), _numerical_fim, _expected_fim_free_beta,
run_module1/2/3, build_comparison_table, run_validation, validate, and main.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import script as module
# ---------------------------------------------------------------------------
_SCRIPT = (
    Path(__file__).parent.parent / "scripts" / "APGI_Somatic_Marker_Identifiability.py"
)


def _load_smi():
    spec = importlib.util.spec_from_file_location(
        "APGI_Somatic_Marker_Identifiability", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["APGI_Somatic_Marker_Identifiability"] = mod
    spec.loader.exec_module(mod)
    return mod


smi = _load_smi()

_sigmoid = smi._sigmoid
_draw_subject_params = smi._draw_subject_params
_generate_session1 = smi._generate_session1
_generate_session2 = smi._generate_session2
_ll_session1 = smi._ll_session1
_ll_session2 = smi._ll_session2
_ll_free_beta = smi._ll_free_beta
_ll_joint = smi._ll_joint
_numerical_fim = smi._numerical_fim
_expected_fim_free_beta = smi._expected_fim_free_beta
run_module1 = smi.run_module1
run_module2 = smi.run_module2
run_module3 = smi.run_module3
build_comparison_table = smi.build_comparison_table
run_validation = smi.run_validation
validate = smi.validate
main = smi.main

RNG = np.random.RandomState(42)


# ===========================================================================
# _sigmoid
# ===========================================================================


class TestSigmoid:
    def test_at_zero_is_half(self):
        assert _sigmoid(np.array([0.0])) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_clipping_does_not_overflow(self):
        # Without clipping, exp(1000) would overflow; _sigmoid clips at ±30
        out = _sigmoid(np.array([1e6, -1e6]))
        assert np.all(np.isfinite(out))

    def test_vectorised(self):
        x = np.linspace(-5, 5, 20)
        out = _sigmoid(x)
        assert out.shape == (20,)
        assert np.all((out >= 0) & (out <= 1))


# ===========================================================================
# _draw_subject_params
# ===========================================================================


class TestDrawSubjectParams:
    def test_returns_dict_with_required_keys(self):
        params = _draw_subject_params(RNG)
        expected = {"Pi_i_baseline", "gamma_V", "gamma_A", "Pi_e", "theta_baseline"}
        assert expected.issubset(set(params.keys()))

    def test_values_positive(self):
        params = _draw_subject_params(RNG)
        for v in params.values():
            assert v > 0


# ===========================================================================
# Generative models
# ===========================================================================


class TestGenerateSession1:
    def setup_method(self):
        self.data = _generate_session1(
            gamma_V=0.6, gamma_A=0.3, sigma_bold=0.2, rng=RNG
        )

    def test_keys_present(self):
        assert {"V", "A", "BOLD_obs", "M_hat_true"} == set(self.data.keys())

    def test_lengths_match(self):
        n = smi.N_CONTEXT_ACTION_PAIRS
        for arr in self.data.values():
            assert len(arr) == n

    def test_bold_obs_influenced_by_gamma(self):
        # Larger gamma_V → BOLD more correlated with V
        d_large = _generate_session1(
            gamma_V=5.0, gamma_A=0.0, sigma_bold=0.01, rng=np.random.RandomState(1)
        )
        d_zero = _generate_session1(
            gamma_V=0.0, gamma_A=0.0, sigma_bold=0.01, rng=np.random.RandomState(1)
        )
        var_large = float(np.var(d_large["BOLD_obs"]))
        var_zero = float(np.var(d_zero["BOLD_obs"]))
        assert var_large > var_zero


class TestGenerateSession2:
    def setup_method(self):
        s1 = _generate_session1(gamma_V=0.6, gamma_A=0.3, sigma_bold=0.2, rng=RNG)
        self.data = _generate_session2(
            Pi_i_baseline=1.0,
            Pi_e=1.0,
            theta_baseline=0.5,
            alpha=smi.DEFAULT_ALPHA_SIGMOID,
            M_hat_true=s1["M_hat_true"],
            rng=RNG,
            n_trials=100,
        )

    def test_keys_present(self):
        assert {"pair_idx", "M_hat", "B", "ze", "zi", "Pi_i_eff"} == set(
            self.data.keys()
        )

    def test_b_is_binary(self):
        assert set(np.unique(self.data["B"])).issubset({0.0, 1.0})

    def test_trial_count(self):
        assert len(self.data["B"]) == 100


# ===========================================================================
# Likelihood functions  (minimisation convention → returns –LL)
# ===========================================================================


class TestLLSession1:
    def _make_data(self):
        rng = np.random.RandomState(7)
        return _generate_session1(0.6, 0.3, 0.2, rng)

    def test_returns_finite_float(self):
        d = self._make_data()
        val = _ll_session1(
            np.array([0.6, 0.3, np.log(0.2)]), d["V"], d["A"], d["BOLD_obs"]
        )
        assert np.isfinite(val)

    def test_correct_params_lower_than_wrong(self):
        d = self._make_data()
        correct = _ll_session1(
            np.array([0.6, 0.3, np.log(0.2)]), d["V"], d["A"], d["BOLD_obs"]
        )
        wrong = _ll_session1(
            np.array([5.0, 5.0, np.log(5.0)]), d["V"], d["A"], d["BOLD_obs"]
        )
        assert correct < wrong

    def test_zero_sigma_returns_large(self):
        d = self._make_data()
        # log_sigma → -inf means sigma → 0 but exp(-1000) ~ 0; function guards this
        val = _ll_session1(np.array([0.0, 0.0, -1000.0]), d["V"], d["A"], d["BOLD_obs"])
        assert val >= 1e10  # guard returns 1e10


class TestLLSession2:
    def _make_data(self):
        rng = np.random.RandomState(8)
        s1 = _generate_session1(0.6, 0.3, 0.2, rng)
        s2 = _generate_session2(
            1.0, 1.0, 0.5, smi.DEFAULT_ALPHA_SIGMOID, s1["M_hat_true"], rng, 200
        )
        return s1, s2

    def test_returns_finite_float(self):
        s1, s2 = self._make_data()
        val = _ll_session2(
            np.array([0.0, 0.0, 0.5]),  # log(1), log(1), theta=0.5
            s2["B"],
            s2["ze"],
            s2["zi"],
            s2["M_hat"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        assert np.isfinite(val)

    def test_correct_params_lower_than_wrong(self):
        s1, s2 = self._make_data()
        correct = _ll_session2(
            np.array([0.0, 0.0, 0.5]),
            s2["B"],
            s2["ze"],
            s2["zi"],
            s2["M_hat"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        wrong = _ll_session2(
            np.array([3.0, 3.0, 5.0]),
            s2["B"],
            s2["ze"],
            s2["zi"],
            s2["M_hat"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        assert correct < wrong


class TestLLFreeBeta:
    def test_returns_finite_float(self):
        rng = np.random.RandomState(9)
        s1 = _generate_session1(0.6, 0.3, 0.2, rng)
        s2 = _generate_session2(
            1.0, 1.0, 0.5, smi.DEFAULT_ALPHA_SIGMOID, s1["M_hat_true"], rng, 100
        )
        val = _ll_free_beta(
            np.array([0.0, 1.0, 0.6, 0.3, 0.0, 0.5]),
            s2["B"],
            s2["ze"],
            s2["zi"],
            s1["V"],
            s1["A"],
            s2["pair_idx"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        assert np.isfinite(val)


class TestLLJoint:
    def test_equals_sum_of_s1_and_s2(self):
        rng = np.random.RandomState(10)
        s1 = _generate_session1(0.6, 0.3, 0.2, rng)
        s2 = _generate_session2(
            1.0, 1.0, 0.5, smi.DEFAULT_ALPHA_SIGMOID, s1["M_hat_true"], rng, 100
        )
        params_s1 = np.array([0.6, 0.3, np.log(0.2)])
        params_s2 = np.array([0.0, 0.0, 0.5])
        ll_s1 = _ll_session1(params_s1, s1["V"], s1["A"], s1["BOLD_obs"])
        ll_s2 = _ll_session2(
            params_s2,
            s2["B"],
            s2["ze"],
            s2["zi"],
            s2["M_hat"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        ll_joint = _ll_joint(
            np.concatenate([params_s1, params_s2]),
            s1["V"],
            s1["A"],
            s1["BOLD_obs"],
            s2["B"],
            s2["ze"],
            s2["zi"],
            s2["M_hat"],
            smi.DEFAULT_ALPHA_SIGMOID,
        )
        assert ll_joint == pytest.approx(ll_s1 + ll_s2, abs=1e-8)


# ===========================================================================
# _numerical_fim
# ===========================================================================


class TestNumericalFIM:
    def test_shape_matches_params(self):
        rng = np.random.RandomState(11)
        s1 = _generate_session1(0.6, 0.3, 0.2, rng)
        params_s1 = np.array([0.6, 0.3, np.log(0.2)])
        fim = _numerical_fim(
            _ll_session1,
            params_s1,
            fn_args=(s1["V"], s1["A"], s1["BOLD_obs"]),
        )
        assert fim.shape == (3, 3)

    def test_approximately_symmetric(self):
        rng = np.random.RandomState(12)
        s1 = _generate_session1(0.6, 0.3, 0.2, rng)
        params_s1 = np.array([0.6, 0.3, np.log(0.2)])
        fim = _numerical_fim(
            _ll_session1,
            params_s1,
            fn_args=(s1["V"], s1["A"], s1["BOLD_obs"]),
        )
        assert np.allclose(fim, fim.T, atol=1e-4)


# ===========================================================================
# _expected_fim_free_beta
# ===========================================================================


class TestExpectedFIMFreeBeta:
    def test_shape_6x6(self):
        rng = np.random.RandomState(0)
        V = rng.randn(40)
        A = rng.randn(40)
        fim = _expected_fim_free_beta(
            log_Pi_i=0.0,
            beta=1.0,
            gamma_V=0.6,
            gamma_A=0.3,
            log_Pi_e=0.0,
            theta=0.5,
            alpha=smi.DEFAULT_ALPHA_SIGMOID,
            V=V,
            A=A,
            n_trials=200,
            seed=0,
        )
        assert fim.shape == (6, 6)

    def test_high_condition_number_with_beta(self):
        # Free-β should produce an ill-conditioned FIM
        rng = np.random.RandomState(0)
        V = rng.randn(40)
        A = rng.randn(40)
        fim = _expected_fim_free_beta(
            log_Pi_i=0.0,
            beta=1.0,
            gamma_V=0.6,
            gamma_A=0.3,
            log_Pi_e=0.0,
            theta=0.5,
            alpha=smi.DEFAULT_ALPHA_SIGMOID,
            V=V,
            A=A,
            n_trials=500,
            seed=0,
        )
        cond = np.linalg.cond(fim)
        assert cond > 100  # should be badly conditioned


# ===========================================================================
# run_module1 / run_module2 / run_module3  (small n for speed)
# ===========================================================================


class TestRunModule1:
    def test_returns_dict_with_passed(self):
        result = run_module1(n_subjects=5, seed=0, verbose=False)
        assert "passed" in result

    def test_convergence_rate_in_range(self):
        result = run_module1(n_subjects=5, seed=0, verbose=False)
        assert 0.0 <= result["convergence_rate"] <= 1.0

    def test_recovery_r_present(self):
        result = run_module1(n_subjects=5, seed=0, verbose=False)
        assert "recovery_r" in result
        assert len(result["recovery_r"]) >= 4

    def test_verbose_does_not_raise(self):
        result = run_module1(n_subjects=5, seed=0, verbose=True)
        assert "passed" in result


class TestRunModule2:
    def test_returns_dict_with_passed(self):
        result = run_module2(seed=0, verbose=False)
        assert "passed" in result

    def test_condition_number_positive(self):
        result = run_module2(seed=0, verbose=False)
        assert result["condition_number"] > 0

    def test_fim_matrix_present(self):
        result = run_module2(seed=0, verbose=False)
        assert "fim_matrix" in result or "fim_diagonal" in result
        assert isinstance(result["condition_number"], float)

    def test_verbose_does_not_raise(self):
        result = run_module2(seed=0, verbose=True)
        assert "passed" in result


class TestRunModule3:
    def test_returns_dict_with_passed(self):
        result = run_module3(n_subjects=5, seed=0, verbose=False)
        assert "passed" in result

    def test_pathological_condition_number_large(self):
        result = run_module3(n_subjects=5, seed=0, verbose=False)
        # key is "condition_number" in module 3 result
        assert result["condition_number"] > 1000

    def test_has_poor_recovery(self):
        result = run_module3(n_subjects=5, seed=0, verbose=False)
        assert result["has_poor_recovery"] is True

    def test_verbose_does_not_raise(self):
        result = run_module3(n_subjects=5, seed=0, verbose=True)
        assert "passed" in result


# ===========================================================================
# build_comparison_table
# ===========================================================================


class TestBuildComparisonTable:
    def _modules(self):
        m1 = run_module1(n_subjects=5, seed=0, verbose=False)
        m2 = run_module2(seed=0, verbose=False)
        m3 = run_module3(n_subjects=5, seed=0, verbose=False)
        return m1, m2, m3

    def test_returns_non_empty_string(self):
        m1, m2, m3 = self._modules()
        table = build_comparison_table(m1, m2, m3)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_condition_number(self):
        m1, m2, m3 = self._modules()
        table = build_comparison_table(m1, m2, m3)
        assert "condition" in table.lower() or "FIM" in table


# ===========================================================================
# run_validation  (smoke — small n)
# ===========================================================================


class TestRunValidation:
    def test_returns_dict_with_passed(self):
        result = run_validation(n_subjects=5, seed=0, verbose=False)
        assert "passed" in result

    def test_protocol_id_present(self):
        result = run_validation(n_subjects=5, seed=0, verbose=False)
        assert result.get("protocol_id") == "VP-SMI"

    def test_named_predictions_present(self):
        result = run_validation(n_subjects=5, seed=0, verbose=False)
        assert (
            "predictions" in result
            or "named_predictions" in result
            or "SMI.1" in str(result)
        )

    def test_verbose_logs_comparison_table(self):
        # Exercises the if verbose: table.splitlines() loop
        result = run_validation(n_subjects=5, seed=0, verbose=True)
        assert "passed" in result


# ===========================================================================
# validate() and main()
# ===========================================================================


class TestValidateAndMain:
    def test_main_returns_passed_key(self):
        result = main(n_subjects=5, seed=0, verbose=False)
        assert "passed" in result

    def test_validate_calls_run_validation(self):
        # validate() is a one-line wrapper; calling it exercises line 1203
        result = validate()
        assert isinstance(result, dict)
        assert "passed" in result
