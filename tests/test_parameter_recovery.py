"""Tests for parameter recovery simulation (Appendix A.4).
Pearson r > 0.75 for both β and Πⁱ.
"""

from apgi.parameter_recovery import (
    generate_synthetic_data,
    recover_parameters,
    run_recovery_simulation,
)


class TestGenerateSyntheticData:
    def test_output_keys(self):
        data = generate_synthetic_data(n_trials=10, beta_true=0.7, pi_i_true=1.0)
        expected = {
            "C_metabolic",
            "V_information",
            "pi_e",
            "z_e",
            "z_i",
            "S_t_observed",
            "ignition",
        }
        assert set(data.keys()) == expected

    def test_lengths(self):
        n = 25
        data = generate_synthetic_data(n_trials=n, beta_true=0.7, pi_i_true=1.0)
        for v in data.values():
            assert len(v) == n

    def test_ignition_is_binary(self):
        data = generate_synthetic_data(
            n_trials=50, beta_true=0.7, pi_i_true=1.0, seed=1
        )
        unique = set(data["ignition"].tolist())
        assert unique.issubset({0, 1})

    def test_reproducibility(self):
        d1 = generate_synthetic_data(n_trials=20, beta_true=0.7, pi_i_true=1.0, seed=42)
        d2 = generate_synthetic_data(n_trials=20, beta_true=0.7, pi_i_true=1.0, seed=42)
        assert (d1["S_t_observed"] == d2["S_t_observed"]).all()


class TestRecoverParameters:
    def test_returns_expected_keys(self):
        data = generate_synthetic_data(
            n_trials=50, beta_true=0.7, pi_i_true=1.0, seed=0
        )
        result = recover_parameters(data, n_restarts=2)
        assert set(result.keys()) == {"beta_hat", "pi_i_hat", "nll", "converged"}
        assert isinstance(result["converged"], bool)

    def test_recovered_values_are_positive(self):
        data = generate_synthetic_data(
            n_trials=50, beta_true=0.7, pi_i_true=1.0, seed=0
        )
        result = recover_parameters(data, n_restarts=2)
        assert result["beta_hat"] > 0
        assert result["pi_i_hat"] > 0

    def test_single_recovery_within_order_of_magnitude(self):
        """A single recovery on 100 trials should be in the right ballpark."""
        data = generate_synthetic_data(
            n_trials=100, beta_true=0.8, pi_i_true=1.2, seed=5
        )
        result = recover_parameters(data, n_restarts=3, seed=5)
        assert 0.1 < result["beta_hat"] < 5.0
        assert 0.1 < result["pi_i_hat"] < 5.0


class TestRunRecoverySimulation:
    """Full simulation criterion: r > 0.75 for both parameters."""

    RECOVERY_CRITERION = 0.75

    def test_recovery_criterion_beta(self):
        results = run_recovery_simulation(
            n_simulations=30,
            n_trials_per_sim=150,
            seed=2024,
        )
        assert (
            results["r_beta"] > self.RECOVERY_CRITERION
        ), f"β recovery r={results['r_beta']:.3f} below criterion {self.RECOVERY_CRITERION}"

    def test_recovery_criterion_pi_i(self):
        results = run_recovery_simulation(
            n_simulations=30,
            n_trials_per_sim=150,
            seed=2024,
        )
        assert (
            results["r_pi_i"] > self.RECOVERY_CRITERION
        ), f"Πⁱ recovery r={results['r_pi_i']:.3f} below criterion {self.RECOVERY_CRITERION}"

    def test_output_structure(self):
        results = run_recovery_simulation(n_simulations=5, n_trials_per_sim=50, seed=1)
        assert set(results.keys()) == {
            "r_beta",
            "r_pi_i",
            "beta_true",
            "beta_hat",
            "pi_i_true",
            "pi_i_hat",
            "converged",
        }
        assert len(results["beta_true"]) == 5
        assert len(results["converged"]) == 5
        assert all(isinstance(c, bool) for c in results["converged"])
