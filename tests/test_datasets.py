"""Tests for apgi.datasets."""

import numpy as np

from apgi import datasets


class TestMakeSampleSession:
    def test_returns_dict_with_required_keys(self):
        result = datasets.make_sample_session(n_trials=10)
        assert isinstance(result, dict)
        required_keys = [
            "pi_e",
            "z_e",
            "pi_i",
            "z_i",
            "C_metabolic",
            "V_information",
        ]
        for key in required_keys:
            assert key in result

    def test_array_lengths_match_n_trials(self):
        n_trials = 25
        result = datasets.make_sample_session(n_trials=n_trials)
        for key in result:
            assert len(result[key]) == n_trials

    def test_values_are_in_expected_ranges(self):
        result = datasets.make_sample_session(n_trials=100)
        assert np.all(result["pi_e"] >= 0.8) and np.all(result["pi_e"] <= 1.5)
        assert np.all(result["z_e"] >= 0.2) and np.all(result["z_e"] <= 1.0)
        assert np.all(result["pi_i"] >= 0.5) and np.all(result["pi_i"] <= 1.5)
        assert np.all(result["z_i"] >= 0.1) and np.all(result["z_i"] <= 0.8)
        assert np.all(result["C_metabolic"] >= 0.5) and np.all(
            result["C_metabolic"] <= 2.0
        )
        assert np.all(result["V_information"] >= 0.1) and np.all(
            result["V_information"] <= 1.0
        )

    def test_reproducibility_with_seed(self):
        result1 = datasets.make_sample_session(n_trials=10, seed=42)
        result2 = datasets.make_sample_session(n_trials=10, seed=42)
        for key in result1:
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_different_seeds_produce_different_results(self):
        result1 = datasets.make_sample_session(n_trials=10, seed=1)
        result2 = datasets.make_sample_session(n_trials=10, seed=2)
        for key in result1:
            assert not np.array_equal(result1[key], result2[key])

    def test_default_n_trials(self):
        result = datasets.make_sample_session()
        for key in result:
            assert len(result[key]) == 50

    def test_arrays_are_float64(self):
        result = datasets.make_sample_session(n_trials=10)
        for key in result:
            assert result[key].dtype == np.float64


class TestMakeSampleDocGroups:
    def test_returns_dict_with_required_keys(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        assert isinstance(result, dict)
        required_keys = [
            "group_labels",
            "group_codes",
            "subject_id",
            "S_t",
            "ignition",
        ]
        for key in required_keys:
            assert key in result

    def test_group_labels_are_strings(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        # numpy string arrays have dtype '<U8' or similar, not object
        assert result["group_labels"].dtype.kind in ("U", "O", "S")
        unique_labels = set(result["group_labels"])
        expected_labels = {"VS_UWS", "MCS", "Controls"}
        assert unique_labels == expected_labels

    def test_group_codes_are_integers(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        assert result["group_codes"].dtype == np.int32
        unique_codes = set(result["group_codes"])
        assert unique_codes == {0, 1, 2}

    def test_subject_id_increments_correctly(self):
        result = datasets.make_sample_doc_groups(n_per_group=3, n_trials=5)
        unique_ids = set(result["subject_id"])
        assert len(unique_ids) == 9  # 3 groups * 3 subjects
        assert max(unique_ids) == 8

    def test_s_t_is_float64(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        assert result["S_t"].dtype == np.float64

    def test_ignition_is_boolean(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        assert result["ignition"].dtype == bool

    def test_total_length_matches_configuration(self):
        n_per_group = 3
        n_trials = 10
        result = datasets.make_sample_doc_groups(
            n_per_group=n_per_group, n_trials=n_trials
        )
        total_length = len(result["group_labels"])
        expected_length = 3 * n_per_group * n_trials  # 3 groups
        assert total_length == expected_length

    def test_s_t_values_are_non_negative(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        assert np.all(result["S_t"] >= 0)

    def test_reproducibility_with_seed(self):
        result1 = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5, seed=42)
        result2 = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5, seed=42)
        for key in result1:
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_different_seeds_produce_different_results(self):
        result1 = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5, seed=1)
        result2 = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5, seed=2)
        # At least one key should differ with different seeds
        any_different = any(
            not np.array_equal(result1[key], result2[key]) for key in result1
        )
        assert any_different

    def test_default_parameters(self):
        result = datasets.make_sample_doc_groups()
        total_length = len(result["group_labels"])
        expected_length = 3 * 20 * 60  # 3 groups * 20 subjects * 60 trials
        assert total_length == expected_length

    def test_group_codes_match_labels(self):
        result = datasets.make_sample_doc_groups(n_per_group=2, n_trials=5)
        for i in range(len(result["group_labels"])):
            label = result["group_labels"][i]
            code = result["group_codes"][i]
            if label == "VS_UWS":
                assert code == 0
            elif label == "MCS":
                assert code == 1
            elif label == "Controls":
                assert code == 2
