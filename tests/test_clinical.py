"""Tests for EnhancedClinicalInterpreter and ConsciousnessLevel."""

import numpy as np
import pytest

from apgi.clinical import (
    ClinicalReport,
    ConsciousnessLevel,
    EnhancedClinicalInterpreter,
)


class TestEnhancedClinicalInterpreterInterpret:
    def test_returns_clinical_report(
        self, sample_S_t, sample_theta_t, sample_C_metabolic
    ):
        interp = EnhancedClinicalInterpreter(window_size=20)
        report = interp.interpret(
            sample_S_t[:20], sample_theta_t[:20], sample_C_metabolic[:20]
        )
        assert isinstance(report, ClinicalReport)

    def test_ignition_rate_in_unit_interval(
        self, sample_S_t, sample_theta_t, sample_C_metabolic
    ):
        interp = EnhancedClinicalInterpreter()
        report = interp.interpret(sample_S_t, sample_theta_t, sample_C_metabolic)
        assert 0.0 <= report.ignition_rate <= 1.0

    def test_ignition_index_positive(
        self, sample_S_t, sample_theta_t, sample_C_metabolic
    ):
        interp = EnhancedClinicalInterpreter()
        report = interp.interpret(sample_S_t, sample_theta_t, sample_C_metabolic)
        assert report.ignition_index > 0

    def test_mismatched_shapes_raise(self):
        interp = EnhancedClinicalInterpreter()
        with pytest.raises(ValueError):
            interp.interpret(np.ones(10), np.ones(11), np.ones(10))

    def test_level_is_valid_enum(self, sample_S_t, sample_theta_t, sample_C_metabolic):
        interp = EnhancedClinicalInterpreter()
        report = interp.interpret(sample_S_t, sample_theta_t, sample_C_metabolic)
        assert report.level in list(ConsciousnessLevel)


class TestConsciousnessLevelClassification:
    """Drive specific ignition-index ranges to verify all four level labels."""

    def _make_report(self, S_val: float, theta_val: float) -> ClinicalReport:
        interp = EnhancedClinicalInterpreter()
        n = 50
        return interp.interpret(
            np.full(n, S_val),
            np.full(n, theta_val),
            np.ones(n),
        )

    def test_unresponsive(self):
        report = self._make_report(S_val=0.2, theta_val=1.0)
        assert report.level == ConsciousnessLevel.UNRESPONSIVE

    def test_minimally_conscious(self):
        report = self._make_report(S_val=0.55, theta_val=1.0)
        assert report.level == ConsciousnessLevel.MINIMALLY_CONSCIOUS

    def test_conscious(self):
        report = self._make_report(S_val=1.0, theta_val=1.0)
        assert report.level == ConsciousnessLevel.CONSCIOUS

    def test_hyper_alert(self):
        report = self._make_report(S_val=2.0, theta_val=1.0)
        assert report.level == ConsciousnessLevel.HYPER_ALERT


class TestInterpretSession:
    def test_returns_list_of_reports(
        self, sample_S_t, sample_theta_t, sample_C_metabolic
    ):
        interp = EnhancedClinicalInterpreter(window_size=20)
        reports = interp.interpret_session(
            sample_S_t, sample_theta_t, sample_C_metabolic
        )
        assert len(reports) == 5
        assert all(isinstance(r, ClinicalReport) for r in reports)

    def test_partial_trailing_window_discarded(self):
        interp = EnhancedClinicalInterpreter(window_size=20)
        reports = interp.interpret_session(np.ones(45), np.ones(45) * 0.8, np.ones(45))
        assert len(reports) == 2

    def test_window_size_one_gives_trial_count(
        self, sample_S_t, sample_theta_t, sample_C_metabolic
    ):
        interp = EnhancedClinicalInterpreter(window_size=1)
        reports = interp.interpret_session(
            sample_S_t, sample_theta_t, sample_C_metabolic
        )
        assert len(reports) == len(sample_S_t)


class TestEnhancedClinicalInterpreterEdgeCases:
    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            EnhancedClinicalInterpreter(window_size=0)

    def test_custom_thresholds_respected(self):
        interp = EnhancedClinicalInterpreter(
            thresholds={
                "minimally_conscious": 100.0,
                "conscious": 200.0,
                "hyper_alert": 300.0,
            }
        )
        report = interp.interpret(np.full(10, 5.0), np.full(10, 1.0), np.ones(10))
        assert report.level == ConsciousnessLevel.UNRESPONSIVE
