"""Tests for APGINormalizer."""

import numpy as np
import pytest

from apgi.normalizer import APGINormalizer


class TestAPGINormalizerZScore:
    def test_fit_transform_zero_mean(self, sample_S_t):
        n = APGINormalizer(method="zscore")
        out = n.fit_transform(sample_S_t)
        assert abs(np.mean(out)) < 1e-10

    def test_fit_transform_unit_std(self, sample_S_t):
        n = APGINormalizer(method="zscore")
        out = n.fit_transform(sample_S_t)
        assert abs(np.std(out) - 1.0) < 1e-6

    def test_inverse_roundtrip(self, sample_S_t):
        n = APGINormalizer(method="zscore")
        normed = n.fit_transform(sample_S_t)
        recovered = n.inverse_transform(normed)
        assert np.allclose(recovered, sample_S_t, atol=1e-10)

    def test_unfitted_transform_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            APGINormalizer().transform(np.array([1.0, 2.0]))

    def test_unfitted_inverse_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            APGINormalizer().inverse_transform(np.array([0.0]))

    def test_is_fitted_flag(self, fitted_normalizer):
        assert fitted_normalizer.is_fitted

    def test_unfitted_is_not_fitted(self):
        assert not APGINormalizer().is_fitted

    def test_repr_contains_method(self, fitted_normalizer):
        assert "zscore" in repr(fitted_normalizer)


class TestAPGINormalizerMinMax:
    def test_fit_transform_range(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        n = APGINormalizer(method="minmax")
        out = n.fit_transform(data)
        # min → ~0, max → ~1 (epsilon prevents exact 1)
        assert out[0] < 1e-6
        assert out[-1] == pytest.approx(1.0, abs=1e-6)

    def test_inverse_roundtrip(self):
        data = np.linspace(1.0, 5.0, 20)
        n = APGINormalizer(method="minmax")
        recovered = n.inverse_transform(n.fit_transform(data))
        assert np.allclose(recovered, data, atol=1e-10)


class TestAPGINormalizerEdgeCases:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            APGINormalizer(method="l2")

    def test_empty_signals_raises(self):
        with pytest.raises(ValueError, match="empty"):
            APGINormalizer().fit(np.array([]))

    def test_single_value_does_not_raise(self):
        n = APGINormalizer(method="zscore")
        n.fit(np.array([3.14]))
        assert n.is_fitted
