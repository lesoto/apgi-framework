"""Shared pytest fixtures for the APGI test suite."""

import numpy as np
import pytest

from apgi.normalizer import APGINormalizer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_S_t(rng):
    """100-trial session of synthetic Sₜ values."""
    return rng.uniform(0.5, 3.0, 100)


@pytest.fixture
def sample_theta_t(rng):
    """Corresponding θₜ values drawn from a slightly different range."""
    return rng.uniform(0.8, 2.5, 100)


@pytest.fixture
def sample_C_metabolic(rng):
    return rng.uniform(0.5, 2.0, 100)


@pytest.fixture
def fitted_normalizer(sample_S_t):
    """APGINormalizer already fitted on sample_S_t."""
    n = APGINormalizer(method="zscore")
    n.fit(sample_S_t)
    return n
