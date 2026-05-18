"""APGINormalizer — signal normalisation for cross-session and cross-participant comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class APGINormalizer:
    """Normalise APGI signals (Sₜ, θₜ) to a common scale.

    Supports z-score and min-max normalisation.  Parameters are estimated
    from a calibration set and then applied to new observations, making
    the normaliser safe to fit once and reuse across recording sessions.

    Parameters
    ----------
    method : {"zscore", "minmax"}
        Normalisation strategy.
    epsilon : float
        Small constant added to the denominator to avoid division by zero.
    """

    def __init__(self, method: str = "zscore", epsilon: float = 1e-8) -> None:
        if method not in ("zscore", "minmax"):
            raise ValueError(f"method must be 'zscore' or 'minmax', got {method!r}")
        self.method = method
        self.epsilon = epsilon
        self._loc: float | None = None
        self._scale: float | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, signals: NDArray) -> "APGINormalizer":
        """Estimate normalisation parameters from a calibration array.

        Args:
            signals: 1-D array of raw signal values (e.g. a session of Sₜ).

        Returns:
            self (for chaining).
        """
        signals = np.asarray(signals, dtype=float).ravel()
        if signals.size == 0:
            raise ValueError("signals must not be empty")

        if self.method == "zscore":
            self._loc = float(np.mean(signals))
            self._scale = float(np.std(signals))
        else:  # minmax
            self._loc = float(np.min(signals))
            self._scale = float(np.max(signals) - np.min(signals))
        return self

    def fit_transform(self, signals: NDArray) -> NDArray:
        """Fit and immediately transform the calibration array."""
        return self.fit(signals).transform(signals)

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, signals: NDArray) -> NDArray:
        """Apply fitted normalisation to new signals.

        Args:
            signals: Array of raw signal values (any shape).

        Returns:
            Normalised array of the same shape.
        """
        if self._loc is None or self._scale is None:
            raise RuntimeError("Call fit() before transform()")
        arr = np.asarray(signals, dtype=float)
        return (arr - self._loc) / (self._scale + self.epsilon)

    def inverse_transform(self, normalised: NDArray) -> NDArray:
        """Reverse the normalisation to recover original-scale values."""
        if self._loc is None or self._scale is None:
            raise RuntimeError("Call fit() before inverse_transform()")
        arr = np.asarray(normalised, dtype=float)
        return arr * (self._scale + self.epsilon) + self._loc

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._loc is not None and self._scale is not None

    def __repr__(self) -> str:
        status = f"loc={self._loc:.4g}, scale={self._scale:.4g}" if self.is_fitted else "unfitted"
        return f"APGINormalizer(method={self.method!r}, {status})"
