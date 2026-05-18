"""Liquid Neural Network (LNN) implementation — Paper 2.

Implements a continuous-time recurrent network whose state evolves as:

    τ · dx/dt = −x + f(W_in · u + W_rec · x + b)

where τ is a learnable time-constant vector, x is the hidden state,
u is the external input, and f is a bounded nonlinearity.
"""

import numpy as np
from numpy.typing import NDArray


class LiquidNeuralNetwork:
    """Continuous-time liquid neural network.

    Parameters
    ----------
    n_inputs : int
        Dimensionality of the external input u.
    n_hidden : int
        Number of liquid (hidden) neurons.
    n_outputs : int
        Dimensionality of the readout y.
    tau : float or NDArray
        Neuron time constants (scalar broadcasts to all neurons).
    spectral_radius : float
        Target spectral radius ρ for the recurrent weight matrix.
    seed : int
        RNG seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        tau: float | NDArray = 10.0,
        spectral_radius: float = 0.9,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.tau = (
            np.full(n_hidden, tau) if np.isscalar(tau) else np.asarray(tau, dtype=float)
        )
        if self.tau.shape != (n_hidden,):
            raise ValueError(f"tau must be scalar or shape ({n_hidden},)")

        # Input weights: sparse uniform in [−1, 1]
        self.W_in = rng.uniform(-1.0, 1.0, (n_hidden, n_inputs))

        # Recurrent weights: sparse Gaussian, scaled to target spectral radius
        W_rec = rng.standard_normal((n_hidden, n_hidden))
        mask = rng.random((n_hidden, n_hidden)) < 0.1  # 10 % connectivity
        W_rec *= mask
        eigenvalues = np.linalg.eigvals(W_rec)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_rec = W_rec * (spectral_radius / current_radius)
        self.W_rec = W_rec

        # Readout weights (linear)
        self.W_out = rng.standard_normal((n_outputs, n_hidden)) * 0.1
        self.b = np.zeros(n_hidden)

        # State
        self.x = np.zeros(n_hidden)

    @staticmethod
    def _nonlinearity(x: NDArray) -> NDArray:
        return np.tanh(x)

    def step(self, u: NDArray, dt: float = 1.0) -> NDArray:
        """Advance the liquid state by one Euler step of size dt.

        Args:
            u: Input vector of shape (n_inputs,).
            dt: Integration time step (ms).

        Returns:
            Readout y of shape (n_outputs,).
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (self.n_inputs,):
            raise ValueError(f"Expected input shape ({self.n_inputs},), got {u.shape}")

        drive = self.W_in @ u + self.W_rec @ self.x + self.b
        dx = (-self.x + self._nonlinearity(drive)) / self.tau
        self.x = self.x + dt * dx
        return self.W_out @ self.x

    def run(self, inputs: NDArray, dt: float = 1.0) -> NDArray:
        """Run the network over a sequence of inputs.

        Args:
            inputs: Array of shape (T, n_inputs).
            dt: Integration time step.

        Returns:
            Readout array of shape (T, n_outputs).
        """
        inputs = np.asarray(inputs, dtype=float)
        T = inputs.shape[0]
        outputs = np.empty((T, self.n_outputs))
        self.reset()
        for t in range(T):
            outputs[t] = self.step(inputs[t], dt)
        return outputs

    def reset(self) -> None:
        """Reset hidden state to zero."""
        self.x = np.zeros(self.n_hidden)
