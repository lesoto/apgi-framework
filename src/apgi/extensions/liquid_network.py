"""Liquid Neural Network (LNN) implementation — Paper 2, §4.

Implements the continuous-time reservoir ODE (Paper 2 §4, Notation Appendix):

    dx/dt = −x/τ_eff(t) + tanh(W_res·x + W_in·u(t))
    τ_eff(t) = τ_baseline / Π(t)

Precision Π(t) acts as time-constant modulation: high precision compresses
τ_eff (sharper, shorter-lived reservoir dynamics); low precision extends it
(broader temporal integration). Ignition is read out as S_t ≡ |W_out·x(t)|
crossing θ_t, followed by a partial post-ignition reset x ← ρ_S·x (Paper 2
§4, "Python implementation" note).
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
        Baseline neuron time constants τ_baseline (scalar broadcasts to all
        neurons). Typical physiological range: 5–50 ms. When a precision
        value Π(t) is passed to :meth:`step`, the *effective* time constant
        used for that step is τ_eff(t) = τ_baseline / Π(t) (Paper 2 §4);
        otherwise τ_baseline is used directly (original behaviour).
    spectral_radius : float
        Target spectral radius ρ_res for the recurrent weight matrix.
        Canonical range [0.7, 0.95] (Paper 2 §4); ρ_res ≈ 0.90 is used as
        the default, matching the paper's reference implementation.
        Values < 1.0 yield stable dynamics; values near 1.0 maximise
        memory capacity (edge-of-chaos regime), with a saddle-node
        bifurcation into sustained ignition predicted near ρ_res ≈ 0.95
        (Paper 2 §5.1, Fig. 3).
    seed : int
        RNG seed for reproducible weight initialisation.

    Examples
    --------
    Single-step inference:

    >>> import numpy as np
    >>> lnn = LiquidNeuralNetwork(n_inputs=3, n_hidden=50, n_outputs=2, seed=0)
    >>> u = np.array([0.5, -0.3, 1.0])
    >>> y = lnn.step(u, dt=1.0)
    >>> y.shape
    (2,)

    Sequence processing (e.g. 100 ms of simulated input at 1 ms resolution):

    >>> rng = np.random.default_rng(42)
    >>> inputs = rng.uniform(-1.0, 1.0, (100, 3))
    >>> outputs = lnn.run(inputs, dt=1.0)
    >>> outputs.shape
    (100, 2)

    Using per-neuron time constants to model heterogeneous populations:

    >>> tau_vec = np.linspace(5.0, 50.0, 50)
    >>> lnn_het = LiquidNeuralNetwork(n_inputs=3, n_hidden=50, n_outputs=2,
    ...                               tau=tau_vec, seed=1)
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

    def step(self, u: NDArray, dt: float = 1.0, pi_t: float | None = None) -> NDArray:
        """Advance the liquid state by one Euler step of size dt.

        dx/dt = −x/τ_eff(t) + tanh(W_res·x + W_in·u(t))

        Args:
            u: Input vector of shape (n_inputs,).
            dt: Integration time step (ms). Stability requires dt < min(tau).
            pi_t: Optional precision Π(t) (Paper 2 §4). When given, the
                effective time constant for this step is
                τ_eff(t) = τ_baseline / Π(t); when None, τ_baseline is used
                directly (original behaviour, Π(t) ≡ 1).

        Returns:
            Readout y of shape (n_outputs,).

        Examples
        --------
        >>> import numpy as np
        >>> lnn = LiquidNeuralNetwork(n_inputs=2, n_hidden=10, n_outputs=1, seed=0)
        >>> y = lnn.step(np.array([1.0, 0.0]), dt=1.0)
        >>> y.shape
        (1,)
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (self.n_inputs,):
            raise ValueError(f"Expected input shape ({self.n_inputs},), got {u.shape}")

        tau_eff = self.tau if pi_t is None else self.tau / pi_t
        drive = self.W_in @ u + self.W_rec @ self.x + self.b
        dx = (-self.x + self._nonlinearity(drive)) / tau_eff
        self.x = self.x + dt * dx
        return self.W_out @ self.x

    def ignite(
        self,
        u: NDArray,
        theta_t: float,
        dt: float = 1.0,
        pi_t: float | None = None,
        rho_S: float = 0.5,
    ) -> tuple[NDArray, bool]:
        """Advance one step and apply the ignition readout + refractory reset.

        S_t ≡ |W_out·x(t)|; ignites when S_t > θ_t; on ignition the reservoir
        state is partially reset: x ← ρ_S·x (Paper 2 §4, "Python
        implementation" note).

        Args:
            u: Input vector of shape (n_inputs,).
            theta_t: Ignition threshold θ_t to compare S_t against.
            dt: Integration time step.
            pi_t: Optional precision Π(t) driving τ_eff modulation.
            rho_S: Post-ignition reservoir retention factor ρ_S ∈ (0, 1).

        Returns:
            Tuple of (readout y, whether ignition fired this step).
        """
        y = self.step(u, dt=dt, pi_t=pi_t)
        fired = bool(np.linalg.norm(y) > theta_t)
        if fired:
            self.x = rho_S * self.x
        return y, fired

    def run(self, inputs: NDArray, dt: float = 1.0) -> NDArray:
        """Run the network over a sequence of inputs.

        Resets hidden state before processing so results are
        independent of prior ``step`` calls.

        Args:
            inputs: Array of shape (T, n_inputs).
            dt: Integration time step.

        Returns:
            Readout array of shape (T, n_outputs).

        Examples
        --------
        >>> import numpy as np
        >>> lnn = LiquidNeuralNetwork(n_inputs=3, n_hidden=20, n_outputs=2, seed=0)
        >>> inputs = np.random.default_rng(0).uniform(-1, 1, (200, 3))
        >>> outputs = lnn.run(inputs, dt=1.0)
        >>> outputs.shape
        (200, 2)
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
