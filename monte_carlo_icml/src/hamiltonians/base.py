"""Abstract Hamiltonian interface.

Every concrete Hamiltonian must implement ``__call__(t, x, p)`` and
``state_dim``.  The interface is kept minimal so that instances work
naturally with ``jax.vmap`` and ``jax.grad``.
"""

import abc
import jax.numpy as jnp


class Hamiltonian(abc.ABC):
    """H(t, x, p) for a differential game or optimal control problem."""

    @abc.abstractmethod
    def __call__(
        self, t: float, x: jnp.ndarray, p: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate the Hamiltonian.

        Parameters
        ----------
        t : scalar time.
        x : state vector, shape ``(n,)`` or ``(batch, n)``.
        p : co-state (spatial gradient), same shape as *x*.

        Returns
        -------
        Scalar or array of H values.
        """
        ...

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        ...

    @property
    def is_quadratic(self) -> bool:
        """True when H = (1/2)|p|^2, enabling exact Cole-Hopf."""
        return False
