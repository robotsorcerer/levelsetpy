"""Multi-GPU distribution with automatic device detection and single-GPU fallback.

Provides utilities for distributing HJ-Gauss solves across available GPUs,
with intelligent fallback to single-GPU (or CPU) mode.

Usage:
    from monte_carlo.src.gpu_distribution import GPUDistributor

    dist = GPUDistributor(auto_detect=True)  # Auto-detect available GPUs
    print(f"Using {dist.n_devices} devices")

    if dist.is_multi_gpu:
        # Use pmap-based distribution
        v = solver.solve_distributed(states, dist)
    else:
        # Fallback to single-GPU vmap
        v = solver.solve(states)
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import devices, pmap
import numpy as np


class GPUDistributor:
    """Manages multi-GPU distribution with single-GPU fallback."""

    def __init__(
        self,
        auto_detect: bool = True,
        force_n_devices: Optional[int] = None,
        device_type: str = "gpu",
    ):
        """
        Parameters
        ----------
        auto_detect : bool
            Automatically detect available devices. If False, uses force_n_devices.
        force_n_devices : int, optional
            Force use of this many devices (overrides auto-detect).
        device_type : str
            Device type: "gpu", "cpu", or "tpu".
        """
        self.device_type = device_type
        self.auto_detect = auto_detect
        self.force_n_devices = force_n_devices

        # Detect available devices (fall back to CPU if device_type unavailable)
        try:
            available_devices = devices(device_type)
        except RuntimeError:
            # Device type not available, fall back to CPU
            available_devices = devices("cpu")
            self.device_type = "cpu"
        self.n_available = len(available_devices)

        if force_n_devices is not None:
            self.n_devices = min(force_n_devices, self.n_available)
        elif auto_detect:
            self.n_devices = self.n_available
        else:
            self.n_devices = 1

        # Select devices to use
        if self.n_devices > 0:
            self.device_list = list(available_devices[:self.n_devices])
        else:
            self.device_list = [devices("cpu")[0]]
            self.n_devices = 1

        self.is_multi_gpu = self.n_devices > 1

        print(f"[GPUDistributor] Using {self.n_devices} devices (available: {self.n_available})")
        if self.is_multi_gpu:
            print(f"  Device list: {[str(d) for d in self.device_list]}")

    def shard_batch(
        self, batch: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
        """Shard a batch across devices.

        Parameters
        ----------
        batch : jnp.ndarray, shape (M, ...)
            Batch array to shard along first axis.

        Returns
        -------
        sharded : jnp.ndarray, shape (n_devices, M_per_device, ...)
            Reshaped batch, padded if necessary.
        original_shape : tuple
            Original batch size for unsharding later.
        """
        M = batch.shape[0]
        M_per_device = (M + self.n_devices - 1) // self.n_devices  # Ceiling division
        M_padded = M_per_device * self.n_devices

        # Pad batch if necessary
        if M < M_padded:
            pad_amount = M_padded - M
            pad_width = [(0, pad_amount)] + [(0, 0)] * (batch.ndim - 1)
            batch_padded = jnp.pad(batch, pad_width, mode="edge")
        else:
            batch_padded = batch

        # Reshape to (n_devices, M_per_device, ...)
        sharded_shape = (self.n_devices, M_per_device) + batch_padded.shape[1:]
        sharded = batch_padded.reshape(sharded_shape)

        return sharded, (M, M_per_device)

    def unshard_batch(
        self, sharded: jnp.ndarray, original_shape: Tuple[int, int]
    ) -> jnp.ndarray:
        """Unshard a batch from devices, removing padding.

        Parameters
        ----------
        sharded : jnp.ndarray, shape (n_devices, M_per_device, ...)
            Sharded result.
        original_shape : tuple (M, M_per_device)
            Original batch size and per-device size.

        Returns
        -------
        unsharded : jnp.ndarray, shape (M, ...)
            Unsharded and unpadded batch.
        """
        M, M_per_device = original_shape
        # Reshape back to (M_padded, ...)
        M_padded = M_per_device * self.n_devices
        unsharded_padded_shape = (M_padded,) + sharded.shape[2:]
        unsharded_padded = sharded.reshape(unsharded_padded_shape)

        # Remove padding
        unsharded = unsharded_padded[:M]
        return unsharded

    def get_pmap_fn(self, base_fn):
        """Wrap a function for pmap distribution.

        Parameters
        ----------
        base_fn : callable
            Function to distribute. Should accept (key, x_per_device, ...).

        Returns
        -------
        pmapped_fn : callable
            Function distributed via pmap across devices.
        """
        if not self.is_multi_gpu:
            # Single device: return base function unchanged
            return base_fn

        # Multi-device: wrap with pmap
        # pmap automatically distributes first axis across devices
        pmapped = pmap(base_fn, axis_name="i")
        return pmapped

    def create_keys_per_device(
        self, seed: int, per_device_batch_size: int
    ) -> jnp.ndarray:
        """Create independent PRNG keys for each device.

        Parameters
        ----------
        seed : int
            Master seed.
        per_device_batch_size : int
            Batch size per device.

        Returns
        -------
        keys_per_device : jnp.ndarray, shape (n_devices, per_device_batch_size, 2)
            Unique keys for each device and batch element.
        """
        key = jax.random.PRNGKey(seed)
        # Split for each device
        keys_per_dev = jax.random.split(key, self.n_devices)
        # Further split for each element in per-device batch
        keys_per_dev_per_elem = jnp.array(
            [jax.random.split(k, per_device_batch_size) for k in keys_per_dev]
        )
        return keys_per_dev_per_elem

    def __repr__(self) -> str:
        mode = "multi-GPU" if self.is_multi_gpu else "single-GPU"
        return f"GPUDistributor({self.n_devices} {self.device_type}s, mode={mode})"
