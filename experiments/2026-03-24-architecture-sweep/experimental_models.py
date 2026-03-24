"""Local model definitions for the architecture sweep experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dnn_arithmetic.models import ReluMLP, ResidualReluMLP

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

Array = jax.Array | np.ndarray


def _small_residual_init(
    key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32
) -> jax.Array:
    """Return a small LeCun-normal initializer for residual down projections.

    Args:
        key: JAX PRNG key.
        shape: Parameter shape.
        dtype: Parameter dtype.

    Returns:
        Initialized weight array.

    """
    base_init = jax.nn.initializers.lecun_normal()
    return 0.1 * base_init(key, shape, dtype)


class DenseReluResNet(nnx.Module):
    """Residual MLP with learned softmax mixing over prior block states.

    Each block forms a learned convex combination of all previous
    residual-stream states, then applies a residual branch on top of that mixed
    state. This changes the skip path itself rather than only the branch input.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual_dim: int,
        inter_dim: int,
        num_blocks: int,
        rngs: nnx.Rngs,
        *,
        down_kernel_init: nnx.Initializer = _small_residual_init,
    ):
        self.in_proj = nnx.Linear(in_dim, residual_dim, rngs=rngs)
        self.out_proj = nnx.Linear(residual_dim, out_dim, rngs=rngs)
        self.block_lin_up = nnx.List[nnx.Linear]()
        self.block_lin_down = nnx.List[nnx.Linear]()
        self.block_mix_logits = nnx.List[nnx.Param[jax.Array]]()

        for block_index in range(num_blocks):
            self.block_lin_up.append(nnx.Linear(residual_dim, inter_dim, rngs=rngs))
            self.block_lin_down.append(
                nnx.Linear(
                    inter_dim,
                    residual_dim,
                    kernel_init=down_kernel_init,
                    rngs=rngs,
                )
            )
            self.block_mix_logits.append(
                nnx.Param(jnp.zeros((block_index + 1,), dtype=jnp.float32))
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_dim = residual_dim
        self.inter_dim = inter_dim
        self.num_blocks = num_blocks

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        history: list[jax.Array] = [self.in_proj(x)]

        for mix_logits, lin_up, lin_down in zip(
            self.block_mix_logits,
            self.block_lin_up,
            self.block_lin_down,
        ):
            weights = jax.nn.softmax(jnp.asarray(mix_logits[...]), axis=0)
            stacked_history = jnp.stack(history, axis=0)
            mixed_input = jnp.tensordot(weights, stacked_history, axes=(0, 0))
            branch = lin_up(mixed_input)
            branch = nnx.relu(branch)
            branch = lin_down(branch)
            history.append(mixed_input + branch)

        # The readout currently uses the latest state; output mixing can be
        # added later as a separate architectural change.
        return self.out_proj(history[-1])

    def init_output_bias(self, y_value: Array) -> None:
        """Initialize the output bias to a representative target value.

        Args:
            y_value: Representative target value in output space.

        """
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias to initialize.")
        bias[...] = jnp.asarray(y_value)

    def get_output_bias(self) -> Array:
        """Return a copy of the current output bias."""
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias.")
        return jnp.array(bias)


__all__ = [
    "DenseReluResNet",
    "ReluMLP",
    "ResidualReluMLP",
]


def dense_resnet_mix_matrix(model: DenseReluResNet) -> np.ndarray:
    """Return the learned dense skip weights as a lower-triangular matrix.

    Args:
        model: Trained dense residual network.

    Returns:
        Array of shape ``(num_blocks, num_blocks)`` where row ``i`` contains the
        normalized skip weights for block ``i + 1`` over states ``h0..h_i``.

    """
    matrix = np.zeros((model.num_blocks, model.num_blocks), dtype=np.float32)
    for row_index, mix_logits in enumerate(model.block_mix_logits):
        weights = jax.nn.softmax(jnp.asarray(mix_logits[...]), axis=0)
        weights_np = np.asarray(weights, dtype=np.float32)
        matrix[row_index, : row_index + 1] = weights_np
    return matrix
