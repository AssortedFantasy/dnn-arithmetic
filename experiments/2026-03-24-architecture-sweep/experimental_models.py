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

        # Output mixing over all states (h0 .. h_{num_blocks}).
        self.output_mix_logits = nnx.Param(
            jnp.zeros((num_blocks + 1,), dtype=jnp.float32)
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
            mixed_input: jax.Array = weights[0] * history[0]
            for w, h in zip(weights[1:], history[1:]):
                mixed_input = mixed_input + w * h
            branch = lin_up(mixed_input)
            branch = nnx.relu(branch)
            branch = lin_down(branch)
            history.append(mixed_input + branch)

        # Learned mixing over all history states for the output.
        out_weights = jax.nn.softmax(jnp.asarray(self.output_mix_logits[...]), axis=0)
        out_state: jax.Array = out_weights[0] * history[0]
        for w, h in zip(out_weights[1:], history[1:]):
            out_state = out_state + w * h
        return self.out_proj(out_state)

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


class DenseReluResNetConcat(nnx.Module):
    """DenseReluResNet variant using concat-based history buffer.

    Instead of a Python list, history is stored as a single array of shape
    ``(slots, batch, residual_dim)`` that grows via ``jnp.concatenate`` each
    block.  Block and output mixing use ``jnp.einsum`` over the stacked buffer.

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

        self.output_mix_logits = nnx.Param(
            jnp.zeros((num_blocks + 1,), dtype=jnp.float32)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_dim = residual_dim
        self.inter_dim = inter_dim
        self.num_blocks = num_blocks

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        # history: (batch, 1, residual_dim)
        history = self.in_proj(x)[:, None, :]

        for mix_logits, lin_up, lin_down in zip(
            self.block_mix_logits,
            self.block_lin_up,
            self.block_lin_down,
        ):
            weights = jax.nn.softmax(jnp.asarray(mix_logits[...]), axis=0)
            # (batch, slots, rdim) @ (slots,) -> (batch, rdim)
            mixed_input = jnp.einsum("s,...s d->...d", weights, history)
            branch = lin_up(mixed_input)
            branch = nnx.relu(branch)
            branch = lin_down(branch)
            new_state = (mixed_input + branch)[:, None, :]
            history = jnp.concatenate([history, new_state], axis=1)

        out_weights = jax.nn.softmax(jnp.asarray(self.output_mix_logits[...]), axis=0)
        out_state = jnp.einsum("s,...s d->...d", out_weights, history)
        return self.out_proj(out_state)

    def init_output_bias(self, y_value: Array) -> None:
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias to initialize.")
        bias[...] = jnp.asarray(y_value)

    def get_output_bias(self) -> Array:
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias.")
        return jnp.array(bias)


class DenseReluResNetPrealloc(nnx.Module):
    """DenseReluResNet variant using a preallocated history buffer.

    History is allocated upfront as ``(batch, num_blocks + 1, residual_dim)``
    and filled in-place via ``.at`` indexing.  This avoids any reallocation
    or concatenation during the forward pass.

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

        self.output_mix_logits = nnx.Param(
            jnp.zeros((num_blocks + 1,), dtype=jnp.float32)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_dim = residual_dim
        self.inter_dim = inter_dim
        self.num_blocks = num_blocks

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        batch_shape = x.shape[:-1]
        # Preallocate: (batch, num_blocks + 1, residual_dim)
        history = jnp.zeros(
            (*batch_shape, self.num_blocks + 1, self.residual_dim),
            dtype=x.dtype,
        )
        h0 = self.in_proj(x)
        history = history.at[:, 0, :].set(h0)

        for idx, (mix_logits, lin_up, lin_down) in enumerate(
            zip(
                self.block_mix_logits,
                self.block_lin_up,
                self.block_lin_down,
            )
        ):
            weights = jax.nn.softmax(jnp.asarray(mix_logits[...]), axis=0)
            # Only mix over slots 0..idx (the filled ones).
            active = jax.lax.dynamic_slice_in_dim(history, 0, idx + 1, axis=1)
            mixed_input = jnp.einsum("s,...s d->...d", weights, active)
            branch = lin_up(mixed_input)
            branch = nnx.relu(branch)
            branch = lin_down(branch)
            history = history.at[:, idx + 1, :].set(mixed_input + branch)

        out_weights = jax.nn.softmax(jnp.asarray(self.output_mix_logits[...]), axis=0)
        out_state = jnp.einsum("s,...s d->...d", out_weights, history)
        return self.out_proj(out_state)

    def init_output_bias(self, y_value: Array) -> None:
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias to initialize.")
        bias[...] = jnp.asarray(y_value)

    def get_output_bias(self) -> Array:
        bias = self.out_proj.bias
        if bias is None:
            raise ValueError("Output projection has no bias.")
        return jnp.array(bias)


__all__ = [
    "DenseReluResNet",
    "DenseReluResNetConcat",
    "DenseReluResNetPrealloc",
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
