"""Model definitions for DNN arithmetic experiments."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

Array = jax.Array | np.ndarray


def _zero_init(
    key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32
) -> jax.Array:
    del key
    return jnp.zeros(shape, dtype=dtype)


def _small_residual_init(
    key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32
) -> jax.Array:
    base_init = jax.nn.initializers.lecun_normal()
    return 0.1 * base_init(key, shape, dtype)


class ReluMLP(nnx.Module):
    """Simple RELU MLP."""

    def __init__(
        self,
        layer_sizes: list[int],
        rngs: nnx.Rngs,
        *,
        kernel_inits: Sequence[nnx.Initializer] | None = None,
        bias_inits: Sequence[nnx.Initializer] | None = None,
    ):
        assert len(layer_sizes) > 1, "ReluMLP must have at least 2 layers."
        n_layers = len(layer_sizes) - 1
        if kernel_inits is not None and len(kernel_inits) != n_layers:
            raise ValueError(
                f"kernel_inits must have length {n_layers}, got {len(kernel_inits)}"
            )
        if bias_inits is not None and len(bias_inits) != n_layers:
            raise ValueError(
                f"bias_inits must have length {n_layers}, got {len(bias_inits)}"
            )

        layers: list[nnx.Linear] = []
        for idx, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            linear_kwargs: dict[str, Any] = {}
            if kernel_inits is not None:
                linear_kwargs["kernel_init"] = kernel_inits[idx]
            if bias_inits is not None:
                linear_kwargs["bias_init"] = bias_inits[idx]
            layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs, **linear_kwargs))

        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)
        return self.layers[-1](x)

    def init_output_bias(self, y_value: Array) -> None:
        """Initialize the output bias to a representative target value.

        Args:
            y_value: Representative target value in output space.

        """
        if len(self.layers) == 0:
            raise ValueError("ReluMLP has no layers to initialize output bias.")
        last_layer = self.layers[-1]
        bias = last_layer.bias
        if bias is None:
            raise ValueError("Output layer has no bias to initialize.")
        bias[...] = jnp.asarray(y_value)

    def get_output_bias(self) -> Array:
        """Return a copy of the current output bias."""
        if len(self.layers) == 0:
            raise ValueError("ReluMLP has no layers.")
        bias = self.layers[-1].bias
        if bias is None:
            raise ValueError("Output layer has no bias.")
        return jnp.array(bias)


class ResidualReluMLP(nnx.Module):
    """Repeated blocks of MLPs with residual connections.
    This is a very standard architecture for building deep networks.
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

        for _ in range(num_blocks):
            self.block_lin_up.append(nnx.Linear(residual_dim, inter_dim, rngs=rngs))
            self.block_lin_down.append(
                nnx.Linear(
                    inter_dim,
                    residual_dim,
                    kernel_init=down_kernel_init,
                    rngs=rngs,
                )
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_dim = residual_dim
        self.inter_dim = inter_dim
        self.num_blocks = num_blocks

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        x = self.in_proj(x)
        for lin_up, lin_down in zip(self.block_lin_up, self.block_lin_down):
            h = lin_up(x)
            h = nnx.relu(h)
            h = lin_down(h)
            x = x + h
        x = self.out_proj(x)
        return x

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


# -----------------------------------------------------------------------------
# Batched prediction (host-side, non-JIT)
# -----------------------------------------------------------------------------


@nnx.jit
def _jit_forward(model: nnx.Module, x: Array) -> Array:
    """JIT-compiled model forward pass."""
    return model(x)  # type: ignore[no-any-return]


def _pick_batch_size(remaining: int) -> int:
    """Choose a fixed batch size to minimise JIT retraces."""
    if remaining >= 256:
        return 256
    if remaining >= 128:
        return 128
    if remaining >= 64:
        return 64
    return remaining


def batched_predict(
    model: nnx.Module,
    x_feat: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Run a model forward pass in fixed-size batches, returning NumPy.

    The model's ``__call__`` is JIT-compiled and retraces whenever the
    input shape changes.  This function chunks the input into a few
    fixed sizes (256, 128, 64, remainder) so JAX compiles at most ~4
    traces regardless of total input size.

    Use this for host-side evaluation, analysis, and snapshot scoring —
    anywhere outside a training loop where batch size may vary.  Do
    **not** call from inside ``@jax.jit`` or ``@nnx.jit`` contexts;
    the Python loop would be unrolled by the tracer.

    Args:
        model: Trained surrogate model (x_feat → y_feat).
        x_feat: Input NumPy array, shape ``(N, D_in)``.
        batch_size: Fixed batch size override.  If ``None``, uses the
            automatic scheme.

    Returns:
        Model predictions, shape ``(N, D_out)``, as a NumPy array.

    """
    if not isinstance(x_feat, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            "batched_predict expects numpy.ndarray input. "
            "Passing jax.Array causes unnecessary device-host round-trips."
        )

    x_np = x_feat
    n = x_np.shape[0]
    if n == 0:
        dummy = jnp.zeros((0, x_np.shape[1]))
        return np.asarray(_jit_forward(model, dummy))

    first_bs = batch_size if batch_size is not None else _pick_batch_size(n)
    first_bs = min(first_bs, n)
    first_out = np.asarray(_jit_forward(model, jnp.asarray(x_np[:first_bs])))

    out = np.empty((n, *first_out.shape[1:]), dtype=first_out.dtype)
    out[:first_bs] = first_out

    offset = first_bs
    remaining = n - first_bs
    while remaining > 0:
        bs = batch_size if batch_size is not None else _pick_batch_size(remaining)
        bs = min(bs, remaining)
        chunk = _jit_forward(model, jnp.asarray(x_np[offset : offset + bs]))
        out[offset : offset + bs] = np.asarray(chunk)
        offset += bs
        remaining -= bs

    return out


__all__ = [
    "ReluMLP",
    "ResidualReluMLP",
    "batched_predict",
]
