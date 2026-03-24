"""Representative diagnostics for residual-network optimization on multiplication."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax import nnx
from flax.nnx import Optimizer

from dnn_arithmetic.loaders import DataIterator
from dnn_arithmetic.models import ReluMLP, ResidualReluMLP, batched_predict
from dnn_arithmetic.plotting import LineSeries, save_line_plot

# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
TRAIN_LOSS_PLOT_PATH = OUTPUT_DIR / "train_loss.png"
TEST_LOSS_PLOT_PATH = OUTPUT_DIR / "test_loss.png"
GRAD_NORM_PLOT_PATH = OUTPUT_DIR / "grad_norm.png"
BRANCH_RATIO_PLOT_PATH = OUTPUT_DIR / "branch_ratio.png"
RESIDUAL_STREAM_DEPTH_PLOT_PATH = OUTPUT_DIR / "residual_stream_depth.png"
UP_GRAD_DEPTH_PLOT_PATH = OUTPUT_DIR / "up_grad_depth.png"
DOWN_GRAD_DEPTH_PLOT_PATH = OUTPUT_DIR / "down_grad_depth.png"

Array = jax.Array | np.ndarray


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for the multiplication benchmark.

    Args:
        x_train: Training inputs.
        y_train: Training targets.
        x_test: Held-out test inputs.
        y_test: Held-out test targets.

    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics for bitwise multiplication predictions.

    Args:
        mse: Mean squared error over output bits.
        bit_accuracy: Fraction of correctly recovered output bits.
        exact_accuracy: Fraction of examples with every output bit correct.
        mean_abs_product_error: Mean absolute integer product error.
        max_abs_product_error: Maximum absolute integer product error.

    """

    mse: float
    bit_accuracy: float
    exact_accuracy: float
    mean_abs_product_error: float
    max_abs_product_error: int


@dataclass(frozen=True)
class RunConfig:
    """Experiment-wide configuration.

    Args:
        bit_width: Operand width in bits.
        train_fraction: Fraction of examples used for training.
        steps: Number of optimizer steps.
        batch_size: Training batch size.
        probe_batch_size: Fixed held-out train batch used for diagnostics.
        learning_rate: Default peak learning rate.
        weight_decay: AdamW kernel weight decay.
        grad_clip_norm: Global gradient clip norm.
        log_every: Interval for train/test loss logging.
        diagnostic_steps: Explicit steps at which to collect detailed snapshots.
        seed: Base seed controlling split, diagnostics batch, and model init.

    """

    bit_width: int
    train_fraction: float
    steps: int
    batch_size: int
    probe_batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    diagnostic_steps: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class ExperimentSpec:
    """One model configuration for the diagnostic sweep.

    Args:
        name: Display name for logs and outputs.
        model_kind: Either "mlp" or "residual".
        learning_rate: Peak AdamW learning rate.
        init_output_bias: Whether to initialize the output bias to the target mean.
        hidden_dims: Hidden widths for MLP experiments.
        residual_dim: Width of the residual stream.
        inter_dim: Bottleneck width inside each residual block.
        num_blocks: Number of residual blocks.
        down_init: Down-projection initialization kind for residual models.

    """

    name: str
    model_kind: str
    learning_rate: float
    init_output_bias: bool
    hidden_dims: tuple[int, ...] = ()
    residual_dim: int = 0
    inter_dim: int = 0
    num_blocks: int = 0
    down_init: str = "small"


@dataclass(frozen=True)
class LayerSnapshot:
    """Depth-wise activation and gradient summary for one layer group.

    Args:
        name: Layer or block identifier.
        stream_rms: RMS of the incoming stream to this unit.
        branch_rms: RMS of the unit output before it is propagated onward.
        branch_to_stream_ratio: Ratio of branch RMS to stream RMS.
        kernel_grad_norm: L2 norm of the kernel gradient.
        kernel_param_norm: L2 norm of the kernel parameters.

    """

    name: str
    stream_rms: float
    branch_rms: float
    branch_to_stream_ratio: float
    kernel_grad_norm: float
    kernel_param_norm: float


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Detailed diagnostic snapshot collected during real training.

    Args:
        step: Optimization step. Step 0 is before any updates.
        loss: Probe-batch mean squared error.
        global_grad_norm: Global L2 norm of all gradients before clipping.
        clip_applied: Whether global clipping would activate at this step.
        layer_stats: Depth-wise statistics.

    """

    step: int
    loss: float
    global_grad_norm: float
    clip_applied: bool
    layer_stats: list[LayerSnapshot]


@dataclass(frozen=True)
class ExperimentRecord:
    """Persisted results for a single architecture run.

    Args:
        spec: Experiment specification.
        train_metrics: Metrics on the training split.
        test_metrics: Metrics on the held-out split.
        step_history: Steps corresponding to train/test loss history.
        train_loss_history: Mean training loss over each logging interval.
        test_loss_history: Test loss at each logging interval.
        elapsed: Wall-clock training time in seconds.
        final_train_loss: Final logged train loss.
        diagnostic_snapshots: Detailed probe-batch diagnostics.

    """

    spec: ExperimentSpec
    train_metrics: EvalMetrics
    test_metrics: EvalMetrics
    step_history: list[int]
    train_loss_history: list[float]
    test_loss_history: list[float]
    elapsed: float
    final_train_loss: float
    diagnostic_snapshots: list[DiagnosticSnapshot]


def _host_array(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True)


def _host_float(value: Any) -> float:
    return float(np.array(jax.device_get(value), dtype=np.float64))


def _state_get(tree: Any, *keys: Any) -> Any:
    """Index a nested state tree while keeping static typing relaxed.

    Args:
        tree: Nested mapping or sequence.
        *keys: Keys or indices to traverse.

    Returns:
        Selected subtree.

    """

    current = tree
    for key in keys:
        current = current[key]
    return current


def _variable_to_host_array(variable: Any) -> np.ndarray:
    """Convert a single NNX variable-like value into a NumPy array.

    Args:
        variable: NNX variable or array-like value.

    Returns:
        Host NumPy array.

    """

    return _host_array(variable[...])


def _zero_init(
    key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32
) -> jax.Array:
    del key
    return jnp.zeros(shape, dtype=dtype)


def _scaled_lecun_normal(scale: float) -> nnx.Initializer:
    """Return a LeCun-normal initializer scaled by a fixed factor.

    Args:
        scale: Scalar multiplier applied to the sampled weights.

    Returns:
        Initializer callable.

    """

    base_init = jax.nn.initializers.lecun_normal()

    def init(
        key: jax.Array, shape: Sequence[int], dtype: Any = jnp.float32
    ) -> jax.Array:
        return scale * base_init(key, shape, dtype)

    return init


def _resolve_down_kernel_init(kind: str) -> nnx.Initializer:
    """Map an initializer name to a residual down-projection initializer.

    Args:
        kind: One of "zero", "small", or "lecun".

    Returns:
        Matching initializer.

    Raises:
        ValueError: If the name is unknown.

    """

    if kind == "zero":
        return _zero_init
    if kind == "small":
        return _scaled_lecun_normal(0.1)
    if kind == "lecun":
        return jax.nn.initializers.lecun_normal()
    raise ValueError(f"Unknown down_init: {kind}")


def _int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Encode integers into little-endian bit vectors.

    Args:
        values: Integer array of shape (N,).
        width: Number of output bits.

    Returns:
        Float32 bit matrix of shape (N, width).

    """

    shifts = np.arange(width, dtype=np.uint32)
    bit_values = (values[:, None] >> shifts[None, :]) & 1
    return bit_values.astype(np.float32)


def _bits_to_int(bit_vectors: np.ndarray) -> np.ndarray:
    """Decode little-endian bit vectors back into integers.

    Args:
        bit_vectors: Binary array of shape (N, width).

    Returns:
        Integer values of shape (N,).

    """

    powers = (1 << np.arange(bit_vectors.shape[1], dtype=np.uint32)).astype(np.uint32)
    return np.sum(bit_vectors.astype(np.uint32) * powers[None, :], axis=1)


def _build_dataset(bit_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct exhaustive multiplication examples for a fixed bit width.

    Args:
        bit_width: Operand width in bits.

    Returns:
        Tuple (x, y) with concatenated operand bits and product bits.

    """

    max_value = 1 << bit_width
    lhs = np.repeat(np.arange(max_value, dtype=np.uint32), max_value)
    rhs = np.tile(np.arange(max_value, dtype=np.uint32), max_value)
    products = lhs * rhs

    x_feat = np.concatenate(
        [_int_to_bits(lhs, bit_width), _int_to_bits(rhs, bit_width)],
        axis=1,
    )
    y_feat = _int_to_bits(products, bit_width * 2)
    return x_feat, y_feat


def _split_dataset(
    x_feat: np.ndarray,
    y_feat: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Shuffle and split the dataset into train and test subsets.

    Args:
        x_feat: Input features.
        y_feat: Target features.
        train_fraction: Fraction of examples used for training.
        seed: NumPy RNG seed.

    Returns:
        Dataclass containing train and test arrays.

    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(x_feat))
    split_index = int(len(x_feat) * train_fraction)

    train_index = permutation[:split_index]
    test_index = permutation[split_index:]
    return DatasetSplit(
        x_train=x_feat[train_index],
        y_train=y_feat[train_index],
        x_test=x_feat[test_index],
        y_test=y_feat[test_index],
    )


def _make_probe_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic training batch for diagnostics.

    Args:
        x_train: Training inputs.
        y_train: Training targets.
        batch_size: Number of examples in the probe batch.
        seed: NumPy RNG seed.

    Returns:
        Fixed diagnostic batch drawn from the training split.

    """

    rng = np.random.default_rng(seed)
    probe_size = min(batch_size, len(x_train))
    indices = rng.permutation(len(x_train))[:probe_size]
    return x_train[indices], y_train[indices]


def _make_diagnostic_steps(steps: int, log_every: int) -> tuple[int, ...]:
    """Construct explicit diagnostic checkpoints.

    Args:
        steps: Total optimization steps.
        log_every: Regular logging cadence.

    Returns:
        Sorted tuple of unique checkpoint steps, including step 0 and the end.

    """

    early_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    regular_steps = list(range(log_every, steps + 1, log_every))
    checkpoints = {step for step in early_steps + regular_steps if 0 <= step <= steps}
    checkpoints.add(steps)
    return tuple(sorted(checkpoints))


def _make_experiments(config: RunConfig) -> list[ExperimentSpec]:
    """Return the compact residual-diagnostics sweep.

    Args:
        config: Experiment configuration.

    Returns:
        Ordered list of model specs.

    """

    base_lr = config.learning_rate
    return [
        ExperimentSpec(
            name="mlp-256x4-bias",
            model_kind="mlp",
            learning_rate=base_lr,
            init_output_bias=True,
            hidden_dims=(256, 256, 256, 256),
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias-zeroinit",
            model_kind="residual",
            learning_rate=base_lr,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            down_init="zero",
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias-smallinit",
            model_kind="residual",
            learning_rate=base_lr,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            down_init="small",
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias-lecuninit",
            model_kind="residual",
            learning_rate=base_lr,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            down_init="lecun",
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias-smallinit-lr1e-3",
            model_kind="residual",
            learning_rate=1e-3,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            down_init="small",
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias-smallinit-lr4e-3",
            model_kind="residual",
            learning_rate=4e-3,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            down_init="small",
        ),
        ExperimentSpec(
            name="res-256-b128-x8-bias-smallinit",
            model_kind="residual",
            learning_rate=base_lr,
            init_output_bias=True,
            residual_dim=256,
            inter_dim=128,
            num_blocks=8,
            down_init="small",
        ),
    ]


def _build_model(
    spec: ExperimentSpec,
    *,
    in_dim: int,
    out_dim: int,
    target_bias: np.ndarray,
    key: jax.Array,
) -> nnx.Module:
    """Create a model instance for one experiment spec.

    Args:
        spec: Model specification.
        in_dim: Input dimension.
        out_dim: Output dimension.
        target_bias: Mean target vector used for optional bias init.
        key: JAX initialization key.

    Returns:
        Initialized NNX model.

    """

    rngs = nnx.Rngs(key)
    if spec.model_kind == "mlp":
        model = ReluMLP([in_dim, *spec.hidden_dims, out_dim], rngs=rngs)
    elif spec.model_kind == "residual":
        model = ResidualReluMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            residual_dim=spec.residual_dim,
            inter_dim=spec.inter_dim,
            num_blocks=spec.num_blocks,
            down_kernel_init=_resolve_down_kernel_init(spec.down_init),
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown model_kind: {spec.model_kind}")

    if spec.init_output_bias:
        model.init_output_bias(target_bias)
    return model


def _find_valid_batch_size(requested: int, data_len: int) -> int:
    """Find a non-ragged batch size that fits the available data.

    Args:
        requested: Desired batch size.
        data_len: Number of available examples.

    Returns:
        Best valid batch size.

    """

    if data_len == 0:
        raise ValueError("Cannot create batches from empty data.")
    if data_len >= requested:
        return requested
    if data_len >= 64:
        return (data_len // 64) * 64
    for size in (32, 16, 8, 4, 2, 1):
        if data_len >= size:
            return size
    return 1


def _cycle_batches(iterator: DataIterator):
    while True:
        for batch in iterator:
            yield batch


def _create_weight_decay_mask_for_params(params: Any) -> Any:
    """Apply AdamW decay only to kernel parameters.

    Args:
        params: Parameter pytree.

    Returns:
        Boolean pytree mask.

    """

    params = nnx.pure(params)

    def is_kernel(path: tuple[Any, ...], _: jax.Array) -> bool:
        if path and isinstance(path[-1], jtu.DictKey):
            return path[-1].key == "kernel"
        return False

    return jtu.tree_map_with_path(is_kernel, params)


def _make_optimizer_tx(
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    total_steps: int,
) -> optax.GradientTransformation:
    """Build the benchmark-matching AdamW schedule for one run.

    Args:
        learning_rate: Peak learning rate.
        weight_decay: AdamW kernel weight decay.
        grad_clip_norm: Global clip norm.
        total_steps: Number of optimizer steps.

    Returns:
        Optax gradient transformation.

    """

    warmup_steps = min(200, max(1, total_steps // 10))
    constant_steps = min(1000, max(1, total_steps // 3))
    remaining_after_warmup = max(0, total_steps - warmup_steps)
    constant_steps = min(constant_steps, remaining_after_warmup)
    decay_steps = max(1, total_steps - warmup_steps - constant_steps)

    schedules = [
        optax.linear_schedule(
            init_value=learning_rate / 1000.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        ),
        optax.constant_schedule(learning_rate),
        optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=decay_steps,
            alpha=0.05,
        ),
    ]
    learning_rate_schedule = optax.join_schedules(
        schedules,
        [warmup_steps, warmup_steps + constant_steps],
    )

    adamw = optax.adamw(
        learning_rate=learning_rate_schedule,
        b1=0.9,
        b2=0.999,
        weight_decay=weight_decay,
        mask=_create_weight_decay_mask_for_params,
    )
    if grad_clip_norm > 0:
        return optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)
    return adamw


@nnx.jit
def _l2_loss(model: nnx.Module, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean squared error summed over the output dimension.

    Args:
        model: Model to evaluate.
        x: Input batch.
        y: Target batch.

    Returns:
        Scalar loss.

    """

    y_pred = model(x)
    per_sample = jnp.sum(jnp.square(y_pred - y), axis=-1)
    return jnp.mean(per_sample)


_l2_val_and_grad = nnx.value_and_grad(_l2_loss)


@nnx.jit
def _train_step(
    model: nnx.Module,
    optimizer: Optimizer[nnx.Module],
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> jax.Array:
    """Run one optimizer step and return the batch loss.

    Args:
        model: Model being optimized.
        optimizer: Wrapped Optax optimizer.
        x_batch: Input batch.
        y_batch: Target batch.

    Returns:
        Scalar batch loss.

    """

    loss, grads = _l2_val_and_grad(model, x_batch, y_batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _eval_step(model: nnx.Module, x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
    """Evaluate a batch loss without updating the model.

    Args:
        model: Model to evaluate.
        x_batch: Input batch.
        y_batch: Target batch.

    Returns:
        Scalar batch loss.

    """

    return _l2_loss(model, x_batch, y_batch)


def _evaluate_loss(
    model: nnx.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    batch_size: int,
    key: jax.Array,
) -> float:
    """Evaluate a model on a full split.

    Args:
        model: Model to evaluate.
        x_eval: Evaluation inputs.
        y_eval: Evaluation targets.
        batch_size: Evaluation batch size.
        key: JAX key for batch iterator shuffling.

    Returns:
        Mean loss over the split.

    """

    iterator = DataIterator([x_eval, y_eval], batch_size=batch_size, key=key)
    total_loss = 0.0
    total_examples = 0
    for x_batch, y_batch in iterator:
        batch_loss = float(_eval_step(model, x_batch, y_batch))
        batch_examples = int(x_batch.shape[0])
        total_loss += batch_loss * batch_examples
        total_examples += batch_examples
    return total_loss / total_examples


def _state_to_array_tree(state: Any) -> Any:
    """Convert an NNX state tree into plain arrays.

    Args:
        state: NNX state tree.

    Returns:
        Matching pytree of NumPy arrays.

    """

    return jax.tree_util.tree_map(_variable_to_host_array, state)


def _tree_l2_norm(tree: Any) -> float:
    """Compute the global L2 norm of an array pytree.

    Args:
        tree: Pytree of arrays.

    Returns:
        Global L2 norm.

    """

    leaves = jax.tree_util.tree_leaves(tree)
    sum_sq = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf, dtype=np.float64)
        sum_sq += float(np.sum(np.square(arr)))
    return float(np.sqrt(sum_sq))


def _rms(values: Array) -> float:
    """Return root-mean-square magnitude.

    Args:
        values: Array-like values.

    Returns:
        RMS magnitude.

    """

    arr = np.asarray(_host_array(values), dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(arr))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio for diagnostic summaries.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.

    Returns:
        Stable ratio, or 0 if the denominator is non-positive.

    """

    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _collect_mlp_layer_stats(
    model: ReluMLP,
    grad_state: Any,
    x_batch: np.ndarray,
) -> list[LayerSnapshot]:
    """Collect activation and gradient summaries for an MLP.

    Args:
        model: MLP model.
        grad_state: Gradient state tree.
        x_batch: Fixed probe batch.

    Returns:
        Per-layer statistics.

    """

    x = jnp.asarray(x_batch)
    param_state: Any = nnx.state(model, nnx.Param)
    stats: list[LayerSnapshot] = []
    for index, layer in enumerate(model.layers[:-1]):
        stream = x
        h = nnx.relu(layer(stream))
        stats.append(
            LayerSnapshot(
                name=f"layer_{index}",
                stream_rms=_rms(stream),
                branch_rms=_rms(h),
                branch_to_stream_ratio=_safe_ratio(_rms(h), _rms(stream)),
                kernel_grad_norm=_tree_l2_norm(
                    _state_to_array_tree(_state_get(grad_state, "layers", index, "kernel"))
                ),
                kernel_param_norm=_tree_l2_norm(
                    _state_to_array_tree(_state_get(param_state, "layers", index, "kernel"))
                ),
            )
        )
        x = h

    last_index = len(model.layers) - 1
    final_stream = x
    final_out = model.layers[-1](final_stream)
    stats.append(
        LayerSnapshot(
            name=f"layer_{last_index}",
            stream_rms=_rms(final_stream),
            branch_rms=_rms(final_out),
            branch_to_stream_ratio=_safe_ratio(_rms(final_out), _rms(final_stream)),
            kernel_grad_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(grad_state, "layers", last_index, "kernel"))
            ),
            kernel_param_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(param_state, "layers", last_index, "kernel"))
            ),
        )
    )
    return stats


def _collect_residual_layer_stats(
    model: ResidualReluMLP,
    grad_state: Any,
    x_batch: np.ndarray,
) -> list[LayerSnapshot]:
    """Collect activation and gradient summaries for a residual MLP.

    Args:
        model: Residual model.
        grad_state: Gradient state tree.
        x_batch: Fixed probe batch.

    Returns:
        Per-layer-group statistics.

    """

    x = model.in_proj(jnp.asarray(x_batch))
    param_state: Any = nnx.state(model, nnx.Param)
    stats: list[LayerSnapshot] = [
        LayerSnapshot(
            name="in_proj",
            stream_rms=_rms(x_batch),
            branch_rms=_rms(x),
            branch_to_stream_ratio=_safe_ratio(_rms(x), _rms(x_batch)),
            kernel_grad_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(grad_state, "in_proj", "kernel"))
            ),
            kernel_param_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(param_state, "in_proj", "kernel"))
            ),
        )
    ]
    for index, (lin_up, lin_down) in enumerate(
        zip(model.block_lin_up, model.block_lin_down)
    ):
        stream = x
        up_act = nnx.relu(lin_up(stream))
        residual = lin_down(up_act)
        stats.append(
            LayerSnapshot(
                name=f"block_{index}_up",
                stream_rms=_rms(stream),
                branch_rms=_rms(up_act),
                branch_to_stream_ratio=_safe_ratio(_rms(up_act), _rms(stream)),
                kernel_grad_norm=_tree_l2_norm(
                    _state_to_array_tree(
                        _state_get(grad_state, "block_lin_up", index, "kernel")
                    )
                ),
                kernel_param_norm=_tree_l2_norm(
                    _state_to_array_tree(
                        _state_get(param_state, "block_lin_up", index, "kernel")
                    )
                ),
            )
        )
        stats.append(
            LayerSnapshot(
                name=f"block_{index}_down",
                stream_rms=_rms(stream),
                branch_rms=_rms(residual),
                branch_to_stream_ratio=_safe_ratio(_rms(residual), _rms(stream)),
                kernel_grad_norm=_tree_l2_norm(
                    _state_to_array_tree(
                        _state_get(grad_state, "block_lin_down", index, "kernel")
                    )
                ),
                kernel_param_norm=_tree_l2_norm(
                    _state_to_array_tree(
                        _state_get(param_state, "block_lin_down", index, "kernel")
                    )
                ),
            )
        )
        x = stream + residual
    final_out = model.out_proj(x)
    stats.append(
        LayerSnapshot(
            name="out_proj",
            stream_rms=_rms(x),
            branch_rms=_rms(final_out),
            branch_to_stream_ratio=_safe_ratio(_rms(final_out), _rms(x)),
            kernel_grad_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(grad_state, "out_proj", "kernel"))
            ),
            kernel_param_norm=_tree_l2_norm(
                _state_to_array_tree(_state_get(param_state, "out_proj", "kernel"))
            ),
        )
    )
    return stats


def _collect_snapshot(
    model: nnx.Module,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    *,
    step: int,
    grad_clip_norm: float,
) -> DiagnosticSnapshot:
    """Collect a detailed diagnostic snapshot for the current model state.

    Args:
        model: Model under inspection.
        probe_x: Fixed diagnostic inputs.
        probe_y: Fixed diagnostic targets.
        step: Current optimization step.
        grad_clip_norm: Clip threshold used for the run.

    Returns:
        Diagnostic snapshot.

    """

    loss, grads = _l2_val_and_grad(model, probe_x, probe_y)
    grad_state = nnx.state(grads)
    grad_tree = _state_to_array_tree(grad_state)
    global_grad_norm = _tree_l2_norm(grad_tree)

    if isinstance(model, ReluMLP):
        layer_stats = _collect_mlp_layer_stats(model, grad_state, probe_x)
    elif isinstance(model, ResidualReluMLP):
        layer_stats = _collect_residual_layer_stats(model, grad_state, probe_x)
    else:
        raise TypeError(f"Unsupported model type: {type(model)!r}")

    return DiagnosticSnapshot(
        step=step,
        loss=_host_float(loss),
        global_grad_norm=global_grad_norm,
        clip_applied=bool(grad_clip_norm > 0.0 and global_grad_norm > grad_clip_norm),
        layer_stats=layer_stats,
    )


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    """Compute regression and exactness metrics for bit predictions.

    Args:
        y_true: Ground-truth bit vectors.
        y_pred: Continuous model predictions.

    Returns:
        Aggregate evaluation metrics.

    """

    mse = float(np.mean(np.square(y_pred - y_true)))
    pred_bits = (y_pred >= 0.5).astype(np.uint8)
    true_bits = y_true.astype(np.uint8)
    bit_matches = np.asarray(np.equal(pred_bits, true_bits), dtype=np.float32)
    exact_matches = np.asarray(np.all(pred_bits == true_bits, axis=1), dtype=np.float32)
    bit_accuracy = float(np.mean(bit_matches))
    exact_accuracy = float(np.mean(exact_matches))

    pred_values = _bits_to_int(pred_bits)
    true_values = _bits_to_int(true_bits)
    abs_product_error = np.abs(
        pred_values.astype(np.int64) - true_values.astype(np.int64)
    )
    return EvalMetrics(
        mse=mse,
        bit_accuracy=bit_accuracy,
        exact_accuracy=exact_accuracy,
        mean_abs_product_error=float(np.mean(abs_product_error)),
        max_abs_product_error=int(np.max(abs_product_error)),
    )


def _run_experiment(
    spec: ExperimentSpec,
    split: DatasetSplit,
    config: RunConfig,
    *,
    seed: int,
) -> ExperimentRecord:
    """Train one model and return benchmark plus diagnostic results.

    Args:
        spec: Experiment specification.
        split: Train/test split.
        config: Global run configuration.
        seed: Base seed for this model run.

    Returns:
        Persistable experiment record.

    """

    init_key, train_key, test_key = jax.random.split(jax.random.key(seed), 3)
    probe_x, probe_y = _make_probe_batch(
        split.x_train,
        split.y_train,
        batch_size=config.probe_batch_size,
        seed=seed + 10_000,
    )
    model = _build_model(
        spec,
        in_dim=split.x_train.shape[1],
        out_dim=split.y_train.shape[1],
        target_bias=split.y_train.mean(axis=0),
        key=init_key,
    )
    optimizer = Optimizer(
        model,
        _make_optimizer_tx(
            learning_rate=spec.learning_rate,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
            total_steps=config.steps,
        ),
        wrt=nnx.Param,
    )

    train_batch_size = _find_valid_batch_size(config.batch_size, len(split.x_train))
    test_batch_size = _find_valid_batch_size(config.batch_size, len(split.x_test))
    train_iter = DataIterator(
        [split.x_train, split.y_train],
        batch_size=train_batch_size,
        key=train_key,
    )
    batch_iter = _cycle_batches(train_iter)

    step_history: list[int] = []
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    diagnostic_snapshots: list[DiagnosticSnapshot] = []
    diagnostic_step_set = set(config.diagnostic_steps)

    if 0 in diagnostic_step_set:
        diagnostic_snapshots.append(
            _collect_snapshot(
                model,
                probe_x,
                probe_y,
                step=0,
                grad_clip_norm=config.grad_clip_norm,
            )
        )

    interval_loss_total = 0.0
    interval_batches = 0
    start_time = time.perf_counter()
    for step in range(config.steps):
        x_batch, y_batch = next(batch_iter)
        batch_loss = float(_train_step(model, optimizer, x_batch, y_batch))
        interval_loss_total += batch_loss
        interval_batches += 1

        step_num = step + 1
        if step_num in diagnostic_step_set:
            diagnostic_snapshots.append(
                _collect_snapshot(
                    model,
                    probe_x,
                    probe_y,
                    step=step_num,
                    grad_clip_norm=config.grad_clip_norm,
                )
            )

        if step_num % config.log_every != 0:
            continue

        train_loss_history.append(interval_loss_total / interval_batches)
        step_history.append(step_num)
        interval_loss_total = 0.0
        interval_batches = 0
        test_loss_history.append(
            _evaluate_loss(
                model,
                split.x_test,
                split.y_test,
                batch_size=test_batch_size,
                key=test_key,
            )
        )

    if interval_batches > 0:
        train_loss_history.append(interval_loss_total / interval_batches)
        step_history.append(config.steps)
        test_loss_history.append(
            _evaluate_loss(
                model,
                split.x_test,
                split.y_test,
                batch_size=test_batch_size,
                key=test_key,
            )
        )

    elapsed = time.perf_counter() - start_time
    train_pred = batched_predict(model, split.x_train, batch_size=config.batch_size)
    test_pred = batched_predict(model, split.x_test, batch_size=config.batch_size)
    train_metrics = _evaluate_predictions(split.y_train, train_pred)
    test_metrics = _evaluate_predictions(split.y_test, test_pred)
    final_train_loss = (
        train_loss_history[-1] if train_loss_history else float("nan")
    )
    return ExperimentRecord(
        spec=spec,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        step_history=step_history,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        elapsed=elapsed,
        final_train_loss=float(final_train_loss),
        diagnostic_snapshots=diagnostic_snapshots,
    )


def _format_metrics(split_name: str, metrics: EvalMetrics) -> str:
    """Format metrics for terminal or summary output.

    Args:
        split_name: Name of the split.
        metrics: Aggregate metrics.

    Returns:
        Formatted metric string.

    """

    return (
        f"{split_name}: mse={metrics.mse:.3e}, "
        f"bit_acc={metrics.bit_accuracy:.4f}, "
        f"exact_acc={metrics.exact_accuracy:.4f}, "
        f"mean_abs_prod_err={metrics.mean_abs_product_error:.3f}, "
        f"max_abs_prod_err={metrics.max_abs_product_error}"
    )


def _record_to_lines(record: ExperimentRecord, num_steps: int) -> list[str]:
    """Render a compact text summary for one experiment.

    Args:
        record: Experiment result.
        num_steps: Number of optimizer steps.

    Returns:
        Summary lines.

    """

    config_suffix = f"lr={record.spec.learning_rate:.2e}"
    if record.spec.model_kind == "residual":
        config_suffix += f", down_init={record.spec.down_init}"

    return [
        record.spec.name,
        (
            f"  steps={num_steps}, elapsed={record.elapsed:.2f}s, "
            f"final_train_loss={record.final_train_loss:.3e}, "
            f"{config_suffix}"
        ),
        f"  {_format_metrics('train', record.train_metrics)}",
        f"  {_format_metrics('test ', record.test_metrics)}",
    ]


def _get_snapshot(record: ExperimentRecord, step: int) -> DiagnosticSnapshot:
    """Fetch a diagnostic snapshot by step.

    Args:
        record: Experiment record.
        step: Optimization step.

    Returns:
        Matching snapshot.

    Raises:
        ValueError: If the snapshot is missing.

    """

    for snapshot in record.diagnostic_snapshots:
        if snapshot.step == step:
            return snapshot
    raise ValueError(f"Missing snapshot for step {step} in {record.spec.name}.")


def _mean_kernel_grad(snapshot: DiagnosticSnapshot, name_fragment: str) -> float:
    """Average kernel gradient norm for matching layers.

    Args:
        snapshot: Diagnostic snapshot.
        name_fragment: Substring used to match layer names.

    Returns:
        Mean gradient norm across matching layers.

    """

    matches = [
        stat.kernel_grad_norm
        for stat in snapshot.layer_stats
        if name_fragment in stat.name
    ]
    if not matches:
        return 0.0
    return float(np.mean(matches))


def _mean_branch_ratio(snapshot: DiagnosticSnapshot, name_fragment: str) -> float:
    """Average branch-to-stream ratio for matching layers.

    Args:
        snapshot: Diagnostic snapshot.
        name_fragment: Substring used to match layer names.

    Returns:
        Mean branch ratio across matching layers.

    """

    matches = [
        stat.branch_to_stream_ratio
        for stat in snapshot.layer_stats
        if name_fragment in stat.name
    ]
    if not matches:
        return 0.0
    return float(np.mean(matches))


def _count_clipped_snapshots(record: ExperimentRecord) -> int:
    """Count how many diagnostic checkpoints exceeded the clip threshold.

    Args:
        record: Experiment record.

    Returns:
        Number of clipped checkpoints.

    """

    return sum(1 for snapshot in record.diagnostic_snapshots if snapshot.clip_applied)


def _block_index(stat_name: str) -> int | None:
    """Extract the residual block index from a layer-stat name.

    Args:
        stat_name: Layer-stat name such as ``block_3_up``.

    Returns:
        Parsed block index, or ``None`` if the name is not block-scoped.

    """

    parts = stat_name.split("_")
    if len(parts) < 3 or parts[0] != "block":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _block_metric_values(
    snapshot: DiagnosticSnapshot,
    *,
    name_suffix: str,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect one block-wise metric vector from a diagnostic snapshot.

    Args:
        snapshot: Diagnostic snapshot.
        name_suffix: Required suffix such as ``"_up"`` or ``"_down"``.
        metric_name: Metric field on ``LayerSnapshot``.

    Returns:
        Pair ``(block_indices, metric_values)``.

    """

    matched: list[tuple[int, float]] = []
    for stat in snapshot.layer_stats:
        if not stat.name.endswith(name_suffix):
            continue
        block_index = _block_index(stat.name)
        if block_index is None:
            continue
        matched.append((block_index, float(getattr(stat, metric_name))))
    matched.sort(key=lambda item: item[0])
    return (
        np.asarray([item[0] for item in matched], dtype=np.float64),
        np.asarray([item[1] for item in matched], dtype=np.float64),
    )


def _depth_profile_steps(total_steps: int) -> list[int]:
    """Choose representative checkpoints for depth-profile plots.

    Args:
        total_steps: Total number of optimization steps.

    Returns:
        Representative checkpoint steps.

    """

    return sorted({min(100, total_steps), min(1000, total_steps), total_steps})


def _selected_depth_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """Choose the key residual runs to visualize by depth.

    Args:
        records: All experiment records.

    Returns:
        Subset of residual runs used for depth-profile plots.

    """

    keep_names = {
        "res-256-b64-x8-bias-zeroinit",
        "res-256-b64-x8-bias-smallinit",
        "res-256-b128-x8-bias-smallinit",
    }
    return [record for record in records if record.spec.name in keep_names]


def _depth_line_series(
    records: list[ExperimentRecord],
    *,
    steps: list[int],
    name_suffix: str,
    metric_name: str,
) -> list[LineSeries]:
    """Build block-depth line series for selected records and checkpoints.

    Args:
        records: Residual experiment records to visualize.
        steps: Representative checkpoints.
        name_suffix: Required layer-stat suffix such as ``"_up"``.
        metric_name: Metric field on ``LayerSnapshot``.

    Returns:
        Plot-ready line series.

    """

    linestyles = ["-", "--", ":"]
    series: list[LineSeries] = []
    for record in records:
        for index, step in enumerate(steps):
            snapshot = _get_snapshot(record, step)
            x_values, y_values = _block_metric_values(
                snapshot,
                name_suffix=name_suffix,
                metric_name=metric_name,
            )
            series.append(
                LineSeries(
                    label=f"{record.spec.name} @ {step}",
                    x_values=x_values,
                    y_values=y_values,
                    linestyle=linestyles[index % len(linestyles)],
                )
            )
    return series


def _stream_endpoint_summary(record: ExperimentRecord, step: int) -> str:
    """Summarize early-vs-late residual-stream RMS and gradient scales.

    Args:
        record: Residual experiment record.
        step: Diagnostic checkpoint.

    Returns:
        Compact text summary for the first and last residual block.

    """

    snapshot = _get_snapshot(record, step)
    _, stream_values = _block_metric_values(
        snapshot,
        name_suffix="_down",
        metric_name="stream_rms",
    )
    _, up_grad_values = _block_metric_values(
        snapshot,
        name_suffix="_up",
        metric_name="kernel_grad_norm",
    )
    _, down_grad_values = _block_metric_values(
        snapshot,
        name_suffix="_down",
        metric_name="kernel_grad_norm",
    )
    return (
        f"  step={step}: stream_rms[first,last]=({stream_values[0]:.3e}, {stream_values[-1]:.3e}), "
        f"up_grad[first,last]=({up_grad_values[0]:.3e}, {up_grad_values[-1]:.3e}), "
        f"down_grad[first,last]=({down_grad_values[0]:.3e}, {down_grad_values[-1]:.3e})"
    )


def _snapshot_line(record: ExperimentRecord, step: int) -> str:
    """Render a compact snapshot summary line.

    Args:
        record: Experiment record.
        step: Requested snapshot step.

    Returns:
        Human-readable snapshot line.

    """

    snapshot = _get_snapshot(record, step)
    if record.spec.model_kind == "residual":
        branch_ratio = _mean_branch_ratio(snapshot, "_down")
        up_grad = _mean_kernel_grad(snapshot, "_up")
        down_grad = _mean_kernel_grad(snapshot, "_down")
        return (
            f"  step={step}: probe_loss={snapshot.loss:.3e}, grad_norm={snapshot.global_grad_norm:.3e}, "
            f"clip={snapshot.clip_applied}, mean_up_grad={up_grad:.3e}, "
            f"mean_down_grad={down_grad:.3e}, mean_down_branch_ratio={branch_ratio:.3e}"
        )
    return (
        f"  step={step}: probe_loss={snapshot.loss:.3e}, grad_norm={snapshot.global_grad_norm:.3e}, "
        f"clip={snapshot.clip_applied}"
    )


def _build_summary_lines(
    config: RunConfig,
    split: DatasetSplit,
    records: list[ExperimentRecord],
) -> list[str]:
    """Build the human-readable summary output.

    Args:
        config: Global run configuration.
        split: Dataset split.
        records: Experiment results.

    Returns:
        Summary lines.

    """

    by_name = {record.spec.name: record for record in records}
    baseline = by_name["res-256-b64-x8-bias-zeroinit"]
    small_init = by_name["res-256-b64-x8-bias-smallinit"]
    lecun_init = by_name["res-256-b64-x8-bias-lecuninit"]
    low_lr = by_name["res-256-b64-x8-bias-smallinit-lr1e-3"]
    high_lr = by_name["res-256-b64-x8-bias-smallinit-lr4e-3"]
    wider = by_name["res-256-b128-x8-bias-smallinit"]
    mlp = by_name["mlp-256x4-bias"]

    baseline_step100 = _get_snapshot(baseline, min(100, config.steps))
    small_init_step100 = _get_snapshot(small_init, min(100, config.steps))
    baseline_step1000 = _get_snapshot(baseline, min(1000, config.steps))
    small_init_step1000 = _get_snapshot(small_init, min(1000, config.steps))
    best_residual = min(
        (record for record in records if record.spec.model_kind == "residual"),
        key=lambda record: record.final_train_loss,
    )

    summary_lines = [
        (
            f"Residual diagnostics: {config.bit_width}-bit unsigned multiplication, "
            f"{len(split.x_train)} train / {len(split.x_test)} test samples"
        ),
        (
            f"Training config: steps={config.steps}, batch_size={config.batch_size}, "
            f"probe_batch_size={config.probe_batch_size}, base_lr={config.learning_rate:.2e}, "
            f"wd={config.weight_decay:.2e}, clip={config.grad_clip_norm:.2f}"
        ),
        (
            f"Diagnostic checkpoints: {', '.join(str(step) for step in config.diagnostic_steps)}"
        ),
        "",
        "Key findings",
        (
            "- Zero-init does change the early dynamics, but not by much after the first phase: "
            f"at step {min(100, config.steps)} the mean residual-up grad is "
            f"{_mean_kernel_grad(baseline_step100, '_up'):.3e} for zero-init versus "
            f"{_mean_kernel_grad(small_init_step100, '_up'):.3e} for small-init, and by step "
            f"{min(1000, config.steps)} the mean down-branch ratio is "
            f"{_mean_branch_ratio(baseline_step1000, '_down'):.3e} versus "
            f"{_mean_branch_ratio(small_init_step1000, '_down'):.3e}."
        ),
        (
            "- Earlier-versus-later block dynamics are now reported explicitly: see the depth profiles "
            "for residual-stream RMS, up-projection gradient norm, and down-projection gradient norm "
            "at steps 100, 1000, and 6000."
        ),
        (
            "- Under the full training schedule, zero-init residual ends at "
            f"final_train_loss={baseline.final_train_loss:.3e}, while small-init ends at "
            f"{small_init.final_train_loss:.3e} and lecun-init ends at {lecun_init.final_train_loss:.3e}."
        ),
        (
            "- Clipping is secondary unless it shows up often: clipped diagnostic checkpoints "
            f"zero-init={_count_clipped_snapshots(baseline)}/{len(baseline.diagnostic_snapshots)}, "
            f"small-init={_count_clipped_snapshots(small_init)}/{len(small_init.diagnostic_snapshots)}."
        ),
        (
            "- Learning-rate sensitivity is measured only after fixing init: small-init @1e-3 ends at "
            f"{low_lr.final_train_loss:.3e}, @2e-3 at {small_init.final_train_loss:.3e}, "
            f"@4e-3 at {high_lr.final_train_loss:.3e}."
        ),
        (
            "- Bottleneck pressure is checked with a wider residual branch: b64 small-init ends at "
            f"{small_init.final_train_loss:.3e}, b128 small-init ends at {wider.final_train_loss:.3e}."
        ),
        (
            "- Best residual variant in this sweep: "
            f"{best_residual.spec.name} with final_train_loss={best_residual.final_train_loss:.3e}; "
            f"MLP control reaches {mlp.final_train_loss:.3e}."
        ),
        "",
        "Selected checkpoints",
        baseline.spec.name,
        _snapshot_line(baseline, min(100, config.steps)),
        _snapshot_line(baseline, min(1000, config.steps)),
        _snapshot_line(baseline, config.steps),
        "",
        small_init.spec.name,
        _snapshot_line(small_init, min(100, config.steps)),
        _snapshot_line(small_init, min(1000, config.steps)),
        _snapshot_line(small_init, config.steps),
        "",
        wider.spec.name,
        _snapshot_line(wider, min(100, config.steps)),
        _snapshot_line(wider, min(1000, config.steps)),
        _snapshot_line(wider, config.steps),
        "",
        "Depth endpoint summaries",
        baseline.spec.name,
        _stream_endpoint_summary(baseline, min(100, config.steps)),
        _stream_endpoint_summary(baseline, min(1000, config.steps)),
        _stream_endpoint_summary(baseline, config.steps),
        "",
        small_init.spec.name,
        _stream_endpoint_summary(small_init, min(100, config.steps)),
        _stream_endpoint_summary(small_init, min(1000, config.steps)),
        _stream_endpoint_summary(small_init, config.steps),
        "",
        wider.spec.name,
        _stream_endpoint_summary(wider, min(100, config.steps)),
        _stream_endpoint_summary(wider, min(1000, config.steps)),
        _stream_endpoint_summary(wider, config.steps),
        "",
        "Full results",
    ]
    for index, record in enumerate(records):
        if index > 0:
            summary_lines.append("")
        summary_lines.extend(_record_to_lines(record, config.steps))
    return summary_lines


def _write_outputs(
    config: RunConfig,
    split: DatasetSplit,
    records: list[ExperimentRecord],
) -> None:
    """Persist machine-readable metrics and text summary.

    Args:
        config: Global run configuration.
        split: Dataset split.
        records: Experiment results.

    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload: dict[str, Any] = {
        "run_dir": str(RUN_DIR),
        "dataset": {
            "operation": "unsigned_multiplication",
            "bit_width": config.bit_width,
            "train_fraction": config.train_fraction,
            "train_examples": len(split.x_train),
            "test_examples": len(split.x_test),
            "input_dim": int(split.x_train.shape[1]),
            "target_dim": int(split.y_train.shape[1]),
        },
        "training": {
            "num_steps": config.steps,
            "batch_size": config.batch_size,
            "probe_batch_size": config.probe_batch_size,
            "default_learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "grad_clip_norm": config.grad_clip_norm,
            "log_every": config.log_every,
            "diagnostic_steps": list(config.diagnostic_steps),
            "seed": config.seed,
        },
        "results": [asdict(record) for record in records],
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2) + "\n")
    SUMMARY_PATH.write_text("\n".join(_build_summary_lines(config, split, records)) + "\n")


def _write_plots(records: list[ExperimentRecord]) -> None:
    """Persist line plots for losses and key diagnostics.

    Args:
        records: Experiment results.

    """

    train_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(record.step_history, dtype=np.float64),
            y_values=np.asarray(record.train_loss_history, dtype=np.float64),
        )
        for record in records
        if record.step_history and record.train_loss_history
    ]
    test_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(record.step_history, dtype=np.float64),
            y_values=np.asarray(record.test_loss_history, dtype=np.float64),
        )
        for record in records
        if record.step_history and record.test_loss_history
    ]
    grad_norm_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(
                [snapshot.step for snapshot in record.diagnostic_snapshots],
                dtype=np.float64,
            ),
            y_values=np.asarray(
                [snapshot.global_grad_norm for snapshot in record.diagnostic_snapshots],
                dtype=np.float64,
            ),
        )
        for record in records
        if record.diagnostic_snapshots
    ]
    branch_ratio_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(
                [snapshot.step for snapshot in record.diagnostic_snapshots],
                dtype=np.float64,
            ),
            y_values=np.asarray(
                [
                    _mean_branch_ratio(snapshot, "_down")
                    for snapshot in record.diagnostic_snapshots
                ],
                dtype=np.float64,
            ),
        )
        for record in records
        if record.spec.model_kind == "residual" and record.diagnostic_snapshots
    ]
    depth_records = _selected_depth_records(records)
    depth_steps = _depth_profile_steps(max(record.step_history[-1] for record in records))
    residual_stream_series = _depth_line_series(
        depth_records,
        steps=depth_steps,
        name_suffix="_down",
        metric_name="stream_rms",
    )
    up_grad_depth_series = _depth_line_series(
        depth_records,
        steps=depth_steps,
        name_suffix="_up",
        metric_name="kernel_grad_norm",
    )
    down_grad_depth_series = _depth_line_series(
        depth_records,
        steps=depth_steps,
        name_suffix="_down",
        metric_name="kernel_grad_norm",
    )

    if train_series:
        save_line_plot(
            TRAIN_LOSS_PLOT_PATH,
            train_series,
            title="Residual diagnostics: train loss",
            x_label="Number of steps",
            y_label="Train MSE",
            y_log=True,
        )
    if test_series:
        save_line_plot(
            TEST_LOSS_PLOT_PATH,
            test_series,
            title="Residual diagnostics: test loss",
            x_label="Number of steps",
            y_label="Test MSE",
            y_log=True,
        )
    if grad_norm_series:
        save_line_plot(
            GRAD_NORM_PLOT_PATH,
            grad_norm_series,
            title="Residual diagnostics: probe gradient norm",
            x_label="Number of steps",
            y_label="Global gradient norm",
            y_log=True,
        )
    if branch_ratio_series:
        save_line_plot(
            BRANCH_RATIO_PLOT_PATH,
            branch_ratio_series,
            title="Residual diagnostics: mean residual branch ratio",
            x_label="Number of steps",
            y_label="Mean down-branch / stream RMS",
            y_log=False,
        )
    if residual_stream_series:
        save_line_plot(
            RESIDUAL_STREAM_DEPTH_PLOT_PATH,
            residual_stream_series,
            title="Residual diagnostics: residual-stream RMS by block",
            x_label="Residual block index",
            y_label="Residual-stream RMS before block",
            y_log=False,
        )
    if up_grad_depth_series:
        save_line_plot(
            UP_GRAD_DEPTH_PLOT_PATH,
            up_grad_depth_series,
            title="Residual diagnostics: up-projection grad norm by block",
            x_label="Residual block index",
            y_label="Up-projection kernel grad norm",
            y_log=True,
        )
    if down_grad_depth_series:
        save_line_plot(
            DOWN_GRAD_DEPTH_PLOT_PATH,
            down_grad_depth_series,
            title="Residual diagnostics: down-projection grad norm by block",
            x_label="Residual block index",
            y_label="Down-projection kernel grad norm",
            y_log=True,
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Parsed CLI namespace.

    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-width", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the representative residual diagnostics sweep."""

    args = _parse_args()
    log_every = max(1, min(100, args.steps))
    config = RunConfig(
        bit_width=args.bit_width,
        train_fraction=args.train_fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        probe_batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=log_every,
        diagnostic_steps=_make_diagnostic_steps(args.steps, log_every),
        seed=args.seed,
    )

    x_feat, y_feat = _build_dataset(config.bit_width)
    split = _split_dataset(
        x_feat,
        y_feat,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    print(
        "Residual diagnostics: "
        f"{config.bit_width}-bit unsigned multiplication, "
        f"{len(split.x_train)} train / {len(split.x_test)} test samples"
    )
    print(
        f"Training config: steps={config.steps}, batch_size={config.batch_size}, "
        f"probe_batch_size={config.probe_batch_size}, base_lr={config.learning_rate:.2e}, "
        f"wd={config.weight_decay:.2e}, clip={config.grad_clip_norm:.2f}"
    )

    records: list[ExperimentRecord] = []
    for index, spec in enumerate(_make_experiments(config)):
        print(f"\nRunning {spec.name} ...")
        record = _run_experiment(spec, split, config, seed=config.seed + index)
        records.append(record)
        for line in _record_to_lines(record, config.steps):
            print(line)

    _write_outputs(config, split, records)
    _write_plots(records)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()