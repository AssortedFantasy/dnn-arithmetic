"""Training utilities for DNN arithmetic experiments."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax import nnx
from flax.nnx import Module, Optimizer

from .loaders import DataIterator, test_train_split

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

logger = logging.getLogger(__name__)

ArrayNp = np.ndarray
ArrayJnp = jax.Array
ModelFactory = Callable[[int, int, jax.Array], Module]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration for training.

    Args:
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        beta_1: Adam beta1.
        beta_2: Adam beta2.
        warmup_steps: Number of warmup steps.
        constant_steps: Number of steps to hold the peak learning rate.
        lr_end_frac: Final learning rate as a fraction of the peak.
        grad_clip_norm: Global gradient clip norm. Set to 0 to disable.

    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    warmup_steps: int = 100
    constant_steps: int = 500
    lr_end_frac: float = 0.01
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a single training run.

    Args:
        num_steps: Number of optimizer steps.
        batch_size: Batch size used by the data iterator.
        test_fraction: Fraction of samples reserved for testing.
        log_every: Record train and test loss every N steps.
        optimizer: Optimizer and schedule configuration.

    """

    num_steps: int = 5000
    batch_size: int = 256
    test_fraction: float = 0.0
    log_every: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class TrainResult:
    """Results from model training.

    Args:
        model: Trained model.
        train_loss_history: Mean train loss over each logging interval.
        test_loss_history: Test loss at each logging interval.
        step_history: Training steps corresponding to the logged losses.
        elapsed: Wall-clock training time in seconds.

    """

    model: Module
    train_loss_history: list[float]
    test_loss_history: list[float]
    step_history: list[int]
    elapsed: float = 0.0


@functools.lru_cache(maxsize=None)
def _make_optimizer_tx(
    config: OptimizerConfig,
    total_steps: int,
) -> optax.GradientTransformation:
    """Build the scheduled AdamW transform for training."""
    warmup_steps = max(0, min(config.warmup_steps, total_steps))
    remaining_after_warmup = max(0, total_steps - warmup_steps)
    constant_steps = max(0, min(config.constant_steps, remaining_after_warmup))
    decay_steps = max(1, total_steps - warmup_steps - constant_steps)

    schedules: list[optax.Schedule] = []
    stage_lengths: list[int] = []

    if warmup_steps > 0:
        schedules.append(
            optax.linear_schedule(
                init_value=config.learning_rate / 1000.0,
                end_value=config.learning_rate,
                transition_steps=warmup_steps,
            )
        )
        stage_lengths.append(warmup_steps)

    if constant_steps > 0:
        schedules.append(optax.constant_schedule(config.learning_rate))
        stage_lengths.append(constant_steps)

    schedules.append(
        optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=decay_steps,
            alpha=config.lr_end_frac,
        )
    )

    if len(schedules) == 1:
        learning_rate = schedules[0]
    else:
        boundaries: list[int] = []
        running_step = 0
        for stage_len in stage_lengths:
            running_step += stage_len
            boundaries.append(running_step)
        learning_rate = optax.join_schedules(schedules, boundaries)

    adamw = optax.adamw(
        learning_rate=learning_rate,
        b1=config.beta_1,
        b2=config.beta_2,
        weight_decay=config.weight_decay,
        mask=_create_weight_decay_mask_for_params,
    )
    if config.grad_clip_norm > 0:
        return optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), adamw)
    return adamw


def _create_weight_decay_mask_for_params(params: Any) -> Any:
    """Apply AdamW decay only to kernel parameters."""
    params = nnx.pure(params)

    def is_kernel(path: tuple[Any, ...], _: jax.Array) -> bool:
        if path and isinstance(path[-1], jtu.DictKey):
            return path[-1].key == "kernel"
        return False

    return jtu.tree_map_with_path(is_kernel, params)


def _init_optimizer(
    model: Module,
    config: OptimizerConfig,
    *,
    total_steps: int,
) -> Optimizer[Module]:
    """Initialize the NNX optimizer wrapper."""
    tx = _make_optimizer_tx(config, total_steps)
    return Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def _l2_loss(model: Module, x: ArrayJnp, y: ArrayJnp) -> ArrayJnp:
    """Mean squared error summed over the output dimension."""
    y_pred = model(x)
    per_sample = jnp.sum(jnp.square(y_pred - y), axis=-1)
    return jnp.mean(per_sample)


_l2_val_and_grad = nnx.value_and_grad(_l2_loss)


@nnx.jit
def _train_step(
    model: Module,
    optimizer: Optimizer[Module],
    x_batch: ArrayJnp,
    y_batch: ArrayJnp,
) -> ArrayJnp:
    """Run one optimizer step and return the batch loss."""
    loss, grads = _l2_val_and_grad(model, x_batch, y_batch)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _eval_step(model: Module, x_batch: ArrayJnp, y_batch: ArrayJnp) -> ArrayJnp:
    """Evaluate one batch loss without updating the model."""
    return _l2_loss(model, x_batch, y_batch)


def _cycle_batches(iterator: DataIterator):
    while True:
        for batch in iterator:
            yield batch


def _find_valid_batch_size(requested: int, data_len: int) -> int:
    """Find a non-ragged batch size that fits the available data."""
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


def _evaluate_model(
    model: Module,
    x_test: ArrayNp,
    y_test: ArrayNp,
    batch_size: int,
    key: jax.Array,
) -> float:
    """Evaluate a model on the held-out split."""
    if len(x_test) == 0:
        return float("nan")

    test_iter = DataIterator([x_test, y_test], batch_size=batch_size, key=key)
    total_loss = 0.0
    total_examples = 0
    for x_batch, y_batch in test_iter:
        batch_loss = float(_eval_step(model, x_batch, y_batch))
        batch_examples = int(x_batch.shape[0])
        total_loss += batch_loss * batch_examples
        total_examples += batch_examples

    if total_examples == 0:
        return float("nan")
    return total_loss / total_examples


def train_model(
    x_feat: ArrayNp,
    y_feat: ArrayNp,
    key: jax.Array,
    config: TrainingConfig,
    model_factory: ModelFactory,
) -> TrainResult:
    """Train a single model on a dataset.

    Args:
        x_feat: Input features.
        y_feat: Target features.
        key: JAX key for initialization and shuffling.
        config: Training configuration.
        model_factory: Callable mapping ``(in_dim, out_dim, key)`` to a model.

    Returns:
        Training result containing the model and logged losses.

    """
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive.")
    if not (0.0 <= config.test_fraction < 1.0):
        raise ValueError("test_fraction must be in [0, 1).")

    x_feat = np.asarray(x_feat)
    y_feat = np.asarray(y_feat)
    if x_feat.ndim != 2 or y_feat.ndim != 2:
        raise ValueError("x_feat and y_feat must be 2D arrays.")
    if len(x_feat) != len(y_feat):
        raise ValueError("x_feat and y_feat must have the same length.")

    in_dim = x_feat.shape[1]
    out_dim = y_feat.shape[1]

    init_key, split_key, train_iter_key, test_iter_key = jax.random.split(key, 4)
    model = model_factory(in_dim, out_dim, init_key)

    if config.test_fraction > 0.0:
        test_data, train_data = test_train_split(
            [x_feat, y_feat],
            test_fraction=config.test_fraction,
            rng=split_key,
        )
        x_test, y_test = test_data
        x_train, y_train = train_data
    else:
        x_train, y_train = x_feat, y_feat
        x_test, y_test = None, None

    if len(x_train) == 0:
        raise ValueError("Training data is empty after the train/test split.")

    train_batch_size = _find_valid_batch_size(config.batch_size, len(x_train))
    train_iter = DataIterator(
        [x_train, y_train],
        batch_size=train_batch_size,
        key=train_iter_key,
    )
    batch_iter = _cycle_batches(train_iter)

    test_batch_size: int | None = None
    if x_test is not None and y_test is not None and len(x_test) > 0:
        test_batch_size = _find_valid_batch_size(config.batch_size, len(x_test))

    optimizer = _init_optimizer(model, config.optimizer, total_steps=config.num_steps)

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    step_history: list[int] = []
    interval_loss_total = 0.0
    interval_batches = 0

    start_time = time.perf_counter()
    for step in range(config.num_steps):
        x_batch, y_batch = next(batch_iter)
        batch_loss = float(_train_step(model, optimizer, x_batch, y_batch))
        interval_loss_total += batch_loss
        interval_batches += 1

        if (step + 1) % config.log_every != 0:
            continue

        train_loss_history.append(interval_loss_total / interval_batches)
        step_history.append(step + 1)
        interval_loss_total = 0.0
        interval_batches = 0

        if (
            x_test is not None
            and y_test is not None
            and len(x_test) > 0
            and test_batch_size is not None
        ):
            test_loss_history.append(
                _evaluate_model(
                    model,
                    x_test,
                    y_test,
                    test_batch_size,
                    test_iter_key,
                )
            )

    if interval_batches > 0:
        train_loss_history.append(interval_loss_total / interval_batches)
        step_history.append(config.num_steps)
        if (
            x_test is not None
            and y_test is not None
            and len(x_test) > 0
            and test_batch_size is not None
        ):
            test_loss_history.append(
                _evaluate_model(
                    model,
                    x_test,
                    y_test,
                    test_batch_size,
                    test_iter_key,
                )
            )

    elapsed = time.perf_counter() - start_time
    final_loss = train_loss_history[-1] if train_loss_history else float("nan")
    logger.debug(
        "Training complete: %d samples, %d steps, %.2fs, final_loss=%.4e",
        len(x_train),
        config.num_steps,
        elapsed,
        final_loss,
    )

    return TrainResult(
        model=model,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        step_history=step_history,
        elapsed=elapsed,
    )


__all__ = [
    "OptimizerConfig",
    "TrainingConfig",
    "TrainResult",
    "ModelFactory",
    "train_model",
]
