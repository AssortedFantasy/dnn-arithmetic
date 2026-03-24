"""Simple benchmark for learning exhaustive 8-bit addition."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
from flax import nnx

from dnn_arithmetic.models import ReluMLP, ResidualReluMLP, batched_predict
from dnn_arithmetic.training import OptimizerConfig, TrainingConfig, train_model


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for the addition benchmark.

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
    """Evaluation metrics for bitwise addition predictions.

    Args:
            mse: Mean squared error over output bits.
            bit_accuracy: Fraction of correctly recovered output bits.
            exact_accuracy: Fraction of examples with every output bit correct.
            mean_abs_sum_error: Mean absolute integer sum error.
            max_abs_sum_error: Maximum absolute integer sum error.

    """

    mse: float
    bit_accuracy: float
    exact_accuracy: float
    mean_abs_sum_error: float
    max_abs_sum_error: int


@dataclass(frozen=True)
class ExperimentSpec:
    """Model configuration for one benchmark run.

    Args:
            name: Display name for the experiment.
            model_kind: Either ``"mlp"`` or ``"residual"``.
            hidden_dims: Hidden widths for MLP experiments.
            residual_dim: Width of the residual stream.
            inter_dim: Bottleneck width inside each residual block.
            num_blocks: Number of residual blocks.
            init_output_bias: Whether to initialize the output bias to the target mean.

    """

    name: str
    model_kind: str
    hidden_dims: tuple[int, ...] = ()
    residual_dim: int = 0
    inter_dim: int = 0
    num_blocks: int = 0
    init_output_bias: bool = False


def _int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Encode integers into little-endian bit vectors.

    Args:
            values: Integer array of shape ``(N,)``.
            width: Number of output bits.

    Returns:
            Float32 bit matrix of shape ``(N, width)``.

    """
    shifts = np.arange(width, dtype=np.uint16)
    bit_values = (values[:, None] >> shifts[None, :]) & 1
    return bit_values.astype(np.float32)


def _bits_to_int(bit_vectors: np.ndarray) -> np.ndarray:
    """Decode little-endian bit vectors back into integers.

    Args:
            bit_vectors: Binary array of shape ``(N, width)``.

    Returns:
            Integer values of shape ``(N,)``.

    """
    powers = (1 << np.arange(bit_vectors.shape[1], dtype=np.uint16)).astype(np.uint16)
    return np.sum(bit_vectors.astype(np.uint16) * powers[None, :], axis=1)


def _build_dataset(bit_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct exhaustive addition examples for a fixed bit width.

    Args:
            bit_width: Operand width in bits.

    Returns:
            Tuple ``(x, y)`` with concatenated operand bits and sum bits.

    """
    max_value = 1 << bit_width
    lhs = np.repeat(np.arange(max_value, dtype=np.uint16), max_value)
    rhs = np.tile(np.arange(max_value, dtype=np.uint16), max_value)
    sums = (lhs + rhs).astype(np.uint16)

    x_feat = np.concatenate(
        [_int_to_bits(lhs, bit_width), _int_to_bits(rhs, bit_width)],
        axis=1,
    )
    y_feat = _int_to_bits(sums, bit_width + 1)
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


def _make_experiments() -> list[ExperimentSpec]:
    """Return the default sweep of model configurations."""
    return [
        ExperimentSpec(
            name="mlp-128x2",
            model_kind="mlp",
            hidden_dims=(128, 128),
            init_output_bias=False,
        ),
        ExperimentSpec(
            name="mlp-256x4-bias",
            model_kind="mlp",
            hidden_dims=(256, 256, 256, 256),
            init_output_bias=True,
        ),
        ExperimentSpec(
            name="res-64-b16-x6",
            model_kind="residual",
            residual_dim=64,
            inter_dim=16,
            num_blocks=6,
            init_output_bias=False,
        ),
        ExperimentSpec(
            name="res-128-b32-x6",
            model_kind="residual",
            residual_dim=128,
            inter_dim=32,
            num_blocks=6,
            init_output_bias=False,
        ),
        ExperimentSpec(
            name="res-128-b32-x6-bias",
            model_kind="residual",
            residual_dim=128,
            inter_dim=32,
            num_blocks=6,
            init_output_bias=True,
        ),
        ExperimentSpec(
            name="res-256-b64-x8-bias",
            model_kind="residual",
            residual_dim=256,
            inter_dim=64,
            num_blocks=8,
            init_output_bias=True,
        ),
    ]


def _build_model_factory(
    spec: ExperimentSpec,
    target_bias: np.ndarray,
) -> Callable[[int, int, jax.Array], nnx.Module]:
    """Create a model factory matching the training API.

    Args:
            spec: Experiment configuration.
            target_bias: Mean target vector used for optional bias init.

    Returns:
            Factory compatible with ``train_model``.

    """

    def factory(in_dim: int, out_dim: int, key: jax.Array) -> nnx.Module:
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
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown model_kind: {spec.model_kind}")

        if spec.init_output_bias:
            model.init_output_bias(target_bias)
        return model

    return factory


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
    bit_accuracy = float(np.mean(pred_bits == true_bits))
    exact_accuracy = float(np.mean(np.all(pred_bits == true_bits, axis=1)))

    pred_values = _bits_to_int(pred_bits)
    true_values = _bits_to_int(true_bits)
    abs_sum_error = np.abs(pred_values.astype(np.int32) - true_values.astype(np.int32))
    return EvalMetrics(
        mse=mse,
        bit_accuracy=bit_accuracy,
        exact_accuracy=exact_accuracy,
        mean_abs_sum_error=float(np.mean(abs_sum_error)),
        max_abs_sum_error=int(np.max(abs_sum_error)),
    )


def _format_metrics(split_name: str, metrics: EvalMetrics) -> str:
    """Format metrics for terminal output."""
    return (
        f"{split_name}: mse={metrics.mse:.3e}, "
        f"bit_acc={metrics.bit_accuracy:.4f}, "
        f"exact_acc={metrics.exact_accuracy:.4f}, "
        f"mean_abs_sum_err={metrics.mean_abs_sum_error:.3f}, "
        f"max_abs_sum_err={metrics.max_abs_sum_error}"
    )


def _run_experiment(
    spec: ExperimentSpec,
    split: DatasetSplit,
    config: TrainingConfig,
    *,
    seed: int,
) -> None:
    """Train one model and print train/test metrics.

    Args:
            spec: Experiment configuration.
            split: Benchmark train/test split.
            config: Training configuration.
            seed: JAX seed for this run.

    """
    model_factory = _build_model_factory(spec, split.y_train.mean(axis=0))
    result = train_model(
        split.x_train,
        split.y_train,
        key=jax.random.key(seed),
        config=config,
        model_factory=model_factory,
    )

    train_pred = batched_predict(
        result.model, split.x_train, batch_size=config.batch_size
    )
    test_pred = batched_predict(
        result.model, split.x_test, batch_size=config.batch_size
    )

    train_metrics = _evaluate_predictions(split.y_train, train_pred)
    test_metrics = _evaluate_predictions(split.y_test, test_pred)
    final_train_loss = (
        result.train_loss_history[-1] if result.train_loss_history else float("nan")
    )

    print(spec.name)
    print(
        f"  steps={config.num_steps}, elapsed={result.elapsed:.2f}s, "
        f"final_train_loss={final_train_loss:.3e}, bias_init={spec.init_output_bias}"
    )
    print(f"  {_format_metrics('train', train_metrics)}")
    print(f"  {_format_metrics('test ', test_metrics)}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the benchmark script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-width", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the 8-bit addition architecture sweep."""
    args = _parse_args()
    x_feat, y_feat = _build_dataset(args.bit_width)
    split = _split_dataset(
        x_feat,
        y_feat,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    config = TrainingConfig(
        num_steps=args.steps,
        batch_size=args.batch_size,
        test_fraction=0.0,
        log_every=max(1, min(100, args.steps)),
        optimizer=OptimizerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=min(100, max(1, args.steps // 10)),
            constant_steps=min(500, max(1, args.steps // 3)),
            lr_end_frac=0.05,
            grad_clip_norm=1.0,
        ),
    )

    print(
        "Benchmark: "
        f"{args.bit_width}-bit addition, {len(split.x_train)} train / {len(split.x_test)} test samples"
    )
    print(
        f"Training config: steps={config.num_steps}, batch_size={config.batch_size}, "
        f"lr={config.optimizer.learning_rate:.2e}, wd={config.optimizer.weight_decay:.2e}"
    )

    experiments = _make_experiments()
    for index, spec in enumerate(experiments):
        _run_experiment(spec, split, config, seed=args.seed + index)


if __name__ == "__main__":
    main()
