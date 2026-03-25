"""Inverse-bottleneck residual sweep: inter_dim > residual_dim."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
from flax import nnx

from dnn_arithmetic.models import ResidualReluMLP, batched_predict
from dnn_arithmetic.plotting import LineSeries, save_line_plot
from dnn_arithmetic.training import OptimizerConfig, TrainingConfig, train_model

RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "inverse_bottleneck_metrics.json"
SUMMARY_PATH = OUTPUT_DIR / "inverse_bottleneck_summary.txt"
TRAIN_LOSS_PLOT = OUTPUT_DIR / "inverse_bottleneck_train_loss.png"
TEST_LOSS_PLOT = OUTPUT_DIR / "inverse_bottleneck_test_loss.png"

SWEEP_SPECS_RAW: list[tuple[int, int, int]] = [
    # (residual_dim, inter_dim, num_blocks)
    (64, 128, 8),
    (32, 128, 8),
    (64, 256, 8),
]


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class EvalMetrics:
    mse: float
    bit_accuracy: float
    exact_accuracy: float
    mean_abs_product_error: float
    max_abs_product_error: int


@dataclass(frozen=True)
class RunConfig:
    bit_width: int
    train_fraction: float
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    log_every: int
    seed: int


@dataclass(frozen=True)
class SweepSpec:
    name: str
    residual_dim: int
    inter_dim: int
    num_blocks: int


@dataclass(frozen=True)
class SweepRecord:
    spec: SweepSpec
    parameter_count: int
    train_metrics: EvalMetrics
    test_metrics: EvalMetrics
    step_history: list[int]
    train_loss_history: list[float]
    test_loss_history: list[float]
    elapsed: float
    final_train_loss: float


def _int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width, dtype=np.uint32)
    bit_values = (values[:, None] >> shifts[None, :]) & 1
    return bit_values.astype(np.float32)


def _bits_to_int(bit_vectors: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(bit_vectors.shape[1], dtype=np.uint32)).astype(np.uint32)
    return np.sum(bit_vectors.astype(np.uint32) * powers[None, :], axis=1)


def _build_dataset(bit_width: int) -> tuple[np.ndarray, np.ndarray]:
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


def _make_sweep_specs() -> list[SweepSpec]:
    return [
        SweepSpec(
            name=f"res-r{r}-b{b}-x{n}",
            residual_dim=r,
            inter_dim=b,
            num_blocks=n,
        )
        for r, b, n in SWEEP_SPECS_RAW
    ]


def _build_model_factory(
    spec: SweepSpec,
    target_bias: np.ndarray,
) -> Callable[[int, int, jax.Array], nnx.Module]:
    def factory(in_dim: int, out_dim: int, key: jax.Array) -> nnx.Module:
        rngs = nnx.Rngs(key)
        model = ResidualReluMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            residual_dim=spec.residual_dim,
            inter_dim=spec.inter_dim,
            num_blocks=spec.num_blocks,
            rngs=rngs,
        )
        model.init_output_bias(target_bias)
        return model

    return factory


def _count_parameters(model: nnx.Module) -> int:
    state = nnx.state(model, nnx.Param)
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(state))


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
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


def _format_summary(config: RunConfig, records: list[SweepRecord]) -> str:
    sorted_records = sorted(records, key=lambda r: r.final_train_loss)
    best = sorted_records[0]
    lines = [
        "Inverse-bottleneck residual sweep: 8-bit unsigned multiplication",
        (
            f"Training config: steps={config.steps}, batch_size={config.batch_size}, "
            f"lr={config.learning_rate:.2e}, wd={config.weight_decay:.2e}, "
            f"clip={config.grad_clip_norm:.2f}, train_fraction={config.train_fraction:.2f}"
        ),
        f"Configs: {[s.name for s in _make_sweep_specs()]}",
        "",
        "Best final train loss",
        (
            f"- {best.spec.name}: final_train_loss={best.final_train_loss:.3e}, "
            f"params={best.parameter_count}, elapsed={best.elapsed:.2f}s"
        ),
        "",
        "Full results (sorted by train loss)",
    ]
    for record in sorted_records:
        final_test_loss = (
            record.test_loss_history[-1] if record.test_loss_history else float("nan")
        )
        lines.extend(
            [
                record.spec.name,
                (
                    f"  params={record.parameter_count}, elapsed={record.elapsed:.2f}s, "
                    f"final_train_loss={record.final_train_loss:.3e}, "
                    f"final_test_loss={final_test_loss:.3e}"
                ),
                (
                    f"  train: mse={record.train_metrics.mse:.3e}, "
                    f"bit_acc={record.train_metrics.bit_accuracy:.4f}, "
                    f"exact_acc={record.train_metrics.exact_accuracy:.4f}, "
                    f"mean_abs_prod_err={record.train_metrics.mean_abs_product_error:.3f}, "
                    f"max_abs_prod_err={record.train_metrics.max_abs_product_error}"
                ),
                (
                    f"  test : mse={record.test_metrics.mse:.3e}, "
                    f"bit_acc={record.test_metrics.bit_accuracy:.4f}, "
                    f"exact_acc={record.test_metrics.exact_accuracy:.4f}, "
                    f"mean_abs_prod_err={record.test_metrics.mean_abs_product_error:.3f}, "
                    f"max_abs_prod_err={record.test_metrics.max_abs_product_error}"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _format_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def _make_progress_callback(
    spec_name: str,
    total_steps: int,
) -> Callable[[int, float, float | None], None]:
    def callback(step: int, train_loss: float, test_loss: float | None) -> None:
        print(
            f"[{spec_name}] step={step}/{total_steps} "
            f"train_loss={train_loss:.3e} test_loss={_format_loss(test_loss)}",
            flush=True,
        )

    return callback


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-width", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return RunConfig(
        bit_width=args.bit_width,
        train_fraction=args.train_fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        seed=args.seed,
    )


def main() -> None:
    """Run the inverse-bottleneck residual sweep and persist outputs."""
    config = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = _make_sweep_specs()
    print(
        f"Inverse-bottleneck sweep: {len(specs)} configurations, "
        f"bit_width={config.bit_width}, steps={config.steps}, "
        f"lr={config.learning_rate:.2e}",
        flush=True,
    )

    x_feat, y_feat = _build_dataset(config.bit_width)
    dataset = _split_dataset(
        x_feat,
        y_feat,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    target_bias = np.mean(dataset.y_train, axis=0, dtype=np.float32)

    optimizer_config = OptimizerConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=max(1, config.steps // 20),
        constant_steps=max(1, config.steps // 2),
        lr_end_frac=0.05,
        grad_clip_norm=config.grad_clip_norm,
    )
    training_config = TrainingConfig(
        num_steps=config.steps,
        batch_size=config.batch_size,
        test_fraction=0.0,
        log_every=config.log_every,
        optimizer=optimizer_config,
    )

    records: list[SweepRecord] = []
    for index, spec in enumerate(specs):
        print(
            f"Starting {spec.name} ({index + 1}/{len(specs)})",
            flush=True,
        )
        init_key = jax.random.key(config.seed + index)
        factory = _build_model_factory(spec, target_bias)
        parameter_count = _count_parameters(
            factory(dataset.x_train.shape[1], dataset.y_train.shape[1], init_key)
        )
        print(f"[{spec.name}] params={parameter_count}", flush=True)
        start_time = time.perf_counter()
        train_result = train_model(
            dataset.x_train,
            dataset.y_train,
            init_key,
            training_config,
            factory,
            eval_data=(dataset.x_test, dataset.y_test),
            on_log=_make_progress_callback(spec.name, config.steps),
        )

        train_pred = batched_predict(train_result.model, dataset.x_train)
        test_pred = batched_predict(train_result.model, dataset.x_test)
        record = SweepRecord(
            spec=spec,
            parameter_count=parameter_count,
            train_metrics=_evaluate_predictions(dataset.y_train, train_pred),
            test_metrics=_evaluate_predictions(dataset.y_test, test_pred),
            step_history=train_result.step_history,
            train_loss_history=train_result.train_loss_history,
            test_loss_history=train_result.test_loss_history,
            elapsed=train_result.elapsed,
            final_train_loss=(
                train_result.train_loss_history[-1]
                if train_result.train_loss_history
                else float("nan")
            ),
        )
        records.append(record)
        final_test_loss = (
            record.test_loss_history[-1] if record.test_loss_history else None
        )
        print(
            (
                f"Finished {spec.name} in {time.perf_counter() - start_time:.2f}s: "
                f"final_train_loss={record.final_train_loss:.3e}, "
                f"final_test_loss={_format_loss(final_test_loss)}"
            ),
            flush=True,
        )

    # Loss curves
    train_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(record.step_history, dtype=np.float64),
            y_values=np.asarray(record.train_loss_history, dtype=np.float64),
        )
        for record in records
    ]
    test_series = [
        LineSeries(
            label=record.spec.name,
            x_values=np.asarray(
                record.step_history[: len(record.test_loss_history)], dtype=np.float64
            ),
            y_values=np.asarray(record.test_loss_history, dtype=np.float64),
        )
        for record in records
    ]
    save_line_plot(
        TRAIN_LOSS_PLOT,
        train_series,
        title="Inverse-Bottleneck Train Loss",
        x_label="Step",
        y_label="Mean Squared Error",
        y_log=True,
    )
    save_line_plot(
        TEST_LOSS_PLOT,
        test_series,
        title="Inverse-Bottleneck Test Loss",
        x_label="Step",
        y_label="Mean Squared Error",
        y_log=True,
    )

    # Persist results
    payload = {
        "run_dir": str(RUN_DIR),
        "dataset": {
            "operation": "unsigned_multiplication",
            "bit_width": config.bit_width,
            "train_fraction": config.train_fraction,
            "train_examples": int(len(dataset.x_train)),
            "test_examples": int(len(dataset.x_test)),
            "input_dim": int(dataset.x_train.shape[1]),
            "target_dim": int(dataset.y_train.shape[1]),
        },
        "sweep": {
            "configs": [asdict(s) for s in specs],
        },
        "training": asdict(config),
        "results": [asdict(record) for record in records],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(_format_summary(config, records), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}", flush=True)
    print(f"Wrote {METRICS_PATH}", flush=True)
    print(f"Wrote {TRAIN_LOSS_PLOT}", flush=True)
    print(f"Wrote {TEST_LOSS_PLOT}", flush=True)


if __name__ == "__main__":
    main()
