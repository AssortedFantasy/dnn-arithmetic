"""Inspect learned dense skip weights for the architecture sweep."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")

import jax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from experimental_models import DenseReluResNet, dense_resnet_mix_matrix
from run import (
    OUTPUT_DIR,
    _build_dataset,
    _build_model_factory,
    _count_parameters,
    _format_loss,
    _make_experiments,
    _make_progress_callback,
    _parse_args,
    _split_dataset,
)

from dnn_arithmetic.training import OptimizerConfig, TrainingConfig, train_model

MIX_SUMMARY_PATH = OUTPUT_DIR / "mixing_summary.txt"
MIX_METRICS_PATH = OUTPUT_DIR / "mixing_metrics.json"


@dataclass(frozen=True)
class MixingRecord:
    """Summary of learned dense skip weights for one model."""

    name: str
    parameter_count: int
    elapsed: float
    final_train_loss: float
    final_test_loss: float | None
    mean_first_state_weight: float
    mean_previous_state_weight: float
    argmax_sources: list[int]
    mix_matrix: list[list[float]]
    heatmap_path: str
    raw_heatmap_path: str


def _rescale_mix_matrix_by_source_count(matrix: np.ndarray) -> np.ndarray:
    """Rescale each row by its source count.

    This turns raw softmax weights into weight relative to the uniform baseline.
    A value of 1 means "uniform attention for this row", values greater than 1
    indicate above-uniform emphasis, and values below 1 indicate suppression.

    Args:
        matrix: Lower-triangular matrix of raw softmax weights.

    Returns:
        Row-rescaled matrix with the same shape.

    """
    scaled = np.array(matrix, copy=True)
    for row_index in range(scaled.shape[0]):
        scaled[row_index, : row_index + 1] *= row_index + 1
    return scaled


def _save_mix_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    output_path: Path,
    colorbar_label: str,
    mode: str,
) -> None:
    """Save a lower-triangular heatmap of the learned skip weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masked = np.ma.array(matrix, mask=np.triu(np.ones_like(matrix, dtype=bool), k=1))
    if mode == "raw":
        cmap = plt.get_cmap("magma").copy()
        image_kwargs = {"vmin": 0.0, "vmax": 1.0, "cmap": cmap}
    elif mode == "rescaled":
        cmap = plt.get_cmap("RdBu_r").copy()
        valid_values = masked.compressed()
        max_value = float(np.max(valid_values)) if valid_values.size > 0 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(1.0, max_value))
        image_kwargs = {"norm": norm, "cmap": cmap}
    else:
        raise ValueError(f"Unknown heatmap mode: {mode}")
    cmap.set_bad(color="#f3f3f3")

    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    image = ax.imshow(masked, origin="upper", **image_kwargs)
    ax.set_title(title)
    ax.set_xlabel("Source state index")
    ax.set_ylabel("Block index")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f"h{index}" for index in range(matrix.shape[1])], rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"block {index + 1}" for index in range(matrix.shape[0])])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _format_summary(records: list[MixingRecord]) -> str:
    """Render a compact text summary of the learned skip weights."""
    lines = [
        "Dense skip-weight analysis",
        "",
    ]
    for record in records:
        lines.extend(
            [
                record.name,
                (
                    f"  params={record.parameter_count}, elapsed={record.elapsed:.2f}s, "
                    f"final_train_loss={record.final_train_loss:.3e}, "
                    f"final_test_loss={_format_loss(record.final_test_loss)}"
                ),
                (
                    f"  mean_first_state_weight={record.mean_first_state_weight:.3f}, "
                    f"mean_previous_state_weight={record.mean_previous_state_weight:.3f}"
                ),
                f"  argmax_sources_by_block={record.argmax_sources}",
                f"  heatmap={record.heatmap_path}",
                f"  raw_heatmap={record.raw_heatmap_path}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Train dense models and visualize their learned skip-weight matrices."""
    config = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        (
            "Dense skip-weight analysis: "
            f"bit_width={config.bit_width}, train_fraction={config.train_fraction:.2f}, "
            f"steps={config.steps}, batch_size={config.batch_size}, "
            f"lr={config.learning_rate:.2e}"
        ),
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

    all_specs = _make_experiments()
    dense_spec_pairs = [
        (index, spec)
        for index, spec in enumerate(all_specs)
        if spec.model_kind == "dense_residual"
    ]
    records: list[MixingRecord] = []
    for display_index, (spec_index, spec) in enumerate(dense_spec_pairs, start=1):
        print(
            f"Starting {spec.name} ({display_index}/{len(dense_spec_pairs)})",
            flush=True,
        )
        init_key = jax.random.key(config.seed + spec_index)
        factory = _build_model_factory(spec, target_bias)
        parameter_count = _count_parameters(
            factory(dataset.x_train.shape[1], dataset.y_train.shape[1], init_key)
        )
        train_result = train_model(
            dataset.x_train,
            dataset.y_train,
            init_key,
            training_config,
            factory,
            eval_data=(dataset.x_test, dataset.y_test),
            on_log=_make_progress_callback(spec.name, config.steps),
        )

        model = train_result.model
        if not isinstance(model, DenseReluResNet):
            raise TypeError(f"Expected DenseReluResNet, got {type(model).__name__}")

        mix_matrix = dense_resnet_mix_matrix(model)
        previous_state_weights = [
            float(mix_matrix[row_index, row_index])
            for row_index in range(mix_matrix.shape[0])
        ]
        heatmap_path = OUTPUT_DIR / f"{spec.name}-mix-weights.png"
        raw_heatmap_path = OUTPUT_DIR / f"{spec.name}-mix-weights-raw.png"
        rescaled_matrix = _rescale_mix_matrix_by_source_count(mix_matrix)
        _save_mix_heatmap(
            rescaled_matrix,
            title=f"{spec.name} skip-weight matrix (rescaled by row width)",
            output_path=heatmap_path,
            colorbar_label="softmax weight x source count",
            mode="rescaled",
        )
        _save_mix_heatmap(
            mix_matrix,
            title=f"{spec.name} skip-weight matrix (raw softmax)",
            output_path=raw_heatmap_path,
            colorbar_label="softmax weight",
            mode="raw",
        )
        final_test_loss = (
            train_result.test_loss_history[-1]
            if train_result.test_loss_history
            else None
        )
        record = MixingRecord(
            name=spec.name,
            parameter_count=parameter_count,
            elapsed=train_result.elapsed,
            final_train_loss=(
                train_result.train_loss_history[-1]
                if train_result.train_loss_history
                else float("nan")
            ),
            final_test_loss=final_test_loss,
            mean_first_state_weight=float(np.mean(mix_matrix[:, 0])),
            mean_previous_state_weight=float(np.mean(previous_state_weights)),
            argmax_sources=[
                int(np.argmax(mix_matrix[row_index, : row_index + 1]))
                for row_index in range(mix_matrix.shape[0])
            ],
            mix_matrix=mix_matrix.tolist(),
            heatmap_path=str(heatmap_path),
            raw_heatmap_path=str(raw_heatmap_path),
        )
        records.append(record)
        print(
            (
                f"Finished {spec.name}: final_train_loss={record.final_train_loss:.3e}, "
                f"final_test_loss={_format_loss(record.final_test_loss)}, "
                f"mean_first_state_weight={record.mean_first_state_weight:.3f}, "
                f"mean_previous_state_weight={record.mean_previous_state_weight:.3f}"
            ),
            flush=True,
        )

    MIX_SUMMARY_PATH.write_text(_format_summary(records), encoding="utf-8")
    MIX_METRICS_PATH.write_text(
        json.dumps({"results": [asdict(record) for record in records]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {MIX_SUMMARY_PATH}", flush=True)
    print(f"Wrote {MIX_METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
