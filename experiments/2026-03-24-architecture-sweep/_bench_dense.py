"""Benchmark: DenseReluResNet vs ResidualReluMLP at r64-b256-x8."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
from experimental_models import DenseReluResNet
from flax import nnx

from dnn_arithmetic.models import ResidualReluMLP, batched_predict
from dnn_arithmetic.training import OptimizerConfig, TrainingConfig, train_model

SEED = 0
STEPS = 6000
BATCH_SIZE = 256
BIT_WIDTH = 8
RESIDUAL_DIM = 64
INTER_DIM = 256
NUM_BLOCKS = 8


def _int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width, dtype=np.uint32)
    bit_values = (values[:, None] >> shifts[None, :]) & 1
    return bit_values.astype(np.float32)


def _bits_to_int(bit_vectors: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(bit_vectors.shape[1], dtype=np.uint32)).astype(np.uint32)
    return np.sum(bit_vectors.astype(np.uint32) * powers[None, :], axis=1)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.square(y_pred - y_true)))
    pred_bits = (y_pred >= 0.5).astype(np.uint8)
    true_bits = y_true.astype(np.uint8)
    bit_acc = float(np.mean(np.equal(pred_bits, true_bits).astype(np.float32)))
    exact_acc = float(
        np.mean(np.all(pred_bits == true_bits, axis=1).astype(np.float32))
    )
    pred_vals = _bits_to_int(pred_bits)
    true_vals = _bits_to_int(true_bits)
    abs_err = np.abs(pred_vals.astype(np.int64) - true_vals.astype(np.int64))
    return {
        "mse": mse,
        "bit_acc": bit_acc,
        "exact_acc": exact_acc,
        "mean_abs_err": float(np.mean(abs_err)),
        "max_abs_err": int(np.max(abs_err)),
    }


def main() -> None:
    max_value = 1 << BIT_WIDTH
    lhs = np.repeat(np.arange(max_value, dtype=np.uint32), max_value)
    rhs = np.tile(np.arange(max_value, dtype=np.uint32), max_value)
    products = lhs * rhs
    x_feat = np.concatenate(
        [_int_to_bits(lhs, BIT_WIDTH), _int_to_bits(rhs, BIT_WIDTH)], axis=1
    )
    y_feat = _int_to_bits(products, BIT_WIDTH * 2)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x_feat))
    split = int(len(x_feat) * 0.75)
    x_train, y_train = x_feat[perm[:split]], y_feat[perm[:split]]
    x_test, y_test = x_feat[perm[split:]], y_feat[perm[split:]]
    target_bias = np.mean(y_train, axis=0, dtype=np.float32)

    opt_config = OptimizerConfig(
        learning_rate=2e-3,
        weight_decay=1e-5,
        warmup_steps=max(1, STEPS // 20),
        constant_steps=max(1, STEPS // 2),
        lr_end_frac=0.05,
        grad_clip_norm=1.0,
    )
    train_config = TrainingConfig(
        num_steps=STEPS,
        batch_size=BATCH_SIZE,
        test_fraction=0.0,
        log_every=100,
        optimizer=opt_config,
    )

    configs: list[tuple[str, type]] = [
        ("ResidualReluMLP", ResidualReluMLP),
        ("DenseReluResNet", DenseReluResNet),
    ]

    for name, cls in configs:

        def factory(
            in_dim: int, out_dim: int, key: jax.Array, _cls: type = cls
        ) -> nnx.Module:
            rngs = nnx.Rngs(key)
            model = _cls(
                in_dim=in_dim,
                out_dim=out_dim,
                residual_dim=RESIDUAL_DIM,
                inter_dim=INTER_DIM,
                num_blocks=NUM_BLOCKS,
                rngs=rngs,
            )
            model.init_output_bias(target_bias)
            return model

        init_key = jax.random.key(SEED)
        param_count = sum(
            int(leaf.size)
            for leaf in jax.tree_util.tree_leaves(
                nnx.state(
                    factory(x_train.shape[1], y_train.shape[1], init_key), nnx.Param
                )
            )
        )

        result = train_model(
            x_train,
            y_train,
            init_key,
            train_config,
            factory,
            eval_data=(x_test, y_test),
        )

        train_m = _evaluate(y_train, batched_predict(result.model, x_train))
        test_m = _evaluate(y_test, batched_predict(result.model, x_test))

        print(f"\n{name} r{RESIDUAL_DIM}-b{INTER_DIM}-x{NUM_BLOCKS}")
        print(f"  params={param_count}, elapsed={result.elapsed:.2f}s")
        print(f"  final_train_loss={result.train_loss_history[-1]:.4e}")
        print(
            f"  train: mse={train_m['mse']:.3e}, bit_acc={train_m['bit_acc']:.4f}, "
            f"exact_acc={train_m['exact_acc']:.4f}, mean_err={train_m['mean_abs_err']:.3f}"
        )
        print(
            f"  test : mse={test_m['mse']:.3e}, bit_acc={test_m['bit_acc']:.4f}, "
            f"exact_acc={test_m['exact_acc']:.4f}, mean_err={test_m['mean_abs_err']:.3f}"
        )


if __name__ == "__main__":
    main()
