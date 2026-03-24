# 8-Bit Unsigned Multiplication Benchmark

This experiment tests whether standard feed-forward and residual ReLU networks can learn exhaustive 8-bit unsigned multiplication from bit-vector inputs.

## Files

- `run.py`: executable experiment entrypoint
- `outputs/metrics.json`: machine-readable run config and results
- `outputs/summary.txt`: compact human-readable summary

## Input Parameterization

### Dataset parameters

- `bit_width`: operand width in bits. Default is `8`.
- `train_fraction`: random fraction of the exhaustive input table used for training. Default is `0.75`.
- `seed`: controls the train/test permutation and per-model initialization seeds.

### Dataset encoding

- Each input example is the concatenation of two little-endian bit vectors.
- For `bit_width = 8`, the input dimension is `16`.
- Targets are the little-endian bits of the exact unsigned product.
- For `bit_width = 8`, the target dimension is `16`.
- The dataset is exhaustive over all operand pairs, so there are `2^bit_width * 2^bit_width` examples.

### Training parameters

- `steps`: optimizer steps per model. Default is `6000`.
- `batch_size`: training and evaluation batch size.
- `learning_rate`: peak AdamW learning rate. Default is `2e-3`.
- `weight_decay`: AdamW kernel weight decay.
- Warmup, constant, and decay schedule settings are derived from `steps` inside `run.py`.

### Model sweep parameters

- `model_kind`: either `mlp` or `residual`.
- `hidden_dims`: hidden widths for MLP runs.
- `residual_dim`: width of the residual stream for residual runs.
- `inter_dim`: bottleneck width inside each residual block.
- `num_blocks`: number of residual blocks.
- `init_output_bias`: whether the output bias is initialized to the mean target bit vector.

## Notes

- This keeps the same exhaustive interpolation setup as the addition benchmark, so comparisons are controlled.
- The default sweep deliberately includes a small 2-layer MLP baseline to check whether multiplication is materially harder than addition under the same bit encoding.

## Running

From the repository root:

```bash
uv run python experiments/2026-03-24-multiplication-8bit/run.py
```