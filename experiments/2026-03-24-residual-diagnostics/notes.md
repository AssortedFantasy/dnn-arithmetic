# Residual Optimization Diagnostics

## Intent

This experiment investigates why the residual MLPs underperform the plain MLP on exhaustive 8-bit unsigned multiplication under the real benchmark training setup.

The motivating benchmark result was that the residual model trained to a much higher final train loss than the MLP after 6000 steps.

The hypotheses are prioritized as:

- initialization of the residual down projection,
- learning-rate sensitivity after fixing initialization,
- bottleneck capacity inside the residual block.

Activation scale, gradient magnitude, and clipping behavior are treated as observed quantities used to interpret those interventions.

The important diagnostic comparison is not just step 0. The main comparisons are early training, mid training, and late training to see whether an initialization difference persists or washes out.

## Files

- `run.py`: executable experiment entrypoint
- `outputs/metrics.json`: machine-readable run config, metrics, and diagnostic snapshots
- `outputs/summary.txt`: compact human-readable summary
- `outputs/train_loss.png`: train loss curves
- `outputs/test_loss.png`: test loss curves
- `outputs/grad_norm.png`: probe-batch global gradient norm curves
- `outputs/branch_ratio.png`: residual down-branch RMS to stream RMS curves

## Setup

### Dataset parameters

- `bit_width`: operand width in bits. Default is `8`.
- `train_fraction`: random fraction of the exhaustive input table used for training. Default is `0.75`.
- `seed`: controls the train/test permutation, diagnostic batch selection, and model initialization.

### Dataset encoding

- Each input example is the concatenation of two little-endian bit vectors.
- For `bit_width = 8`, the input dimension is `16`.
- Targets are the little-endian bits of the exact unsigned product.
- For `bit_width = 8`, the target dimension is `16`.
- The dataset is exhaustive over all operand pairs, so there are `2^bit_width * 2^bit_width` examples.

### Training parameters

- `steps`: optimizer steps per model. Default is `6000`.
- `batch_size`: training and evaluation batch size. Default is `256`.
- `probe_batch_size`: fixed training batch used only for diagnostic snapshots. Default is `1024`.
- `learning_rate`: base peak AdamW learning rate. Default is `2e-3`.
- `weight_decay`: AdamW kernel weight decay. Default is `1e-5`.
- `grad_clip_norm`: global gradient clip norm. Default is `1.0`.

### Optimizer schedule

The schedule is intentionally matched to the multiplication benchmark:

- linear warmup from `lr / 1000` to `lr`,
- constant region,
- cosine decay to `0.05 * lr`.

Warmup and constant durations are derived from `steps` in the same way as the benchmark script.

## Diagnostic design

This experiment does not train on a fixed batch. It trains on the normal shuffled train split and only uses a fixed probe batch to measure diagnostics at selected checkpoints.

At each diagnostic checkpoint it records:

- probe-batch loss,
- global gradient norm before clipping,
- whether the clip threshold would be exceeded,
- per-layer or per-block kernel gradient norms,
- per-layer or per-block parameter norms,
- activation RMS,
- residual branch to stream RMS ratios.

The diagnostic checkpoints include dense early steps and regular checkpoints through the full run, so the experiment can capture both the initialization regime and the long-horizon behavior.

For the residual naming, `x8` means `8` residual blocks. Each block contains two learned linear layers: an up projection `residual_dim -> inter_dim` and a down projection `inter_dim -> residual_dim`.

## Variants

- `mlp-256x4-bias`: plain MLP control.
- `res-256-b64-x8-bias-zeroinit`: current residual baseline with zero down-projection initialization.
- `res-256-b64-x8-bias-smallinit`: same residual architecture with small non-zero down-projection initialization.
- `res-256-b64-x8-bias-lecuninit`: same residual architecture with standard LeCun initialization on the down projection.
- `res-256-b64-x8-bias-smallinit-lr1e-3`: small-init residual with lower learning rate.
- `res-256-b64-x8-bias-smallinit-lr4e-3`: small-init residual with higher learning rate.
- `res-256-b128-x8-bias-smallinit`: small-init residual with a wider bottleneck to test capacity pressure.

## Interpretation rules

- If the zero-init residual has strongly suppressed residual-up gradients at step 0 and the small-init or standard-init variants materially improve final train loss, initialization is the primary issue.
- If zero-init and small-init look meaningfully different only at step 0 but similar by early or mid training, initialization is a real effect but not the main cause of the final gap.
- If the initialization change helps but residuals still lag while branch ratios and gradient norms look healthy, the remaining issue is likely bottleneck capacity.
- If changing learning rate around the small-init residual moves final train loss substantially, learning-rate sensitivity is a secondary contributor.
- If clipping is rare across diagnostic checkpoints, clipping should not be treated as the main explanation.

## Running

From the repository root:

```bash
uv run python experiments/2026-03-24-residual-diagnostics/run.py
```