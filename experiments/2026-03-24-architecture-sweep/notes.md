# Architecture Sweep

## Intent

This experiment compares the plain MLP baseline against residual alternatives that keep a similar residual width but change how the residual state itself is propagated across depth.

The first new architecture is a dense ReLU ResNet. Instead of propagating only the latest residual stream, block `i` forms a learned softmax-normalized weighted sum of all earlier residual-stream states and uses that mixed state as the block skip path. The goal is to reduce information dilution by letting later blocks reconstruct their input state from the full history rather than from a single accumulated stream.

The sweep is intended to answer three practical questions:

- does the dense weighted-skip architecture train faster or slower than the plain residual baseline,
- does it improve final train loss or test loss at matched width and bottleneck,
- does any gain come from the architecture change itself or only from widening the bottleneck.

## Files

- `run.py`: executable experiment entrypoint
- `run2.py`: dense skip-weight analysis and heatmap generation
- `experimental_models.py`: local architecture definitions for this sweep
- `outputs/metrics.json`: machine-readable config and results
- `outputs/summary.txt`: compact human-readable summary
- `outputs/train_loss.png`: train-loss curves
- `outputs/test_loss.png`: test-loss curves
- `outputs/mixing_summary.txt`: compact learned-weight summary for dense models
- `outputs/*-mix-weights.png`: lower-triangular heatmaps of learned dense skip weights

## Dataset and training setup

- task: exhaustive 8-bit unsigned multiplication
- inputs: concatenated little-endian bit vectors for the two operands
- targets: little-endian product bits
- default train fraction: `0.75`
- default optimizer: AdamW with the same warmup, hold, decay shape used by the residual diagnostics run
- default learning rate: `2e-3`
- default batch size: `256`
- default steps: `6000`
- default clip norm: `1.0`

## Initial sweep

- `mlp-256x4-bias`: plain MLP control
- `res-256-b64-x8-bias`: residual ReLU baseline at the fair-parameter bottleneck
- `res-256-b128-x8-bias`: wider residual baseline to separate architecture effects from pure capacity
- `dense-res-256-b64-x8-bias`: dense weighted-skip residual model at the fair-parameter bottleneck
- `dense-res-256-b128-x8-bias`: wider dense weighted-skip residual model

## Dense weighted-skip definition

Let `h0` be the projected input state. For block `i`, maintain learned logits over `h0, ..., h{i-1}`. After softmax normalization, use those weights to form

`mix_i = sum_j alpha_{i,j} h_j`

Then compute the next state from that mixed state:

`h_i = mix_i + f_i(mix_i)`

with `f_i` implemented as `Linear(residual_dim, inter_dim)`, `ReLU`, `Linear(inter_dim, residual_dim)`.

The output projection currently reads only the latest state `h_B`. Output-side mixing over the full history is intentionally left out for now so the experiment isolates the effect of dense skip propagation inside the stack.

## Running

From the repository root:

```bash
uv run python experiments/2026-03-24-architecture-sweep/run.py
```

To inspect the learned dense skip weights directly:

```bash
uv run python experiments/2026-03-24-architecture-sweep/run2.py
```