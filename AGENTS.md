# AGENTS.md

* This project uses Python >= 3.12, this means that typing with Dict, List, or Optional are all deprecated. Use plain dict, list, | None instead.

* Virtual enviroments are managed with `uv`.

* You should try to use types. Avoid dict typing and instead use dataclasses. Some library code results in type errors due to a lack of annotations and that is acceptable.

* Do not leave deprecated code for backwards compatibility. It causes confusion and maintenance challenges. This is a research project, not a dependency, priority is low technical debt and fast iteration speed.

* Don't create functions that take in a hodgepodge of different types for one argument. Be specific about what types you expect. For example don't create functions that take in any number of numeric types.

* Use Google Style for Python docstrings. Example:

```
def square_root(n):
    """Calculate the square root of a number.

    Args:
        n: the number to get the square root of.
    Returns:
        the square root of n.

    """
```

* Flax and NNX are being used for neural networks.

* Project layout conventions:
    * Reusable library code belongs under `src/dnn_arithmetic/`.
    * Experiments belong under `experiments/` in dated folders, for example `experiments/2026-03-24-addition-8bit/`.
    * Each experiment folder should keep `run.py`, `notes.md`, and an `outputs/` directory together.
    * `notes.md` should document the experiment intent, important observations, and how inputs / targets / model sweeps are parameterized.
    * `outputs/` should stay simple and contain lightweight generated artifacts such as `summary.txt`, `metrics.json`, and plots if needed.
    * Avoid putting one-off experiment code in top-level `scripts/` when it should live in a dated experiment folder.
