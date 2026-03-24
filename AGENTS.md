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
