"""Iteration utilities for NumPy-hosted datasets."""

from collections import deque
from collections.abc import Callable, Sequence
from logging import getLogger
from typing import cast

import chex
import jax
import numpy as np

ArrayNp = np.ndarray
ArrayJnp = jax.Array
Array = ArrayNp | ArrayJnp


def jax_key_to_numpy_rng(key: chex.PRNGKey) -> np.random.Generator:
    """Convert a JAX PRNG key into a NumPy generator.

    Args:
        key: JAX PRNG key.

    Returns:
        NumPy random generator seeded from the key data.

    """
    return np.random.default_rng(np.asarray(jax.random.key_data(key)))


def identity(x: Array) -> Array:
    return x


logger = getLogger(__name__)

# With lots of benchmarking, turns out that just using bog standard device_put is fastest.
# device_put = cast(Callable[[Array], ArrayJnp], jax.jit(identity))
device_put = cast(Callable[[Array], ArrayJnp], jax.device_put)


class DataIterator:
    def __init__(
        self,
        data: Sequence[ArrayNp],
        batch_size: int,
        key: chex.PRNGKey,
        prefetch: int = 2,
    ):
        self.warned_about_small_data = False
        self.data = list(np.array(d) for d in data)
        self.batch_size = batch_size
        self.len_data = len(data[0])
        self.rng = jax_key_to_numpy_rng(key)
        self.prefetch = prefetch
        self.queue: deque[list[ArrayJnp]] = deque(maxlen=prefetch)

        for i, d in enumerate(self.data):
            if len(d) != self.len_data:
                raise ValueError("All data arrays must have the same length.")
            if len(d.shape) < 1:
                raise ValueError("Bad shape for data array.")
            if len(d.shape) == 1:
                self.data[i] = d.reshape(-1, 1)

    def shuffle(self):
        """Shuffle all stored arrays in unison."""
        shuffle = self.rng.permutation(self.len_data)
        for i, d in enumerate(self.data):
            self.data[i] = d[shuffle]

    def truncate(self, n: int, shuffle: bool = True):
        if n <= 0:
            raise ValueError("n must be non-negative.")

        if shuffle:
            self.shuffle()

        if n > self.len_data:
            return

        self.data = [d[:n] for d in self.data]
        self.len_data = n

    def append_data(self, new_data: Sequence[ArrayNp]):
        new_data = list(np.array(d) for d in new_data)
        new_len_data = len(new_data[0])

        # Validate new data
        if len(new_data) != len(self.data):
            raise ValueError(
                "Number of new data arrays must match existing data arrays."
            )

        for i, d in enumerate(new_data):
            if len(d) != new_len_data:
                raise ValueError("All new data arrays must have the same length.")
            if len(d.shape) < 1:
                raise ValueError("Bad shape for new data array.")
            if len(d.shape) == 1:
                new_data[i] = d.reshape(-1, 1)
            if d.shape[1:] != self.data[i].shape[1:]:
                raise ValueError(
                    f"Shape mismatch for array {i}: expected {self.data[i].shape[1:]}, got {d.shape[1:]}."
                )

        # Append the new data
        for i, d in enumerate(new_data):
            self.data[i] = np.concatenate([self.data[i], d], axis=0)

        self.len_data += new_len_data

    def __iter__(self):
        self.shuffle()

        if self.len_data < self.batch_size:
            if not self.warned_about_small_data:
                self.warned_about_small_data = True

                logger.warning(
                    "Data length (%d) is less than batch size (%d)! Cannot form any batches.",
                    self.len_data,
                    self.batch_size,
                )

        gen = (
            [device_put(d[i : i + self.batch_size]) for d in self.data]
            for i in range(
                0,
                # No ragged batches
                self.len_data - self.batch_size + 1,
                self.batch_size,
            )
        )

        try:
            for _ in range(self.prefetch):
                self.queue.append(next(gen))

            while True:
                yield self.queue.popleft()
                self.queue.append(next(gen))
        except StopIteration:
            while self.queue:
                yield self.queue.popleft()

    def __len__(self):
        return self.len_data // self.batch_size


def test_train_split(data: Sequence[ArrayNp], test_fraction: float, rng: chex.PRNGKey):
    """
    Splits a sequence of data arrays into test and train sets after random shuffling.
    This function takes multiple data arrays (assumed to be aligned by index), shuffles them
    uniformly using the provided random key, and splits each into a test portion and a train
    portion based on the specified test fraction.
    Args:
        data (Sequence[ArrayNp]): A sequence of NumPy arrays, all of which must have the same
            length. Each array represents a feature or dataset component.
        test_fraction (float): The fraction of the data to allocate to the test set. Must be
            between 0 and 1 (exclusive).
        rng (chex.PRNGKey): A JAX PRNG key used to generate a random permutation for shuffling
            the data.
    Returns:
        tuple[list[ArrayNp], list[ArrayNp]]: A tuple containing two lists:
            - test: A list of arrays, each containing the test portion of the corresponding input
              array.
            - train: A list of arrays, each containing the train portion of the corresponding
              input array.
    """

    len_data = len(data[0])
    test: list[ArrayNp] = []
    train: list[ArrayNp] = []
    np_rng = jax_key_to_numpy_rng(rng)
    permutation = np_rng.permutation(len_data)

    for d in data:
        assert len(d) == len_data, "All data arrays must have the same length."
        d = d[permutation]
        test.append(d[: int(len_data * test_fraction)])
        train.append(d[int(len_data * test_fraction) :])
    return test, train


__all__ = [
    "DataIterator",
    "test_train_split",
]
