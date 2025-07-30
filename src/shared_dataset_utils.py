"""Shared dataset-building helpers used by both ATAC-pretraining and
RAMPAGE-fine-tuning utilities.

The original `training_utils_*` modules each contained nearly identical
code for building `tf.data` pipelines.  These helpers consolidate that
boilerplate so future changes (e.g. compression type, deterministic
settings, prefetch policy) only need to be made in one place.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import tensorflow as tf

__all__ = [
    "build_tfrecord_dataset",
]


AUTOTUNE = tf.data.AUTOTUNE


def _resolve_files(pattern: str) -> tf.data.Dataset:
    """Return a `Dataset` of file paths matching *pattern* (no shuffling)."""
    return tf.data.Dataset.list_files(tf.io.gfile.glob(pattern), shuffle=False)


def build_tfrecord_dataset(
    gcs_path: str,
    split: str,
    wildcard: str,
    deserialize_fn: Callable[[tf.Tensor], tf.Tensor],
    *,
    batch: int,
    options: tf.data.Options,
    deterministic: bool,
    num_parallel_calls: int | None = None,
    repeat: int = 1,
    take: Optional[int] = None,
) -> tf.data.Dataset:
    """Construct a batched & prefetched TFRecord dataset.

    Parameters
    ----------
    gcs_path
        Root GCS/local path (folder containing *split* sub-folder).
    split
        One of "train" / "valid" / etc.; appended to *gcs_path*.
    wildcard
        File-name pattern (e.g. "*.tfr" or "*.tfrecord").
    deserialize_fn
        Function that converts a raw record → model inputs.
    batch
        Examples per batch.
    options
        A `tf.data.Options` configured by the caller (e.g. sharding policy).
    deterministic
        Forwarded to `Dataset.map` to allow nondeterministic order for speed.
    num_parallel_calls
        How many parallel cores for `map`; defaults to `tf.data.AUTOTUNE`.
    repeat
        How many times to repeat the dataset (e.g. epochs or infinite).
    take
        If provided, trims the dataset to *take* elements before batching.
    """

    if num_parallel_calls is None:
        num_parallel_calls = AUTOTUNE

    pattern = os.path.join(gcs_path, split, wildcard)
    files = _resolve_files(pattern)

    ds = tf.data.TFRecordDataset(
        files, compression_type="ZLIB", num_parallel_reads=AUTOTUNE
    )
    ds = ds.with_options(options)

    # Deserialise / preprocess
    ds = ds.map(
        deserialize_fn,
        deterministic=deterministic,
        num_parallel_calls=num_parallel_calls,
    )

    if take is not None:
        ds = ds.take(take)

    if repeat != 1:
        ds = ds.repeat(repeat)

    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds 