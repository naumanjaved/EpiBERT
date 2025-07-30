"""Shared helper utilities for EpiBERT training scripts.

These functions are duplicated across the legacy ATAC-pretraining and
RAMPAGE-fine-tuning utilities.  Centralising them here reduces copy-paste
maintenance.  Public names and behaviour are unchanged so importing code
continues to work.
"""

from __future__ import annotations

from typing import List, Tuple
import tensorflow as tf

__all__ = [
    "parse_bool_str",
    "tf_tpu_initialize",
    "early_stopping",
    "one_hot",
    "log2",
]


def parse_bool_str(val: str) -> bool:  # noqa: D401 – simple util
    """Interpret common string representations of booleans.

    Any of "False", "false", "FALSE", or "F" (case sensitive) returns
    ``False``; everything else returns ``True``.
    """
    return val not in {"False", "false", "FALSE", "F"}


# -----------------------------------------------------------------------------
# TPU / strategy helpers
# -----------------------------------------------------------------------------

def tf_tpu_initialize(tpu_name: str | None, zone: str | None):
    """Return a `tf.distribute.Strategy` for the requested TPU or default GPUs."""
    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name, zone=zone
        )
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
    except ValueError:
        # Fallback to MirroredStrategy (multi-GPU) or default strategy.
        strategy = tf.distribute.get_strategy()
    return strategy


# -----------------------------------------------------------------------------
# Training-loop helpers
# -----------------------------------------------------------------------------

def early_stopping(
    current_val_loss: float,
    logged_val_losses: List[float],
    best_epoch: int,
    patience: int,
    patience_counter: int,
    min_delta: float,
):
    """Generic early-stopping utility (identical logic for both tasks)."""
    logged_val_losses.append(current_val_loss)
    if current_val_loss < min(logged_val_losses):
        best_epoch = len(logged_val_losses) - 1
        patience_counter = 0
    else:
        patience_counter += 1

    should_stop = patience_counter >= patience and (
        min(logged_val_losses) - current_val_loss
    ) < min_delta
    return should_stop, best_epoch, patience_counter


# -----------------------------------------------------------------------------
# Misc tensor helpers
# -----------------------------------------------------------------------------

def one_hot(sequence: tf.Tensor) -> tf.Tensor:
    """One-hot encode an `A/C/G/T/N` DNA sequence tensor."""
    vocabulary = tf.constant(["A", "C", "G", "T"])
    mapping = tf.constant([0, 1, 2, 3])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocabulary, mapping), default_value=4
    )

    input_characters = tf.strings.upper(tf.strings.unicode_split(sequence, "UTF-8"))
    return tf.one_hot(table.lookup(input_characters), depth=5, dtype=tf.float32)[:, :4]


def log2(x: tf.Tensor) -> tf.Tensor:
    """Compute log base-2 in a graph-friendly way."""
    return tf.math.log(x) / tf.math.log(tf.constant(2, dtype=x.dtype)) 