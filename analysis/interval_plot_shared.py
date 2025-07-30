"""Shared interval and plotting utilities used across analysis notebooks.

This module consolidates code that was duplicated in
`interval_and_plotting_utilities.py` and `interval_and_plotting_rampage.py`.
All functions are copied verbatim (minor docstring tweaks) so callers require
no changes beyond importing from this module.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import strings as tfs
import matplotlib.pyplot as plt
from kipoiseq import Interval
import pyfaidx
from typing import Dict, Tuple

__all__ = [
    "one_hot",
    "get_per_base_score_f",
    "process_bedgraph",
    "resize_interval",
    "plot_tracks",
    "FastaStringExtractor",
]


def one_hot(sequence: str) -> tf.Tensor:
    """One-hot encode a DNA sequence (A/C/G/T => 4-vector, others => 0)."""

    vocabulary = tf.constant(["A", "C", "G", "T"])
    mapping = tf.constant([0, 1, 2, 3])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocabulary, mapping), default_value=4
    )

    input_characters = tf.strings.upper(tf.strings.unicode_split(sequence, "UTF-8"))
    return tf.one_hot(table.lookup(input_characters), depth=5, dtype=tf.float32)[:, :4]


def get_per_base_score_f(start, end, score, base):
    """Fill `base[start[i]:end[i]] = score[i]` for each interval i."""
    for k in range(start.shape[0]):
        base[start[k] : end[k]] = score[k]
    return base


def process_bedgraph(interval, df):
    """Return per-base coverage for *interval* from bedGraph slice *df*."""

    df["start"] = df["start"].astype("int64") - int(interval.start)
    df["end"] = df["end"].astype("int64") - int(interval.start)

    df.loc[df.start < 0, "start"] = 0
    df.loc[df.end > len(interval), "end"] = len(interval)

    per_base = np.zeros(len(interval), dtype=np.float64)
    return get_per_base_score_f(
        df["start"].to_numpy().astype(np.int_),
        df["end"].to_numpy().astype(np.int_),
        df["score"].to_numpy().astype(np.float64),
        per_base,
    )


def resize_interval(interval_str: str, size: int) -> Tuple[str, int, int]:
    """Center-crop *interval_str* ("chr:start-end") to *size* bp."""
    chrom, rest = interval_str.split(":")
    start_s, end_s = rest.split("-")
    start, end = int(start_s.replace(",", "")), int(end_s.replace(",", ""))
    mid = (start + end) // 2
    return chrom, mid - size // 2, mid + size // 2


def plot_tracks(tracks: Dict[str, Tuple[np.ndarray, str]], start: int, end: int, y_lim: float, *, height: float = 1.5):
    """Convenience helper to plot multiple coverage tracks."""
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(24, height * (len(tracks) + 1)), sharex=True)
    for ax, (title, (data, color)) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(data)), data, color=color)
        ax.set_title(title)
        ax.set_ylim((0, y_lim))
    plt.tight_layout()


class FastaStringExtractor:
    """Lightweight FASTA wrapper returning upper-case sequences as strings."""

    def __init__(self, fasta_file: str):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        chrom_len = self._chromosome_sizes[interval.chrom]
        trimmed = Interval(interval.chrom, max(interval.start, 0), min(interval.end, chrom_len))
        seq = str(
            self.fasta.get_seq(trimmed.chrom, trimmed.start + 1, trimmed.stop).seq
        ).upper()
        pad_up = "N" * max(-interval.start, 0)
        pad_down = "N" * max(interval.end - chrom_len, 0)
        return pad_up + seq + pad_down

    def close(self):
        self.fasta.close() 