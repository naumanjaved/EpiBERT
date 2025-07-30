import argparse
import collections
import gzip
import math
import os
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import pybedtools as pybt
import tabix as tb

from datetime import datetime
from tensorflow import strings as tfs
from tensorflow.keras import initializers as inits
from scipy import stats
from scipy.signal import find_peaks

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
from kipoiseq import Interval
import pyfaidx
import kipoiseq

import logomaker

# ---------------------------------------------------------------------
# Shared utilities (avoid duplicate code across analysis scripts)
# ---------------------------------------------------------------------
from analysis.interval_plot_shared import (
    one_hot,
    process_bedgraph,
    get_per_base_score_f,
    resize_interval,
    plot_tracks,
    FastaStringExtractor,
)

def process_bedgraph(interval, df):
    """
    Extracts per-base coverage information for a given genomic interval from a dataframe.

    Args:
        interval (pybedtools.Interval): Genomic interval.
        df (pd.DataFrame): Dataframe containing 'start', 'end', and 'score' columns.

    Return np.ndarray: Array of per-base scores within the interval length.
    """
    df['start'] = df['start'].astype('int64') - int(interval.start)
    df['end'] = df['end'].astype('int64') - int(interval.start)

    # Ensure values are within interval bounds
    df.loc[df.start < 0, 'start'] = 0
    df.loc[df.end > len(interval), 'end'] = len(interval)

    per_base = np.zeros(len(interval), dtype=np.float64)
    return get_per_base_score_f(
        df['start'].to_numpy().astype(np.int_),
        df['end'].to_numpy().astype(np.int_),
        df['score'].to_numpy().astype(np.float64),
        per_base
    )


def get_per_base_score_f(start, end, score, base):
    """
    Fills a base array with scores over specified intervals.

    Args:
        start (np.ndarray): Start indices of intervals.
        end (np.ndarray): End indices of intervals.
        score (np.ndarray): Scores for each interval.
        base (np.ndarray): Base array to be updated.

    Return np.ndarray: Updated base array with scores.
    """
    for k in range(start.shape[0]):
        base[start[k]:end[k]] = score[k]
    return base


def return_bg_interval(atac_bedgraph, chrom, interval_start, interval_end, num_bins, resolution):
    """
    Processes ATAC-seq bedgraph data for a specified genomic interval.

    Args:
        atac_bedgraph (str): Path to the ATAC-seq bedgraph file.
        chrom (str): Chromosome.
        interval_start (int): Start position of the interval.
        interval_end (int): End position of the interval.
        num_bins (int): Number of bins for resolution adjustment.
        resolution (int): Bin resolution.

    Return tf.Tensor: Binned and summed ATAC-seq signal as a TensorFlow constant.
    """

    interval_str = '\t'.join([chrom, str(interval_start), str(interval_end)])
    interval_bed = pybt.BedTool(interval_str, from_string=True)
    interval = interval_bed[0]

    atac_bedgraph_bed = tb.open(atac_bedgraph)
    atac_subints = atac_bedgraph_bed.query(chrom, interval_start, interval_end)
    atac_subints_df = pd.DataFrame([rec for rec in atac_subints])

    if atac_subints_df.empty:
        return tf.constant([0.0] * num_bins, dtype=tf.float32)

    atac_subints_df.columns = ['chrom', 'start', 'end', 'score']
    atac_bedgraph_out = process_bedgraph(interval, atac_subints_df)

    atac_processed = np.sum(
        np.reshape(atac_bedgraph_out, [num_bins, resolution]), axis=1, keepdims=True
    )
    return tf.constant(atac_processed, dtype=tf.float32)


def resize_interval(interval_str, size):
    """
    Resizes a genomic interval to a specified length.

    Args:
        interval_str (str): Genomic interval in the format 'chrom:start-stop'.
        size (int): Desired size of the interval.

    Return tuple: Chromosome, start, and end of the resized interval.
    """
    chrom = interval_str.split(':')[0]
    start = int(interval_str.split(':')[1].split('-')[0].replace(',', ''))
    stop = int(interval_str.split(':')[1].split('-')[1].replace(',', ''))

    mid_point = (start + stop) // 2
    new_start = mid_point - size // 2
    new_stop = mid_point + size // 2

    return chrom, new_start, new_stop


def plot_tracks(tracks, start, end, y_lim, height=1.5):
    """
    Plots multiple genomic tracks over a specified interval.

    Args:
        tracks (dict): Dictionary where keys are track titles and values are tuples (data, color).
        start (int): Start of the interval.
        end (int): End of the interval.
        y_lim (float): Y-axis limit for the plots.
        height (float, optional): Height of each plot. Defaults to 1.5.

    Return None
    """
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(24, height * (len(tracks) + 1)), sharex=True)
    for ax, (title, (data, color)) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(data)), data, color=color)
        ax.set_title(title)
        ax.set_ylim((0, y_lim))
    plt.tight_layout()

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


class EpiBERT_model:
    """
    Manages the EpiBERT model, including initialization, prediction,
    and gradient-based input attribution.
    """

    def __init__(self, strategy, model, model_checkpoint):
        """
        Initializes the EpiBERT model by loading weights and setting up a test input.

        Args:
            strategy (tf.distribute.Strategy): TensorFlow distribution strategy.
            model (tf.keras.Model): The EpiBERT model.
            model_checkpoint (str): Path to the model checkpoint for restoring weights.
        """
        self.model = model

        # Create dummy inputs for initializing the model
        dummy_seq = tf.data.Dataset.from_tensor_slices([tf.ones((524288, 4), dtype=tf.float32)] * 6)
        dummy_atac = tf.data.Dataset.from_tensor_slices([tf.ones((131072, 1), dtype=tf.float32)] * 6)
        dummy_motif = tf.data.Dataset.from_tensor_slices([tf.ones((1, 693), dtype=tf.float32)] * 6)
        combined_dataset = tf.data.Dataset.zip((dummy_seq, dummy_atac, dummy_motif))
        batched_dataset = combined_dataset.batch(6)
        dist = strategy.experimental_distribute_dataset(batched_dataset)
        dist_it = iter(dist)

        print('Loading model...')

        @tf.function
        def build(input_dummy):
            self.model(input_dummy, training=False)

        # Run a test input to initialize the model
        strategy.run(build, args=(next(dist_it),))
        print('Test input ran successfully.')

        # Restore model weights
        ckpt = tf.train.Checkpoint(model=self.model)
        status = ckpt.restore(model_checkpoint).expect_partial()
        status.assert_existing_objects_matched()
        print('Model weights loaded.')

    def predict_on_batch_dist(self, strategy, inputs):
        """
        Runs distributed predictions on a batch of inputs.

        Args:
            strategy (tf.distribute.Strategy): TensorFlow distribution strategy.
            inputs (iterator): Distributed dataset iterator.

        Return Tuple of model outputs and intermediate attention matrices.
        """
        @tf.function
        def run_model(inputs):
            return self.model.predict_on_batch(inputs)

        outputs = strategy.run(run_model, args=(next(inputs),))
        return outputs

    def contribution_input_grad_dist(self, strategy, model_inputs, gradient_mask):
        """
        Computes gradient-based contributions of inputs to model predictions.

        Args:
            strategy (tf.distribute.Strategy): TensorFlow distribution strategy.
            model_inputs (tuple): Tuple of model inputs (seq, atac, tf_activity).
            gradient_mask (tf.Tensor): Mask specifying the region of interest for gradients.

        Return seq, seq_grads_all, atac_grads, prediction, att_matrices
        """
        @tf.function
        def compute_gradients(model_inputs, gradient_mask):
            seq, atac, _ = model_inputs
            gradient_mask = tf.cast(gradient_mask, dtype=tf.float32)
            gradient_mask_mass = tf.reduce_sum(gradient_mask)

            with tf.GradientTape() as tape:
                tape.watch(seq)
                tape.watch(atac)
                prediction, *rest = self.model.predict_on_batch(model_inputs)
                prediction_mask = tf.reduce_sum(gradient_mask * prediction) / gradient_mask_mass

            input_grads = tape.gradient(prediction_mask, model_inputs)
            seq_grads, atac_grads = input_grads[0], input_grads[1]
            atac_grads = atac_grads[0, :, 0] * atac[0, :, 0]  # Element-wise product
            return seq, seq_grads, atac_grads, prediction, rest

        seq, seq_grads, atac_grads, prediction, att_matrices = \
            strategy.run(compute_gradients, args=(next(model_inputs), gradient_mask))

        # Adjust sequence gradients to length 524287
        seq_grads_all = self._adjust_sequence_gradients(seq_grads)
        return seq, seq_grads_all, atac_grads, prediction, att_matrices

    def _adjust_sequence_gradients(self, seq_grads_orig):
        """
        Adjusts sequence gradients for alignment across augmentations.

        Args:
            seq_grads_orig (tf.Tensor): Original sequence gradients.

        Return List of adjusted sequence gradients.
        """
        seq_grads_min1 = seq_grads_orig.values[0][0, 2:, :]
        seq_grads = seq_grads_orig.values[1][0, 1:-1, :]
        seq_grads_max1 = seq_grads_orig.values[2][0, :-2, :]

        seq_grads_min1_r = tf.reverse(
            tf.gather(seq_grads_orig.values[3][0, :-2, :], [3, 2, 1, 0], axis=-1), axis=[0]
        )
        seq_grads_r = tf.reverse(
            tf.gather(seq_grads_orig.values[4][0, 1:-1, :], [3, 2, 1, 0], axis=-1), axis=[0]
        )
        seq_grads_max1_r = tf.reverse(
            tf.gather(seq_grads_orig.values[5][0, 2:, :], [3, 2, 1, 0], axis=-1), axis=[0]
        )

        return [seq_grads_min1, seq_grads, seq_grads_max1, seq_grads_min1_r, seq_grads_r, seq_grads_max1_r]



class EpiBERT_model_nostrat:
    """
    A wrapper for using two EpiBERT models without distributed TensorFlow strategy.
    Supports ensemble prediction, caQTL scoring, and gradient-based attribution.
    """

    def __init__(self, model1, model2, model1_checkpoint, model2_checkpoint):
        """
        Initializes the two EpiBERT models and loads their checkpoints.

        Args:
            model1 (tf.keras.Model): First EpiBERT model.
            model2 (tf.keras.Model): Second EpiBERT model.
            model1_checkpoint (str): Path to the checkpoint for model1.
            model2_checkpoint (str): Path to the checkpoint for model2.
        """
        self.model1 = model1
        self.model2 = model2

        # Dummy input to build model
        dummy_seq = tf.ones((1, 524288, 4), dtype=tf.float32)
        dummy_atac = tf.ones((1, 131072, 1), dtype=tf.float32)
        dummy_motif = tf.ones((1, 1, 693), dtype=tf.float32)
        _ = model1((dummy_seq, dummy_atac, dummy_motif), training=False)
        _ = model2((dummy_seq, dummy_atac, dummy_motif), training=False)

        # Load weights
        tf.train.Checkpoint(model=self.model1).restore(model1_checkpoint).expect_partial().assert_existing_objects_matched()
        tf.train.Checkpoint(model=self.model2).restore(model2_checkpoint).expect_partial().assert_existing_objects_matched()
        print("Loaded checkpoints.")

    def predict_on_batch(self, inputs):
        """
        Predicts output by averaging results from both models.

        Args:
            inputs (tuple): Input tuple (seq, atac, motif).

        Return tf.Tensor: Averaged prediction output [length,].
        """
        output1, _ = self.model1.predict_on_batch(inputs)
        output2, _ = self.model2.predict_on_batch(inputs)
        return tf.reduce_mean(tf.stack([output1, output2], axis=0), axis=0)[0, :, 0]

    def ca_qtl_score(self, inputs, inputs_mut):
        """
        Computes caQTL scores and averaged predicted signal between ref and alt alleles.

        Args:
            inputs (tuple): Tuple of (forward, reverse) inputs for the reference allele.
            inputs_mut (tuple): Tuple of (forward, reverse) inputs for the mutant allele.

        Return Tuple[tf.Tensor, tf.Tensor, float]: WT signal, mutant signal, caQTL score.
        """
        def model_caqtl_score(model):
            fwd, rev = model.predict_on_batch(inputs[0]), model.predict_on_batch(inputs[1])
            fwd_mut, rev_mut = model.predict_on_batch(inputs_mut[0]), model.predict_on_batch(inputs_mut[1])

            # Region of interest: positions 2044 to 2047
            wt_score = tf.reduce_sum(fwd[0][0, 2044:2047, 0])
            wt_rev_score = tf.reduce_sum(tf.reverse(rev[0], axis=[1])[0, 2044:2047, 0])
            mut_score = tf.reduce_sum(fwd_mut[0][0, 2044:2047, 0])
            mut_rev_score = tf.reduce_sum(tf.reverse(rev_mut[0], axis=[1])[0, 2044:2047, 0])

            caqtl = ((mut_score - wt_score) + (mut_rev_score - wt_rev_score)) / 2.0
            wt_signal = (fwd[0][0, :, 0] + tf.reverse(rev[0], axis=[1])[0, :, 0]) / 2.0
            mut_signal = (fwd_mut[0][0, :, 0] + tf.reverse(rev_mut[0], axis=[1])[0, :, 0]) / 2.0

            return wt_signal, mut_signal, caqtl.numpy()

        # Score from each model
        wt1, mut1, score1 = model_caqtl_score(self.model1)
        wt2, mut2, score2 = model_caqtl_score(self.model2)

        # Ensemble average
        output_wt = (wt1 + wt2) / 2.0
        output_mut = (mut1 + mut2) / 2.0
        caqtl_score = (score1 + score2) / 2.0

        return output_wt, output_mut, caqtl_score

    def contribution_input_grad(self, model_inputs, gradient_mask):
        """
        Computes input gradients (sequence and ATAC) using model1 only.

        Args:
            model_inputs (tuple): Input tuple (seq, atac, motif).
            gradient_mask (tf.Tensor): Mask for region to compute gradient over.

        Return Tuple: (sequence, sequence gradients, ATAC gradients, prediction, attention matrices)
        """
        @tf.function
        def compute_grad(model_inputs, mask):
            seq, atac, _ = model_inputs
            mask = tf.cast(mask, dtype=tf.float32)
            mask_sum = tf.reduce_sum(mask)

            with tf.GradientTape() as tape:
                tape.watch(seq)
                tape.watch(atac)

                prediction, attn = self.model1.predict_on_batch(model_inputs)
                masked_pred = tf.reduce_sum(prediction * mask) / mask_sum

            grads = tape.gradient(masked_pred, model_inputs)
            seq_grads = grads[0]
            atac_grads = grads[1][0, :, 0] * atac[0, :, 0]
            return seq, seq_grads, atac_grads, prediction, attn
        return compute_grad(model_inputs, gradient_mask)

def process_and_load_data(file_path):
    """
    Processes a motif activity file and returns a normalized TensorFlow tensor.
    - Reads a tab-delimited file.
      - Filters lines containing both 'consensus_pwms.meme' and 'AC'.
      - Extracts the third field (name) and the negated fifteenth field (activity score).
      - Sorts entries alphabetically by name.
      - Normalizes activity scores to [0, 1].
    Args:
        file_path (str): Path to the motif activity file.
    Return tf.Tensor: Normalized motif activity scores as a 1D float32 tensor.
    """
    processed_data = []

    # Read and process each line
    with open(file_path, 'r') as file:
        for line in file:
            if 'consensus_pwms.meme' in line and 'AC' in line:
                fields = line.strip().split('\t')
                try:
                    name = str(fields[2])
                    score = -1 * float(fields[14])
                    processed_data.append((name, score))
                except (IndexError, ValueError):
                    continue  # Skip malformed lines

    # Sort by motif name (field 3)
    processed_data.sort(key=lambda x: x[0])

    # Extract and normalize scores
    motif_scores = tf.constant([score for _, score in processed_data], dtype=tf.float32)
    min_val = tf.reduce_min(motif_scores)
    max_val = tf.reduce_max(motif_scores)

    # Normalize to [0, 1]
    normalized_scores = (motif_scores - min_val) / (max_val - min_val)
    return normalized_scores


def return_all_inputs(interval, atac_dataset, SEQUENCE_LENGTH, num_bins, resolution, motif_activity,
                      crop_size, output_length, fasta_extractor, mask_indices_list, strategy):
    """
    Processes input data and generates distributed datasets for model evaluation.
    Handles chromosomal interval resizing, ATAC-seq data extraction,
    one-hot sequence encoding, masking operations, and test-time augmentations.
    Args:
        interval (str): Genomic interval (e.g., 'chr1:100000-200000').
        atac_dataset (dataset): ATAC-seq dataset for extracting signals.
        SEQUENCE_LENGTH (int): Length of the sequence window to extract.
        num_bins (int): Number of bins for ATAC-seq signal.
        resolution (int): Resolution of data.
        motif_activity (array): Motif activity matrix.
        crop_size (int): Amount to crop from each side of the output.
        output_length (int): Length of the final output signal.
        fasta_extractor (object): Fasta file extractor for sequence extraction.
        mask_indices_list (str): Comma-separated list of masked index ranges.
        strategy (tf.distribute.Strategy): TensorFlow distribution strategy.
    Return dist_it (iterator): Distributed dataset iterator for training.
        target_atac (tf.Tensor): Cropped target ATAC-seq signal.
        masked_atac_reshape (tf.Tensor): Cropped masked ATAC-seq signal.
        mask (tf.Tensor): Cropped mask array.
        mask_centered (tf.Tensor): Cropped centered mask array.
    """
    # Extract chromosomal interval and corresponding ATAC-seq data
    chrom, start, stop = resize_interval(interval, SEQUENCE_LENGTH)
    atac_arr = return_bg_interval(atac_dataset, chrom, start, stop, num_bins, resolution)
    atac_arr += tf.abs(tf.random.normal(atac_arr.shape, mean=1e-4, stddev=1e-4, dtype=tf.float32))

    # Process and prepare motif activity data
    motif_activity = tf.expand_dims(process_and_load_data(motif_activity), axis=0)

    # Generate one-hot encoded sequence for extended interval
    extended_interval = kipoiseq.Interval(chrom, start - 1, stop + 1)
    sequence_one_hot_orig = tf.constant(one_hot(fasta_extractor.extract(extended_interval)), dtype=tf.float32)

    # Create masking arrays
    mask, mask_centered, atac_mask = create_masks(mask_indices_list, SEQUENCE_LENGTH)

    # Apply masking to ATAC-seq data
    masked_atac = apply_mask_to_atac(atac_arr, atac_mask, clip_value=150.0)

    # Reshape and crop masked ATAC-seq data
    masked_atac_reshape = reshape_and_crop(masked_atac, crop_size, output_length)

    # Reshape and crop target ATAC-seq data
    target_atac = process_target_atac(atac_arr, crop_size, output_length)

    # Perform test-time augmentations
    sequences, sequences_rev = generate_augmented_sequences(sequence_one_hot_orig, output_length)
    masked_atac_rev = tf.reverse(masked_atac, axis=[0])

    # Create datasets for distribution
    seqs = tf.data.Dataset.from_tensor_slices(sequences + sequences_rev)
    atacs = tf.data.Dataset.from_tensor_slices([masked_atac] * 3 + [masked_atac_rev] * 3)
    motifs = tf.data.Dataset.from_tensor_slices([motif_activity] * 6)

    combined_dataset = tf.data.Dataset.zip((seqs, atacs, motifs))
    batched_dataset = combined_dataset.batch(6).repeat(2)
    dist = strategy.experimental_distribute_dataset(batched_dataset)

    # Crop and finalize masks
    mask = crop_mask(mask, crop_size, output_length)


def return_inputs_caqtl_score(variant, atac_dataset, motif_activity,
                              fasta_extractor, SEQUENCE_LENGTH=524288, num_bins=131072,
                              resolution=4, crop_size=2, output_length=4096,
                              mask_indices_list='2043-2048'):
    """
    Processes genomic and ATAC-seq data to prepare inputs for cAQTL analysis.

    Args:
        variant (list): Variant information [chrom:pos].
        atac_dataset (dataset): ATAC-seq dataset for extracting signals.
        motif_activity (array): Motif activity matrix.
        fasta_extractor (object): Fasta file extractor for sequence extraction.
        SEQUENCE_LENGTH (int): Length of the sequence window to extract.
        num_bins (int): Number of bins for ATAC-seq signal.
        resolution (int): Resolution of data.
        crop_size (int): Amount to crop from each side of the output.
        output_length (int): Length of the final output signal.
        mask_indices_list (str): Comma-separated list of masked index ranges.

    Return tuple: Prepared inputs for the model, including wild-type and mutated sequences,
               ATAC-seq signals, motif activity, target ATAC, and masks.
    """

    # Extract and resize variant interval
    var_chrom, var_start = variant[0].split(':')[0], int(variant[0].split(':')[1])
    var_end = var_start + 1
    chrom, start, stop = resize_interval(f"{var_chrom}:{var_start}-{var_end}", SEQUENCE_LENGTH)

    # Get ATAC-seq signal for the interval
    atac_arr = return_bg_interval(atac_dataset, chrom, start, stop, num_bins, resolution)

    # Process motif activity
    motif_activity = tf.expand_dims(process_and_load_data(motif_activity), axis=0)

    # Extract wild-type sequence
    interval = kipoiseq.Interval(chrom, start, stop)
    sequence_one_hot_orig = one_hot(fasta_extractor.extract(interval)).numpy()

    # Modify sequence to include the variant
    chrom, pos, alt = parse_var(variant)
    sub_pos = int(pos) - start - 1
    sequence_one_hot_orig_mod = np.concatenate(
        (sequence_one_hot_orig[:sub_pos, :], one_hot(alt), sequence_one_hot_orig[sub_pos+1:, :]), axis=0
    )

    # Convert sequences to tensors and create reverse complements
    sequence_one_hot_orig = tf.constant(sequence_one_hot_orig, dtype=tf.float32)
    sequence_one_hot_orig_rev = tf.reverse(
        tf.gather(sequence_one_hot_orig, [3, 2, 1, 0], axis=-1), axis=[0]
    )

    sequence_one_hot_orig_mod = tf.constant(sequence_one_hot_orig_mod, dtype=tf.float32)
    sequence_one_hot_orig_mod_rev = tf.reverse(
        tf.gather(sequence_one_hot_orig_mod, [3, 2, 1, 0], axis=-1), axis=[0]
    )

    # Create masks for specified index ranges
    mask, mask_centered, atac_mask = create_masks(SEQUENCE_LENGTH, mask_indices_list)
    masked_atac = apply_mask_to_atac(atac_arr, atac_mask)

    # Reverse and reshape masked ATAC-seq signals
    masked_atac_rev = tf.reverse(masked_atac, axis=[0])
    masked_atac_reshape = reshape_and_crop(masked_atac, crop_size, output_length)

    # Process and crop target ATAC-seq signals
    target_atac = process_target_atac(atac_arr, crop_size, output_length)

    # Crop masks to match the output length
    mask = crop_mask(mask, crop_size, output_length)
    mask_centered = crop_mask(mask_centered, crop_size, output_length)

    # Prepare inputs for wild-type and mutated sequences
    inputs = prepare_inputs(sequence_one_hot_orig, masked_atac, motif_activity, sequence_one_hot_orig_rev, masked_atac_rev)
    inputs_mut = prepare_inputs(sequence_one_hot_orig_mod, masked_atac, motif_activity, sequence_one_hot_orig_mod_rev, masked_atac_rev)

    return inputs, inputs_mut, masked_atac, motif_activity, target_atac, \
           masked_atac_reshape[:, 0], mask[0, :, 0], mask_centered, (chrom, start, stop)


def create_masks(SEQUENCE_LENGTH, mask_indices_list):
    """Creates binary masks for specified index ranges."""
    mask = np.zeros((1,SEQUENCE_LENGTH//128,1))
    mask_centered = np.zeros((1,SEQUENCE_LENGTH//128,1))
    atac_mask = np.ones((SEQUENCE_LENGTH//128,1))
    for entry in mask_indices_list.split(','):
        mask_start = int(entry.split('-')[0])
        mask_end = int(entry.split('-')[1])

        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start,mask_end):
                mask[0,k,0]=1
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start+5,mask_end-3):
                mask_centered[0,k,0]=1
        for k in tf.range(mask_start,mask_end):
            atac_mask[k,0] = 0.0

    atac_mask = tf.constant(atac_mask,dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1,32]),[-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)

    return mask, mask_centered, atac_mask

def apply_mask_to_atac(atac_arr, atac_mask, clip_value=150.0):
    masked_atac = atac_arr * atac_mask
    diff = tf.sqrt(tf.nn.relu(masked_atac - clip_value))
    return tf.clip_by_value(masked_atac, 0.0, clip_value) + diff


def reshape_and_crop(masked_atac, crop_size, output_length):
    """Reshapes and crops masked ATAC-seq signals."""
    reshaped = tf.reduce_sum(tf.reshape(masked_atac, [-1, 32]), axis=1, keepdims=True)
    return tf.slice(reshaped, [crop_size, 0], [output_length - 2 * crop_size, -1])


def process_target_atac(atac_arr, crop_size, output_length):
    """Processes and crops target ATAC-seq signals."""
    target_atac = tf.reduce_sum(tf.reshape(atac_arr, [-1, 32]), axis=1, keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(target_atac - 2000.0))
    target_atac = tf.clip_by_value(target_atac, 0.0, 2000.0) + diff
    return tf.slice(target_atac, [crop_size, 0], [output_length - 2 * crop_size, -1])


def crop_mask(mask, crop_size, output_length):
    """Crops the mask to match the desired output length."""
    return tf.slice(mask, [0, crop_size, 0], [-1, output_length - 2 * crop_size, -1])


def prepare_inputs(seq, atac, motif, seq_rev, atac_rev):
    """Prepares inputs for both wild-type and mutated sequences."""
    return (
        (
            tf.expand_dims(seq, axis=0),
            tf.expand_dims(atac, axis=0),
            tf.expand_dims(motif, axis=0)
        ),
        (
            tf.expand_dims(seq_rev, axis=0),
            tf.expand_dims(atac_rev, axis=0),
            tf.expand_dims(motif, axis=0)
        )
    )

def generate_augmented_sequences(sequence_one_hot, output_length):
    slices = [
        tf.slice(sequence_one_hot, [offset, 0], [output_length * 128, -1])
        for offset in range(3)
    ]

    slices_rev = [
        tf.reverse(tf.gather(seq, [3, 2, 1, 0], axis=-1), axis=[0])
        for seq in slices
    ]

    return slices, slices_rev

def plot_logo(matrix, y_min, y_max):
    """
    Plots a sequence logo using Logomaker.

    Args:
        matrix (np.ndarray or list): A matrix of shape (L, 4) with base frequencies or saliencies.
        y_min (float): Minimum y-axis value.
        y_max (float): Maximum y-axis value.
    """
    df = pd.DataFrame(matrix, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'

    logo = logomaker.Logo(df)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True, bounds=[y_min, y_max])
    logo.ax.set_ylim([y_min, y_max])
    logo.ax.set_yticks([])
    logo.ax.set_yticklabels([])
    logo.ax.set_ylabel('saliency', labelpad=-1)


def write_bg(window, seq_length, crop, input_arr, output_res, out_file_name):
    """
    Writes bedgraph-style coverage from array to a file.

    Args:
        window (str): Genomic interval string (e.g., 'chr1:10000-20000').
        seq_length (int): Total length of sequence.
        crop (int): Amount of cropping applied.
        input_arr (array-like): Coverage values to write.
        output_res (int): Resolution per bin.
        out_file_name (str): Output file path.
    """
    chrom, start, _ = resize_interval(window, seq_length)
    start += crop * output_res

    with open(out_file_name, 'w') as out_file:
        for k, value in enumerate(input_arr):
            start_interval = k * output_res + start
            end_interval = (k + 1) * output_res + start
            out_file.write(f"{chrom}\t{start_interval}\t{end_interval}\t{value}\n")


def return_all_inputs_no_strategy(interval, atac_dataset, SEQUENCE_LENGTH, num_bins, resolution,
                                   motif_activity, crop_size, output_length, fasta_extractor, mask_indices_list):
    """
    Prepares model input tensors without a distribution strategy.

    Args:
        interval (str): Genomic interval string.
        atac_dataset: ATAC-seq dataset.
        SEQUENCE_LENGTH (int): Total input sequence length.
        num_bins (int): Number of bins for ATAC signal.
        resolution (int): Resolution of ATAC signal.
        motif_activity (str): Path to motif activity file.
        crop_size (int): Cropping size in output bins.
        output_length (int): Final output sequence length.
        fasta_extractor: Object to extract sequences from FASTA.
        mask_indices_list (str): Mask index ranges in format 'start-end,...'.

    Return tuple: Sequence input, masked ATACs, motif activity, target ATAC, reshaped masked ATAC,
               forward/reverse masks, centered masks.
    """
    chrom, start, stop = resize_interval(interval, SEQUENCE_LENGTH)
    atac_arr = return_bg_interval(atac_dataset, chrom, start, stop, num_bins, resolution)
    atac_arr += tf.abs(tf.random.normal(atac_arr.shape, mean=1e-4, stddev=1e-4))

    motif_activity = tf.expand_dims(process_and_load_data(motif_activity), axis=0)
    interval = kipoiseq.Interval(chrom, start - 1, stop + 1)
    sequence_one_hot_orig = tf.constant(one_hot(fasta_extractor.extract(interval)), dtype=tf.float32)
    sequence_one_hot = tf.slice(sequence_one_hot_orig, [1, 0], [output_length * 128, -1])

    mask, mask_centered, atac_mask = create_masks(mask_indices_list, SEQUENCE_LENGTH)
    masked_atac = apply_mask_to_atac(atac_arr, atac_mask, clip_value=150.0)
    masked_atac_rev = tf.reverse(masked_atac, axis=[0])

    masked_atac_reshape = reshape_and_crop(masked_atac, crop_size, output_length)
    target_atac = process_target_atac(atac_arr, crop_size, output_length)

    mask = crop_mask(mask, crop_size, output_length)
    mask_centered = crop_mask(mask_centered, crop_size, output_length)
    mask_rev = tf.reverse(mask, axis=[0])
    mask_centered_rev = tf.reverse(mask_centered, axis=[0])

    return sequence_one_hot, (masked_atac, masked_atac_rev), motif_activity, \
           target_atac, masked_atac_reshape, (mask, mask_rev), (mask_centered, mask_centered_rev)


def return_grads(model1, model2):
    """
    Returns a function to compute ensemble input gradients using two models.

    Args:
        model1, model2: Trained EpiBERT models.

    Return Callable: A tf.function for computing gradients given model inputs and mask.
    """
    @tf.function
    def contribution_input_grad_dist(inputs, gradient_mask):
        sequence, rev_seq, atac, rev_atac, mask, mask_rev, target, target_rev, motif_activity, interval_id, cell_type = inputs
        gradient_mask = tf.cast(gradient_mask, dtype=tf.float32)
        gradient_mask_mass = tf.reduce_sum(gradient_mask)

        def compute(model, seq, atac):
            with tf.GradientTape() as tape:
                tape.watch(seq)
                tape.watch(atac)
                pred, *_ = model.predict_on_batch((seq, atac, motif_activity))
                loss = tf.reduce_sum(gradient_mask * pred) / gradient_mask_mass
            grads = tape.gradient(loss, (seq, atac))
            return grads[0]

        grads_fwd_1 = compute(model1, sequence, atac)
        grads_rev_1 = tf.reverse(tf.gather(compute(model1, rev_seq, rev_atac), [3, 2, 1, 0], axis=-1), axis=[0])
        grads_fwd_2 = compute(model2, sequence, atac)
        grads_rev_2 = tf.reverse(tf.gather(compute(model2, rev_seq, rev_atac), [3, 2, 1, 0], axis=-1), axis=[0])

        all_grads = grads_fwd_1 + grads_rev_1 + grads_fwd_2 + grads_rev_2
        sub_grads = all_grads[:, 261888:262400, :] / 4.0
        sub_grads /= tf.reduce_max(tf.abs(sub_grads))
        seq_sub = sequence[:, 261888:262400, :]

        return seq_sub, sub_grads

    return contribution_input_grad_dist
def deserialize_test(serialized_example, g, use_motif_activity, mask, atac_mask,
                     input_length=196608, max_shift=10, output_length_ATAC=49152,
                     output_length=1536, crop_size=2, output_res=128,
                     mask_size=1536, log_atac=False, use_atac=True, use_seq=True):
    """
    Parses and deserializes a single serialized example from a TFRecord.
    Applies stochastic shifting, normalization, masking, and returns inputs.

    Return tuple: One-hot sequence, reverse sequence, masked ATAC, reversed ATAC,
               masks, reversed masks, targets, reversed targets, motif activity,
               interval ID, cell type.
    """
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'motif_activity': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([], tf.string),
        'cell_type': tf.io.FixedLenFeature([], tf.string)
    }

    seq_shift = 5
    data = tf.io.parse_example(serialized_example, feature_map)

    sequence = one_hot(tf.strings.substr(data['sequence'], seq_shift, 524288))

    atac = tf.io.parse_tensor(data['atac'], out_type=tf.float16)
    atac = tf.cast(atac, tf.float32) + tf.abs(g.normal(atac.shape, mean=1e-4, stddev=1e-4))
    atac_target = atac

    motif_activity = tf.cast(tf.io.parse_tensor(data['motif_activity'], out_type=tf.float16), tf.float32)
    motif_activity = (motif_activity - tf.reduce_min(motif_activity)) / (tf.reduce_max(motif_activity) - tf.reduce_min(motif_activity))
    motif_activity = tf.expand_dims(motif_activity, axis=0)

    interval_id = tf.io.parse_tensor(data['interval'], out_type=tf.int32)
    cell_type = tf.io.parse_tensor(data['cell_type'], out_type=tf.int32)

    atac_mask = tf.constant(atac_mask, dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1, 32]), [-1])
    atac_mask = tf.expand_dims(atac_mask, axis=1)

    masked_atac = atac * atac_mask
    masked_atac = tf.clip_by_value(masked_atac, 0.0, 150.0) + tf.sqrt(tf.nn.relu(masked_atac - 150.0))

    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1, 32]), axis=1, keepdims=True)
    atac_out = tf.clip_by_value(atac_out, 0.0, 2000.0) + tf.sqrt(tf.nn.relu(atac_out - 2000.0))
    atac_out = tf.slice(atac_out, [crop_size, 0], [output_length - 2 * crop_size, -1])

    mask = tf.slice(mask, [crop_size, 0], [output_length - 2 * crop_size, -1])

    rev_seq = tf.reverse(tf.gather(sequence, [3, 2, 1, 0], axis=-1), axis=[0])
    masked_atac_rev = tf.reverse(masked_atac, axis=[0])
    mask_rev = tf.reverse(mask, axis=[0])
    atac_out_rev = tf.reverse(atac_out, axis=[0])

    return tf.cast(sequence, tf.bfloat16), tf.cast(rev_seq, tf.bfloat16), \
           tf.cast(masked_atac, tf.bfloat16), tf.cast(masked_atac_rev, tf.bfloat16), \
           tf.cast(mask, tf.int32), tf.cast(mask_rev, tf.int32), \
           tf.cast(atac_out, tf.float32), tf.cast(atac_out_rev, tf.float32), \
           tf.cast(motif_activity, tf.bfloat16), tf.cast(interval_id, tf.int32), tf.cast(cell_type, tf.int32)


def return_dataset(gcs_path, batch, input_length, output_length_ATAC, output_length, crop_size,
                   output_res, max_shift, options, num_parallel, mask, atac_mask,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   use_motif_activity, g):
    """
    Returns a tf.data.Dataset built from GCS path and preprocessed using deserialize_test.
    """
    files = tf.io.gfile.glob(gcs_path)
    dataset = tf.data.TFRecordDataset(files, compression_type='ZLIB', num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    dataset = dataset.map(lambda record: deserialize_test(
        record, g, use_motif_activity, mask, atac_mask, input_length, max_shift,
        output_length_ATAC, output_length, crop_size, output_res, random_mask_size,
        log_atac, use_atac, use_seq),
        deterministic=True, num_parallel_calls=num_parallel)

    return dataset.repeat(2).batch(batch).prefetch(tf.data.AUTOTUNE)


def return_distributed_iterators(gcs_path, global_batch_size, input_length, max_shift, output_length_ATAC,
                                  output_length, crop_size, output_res, num_parallel_calls, strategy,
                                  options, random_mask_size, mask, atac_mask, log_atac, use_atac,
                                  use_seq, seed, use_motif_activity, g):
    """
    Returns a distributed iterator for test data.
    """
    dataset = return_dataset(gcs_path, global_batch_size, input_length, output_length_ATAC,
                              output_length, crop_size, output_res, max_shift, options,
                              num_parallel_calls, mask, atac_mask, random_mask_size,
                              log_atac, use_atac, use_seq, seed, use_motif_activity, g)

    return iter(strategy.experimental_distribute_dataset(dataset))


def parse_var(variant):
    """Parses variant input like ['chr1:12345-12346', 'A'] to components."""
    chrom = variant[0].split(':')[0]
    pos = int(variant[0].split(':')[1].split('-')[0].replace(',', ''))
    return chrom, pos, variant[1]


def parse_var_long(variant):
    """Parses variant input with ALT sequence and returns length."""
    chrom, pos, alt = parse_var(variant)
    return chrom, pos, len(alt), alt


def parse_gtf_collapsed(gtf_file, chromosome, start, end):
    genes = {}
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            columns = line.strip().split("\t")
            if len(columns) < 9:
                continue
            feature_type = columns[2]
            chrom = columns[0]
            feature_start = int(columns[3])
            feature_end = int(columns[4])
            if chrom != chromosome or feature_end < start or feature_start > end:
                continue

            # Parse attributes in the 9th column
            attributes = {}
            for attr in columns[8].split(';'):
                attr = attr.strip()
                if attr:
                    parts = attr.split(' ', 1)
                    if len(parts) == 2:
                        key, value = parts
                        attributes[key.strip()] = value.strip('"')

            gene_name = attributes.get("gene_name", "NA")

            if feature_type == "exon":
                if gene_name not in genes:
                    genes[gene_name] = {
                        "start": feature_start,
                        "end": feature_end,
                        "exons": [],
                    }
                # Extend the gene's range
                genes[gene_name]["start"] = min(genes[gene_name]["start"], feature_start)
                genes[gene_name]["end"] = max(genes[gene_name]["end"], feature_end)
                # Add the exon
                genes[gene_name]["exons"].append((feature_start, feature_end))

    # Merge overlapping exons for each gene
    for gene in genes.values():
        gene["exons"] = merge_intervals(gene["exons"])

    return genes


def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    # Sort intervals by start position
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            # Merge intervals
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def plot_collapsed_gene_track(ax, genes, start, end, chromosome):
    """
    Plots collapsed gene structures on a Matplotlib axis.
    """
    y = 0  # Track height for stacking genes
    for gene_name, gene_data in genes.items():
        # Plot the entire gene range
        ax.plot([gene_data["start"], gene_data["end"]], [y, y], color="black", linewidth=1)
        # Plot each merged exon
        for exon_start, exon_end in gene_data["exons"]:
            ax.add_patch(plt.Rectangle(
                (exon_start, y - 0.2), exon_end - exon_start, 0.4, color="blue", alpha=0.7
            ))
        # Add gene name with larger font size
        ax.text((gene_data["start"] + gene_data["end"]) / 2, y + 0.4,
                gene_name, fontsize=10, ha="center", va="bottom", color="black")
        y -= 1  # Move to the next track

    ax.set_xlim(start, end)
    ax.set_ylim(y, 1)
    ax.set_yticks([])  # Remove y-axis ticks and labels
    ax.spines['top'].set_visible(False)  # Remove the top border
    ax.spines['right'].set_visible(False)  # Remove the right border
    ax.spines['left'].set_visible(False)  # Remove the left border
    ax.spines['bottom'].set_visible(False)  # Remove the bottom border

def plot_tracks_with_genes(tracks, gtf_file, interval, y_lim, height=1.5):
    """
    Plots tracks along with a collapsed gene model from a GTF.
    """
    chromosome,start,end=interval
    # Parse the GTF file to extract collapsed genes
    genes = parse_gtf_collapsed(gtf_file, chromosome, start, end)

    # Create subplots
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(24, height * (len(tracks) + 1)), sharex=True)

    # Plot collapsed gene track
    plot_collapsed_gene_track(axes[0], genes, start, end,chromosome)

    # Plot other tracks
    for ax, (title, y) in zip(axes[1:], tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(y[0])), y[0], color=y[1])
        ax.set_title(title)
        ax.set_ylim((0, y_lim))

    # Add chromosome label at the bottom of the whole figure
    label = f"{chromosome}: {start} - {end}"
    fig.text(0.5, -0.05, label, ha="center", va="center", fontsize=12, color="black")

    plt.tight_layout()
    plt.show()


def get_caqtl_score(output, output_mut):
    """
    Returns caQTL delta score from 2044-2047 window.
    """
    return tf.reduce_sum(output_mut[2044:2047]) - tf.reduce_sum(output[2044:2047])