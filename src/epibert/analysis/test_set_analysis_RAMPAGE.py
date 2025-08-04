import time
import os
import subprocess
import sys
sys.path.insert(0, '/home/jupyter/repos/genformer')
import re
import argparse
import collections
import gzip
import math
import shutil

import numpy as np
import time
from datetime import datetime
import random

import seaborn as sns

import logging
os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf

import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  
## custom modules
import src.models.aformer_atac_rna as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers

import training_utils_atac_rna as training_utils

from scipy import stats

def gaussian_kernel(size: int, std: float):
    """Generate a Gaussian kernel for smoothing."""
    d = tf.range(-(size // 2), (size // 2) + 1, dtype=tf.float32)
    gauss_kernel = tf.exp(-tf.square(d) / (2*std*std))
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    return gauss_kernel[..., tf.newaxis, tf.newaxis]

@tf.function
def deserialize_test(serialized_example, g_val,
                   input_length = 524288, max_shift = 4, output_length_ATAC = 262144,
                   output_length = 4096, crop_size = 1600, output_res = 128,
                   atac_mask_dropout = 0.01, mask_size = 896):

    """
    Deserialize a serialized example from a TFRecordFile to return validation data. 
    For validation, we will mask one peak of the input atac profile  and apply atac_mask_dropout 
    percentage to remaining profile will also be masked. Also, the sequence/targets will be 
    randomly reverse complemented.

    Parameters:
    - serialized_example: Serialized example from a TFRecord file.
    - g: TensorFlow random number generator.
    - use_motif_activity: Whether to use TF activity  or provide random noise for ablations
    - input_length: expected length of input sequence
    - max_shift: maximum number of bases to shift the input sequence
    - output_length_ATAC: expected length of the ATAC profile
    - output_length: expected length of the target ATAC profile (should be output_length_ATAC // output_res)
    - crop_size: number of bins to crop from the edges of the target ATAC profile
    - output_res: resolution of the target ATAC profile, default is 128
    - atac_mask_dropout: percentage of the target ATAC profile to mask
    - mask_size: size of the masked region in the target ATAC profile, expressed as number of bp
    - log_atac: whether to log transform the input ATAC profile
    - use_atac: whether to use the input ATAC profile
    - use_seq: whether to use the input sequence
    Returns:
    - Tuple of processed sequence, masked ATAC, mask, ATAC output, and TF activity tensors.
    """

    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'motif_activity': tf.io.FixedLenFeature([], tf.string),
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'rna': tf.io.FixedLenFeature([], tf.string), # rampage profile, input
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'processed_gene_token': tf.io.FixedLenFeature([], tf.string),
        'cell_encoding': tf.io.FixedLenFeature([], tf.string)
    }

    ## now parse out the actual data
    # parse out the actual data 
    data = tf.io.parse_example(serialized_example, feature_map)

    # atac input, cast to float32 
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'], out_type=tf.float32), [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
    atac = atac #+ tf.math.abs(g_val.normal(atac.shape,mean=1.0e-04,stddev=1.0e-04,dtype=tf.float32))
    atac_target = atac ## store the target ATAC, as we will subsequently directly manipulate atac for masking

    # rna output, cast to float32 
    rna = tf.ensure_shape(tf.io.parse_tensor(data['rna'], out_type=tf.float32), [output_length,1])
    rna = tf.clip_by_value(rna, clip_value_min=0.0, clip_value_max=65500.0)
    rna = tf.math.pow(rna,0.50)
    rna = tf.expand_dims(rna, axis=0)
    gauss_kernel = gaussian_kernel(3, 1)
    gauss_kernel = tf.cast(gauss_kernel, dtype=tf.float32) 
    rna = tf.nn.conv1d(rna, filters=gauss_kernel, stride=1, padding='SAME')
    rna = tf.squeeze(rna, axis=0)
    rna = tf.slice(rna, [crop_size,0], [output_length-2*crop_size,-1]) # crop at the outset

    # tss tokens, cast to float32 
    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'], out_type=tf.int32), [output_length])
    tss_tokens = tf.expand_dims(tss_tokens,axis=1)
    tss_tokens = tf.cast(tss_tokens,dtype=tf.float32)
    tss_tokens = tf.slice(tss_tokens, [crop_size,0], [output_length-2*crop_size,-1]) # crop at the outset 

    # set up a semi-random seem based on the number of
    # peaks and atac signal in the window
    # get peaks centers 
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'], out_type=tf.int32), [output_length])
    peaks_center = tf.expand_dims(peaks_center,axis=1)
    peaks_c_crop = tf.slice(peaks_center, [crop_size,0], [output_length-2*crop_size,-1]) # crop at the outset

    gene_token= tf.io.parse_tensor(data['processed_gene_token'],
                                   out_type=tf.int32)

    cell_type = tf.io.parse_tensor(data['cell_encoding'],
                                  out_type=tf.int32)

    peaks_sum = tf.reduce_sum(peaks_center)

    randomish_seed = peaks_sum + tf.cast(tf.reduce_sum(atac),dtype=tf.int32)

    shift = 2
    
    rev_comp = tf.math.round(g_val.uniform([], 0, 1)) #switch for random reverse complementation
    
    # sequence, get substring based on sequence shift, one_hot
    sequence = one_hot(tf.strings.substr(data['sequence'], shift,input_length))

    # motif activity, cast to float32 and expand dims to allow for processing by model input FC layers
    motif_activity = tf.ensure_shape(tf.io.parse_tensor(data['motif_activity'], out_type=tf.float32), [693])
    motif_activity = tf.cast(motif_activity,dtype=tf.float32)
    motif_activity = tf.expand_dims(motif_activity,axis=0)
    min_val = tf.reduce_min(motif_activity)
    max_val = tf.reduce_max(motif_activity)
    motif_activity = (motif_activity - min_val) / (max_val - min_val)
    
    # generate ATAC mask 
    full_comb_mask, full_comb_mask_store,full_comb_unmask_store = mask_ATAC_profile(
                                                output_length_ATAC,
                                                output_length,
                                                crop_size,
                                                mask_size,
                                                output_res,
                                                peaks_c_crop,
                                                randomish_seed,
                                                1, # set atac_mask_int to 1 to prevent increased masking used in training
                                                atac_mask_dropout)

    masked_atac = atac * full_comb_mask ## apply the mask to the input profile

    ## input ATAC clipping
    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff

    # generate the output atac profile by summing the inputs to a desired resolution
    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape))) # soft clip the targets
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out, [crop_size,0], [output_length-2*crop_size,-1]) # crop to desired length


    return tf.cast(tf.ensure_shape(sequence, 
                                   [input_length,4]),dtype=tf.bfloat16), \
                tf.cast(masked_atac,dtype=tf.bfloat16), \
                tf.cast(full_comb_mask_store,dtype=tf.int32), \
                tf.cast(full_comb_unmask_store,dtype=tf.int32), \
                tf.cast(atac_out,dtype=tf.float32), \
                tf.cast(motif_activity,dtype=tf.bfloat16), \
                tf.cast(rna,dtype=tf.float32), \
                tss_tokens, \
                            gene_token, cell_type


def return_dataset(gcs_path, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, atac_mask_dropout,
                   random_mask_size, seed, g):

    """
    return a tf dataset object for given gcs path
    """

    list_files = (tf.io.gfile.glob(gcs_path))

    files = tf.data.Dataset.list_files(list_files,shuffle=False)
    dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    dataset = dataset.map(lambda record: deserialize_test(
                                                record, g,
                                                input_length, max_shift,
                                                output_length_ATAC, output_length,
                                                crop_size, output_res,
                                                atac_mask_dropout, random_mask_size),
                      deterministic=True,
                      num_parallel_calls=num_parallel)

    return dataset.take(1731*8).batch(batch).repeat(2).prefetch(tf.data.AUTOTUNE)


def return_distributed_iterators(gcs_path, global_batch_size,
                                 input_length, max_shift, output_length_ATAC,
                                 output_length, crop_size, output_res,
                                 num_parallel_calls, strategy,
                                 options,
                                 atac_mask_dropout,
                                 random_mask_size,seed,g):

    test_data = return_dataset(gcs_path, global_batch_size, input_length,
                             output_length_ATAC, output_length, crop_size,
                             output_res, max_shift, options, num_parallel_calls,atac_mask_dropout,
                               random_mask_size,seed, g)

    test_dist = strategy.experimental_distribute_dataset(test_data)
    test_data_it = iter(test_dist)

    return test_data_it


def return_test_build(model,metric_dict, strategy):
    
    metric_dict['RNA_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['RNA_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    
    @tf.function(reduce_retracing=True)
    def dist_test_step(inputs):
        sequence, atac,mask,unmask, target,motif_activity,rna, tss_tokens, gene_token,cell_type=inputs

        
        input_tuple = sequence,atac,motif_activity

        output_profile = model(input_tuple, training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32

        target = tf.math.reduce_sum((rna * tss_tokens)[:,:,0], axis=1)
        output = tf.math.reduce_sum((output_profile * tss_tokens)[:,:,0], axis=1)
        
        
        metric_dict['RNA_PearsonR'].update_state(rna, output_profile)
        metric_dict['RNA_R2'].update_state(rna, output_profile)
        """
        inputs_rev = seq_rev,atac_rev,motif_activity

        output_rev = model(inputs_rev, training=False)
        output_rev = tf.cast(output_rev,dtype=tf.float32) # ensure cast to float32

        target_rev = tf.math.reduce_sum((rna_rev * tss_tokens_rev)[:,:,0], axis=1)
        output_rev = tf.math.reduce_sum((output_rev * tss_tokens_rev)[:,:,0], axis=1)
        """
        #target_mean = (target + target_rev) / 2.0
        #output_mean = (output + output_rev) / 2.0
        
        return target, output, gene_token, cell_type
    
    

    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,atac,motif_activity=inputs

            input_tuple = sequence,atac,motif_activity

            output_profile = model(input_tuple, training=False)

        #for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
        strategy.run(val_step, args=(next(iterator),))

    return dist_test_step, build_step


def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=4)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters),
                      depth = 5,
                      dtype=tf.float32)[:, :4]
    return out




def mask_ATAC_profile(output_length_ATAC, output_length, crop_size, mask_size,output_res, 
                        peaks_c_crop, randomish_seed, atac_mask_int, atac_mask_dropout):
    """
    Creates a random mask for the input ATAC profile
    The mask consists of 1s and 0s, where 1s indicate masked regions.
    Mask is generated by selecting one peak to mask in the window if present,
    and randomly masking atac_mask_dropout % of the remaining bins

    Parameters:
    - output_length_ATAC: the length of the input ATAC profile (at 4 bp resolution)
    - output_length: The length of the output ATAC profile
    - crop_size: The size of the cropped region to take from each edge of output ATAC profile
    - mask_size: The size of the masked region in basepairs
    - output_res: The resolution of the output target ATAC profile
    - peaks_c_crop: The center of the peaks in the output ATAC profile
    - randomish_seed: A random seed to use for the random masking
    - atac_mask_dropout: The dropout rate for the ATAC masking
    Returns:
    - The mask as well as inverted mask. Inverted mask is used to set the masked regions to 0. 
    """

    num_mask_bins = mask_size // output_res ## calculate the number of bins that will be masked in each region
    out_length_cropped = output_length-2*crop_size
    if out_length_cropped % num_mask_bins != 0:
        raise ValueError('ensure that masking size divided by output res is a factor of the cropped output length')
    center = (output_length-2*crop_size)//2 # compute center of the output window

    
    # -------------------------- here we set up masking one of the peaks ----------------------------------------
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0] # get indices of peak centers, randomly select one
    mask_indices_temp = tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5])

    if tf.size(mask_indices_temp) > 0: # in the case where we did actually have peaks in the window
        ridx = tf.concat([mask_indices_temp],axis=0) # adjust dimensions by concatenating selected indices
        start_index = ridx[0] - num_mask_bins // 2 + crop_size # compute start index of masked PEAK region
        end_index = ridx[0] + 1 + num_mask_bins // 2 + crop_size # compute end index of masked PEAK region
        indices = tf.range(start_index, end_index) # get range of indices
        mask = (indices >= 0) & (indices < output_length) # mask to apply to indices to ensure they are valid
        filtered_indices = tf.boolean_mask(indices, mask) # apply index mask
        mask_indices = tf.cast(tf.reshape(filtered_indices, [-1, 1]), dtype=tf.int64) # now we get the full indices
        st=tf.SparseTensor( # create sparse tensor with 1s at masked indices
            indices=mask_indices,
            values=tf.ones([tf.shape(mask_indices)[0]], dtype=tf.float32),
            dense_shape=[output_length])
        dense_peak_mask=tf.sparse.to_dense(st)
        dense_peak_mask=1.0-dense_peak_mask # invert the mask to actually use
    else:
        dense_peak_mask = tf.ones([output_length],dtype=tf.float32) # if no peaks in window, no mask
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1) # adjust dimenions to multiply by ATAC profile

    # -----------------------here we set up the RANDOM mask (not peak based) ------------------------------------
    edge_append = tf.ones((crop_size,1),dtype=tf.float32) # we don't want the mask on the edge, so set these to 1

    # we will create the mask by first initializing a tensor of 1s, with shape 
    # based on the output cropped length and the number of bins to mask in each region. 
    # e.g., if we want to mask 
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)

    if ((atac_mask_int == 0)): # with 1/atac_mask_int probability, increase random masking percentage 
        atac_mask_dropout = 3 * atac_mask_dropout
    
    # here create the random mask using dropout 
    atac_mask=tf.nn.experimental.stateless_dropout(atac_mask,
                                              rate=(atac_mask_dropout),
                                              seed=[0,randomish_seed-5]) / (1. / (1.0-(atac_mask_dropout)))
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)


    # -----------------------here we COMBINE atac and peak mask -------------------------------------------------
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2) # if either mask is 0, mask value set to 0
    full_comb_mask_store = 1.0 - full_comb_mask # store the mask after inverting it to get the masked region indices

    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)

    if crop_size > 0:
        full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask

    ## generate random tensor of un-masked positions, with number of positions equal to the originally masked number of positions
    num_masked = tf.cast(tf.math.reduce_sum(full_comb_mask_store),dtype=tf.int32)
    zero_indices = tf.where(full_comb_mask_store == 0) # Step 2: Find indices of 0s in full_comb_mask_store
    selected_indices = tf.random.experimental.stateless_shuffle(zero_indices,seed = [0,randomish_seed+1])[:num_masked] # Step 3: Randomly select equal number of indices as there are 1s in full_comb_mask_store
    full_comb_unmask_store = tf.zeros_like(full_comb_mask_store,dtype=tf.int32) # Step 4: Create full_comb_unmask_store with all zeros
    updates = tf.ones_like(selected_indices[:, 0], dtype=tf.int32) # Update full_comb_unmask_store to have 1s at selected indices
    full_comb_unmask_store = tf.tensor_scatter_nd_update(full_comb_unmask_store, selected_indices, updates) # Since selected_indices is a 2D tensor with shape [num_ones, tensor_rank], we need to update using scatter_nd

    return full_comb_mask, full_comb_mask_store, full_comb_unmask_store