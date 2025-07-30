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
sys.path.insert(0, '/home/jupyter/repos/genformer')
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

import src.models.aformer_atac_rna as genformer
import logomaker

# ---------------------------------------------------------------------
# Shared utilities (deduplicated across analysis scripts)
# ---------------------------------------------------------------------
from analysis.interval_plot_shared import (
    one_hot,
    process_bedgraph,
    get_per_base_score_f,
    resize_interval,
    plot_tracks,
    FastaStringExtractor,
)

def return_atac_interval(atac_bedgraph,
                         chrom,interval_start,interval_end,num_bins,resolution):
    
    interval_str = '\t'.join([chrom, 
                              str(interval_start),
                              str(interval_end)])
    interval_bed = pybt.BedTool(interval_str, from_string=True)
    interval = interval_bed[0]
    
    atac_bedgraph_bed = tb.open(atac_bedgraph)
    ### atac processing ######################################-
    atac_subints= atac_bedgraph_bed.query(chrom,
                                          interval_start,
                                          interval_end)
    atac_subints_df = pd.DataFrame([rec for rec in atac_subints])


    # if malformed line without score then disard
    if (len(atac_subints_df.index) == 0):
        atac_bedgraph_out = np.array([0.0] * (num_bins))
    else:
        atac_subints_df.columns = ['chrom', 'start', 'end', 'score']
        atac_bedgraph_out = process_bedgraph(
            interval, atac_subints_df)
        
    atac_processed = atac_bedgraph_out
    atac_processed = np.reshape(atac_processed, [num_bins,resolution])
    atac_processed = np.sum(atac_processed,axis=1,keepdims=True)
    
    atac_processed = tf.constant(atac_processed,dtype=tf.float32)
    
    return atac_processed

    

def plot_logo(matrix,y_min,y_max):
    
    df = pd.DataFrame(matrix, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    
    # create Logo object
    nn_logo = logomaker.Logo(df)

    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True, bounds=[y_min, y_max])

    # style using Axes methods
    nn_logo.ax.set_ylim([y_min, y_max])
    nn_logo.ax.set_yticks([])
    nn_logo.ax.set_yticklabels([])
    nn_logo.ax.set_ylabel('saliency', labelpad=-1)
    


def write_bg(window,seq_length, crop,input_arr,output_res,out_file_name):
    
    chrom,start,stop = resize_interval(window,seq_length)
    start = start + crop * output_res
    
    out_file = open(out_file_name, 'w')
    for k, value in enumerate(input_arr):
        start_interval = k * output_res + start
        end_interval = (k+1) * output_res + start

        line = [str(chrom),
                str(start_interval), str(end_interval),
                str(value)]
        
        out_file.write('\t'.join(line) + '\n')
    out_file.close()
    
    
    
def return_eg(interval, eg_dataset, SEQUENCE_LENGTH):

    chrom,start,stop = resize_interval(interval,SEQUENCE_LENGTH)
    rna_arr = return_atac_interval(eg_dataset,chrom,
                                    start,stop,SEQUENCE_LENGTH,1)

    return rna_arr



class enformer_model:
    def __init__(self):
        model = enformer.Enformer()
        self.model = model
        dummy_seq = tf.data.Dataset.from_tensor_slices([tf.ones((196608,4),dtype=tf.float32)] *6)
        combined_dataset = tf.data.Dataset.zip((dummy_seq))
        batched_dataset = combined_dataset.batch(6)
        dist = strategy.experimental_distribute_dataset(batched_dataset)
        dist_it = iter(dist)
        print('loading')
        @tf.function
        def build(input_dummy):
            self.model(input_dummy,is_training=False)
        strategy.run(build, args=(next(dist_it),))
        
        options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
        checkpoint = tf.train.Checkpoint(module=model)
        tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        latest = tf.train.latest_checkpoint("sonnet_weights")
        checkpoint.restore(latest,options=options)#.assert_existing_objects_matched()
        
    
    def predict_on_batch_dist(self, strategy, inputs):
        @tf.function
        def run_model(inputs):
            output = self.model(model_inputs,is_training=False)['human']
            return output
            
        output = strategy.run(run_model, args=(next(inputs),))
        
        return output
    
    def contribution_input_grad(self, strategy, model_inputs, gradient_mask):
        @tf.function
        def contribution_input_grad_dist(model_inputs,gradient_mask):
            
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
            gradient_mask_mass = tf.reduce_sum(gradient_mask)

            with tf.GradientTape() as input_grad_tape:
                input_grad_tape.watch(model_inputs)
                output = self.model(model_inputs,is_training=False)['human'][:,:,4828]
                
                gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)

                prediction_mask = tf.reduce_sum(gradient_mask *
                                                output) / gradient_mask_mass
                
            input_grads = input_grad_tape.gradient(prediction_mask, model_inputs)

            return model_inputs, input_grads
        
        model_inputs, input_grads = \
            strategy.run(contribution_input_grad_dist, args = (next(model_inputs),gradient_mask))
        
        return model_inputs, input_grads
    
    
def return_inputs_enformer(interval,mask_indices_list,fasta_extractor, strategy):

    SEQUENCE_LENGTH=196608
    chrom,start,stop = resize_interval(interval,SEQUENCE_LENGTH)
    interval = kipoiseq.Interval(chrom, start, stop)
    sequence_one_hot_orig = tf.constant(one_hot(fasta_extractor.extract(interval)),dtype=tf.float32)

    mask = np.zeros((1,SEQUENCE_LENGTH//128,1))
    for entry in mask_indices_list.split(','): 
        
        mask_start = int(entry.split('-')[0])
        mask_end = int(entry.split('-')[1])
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start,mask_end):
                mask[0,k,0]=1
    
    # Create a dataset for each tensor
    seqs = tf.data.Dataset.from_tensor_slices([sequence_one_hot_orig]*6)
    
    # Zip the datasets together
    combined_dataset = tf.data.Dataset.zip((seqs))

    batched_dataset = combined_dataset.batch(6).repeat(2)

    # Convert the batched dataset to an iterator
    dist = strategy.experimental_distribute_dataset(batched_dataset)
    dist_it = iter(dist)
    crop_size=320
    output_length=1536
    mask = tf.slice(mask, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    #mask_centered = tf.slice(mask_centered, [0,crop_size,0],[-1,output_length-2*crop_size,-1])
    return dist_it, mask#, mask_centered