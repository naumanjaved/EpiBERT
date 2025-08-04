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
import matplotlib.pyplot as plt
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
import src.models.aformer_atac as genformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers

import training_utils_atac as training_utils

from scipy import stats
import kipoiseq

import analysis.interval_plotting_consolidated as utils

SEQUENCE_LENGTH=524288

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-1')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic=False
    mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(True)
    
    BATCH_SIZE_PER_REPLICA = 1 
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    mask_indices = list(range(2041,2053))
    
    mask_start = mask_indices[0]
    mask_end = mask_indices[-1]
    
    mask = np.zeros((524288//128,1))
    for k in range(524288//128):
        if k in range(mask_start,mask_end):
            mask[k,0]=1
    mask = tf.constant(mask)
            
    atac_mask = np.ones((524288//128,1))
    for k in tf.range(mask_start,mask_end):
        atac_mask[k,0] = 0.0
    atac_mask = tf.constant(atac_mask)

    
with strategy.scope():
    model1 = genformer.genformer(kernel_transformation='relu_kernel_transformation',
                                    dropout_rate=0.20,
                                    pointwise_dropout_rate=0.10,
                                    input_length=524288,
                                    output_length=4096,
                                    final_output_length=4092,
                                    num_heads=8,
                                    numerical_stabilizer=0.0000001,
                                    max_seq_length=4096,
                                    #nb_random_features=256,
                                    seed=19,
                                    norm=True,
                                    BN_momentum=0.90,
                                    normalize = True,
                                     use_rot_emb = True,
                                    num_transformer_layers=8,
                                    final_point_scale=6,
                                    filter_list_seq=[512,640,640,768,896,1024],
                                    filter_list_atac=[32,64])
    checkpoint_path="gs://genformer_europe_west_copy/atac_pretrain/models/genformer_524k_LR-1.0e-04_C-512_640_640_768_896_1024_T-8_motif-True_3_1abkvdo1/ckpt-45"
    
    dummy_seq = tf.data.Dataset.from_tensor_slices([tf.ones((524288,4),dtype=tf.float32)] *8)
    dummy_atac = tf.data.Dataset.from_tensor_slices([tf.ones((131072,1),dtype=tf.float32)]*8)
    dummy_motif = tf.data.Dataset.from_tensor_slices([tf.ones((1,693),dtype=tf.float32)]*8)
    combined_dataset = tf.data.Dataset.zip((dummy_seq, dummy_atac, dummy_motif))
    batched_dataset = combined_dataset.batch(6)
    dist = strategy.experimental_distribute_dataset(batched_dataset)
    dist_it = iter(dist)
    print('loading')
    @tf.function
    def build(input_dummy):
        self.model(input_dummy,training=False)
    strategy.run(build, args=(next(dist_it),))

    print('ran test input')
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    print('loaded weights')
    
    file_name = "gs://temporary-bucket-javed/genformer_atac_pretrain/524k/genformer_atac_pretrain_globalacc_conv_fpm_PC3/test/HG_PC_3.tfr"
    
    test_data_it = utils.return_distributed_iterators(file_name,GLOBAL_BATCH_SIZE, 524288,
                                                       5, 131072, 4096, 2,
                                                       128, 4, strategy, options, 1536, 
                                                       mask, atac_mask, False, True, True, 
                                                       6, True, g)
    
    

    mask_indices='2041-2053'
    mask = np.zeros((1,SEQUENCE_LENGTH//128,1))
    mask_centered = np.zeros((1,SEQUENCE_LENGTH//128,1))
    for entry in mask_indices.split(','): 
        mask_start = int(entry.split('-')[0])
        mask_end = int(entry.split('-')[1])
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start,mask_end):
                mask[0,k,0]=1
        for k in range(SEQUENCE_LENGTH//128):
            if k in range(mask_start+5,mask_end-3):
                mask_centered[0,k,0]=1
    mask_centered = tf.constant(mask_centered)
    
    contribution_input_grad_dist = utils.return_grads(model,mask_centered)
    
    
    grads_list = []
    seqs_list = []
    
    for k in range(5):
        
        seqs,grads = strategy.run(contribution_input_grad_dist, args=(next(test_data_it),))
        for x in strategy.experimental_local_results(seqs):
            seqs_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(grads):
            grads_list.append(tf.reshape(x, [-1]))
            
    print(grads_list)
    print(seqs_list)
            
    
    

    
    
    

