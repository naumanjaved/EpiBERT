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
import pandas as pd

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
import src.models.aformer_atac_rna as genformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers

import training_utils_atac as training_utils

from scipy import stats
import kipoiseq

import analysis.interval_plotting_consolidated as utils

import sys

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=sys.argv[1])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic=False
    mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(True)
    
    model = genformer.genformer(kernel_transformation='relu_kernel_transformation',
                                    dropout_rate=0.20,
                                    pointwise_dropout_rate=0.10,
                                    input_length=524288,
                                    output_length=4096,
                                    final_output_length=896,
                                    num_heads=8,
                                    numerical_stabilizer=0.0000001,
                                    max_seq_length=4096,
                                    seed=19,
                                    norm=True,
                                    BN_momentum=0.90,
                                    normalize = True,
                                     use_rot_emb = True,
                                    num_transformer_layers=8,
                                    final_point_scale=6,
                                    filter_list_seq=[512,640,640,768,896,1024],
                                    filter_list_atac=[32,64],
                                    predict_atac=True)
    #checkpoint_path="gs://genformer_europe_west_copy/524k/rampage_finetune/models/genformer_524k_LR1-5.0e-04_LR2-5.0e-04_C-512_640_640_768_896_1024_T-8_motif-True_9_m7o9qhwt/ckpt-16"
    checkpoint_path="gs://genformer_europe_west_copy/524k/rampage_finetune/models/genformer_524k_LR1-5.0e-04_LR2-5.0e-04_C-512_640_640_768_896_1024_T-8_motif-True_1_yxy25j6h/ckpt-35"
    #checkpoint_path="gs://genformer_europe_west_copy/rampage_finetune/524k/models/genformer_524k_LR1-1.0e-04_LR2-1.0e-04_C-512_640_640_768_896_1024_T-8_motif-True_4_25oy5umk/ckpt-42"
    genformer = utils.genformer_model(strategy, model, checkpoint_path)

    
SEQUENCE_LENGTH=524288
resolution=4
num_bins = SEQUENCE_LENGTH // resolution
fasta_file = '/home/jupyter/reference/hg38_erccpatch.fa'
fasta_extractor = utils.FastaStringExtractor(fasta_file)

genes_dict = {"HNRNPA1" : "chr12:54279939-54281439", "NFE2" : "chr12:54300287-54301787", "COPZ1" : "chr12:54324339-54325839",
                "ITGA5" : "chr12:54418516-54420016", "WDR83OS" : "chr19:12668901-12670401", "DHPS" : "chr19:12681137-12682637",
                "C19orf43" : "chr19:12734025-12735525", "JUNB" : "chr19:12790745-12792245", "PRDX2" : "chr19:12801160-12802660",
                "RNASEH2A" : "chr19:12805863-12807363", "DNASE2" : "chr19:12880771-12882271", "KLF1" : "chr19:12886453-12887953",
                "CALR" : "chr19:12937849-12939349", "RAD23A" : "chr19:12945063-12946563", "LYL1" : "chr19:13102410-13103910",
                "FUT1" : "chr19:48752910-48754410", "BCAT2" : "chr19:48810313-48811813", "PPP1R15A" : "chr19:48871641-48873141",
                "NUCB1" : "chr19:48899299-48900799", "BAX" : "chr19:48954109-48955609", "FTL" : "chr19:48964558-48966058",
                "SEC61A1" : "chr3:128051618-128053118", "RPN1" : "chr3:128650126-128651626", "RAB7A" : "chr3:128725385-128726885",
                "CNBP" : "chr3:129183217-129184717", "H1FX" : "chr3:129315527-129317027", "MYC" : "chr8:127735318-127736818",
                "CCDC26" : "chr8:129574268-129575768", "GATA1" : "chrX:48785823-48787323", "HDAC6" : "chrX:48801265-48802765",
                "PQBP1" : "chrX:48897161-48898661", "PLP2" : "chrX:49171082-49172586"}

file = '/home/jupyter/datasets/eg/hg38_eg.bed'

import random
seed=random.randint(1,100)
atac_file="/home/jupyter/datasets/ATAC/HG_K562.bed.gz"
rna_file = "/home/jupyter/datasets/ATAC/HG_K562.rampage.bed.gz"
tf_arr='/home/jupyter/datasets/ATAC/HG_K562.tsv'
output_length = num_bins // 32 
crop_size=1600
mask_indices='2041-2053'
seed=6

for gene in genes_dict.keys():
    command = "grep '" + gene + "' " + file + " | sort -k1,1 -k2,2n > temp_files/" + gene + ".eg.bed"
    subprocess.call(command,shell=True)
    
    command = '''awk '{OFS="\t"}{print $1,$2,$3,NR}' temp_files/''' + gene + '.eg.bed' + ' > temp_files/' + gene + '.eg.encoded.bed'
    subprocess.call(command,shell=True)
    
    command = 'bgzip temp_files/' + gene + '.eg.encoded.bed'
    subprocess.call(command,shell=True)
    
    command = 'tabix temp_files/' + gene + '.eg.encoded.bed.gz'
    subprocess.call(command,shell=True)
    
    interval_center = genes_dict[gene]
    
    
    inputs, masked_atac, target_atac,target_atac_uncropped,rna_arr,masked_atac_reshape, mask, mask_centered = \
    utils.return_all_inputs_simple(interval_center, atac_file,rna_file, SEQUENCE_LENGTH,
                      num_bins, resolution,tf_arr,crop_size,output_length,
                      fasta_extractor,mask_indices,strategy)

    seq, seq_grads, atac_grads, prediction, att_matrices,att_matrices_norm = genformer.contribution_input_grad_dist_simple(strategy,inputs,mask)
    
    grad_input = tf.abs(atac_grads.values[0][:,0]) * masked_atac[:,0]

    reshaped_grad = tf.reduce_sum(tf.reshape(grad_input,[4096,32]),
                                     axis=1)
    
    seq_grad_input = tf.reduce_sum(tf.reshape(tf.reduce_sum(tf.abs(seq_grads) * seq.values[0][0,:,:],axis=1),
                                [-1,128]),axis=1)


    eg = utils.return_eg(interval_center, '/home/jupyter/repos/genformer/analysis/enhancers/temp_files/' + gene + '.eg.encoded.bed.gz',
                         524288)

    eg_grouped = tf.reduce_max(tf.reshape(eg,[4096,128]),
                                         axis=1)

    eg_grouped = eg_grouped.numpy()
    eg_unique = np.unique(eg_grouped)

    atac_grads_scaled = reshaped_grad / tf.reduce_max(reshaped_grad)
    seq_grad_input_scaled = seq_grad_input / tf.reduce_max(seq_grad_input)
    out_tsv = '/home/jupyter/repos/genformer/analysis/enhancers/out_gradients_35/' + gene + '.eg.preds.summed.tsv'
    lst = []
    lst1 = []
    for k in eg_unique[eg_unique != 0]:
        indices = list(np.where(eg_grouped == k)[0])

        atac_grads_scaled_vals = tf.reduce_sum(tf.gather(atac_grads_scaled, indices)).numpy()
        seq_grads_scaled = tf.reduce_sum(tf.gather(seq_grad_input_scaled,indices)).numpy()

        lst.append(int(k))
        lst1.append(atac_grads_scaled_vals+seq_grads_scaled)

    df = pd.DataFrame({'encoding': lst, 'grad_out': lst1})

    df.to_csv(out_tsv,header=True,index=False,sep='\t')
    
    
    atac_grads_scaled = reshaped_grad #/ tf.reduce_max(reshaped_grad)
    seq_grad_input_scaled = seq_grad_input #/ tf.reduce_max(seq_grad_input)
    out_tsv = '/home/jupyter/repos/genformer/analysis/enhancers/out_gradients_35/' + gene + '.eg.preds.summed.non_scaled.tsv'
    lst = []
    lst1 = []
    for k in eg_unique[eg_unique != 0]:
        indices = list(np.where(eg_grouped == k)[0])

        atac_grads_scaled_vals = tf.reduce_sum(tf.gather(atac_grads_scaled, indices)).numpy()
        seq_grads_scaled = tf.reduce_sum(tf.gather(seq_grad_input_scaled,indices)).numpy()

        lst.append(int(k))
        lst1.append(atac_grads_scaled_vals+seq_grads_scaled)

    df = pd.DataFrame({'encoding': lst, 'grad_out': lst1})

    df.to_csv(out_tsv,header=True,index=False,sep='\t')
