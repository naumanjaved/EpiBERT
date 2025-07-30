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
import utils

import tabix as tb

import tensorflow as tf

import pybedtools as pybt
import pyximport; pyximport.install()
import cython_fxn
from datetime import datetime
from tensorflow import strings as tfs
from tensorflow.keras import initializers as inits

from datetime import datetime


# ===========================================================================#

def parse_bool_str(input_str):
    if input_str == 'False':
        return False
    else:
        return True
    
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=5) # makes N correspond to all 0s

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])
    
    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=1)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def random_encode(input_tuple):
    sequence, randint = input_tuple
    if randint == 0:
        return one_hot(sequence)
    else:
        return rev_comp_one_hot(sequence)
    

def main():
     # ======================== arg parse ===================================#
    parser = argparse.ArgumentParser(
        description='Process input atac/rna bedgraphs for genformer')
   
    # input files definition
    parser.add_argument('genome_fasta', 
                        help='genome fasta')
    parser.add_argument('genome_file',
                        help='genome chrom sizes, bedtools format')
    parser.add_argument('atac_bedgraph', 
                        help='atac_bedgraph')
    parser.add_argument('intervals')
    parser.add_argument('peaks',
                        help='peaks')
    parser.add_argument('motif_activity',
                        help='motif_activity')
    parser.add_argument('--input_sequence_length', 
                        dest='input_sequence_length',type=int)
    parser.add_argument('--shift_amt',
                        dest="shift_amt",
                        type=int,
                        help='shift_amt')
    parser.add_argument('--gcs_bucket',
                        dest='bucket',
                        help='cloud bucket to write files to')
    parser.add_argument('--base_name',
                        dest='base_name', 
                        help='base name for output')
    parser.add_argument('--dir_name',
                        dest='dir_name', 
                        help='out dir name')
    parser.add_argument('--cell_type',
                        dest='cell_type',
                        help='cell_type')
    parser.add_argument('--cell_type_map',
                        dest='cell_type_map',
                        help='cell_type_map')
    parser.add_argument('--output_res',
                        dest="output_res",
                        type=int,
                        help='output_res')
    parser.add_argument('--output_res_atac',
                        dest="output_res_atac",
                        type=int,
                        help='output_res_atac')
    parser.add_argument('--num_bins',
                        dest="num_bins",
                        type=int,
                        help='num_bins')
    parser.add_argument('--peak_expand',
                        dest="peak_expand",
                        type=int,
                        help='peak_expand')
    
    args = parser.parse_args()
     # ======================== end arg parse ================================#      
    # =============== verify other input params/ ==============================#

    if ('gs://' not in args.bucket):
        parser.error('enter a valid GCS bucket with write permissions')     
        
    # ========================document parameters ===========================#
    stats_file_name = args.bucket + '/' + \
                        args.dir_name + '/metadata/' + \
                            args.base_name + '.stats'
    stats_file = open('stats.out', 'w')

    stats_file.write('gcs_bucket:' + str(args.bucket) + '\n')
    stats_file.write('base_name:' + str(args.base_name) + '\n')
    stats_file.write('dir_name:' + str(args.dir_name) + '\n')
    
    # convert bedgraphs
    atac_bedgraph_bed = tb.open(args.atac_bedgraph)
    
    
    ### genome df 
    genome_df = pd.read_csv(args.genome_file, header=None, sep ='\t')
    genome_df.columns = ['chrom', 'size']
    
    ### get cell type encoding
    cell_type_map = utils.parse_cell_map(args.cell_type_map)
    cell_encoding = cell_type_map[args.cell_type]

    ### interval encoding
    interval_map = utils.parse_interval_map(args.intervals)
    
    counter = 0
    # set TF write options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    
    ### expand peaks to desired length
    ext = str(args.peak_expand)
    subprocess.call("cat " + args.peaks + ''' | awk '{OFS="\t"}{print $1,$2-''' + ext + ''',$3+''' + ext + '''}' | sort -k1,1 -k2,2n > peaks_adjust.bed''',shell=True)
    subprocess.call("bgzip peaks_adjust.bed",shell=True)     ### now bgzip
    subprocess.call("tabix peaks_adjust.bed.gz",shell=True) ### tabix
    peaks_bed = tb.open("peaks_adjust.bed.gz")
    #subprocess.call("zcat peaks_adjust.bed.gz | head -10",shell=True)
    
    ### get peak center
    subprocess.call("cat " + args.peaks + ''' | awk '{OFS="\t"}{print $1,int(($2+$3)/2)-1,int(($2+$3)/2)}' | sort -k1,1 -k2,2n > peaks_center.bed''',shell=True)
    subprocess.call("bgzip peaks_center.bed",shell=True)     ### now bgzip
    subprocess.call("tabix peaks_center.bed.gz",shell=True) ### tabix
    peaks_center = tb.open("peaks_center.bed.gz")
    #subprocess.call("zcat peaks_center.bed.gz | head -10",shell=True)
    
    ####### load in motif activity
    command = "grep 'consensus_pwms.meme' " + args.motif_activity + " | grep 'AC' | awk '{print $3,$15}' | sort -k1,1 | awk '{print $2*-1}' > ranks.tsv"
    subprocess.call(command,shell=True)
    # Use temporary file in current directory or specified output directory
    ranks_file = os.path.join(os.getcwd(), "ranks.tsv")
    in_file = open(ranks_file, 'r')
    lst = []
    for line in in_file.readlines():
        lst.append(float(line.rstrip('\n').split('\t')[-1]))
    np_arr = np.asarray(lst)
    in_file.close()
    
    print('initialized all file loading, now writing intervals')
    
    for k, subset in enumerate(['test']):
        
        # same order as list in loop definition
        subprocess.call("grep " + subset + " " + args.intervals + " > " + subset + ".bed",
                        shell=True)
        sub_interval = pybt.BedTool(subset + '.bed').to_dataframe()
        
        sub_interval = sub_interval.sample(frac=1,random_state=int(str(datetime.now().timestamp())[-3:])).reset_index(drop=True)

        tfr_file = args.bucket + '/' + args.dir_name + '/' + \
            subset + '/' + args.base_name + '.tfr'
        
        with tf.io.TFRecordWriter(tfr_file, options=tf_opts) as writer:

            for index, row in sub_interval.iterrows():
                ### get interval encoding
                interval_str = '\t'.join([row['chrom'], 
                                          str(row['start']),
                                          str(row['end'])])
                interval_encoded=interval_map[interval_str]
                        
                ### now do sequence shifting
                length = (int(row['end']) - int(row['start']))
                        
                target_length = length + 2 * args.shift_amt
                
                interval_start = int(row['start']) - args.shift_amt
                interval_end = int(row['end']) + args.shift_amt
                chrom = str(row['chrom'].strip())

                ### correction if we shift past edge of chromosome
                chrom_size = int(genome_df.loc[genome_df['chrom']==chrom].iloc[0]['size'])
                
                if interval_start < 0:
                    interval_start = 0
                    interval_end = length + 2 * args.shift_amt
                if interval_end > chrom_size:
                    interval_end = chrom_size
                    interval_start = chrom_size - length - (2* args.shift_amt)

                interval_str = '\t'.join([chrom, 
                                          str(interval_start),
                                          str(interval_end)])
                interval_bed = pybt.BedTool(interval_str, from_string=True)
                interval = interval_bed[0]

                ### atac processing ######################################-
                atac_subints= atac_bedgraph_bed.query(chrom,
                                                      interval_start,
                                                      interval_end)
                atac_subints_df = pd.DataFrame([rec for rec in atac_subints])


                # if malformed line without score then disard
                if (len(atac_subints_df.index) == 0):
                    atac_bedgraph_out = np.array([0.0] * (target_length))
                else:
                    atac_subints_df.columns = ['chrom', 'start', 'end', 'score']
                    atac_bedgraph_out = utils.process_bedgraph(
                                        interval, atac_subints_df)
                    
                    
                ### peaks-processing ######################################-
                peaks_subints = peaks_bed.query(chrom,
                                                interval_start,
                                                interval_end)
                peaks_subints_df = pd.DataFrame([rec for rec in peaks_subints])
                if (len(peaks_subints_df.index) == 0):
                    peaks_bedgraph = np.array([0.0] * (args.num_bins))
                else:
                    peaks_subints_df.columns = ['chrom', 'start', 'end']

                    peaks_subints_df['start'] = peaks_subints_df['start'].astype('int64') - int(interval.start)
                    peaks_subints_df['end'] = peaks_subints_df['end'].astype('int64') - int(interval.start)
                    peaks_bedgraph = utils.get_TSS_token(interval, peaks_subints_df,
                                                         args.num_bins,args.output_res, args.shift_amt)
                peaks_processed = peaks_bedgraph
                
                ### peaks-center-processing ######################################-
                peaks_c_subints = peaks_center.query(chrom,
                                                interval_start,
                                                interval_end)
                peaks_c_subints_df = pd.DataFrame([rec for rec in peaks_c_subints])
                if (len(peaks_c_subints_df.index) == 0):
                    peaks_c_bedgraph = np.array([0.0] * (args.num_bins))
                else:
                    peaks_c_subints_df.columns = ['chrom', 'start', 'end']

                    peaks_c_subints_df['start'] = peaks_c_subints_df['start'].astype('int64') - int(interval.start)
                    peaks_c_subints_df['end'] = peaks_c_subints_df['end'].astype('int64') - int(interval.start)
                    peaks_c_bedgraph = utils.get_TSS_token(interval, peaks_c_subints_df,
                                                         args.num_bins,args.output_res,args.shift_amt)
                peaks_c_processed = peaks_c_bedgraph

                

                # get sequence for specific region
                sequence = utils.return_sequence(interval, args.genome_fasta)
                #sequence_rev_comp = utils.rev_complement(sequence)
                
                num_bins_atac = args.input_sequence_length // args.output_res_atac

                processed_sequence = sequence.upper()
                atac_processed = atac_bedgraph_out
                atac_processed = atac_processed[5:5+args.input_sequence_length] # adjust for random +/- 5 bp sequence shift
                atac_processed = np.reshape(atac_processed, [num_bins_atac,args.output_res_atac])
                atac_processed = np.sum(atac_processed,axis=1,keepdims=True)
                
                feature = {
                            'atac': _tensor_feature_float(np.round(atac_processed,2)),
                            'peaks': _tensor_feature_int(peaks_processed),
                            'peaks_center': _tensor_feature_int(peaks_c_processed),
                            'sequence': _bytes_feature(processed_sequence.encode()),
                            'interval': _tensor_feature_int(interval_encoded),
                            'motif_activity': _tensor_feature_float(np_arr),
                            'cell_type': _tensor_feature_int(cell_encoding)
                            }
                
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

                writer.write(example_proto)
                
                counter += 1
                if counter % 25 == 0:
                    print(counter)

            print('total:' + str(counter))

         
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _tensor_feature_float(numpy_array):
    serialized_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array, dtype=tf.float16))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))

def _tensor_feature_int(numpy_array):
    serialized_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array, dtype=tf.int32))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

##########################################################################
if __name__ == '__main__':
    main()

    