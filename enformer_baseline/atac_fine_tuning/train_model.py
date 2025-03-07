import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
import time
import pandas as pd
from datetime import datetime
import random

#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import sonnet as snt
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

## custom modules
import enformer_vanilla as enformer
import metrics as metrics
import training_utils as training_utils

import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
from scipy import stats

import optimizers

def parse_bool_str(input_str):
    if input_str == 'False':
        return False
    else:
        return True

 ## reformat 
# ===========================================================================#

def main():
    # ============== arg parse ==============================================# 
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()
    
    #================ init ==================================================# 
    
    ### make sure gcloud auth set to picard-testing-176520
        
    ### make sure TPU started

    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'hg_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'lr_base1': {
                    'values':[float(x) for x in args.lr_base1.split(',')]
                },
                'lr_base2': {
                    'values':[float(x) for x in args.lr_base2.split(',')]
                },
                'epsilon': {
                    'values':[args.epsilon]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'use_enformer_weights': {
                    'values':[parse_bool_str(x) for x in args.use_enformer_weights.split(',')]
                }
                }

    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name)
        g = tf.random.Generator.from_seed(datetime.now().timestamp())
        ## rest must be w/in strategy scope
        with strategy.scope():
            config_defaults = {
                "lr_base": 0.01 ### will be overwritten
            }
            
            ### log training parameters
            wandb.init(config=config_defaults, 
                       project= args.wandb_project, 
                       entity=args.wandb_user)
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.input_length=args.input_length
            wandb.config.num_epochs=args.num_epochs
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples=args.val_examples
            
            run_name = '_'.join(['ENFORMER',
                                 'LR1-' + str(wandb.config.lr_base1),
                                 'LR2-' + str(wandb.config.lr_base2),
                                 'GC-' + str(wandb.config.gradient_clip),
                                 'init-' + str(wandb.config.use_enformer_weights),
                                 args.model_save_basename])
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            
            wandb.run.name = run_name + "_" + date_string
            base_name = wandb.config.model_save_basename + "_" + wandb.run.name
            
            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=1
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print(GLOBAL_BATCH_SIZE)
            num_train=wandb.config.train_examples
            num_val=wandb.config.val_examples

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : num_val // GLOBAL_BATCH_SIZE + 1},
                                allow_val_change=True)
            wandb.config.update({"total_steps": num_train // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            
            
            tr_data_it,val_data_it =  \
                training_utils.return_distributed_iterators(args.gcs_path,
                                                            GLOBAL_BATCH_SIZE,
                                                            196608,
                                                            10,
                                                            1536,
                                                            args.num_targets,
                                                            args.num_parallel,
                                                            args.num_epochs,
                                                            strategy,
                                                            options,
                                                            g)

                
            
            enformer_model = enformer.Enformer(output_heads_dict = {'human': args.num_targets})

            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            
            checkpoint_name = wandb.config.model_save_dir + "/" + \
                            wandb.config.model_save_basename + "_" + date_string + "_" + wandb.run.name

            model_checkpoint = tf.train.Checkpoint(module=enformer_model)

            scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=1.0)
            scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler1)
            scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=1.0)
            scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                         warmup_steps=math.ceil(wandb.config.warmup_frac*wandb.config.total_steps),
                                         decay_schedule_fn=scheduler2)
            optimizer1 = tfa.optimizers.AdamW(learning_rate=scheduler1, weight_decay=1.0e-06,
                                             epsilon=wandb.config.epsilon)
            
            optimizer2 = tfa.optimizers.AdamW(learning_rate=scheduler2,weight_decay=1.0e-06,
                                                  epsilon=wandb.config.epsilon)
            optimizers_in = optimizer1,optimizer2

            metric_dict = {}
            

            train_step,val_step,build_step,metric_dict = training_utils.return_train_val_functions(enformer_model,
                                                                                                                optimizers_in,
                                                                                                                strategy,
                                                                                                                metric_dict,
                                                                                                                NUM_REPLICAS,
                                                                                                                wandb.config.gradient_clip)
            
            
            ### main training loop
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            
            for epoch_i in range(1, wandb.config.num_epochs+1):
                if epoch_i == 1:
                    # run once to build the model w/o updating anything
                    print('building model...')
                    build_step(val_data_it)
                    
                    if wandb.config.use_enformer_weights:
                        if args.enformer_checkpoint_path is not None:
                            print('loading enformer weights...')
                            options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
                            checkpoint = tf.train.Checkpoint(module=enformer_model)
                            tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                            latest = tf.train.latest_checkpoint(args.enformer_checkpoint_path)
                            checkpoint.restore(latest,options=options)#.assert_existing_objects_matched()
                    total_params = 0
                    for k in enformer_model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params))


                # main training step 
                print('starting epoch_', str(epoch_i))
                start = time.time()
                for step in range(wandb.config.train_steps):
                    strategy.run(train_step, args=(next(tr_data_it),))
                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('human_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                wandb.log({'human_train_loss': metric_dict['hg_tr'].result().numpy()},
                          step=epoch_i)
                tr_pearsonsR=metric_dict['tr_pearsonsR'].result()['PearsonR'].numpy()
                wandb.log({'human_tr_tracks_pearsons': np.nanmean(tr_pearsonsR)},
                          step=epoch_i)
                print('human tr_pearsonsR: ' + str(np.nanmean(tr_pearsonsR)))
                tr_R2=metric_dict['tr_R2'].result()['R2'].numpy()
                wandb.log({'human_tr_tracks_R2': np.nanmean(tr_R2)},
                          step=epoch_i)
                print('human tr_r2: ' + str(np.nanmean(tr_R2)))

                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                for step in range(wandb.config.val_steps):
                    strategy.run(val_step, args=(next(val_data_it),))
                
                print('human_val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                val_losses.append(metric_dict['hg_val'].result().numpy())
                wandb.log({'human_val_loss': metric_dict['hg_val'].result().numpy()},
                          step=epoch_i)
                pearsonsR=metric_dict['pearsonsR'].result()['PearsonR'].numpy()

                val_pearsons.append(pearsonsR)
                wandb.log({'human_val_tracks_pearsons': np.nanmean(pearsonsR)},
                          step=epoch_i)
                print('human pearsonsR: ' + str(np.nanmean(pearsonsR)))

                R2=metric_dict['R2'].result()['R2'].numpy()
                wandb.log({'human_val_tracks_R2': np.nanmean(R2)},
                          step=epoch_i)
                print('human r2: ' + str(np.nanmean(R2)))
 

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation')
                print('validation duration(mins): ' + str(duration))
                
                if (epoch_i > 2):
                    stop_criteria,patience_counter,best_epoch = \
                        training_utils.early_stopping(current_val_loss=val_losses[-1],
                                                        logged_val_losses=val_losses,
                                                        current_pearsons=val_pearsons[-1],
                                                        logged_pearsons=val_pearsons,
                                                        current_epoch=epoch_i,
                                                        best_epoch=best_epoch,
                                                        save_freq=args.savefreq,
                                                        patience=wandb.config.patience,
                                                        patience_counter=patience_counter,
                                                        min_delta=wandb.config.min_delta,
                                                        model_checkpoint=model_checkpoint,
                                                        checkpoint_name=checkpoint_name)
                #plt.close('all')
                    print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            checkpoint.save(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")
            #enformer_model.save_weights(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
