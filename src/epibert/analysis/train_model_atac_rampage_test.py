import time
import os
import argparse
import wandb
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import sys

# Environment configuration for TensorFlow
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'


sys.path.insert(0, os.path.expanduser("~") + '/EpiBERT')
## custom modules
import src.models.epibert_rampage_finetune as epibert # can toggle which model you want here
import src.optimizers as optimizers
import analysis.training_utils_atac_rampage_test as training_utils
import src.schedulers as schedulers

# Function to parse boolean string values
def parse_bool_str(input_str):
    if input_str in ['False', 'false', 'FALSE', 'F']:
        return False
    return True

# Main function definition
def main():
    # set up argument parser
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()

    if (parse_bool_str(args.load_init) and (args.checkpoint_path is not None)):
        input_ckpt = args.checkpoint_path.split('/')[-2]
        seed = args.seed
        run_id = input_ckpt.split('_')[-1]
        num_transformer_layers = input_ckpt.split('_')[-4].split('-')[-1]
        use_motif_activity=input_ckpt.split('_')[-3].split('-')[-1]
        filter_list_seq = input_ckpt.split('_')[-10:-4]
        filter_list_seq[0] = filter_list_seq[0].split('-')[1]
        filter_list_seq = ','.join(filter_list_seq)
        lr_base1 = '-'.join(input_ckpt.split('_')[-12].split('-')[1:])
        lr_base2 = '-'.join(input_ckpt.split('_')[-11].split('-')[1:])
    else:
        seed = args.seed
        run_id = args.run_id
        num_transformer_layers = args.num_transformer_layers
        use_motif_activity=args.use_motif_activity
        filter_list_seq = args.filter_list_seq
        lr_base1 = args.lr_base1
        lr_base2 = args.lr_base2

    #def sweep_train(config_defaults=None):
    strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone) # initialize TPU
    mixed_precision.set_global_policy('mixed_bfloat16')
    g = tf.random.Generator.from_seed(seed) # training data random seed init
    g_val = tf.random.Generator.from_seed(args.val_data_seed) # validation data random seed init
    g_val_ho= tf.random.Generator.from_seed(args.val_data_seed) # holdout validation data random seed init

    mod_run_name = '_'.join([args.model_save_basename,
                                str(args.input_length / 1000)[:4].rstrip('.') + 'k',
                            'LR1-' + str(lr_base1),
                            'LR2-' + str(lr_base2),
                            'C-' + filter_list_seq.replace(',','_'),
                            'T-' + str(num_transformer_layers),
                            'motif-' + str(use_motif_activity)])
    date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
    date_string = date_string.replace(' ','_')
    date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
    date_string = date_string.replace(' ','_')

    # defining sweep options, parameters are specified by execute_sweep.sh
    # -----------------------------------------------------------------------------------------------------------
    config = {
            'input_length':  int(args.input_length),
            'output_length':  int(args.output_length),
            'output_length_ATAC': int(args.output_length_ATAC),
            'final_output_length': int(args.final_output_length),
            'output_res':  int(args.output_res),
            'dropout_rate':  float(args.dropout_rate),
            'pointwise_dropout_rate':  float(args.pointwise_dropout_rate),
            'lr_base1': float(lr_base1),
            'lr_base2': float(lr_base2),
            'gradient_clip':  float(args.gradient_clip),
            'decay_frac':  float(args.decay_frac),
            'num_transformer_layers':  int(num_transformer_layers),
            'num_heads':  int(args.num_heads),
            'kernel_transformation': args.kernel_transformation,
            'epsilon': float(args.epsilon),
            'load_init': parse_bool_str(args.load_init),
            'filter_list_seq':  [int(x) for x in filter_list_seq.split(',')],
            'filter_list_atac': [int(x) for x in args.filter_list_atac.split(',')],
            'BN_momentum': float(args.BN_momentum),
            'num_epochs': int(args.num_epochs),
            'atac_mask_dropout': float(args.atac_mask_dropout),
            'atac_mask_dropout_val': float(args.atac_mask_dropout_val),
            'rectify': parse_bool_str(args.rectify),
            'log_atac': parse_bool_str(args.log_atac),
            'use_atac': parse_bool_str(args.use_atac),
            'use_seq': parse_bool_str(args.use_seq),
            'random_mask_size': int(args.random_mask_size),
            'final_point_scale': int(args.final_point_scale),
            'seed': int(seed),
            'val_data_seed': int(args.val_data_seed),
            'atac_corrupt_rate':  int(args.atac_corrupt_rate),
            'use_motif_activity':  parse_bool_str(use_motif_activity),
            'loss_type':  str(args.loss_type),
            'total_weight_loss':  float(args.total_weight_loss),
            'use_rot_emb': parse_bool_str(args.use_rot_emb),
            'checkpoint_path': args.checkpoint_path,
            'checkpoint_path_FT': args.checkpoint_path_FT,
            'tpu': args.tpu_name,
            'gcs_path': args.gcs_path,
            'gcs_path_holdout': args.gcs_path_holdout,
            'decay_steps': args.decay_steps,
            'test_examples': args.test_examples,
            'test_examples_ho': args.test_examples_ho,
            'test_TSS': args.test_TSS,
            'test_TSS_ho': args.test_TSS_ho,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'model_save_dir': args.model_save_dir,
            'model_save_basename': args.model_save_basename,
            'max_shift': int(args.max_shift),
            'crop_size': (int(args.output_length) - int(args.final_output_length))//2,
            'reset_optimizer_state': parse_bool_str(args.reset_optimizer_state),
            'warmup_steps': float(args.warmup_steps),
            'return_constant_lr': parse_bool_str(args.return_constant_lr),
            'atac_scale': float(args.atac_scale),
            'load_init_FT': parse_bool_str(args.load_init_FT),
            'predict_atac': parse_bool_str(args.predict_atac)
    }

    wandb.init(config=config,
                project= args.wandb_project,
                id=run_id,
                name = None if not parse_bool_str(args.load_init) else mod_run_name + "_" + str(args.seed),
                entity=args.wandb_user,
                resume="allow" if not (parse_bool_str(args.load_init) and (run_id is not None)) else "allow")
    run_id_unique = wandb.run.id if run_id is None else run_id
    print('run_id:' + run_id_unique)
    wandb.run.name = mod_run_name + "_" + str(args.seed) + "_" + run_id_unique

    with strategy.scope(): ## keep remainder of parameter initialization within TPU/GPU strategy scope
        # TFrecord dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy=\
            tf.data.experimental.AutoShardPolicy.FILE
        options.deterministic=False
        options_val = tf.data.Options()
        options_val.experimental_distribute.auto_shard_policy=\
            tf.data.experimental.AutoShardPolicy.DATA
        options_val.deterministic=False
        mixed_precision.set_global_policy('mixed_bfloat16')

        NUM_REPLICAS = strategy.num_replicas_in_sync
        BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
        GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS # num total examples per step across all replicas
        print('global batch size:', GLOBAL_BATCH_SIZE)

        wandb.config.update({"test_steps": 1+(wandb.config.test_examples // GLOBAL_BATCH_SIZE)},
                            allow_val_change=True)
        wandb.config.update({"test_steps_ho": 1+(wandb.config.test_examples_ho // GLOBAL_BATCH_SIZE)},
                            allow_val_change=True)
        wandb.config.update({"tss_steps" : 1+(wandb.config.test_TSS // GLOBAL_BATCH_SIZE)},
                            allow_val_change=True)
        wandb.config.update({"tss_steps_ho" : 1+(wandb.config.test_TSS_ho // GLOBAL_BATCH_SIZE)},
                            allow_val_change=True)
        # create the dataset iterators, one for training, one for holdout validation  
        test_it,test_it_ho,tss_it,tss_it_ho,tss_it_ho_b = \
                training_utils.return_distributed_iterators(wandb.config.gcs_path, wandb.config.gcs_path_holdout,
                                                            GLOBAL_BATCH_SIZE, wandb.config.input_length,
                                                            wandb.config.max_shift, wandb.config.output_length_ATAC,
                                                            wandb.config.output_length, wandb.config.crop_size,
                                                            wandb.config.output_res, args.num_parallel, 48,
                                                            strategy, options,options_val, wandb.config.atac_mask_dropout,
                                                            wandb.config.atac_mask_dropout_val,
                                                            wandb.config.random_mask_size, wandb.config.log_atac,
                                                            wandb.config.use_atac, wandb.config.use_seq, wandb.config.seed,
                                                            wandb.config.val_data_seed, wandb.config.atac_corrupt_rate,
                                                            wandb.config.test_steps, wandb.config.use_motif_activity,
                                                            g, g_val,g_val_ho)
        
        
        
        print('created dataset iterators')
        # initialize model
        model = epibert.epibert(kernel_transformation=wandb.config.kernel_transformation,
                                dropout_rate=wandb.config.dropout_rate,
                                pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                input_length=wandb.config.input_length,
                                output_length=wandb.config.output_length,
                                final_output_length=wandb.config.final_output_length,
                                num_heads=wandb.config.num_heads,
                                numerical_stabilizer=0.0000001,
                                max_seq_length=wandb.config.output_length,
                                norm=True,
                                BN_momentum=wandb.config.BN_momentum,
                                normalize = True,
                                seed = wandb.config.val_data_seed,
                                num_transformer_layers=wandb.config.num_transformer_layers,
                                final_point_scale=wandb.config.final_point_scale,
                                filter_list_seq=wandb.config.filter_list_seq,
                                filter_list_atac=wandb.config.filter_list_atac,
                                predict_atac=wandb.config.predict_atac,
                                use_rot_emb=wandb.config.use_rot_emb)

        print('initialized model')

        # initialize optimizer with warmup and cosine decay
        init_learning_rate=1.0e-07
        optimizer1 = tf.keras.optimizers.AdamW(learning_rate=init_learning_rate,
                                                epsilon=wandb.config.epsilon,
                                                weight_decay=1.0e-04,
                                                global_clipnorm=wandb.config.gradient_clip)
        optimizer1.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm',
                                                        'BN', 'LN', 'LayerNorm','BatchNorm'])
        
        optimizer2 = tf.keras.optimizers.AdamW(learning_rate=init_learning_rate,
                                                epsilon=wandb.config.epsilon,
                                                weight_decay=1.0e-4,
                                                global_clipnorm=wandb.config.gradient_clip)
        optimizer2.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm',
                                                        'BN', 'LN', 'LayerNorm','BatchNorm'])

        metric_dict = {} # initialize dictionary to store metrics


        optimizers_in = optimizer1, optimizer2
        batch_num = tf.Variable(0, name="batch_num")
        optimizer_step_track = tf.Variable(0, name="optimizer_step_track")
        best_val_loss = tf.Variable(1000.0, name="best_val_loss")
        ckpt = tf.train.Checkpoint(batch_num=batch_num,
                                    optimizer1=optimizer1,
                                    optimizer2=optimizer2,
                                    model=model,
                                    best_val_loss=best_val_loss,
                                    optimizer_step_track=optimizer_step_track)
        
        checkpoint_dir = os.path.join(wandb.config.model_save_dir, wandb.run.name)
        if wandb.config.load_init:
            checkpoint_dir = wandb.config.checkpoint_path
            wandb.run.name = wandb.config.checkpoint_path.split('/')[-1]

        manager = tf.train.CheckpointManager(ckpt,
                                                directory=checkpoint_dir,
                                                max_to_keep=25)
        
        # initialize functions for training and validation steps
        test_step, test_step_ho, val_step, val_step_ho,build_step, metric_dict = \
            training_utils.return_train_val_functions(
                model=model,
                optimizers_in=optimizers_in,
                strategy=strategy,
                metric_dict=metric_dict,
                num_replicas=NUM_REPLICAS,
                loss_type=wandb.config.loss_type,
                total_weight=wandb.config.total_weight_loss,
                atac_scale=wandb.config.atac_scale,
                predict_atac=wandb.config.predict_atac
            )
        

        print('building model...')
        build_step(tss_it_ho_b)
        total_params = 0
        for k in model.trainable_variables:
            var = k.values[0]
            total_params += tf.size(var)
        print('built model, total params: ' + str(total_params))

        wandb.config.update({"num_epochs_to_start": 0}, allow_val_change=True)
        if wandb.config.load_init:
            status = ckpt.restore(wandb.config.checkpoint_path)
            print('restored from checkpoint')
            print('restart training at epoch: ' + str(1+ batch_num.numpy()))
            print('restart at data batch: ' + str(batch_num.numpy()))
            wandb.config.update({"num_epochs_to_start": batch_num.numpy()}, 
                                allow_val_change=True)
        if wandb.config.load_init_FT:
            ckpt_FT=tf.train.Checkpoint(model=model)
            status = ckpt_FT.restore(wandb.config.checkpoint_path_FT)
            print('restored from checkpoint for fine-tuning')

        print(wandb.config)

        target_list=[]
        output_list=[]
        cell_types=[]
        for k in range(wandb.config.test_steps):
            target_rna,output_rna,cell_type = strategy.run(test_step, args=(next(test_it),))
            for x in strategy.experimental_local_results(target_rna): # flatten the true values
                target_list.append(x[0,:,0])
            for x in strategy.experimental_local_results(output_rna): # flatten the pred values
                output_list.append(x[0,:,0])
            for x in strategy.experimental_local_results(cell_type): # flatten the pred values
                cell_types.append(x)

        np.save('target.npy', np.array(target_list))
        np.save('output.npy', np.array(output_list))
        np.save('cell_types_ho.npy', np.array(cell_types))

        print('RNA_PearsonR_test: ' + str(metric_dict['RNA_PearsonR_test'].result()['PearsonR'].numpy()))
        print('RNA_R2_test: ' + str(metric_dict['RNA_R2_test'].result()['R2'].numpy()))

        target_list=[]
        output_list=[]
        cell_types=[]
        for k in range(wandb.config.test_steps_ho):
            target_rna,output_rna,cell_type=strategy.run(test_step_ho, args=(next(test_it_ho),))
            for x in strategy.experimental_local_results(target_rna): # flatten the true values
                target_list.append(x[0,:,0])
            for x in strategy.experimental_local_results(output_rna): # flatten the pred values
                output_list.append(x[0,:,0])
            for x in strategy.experimental_local_results(cell_type): # flatten the pred values
                cell_types.append(x)

        np.save('target_ho.npy', np.array(target_list))
        np.save('output_ho.npy', np.array(output_list))
        np.save('cell_types.npy', np.array(cell_types))

        print('RNA_PearsonR_test_ho: ' + str(metric_dict['RNA_PearsonR_test_ho'].result()['PearsonR'].numpy()))
        print('RNA_R2_test_ho: ' + str(metric_dict['RNA_R2_test_ho'].result()['R2'].numpy()))

        wandb.log({'RNA_PearsonR_test': metric_dict['RNA_PearsonR_test'].result()['PearsonR'].numpy(),
                    'RNA_R2_test': metric_dict['RNA_R2_test'].result()['R2'].numpy(),
                    'RNA_PearsonR_test_ho': metric_dict['RNA_PearsonR_test_ho'].result()['PearsonR'].numpy(),
                    'RNA_R2_test_ho': metric_dict['RNA_R2_test_ho'].result()['R2'].numpy()},step=1)
        
        if wandb.config.predict_atac:
            print('ATAC_PearsonR_test: ' + str(metric_dict['ATAC_PearsonR_test'].result()['PearsonR'].numpy()))
            print('ATAC_R2_test: ' + str(metric_dict['ATAC_R2_test'].result()['R2'].numpy()))
            print('ATAC_PearsonR_test_ho: ' + str(metric_dict['ATAC_PearsonR_test_ho'].result()['PearsonR'].numpy()))
            print('ATAC_R2_test_ho: ' + str(metric_dict['ATAC_R2_test_ho'].result()['R2'].numpy()))
            wandb.log({'ATAC_PearsonR_test': metric_dict['ATAC_PearsonR_test'].result()['PearsonR'].numpy(),
                    'ATAC_R2_test': metric_dict['ATAC_R2_test'].result()['R2'].numpy(),
                    'ATAC_PearsonR_test_ho': metric_dict['ATAC_PearsonR_test_ho'].result()['PearsonR'].numpy(),
                    'ATAC_R2_test_ho': metric_dict['ATAC_R2_test_ho'].result()['R2'].numpy()},
                        step=1)


        pred_list = [] # list to store predictions
        true_list = [] # list to store true values
        cell_type_list = [] # list to store predictions
        gene_list = [] # list to store true values
        target_list = [] # list to store predictions
        output_list = [] # list to store true values
        for k in range(wandb.config.tss_steps):
            true, pred,gene,cell_type, target_rna, output_rna = strategy.run(val_step, args=(next(tss_it),))
            for x in strategy.experimental_local_results(true): # flatten the true values
                true_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(pred): # flatten the pred values
                pred_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(cell_type): # flatten the true values
                cell_type_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(gene): # flatten the pred values
                gene_list.append(tf.reshape(x, [-1]))

        cell_specific_corrs, gene_specific_corrs,results_df = training_utils.make_plots(tf.concat(true_list,0),
                                                            tf.concat(pred_list,0),
                                                            tf.concat(cell_type_list,0),
                                                            tf.concat(gene_list,0), 5000)


        print('cell_specific_correlation_test: ' + str(cell_specific_corrs))
        print('gene_specific_correlation_test: ' + str(gene_specific_corrs))
        wandb.log({'cell_specific_correlation': cell_specific_corrs,
                    'gene_specific_correlation': gene_specific_corrs},
                    step=1)
        results_df.to_csv('rampage_preds_test_genes.tsv',sep='\t',index=False,header=True)

        pred_list = [] # list to store predictions
        true_list = [] # list to store true values
        cell_type_list = [] # list to store predictions
        gene_list = [] # list to store true values
        target_list = [] # list to store predictions
        output_list = [] # list to store true values
        for k in range(wandb.config.tss_steps_ho):
            true, pred,gene,cell_type, target_rna, output_rna = strategy.run(val_step_ho, args=(next(tss_it_ho),))
            for x in strategy.experimental_local_results(true): # flatten the true values
                true_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(pred): # flatten the pred values
                pred_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(cell_type): # flatten the true values
                cell_type_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(gene): # flatten the pred values
                gene_list.append(tf.reshape(x, [-1]))


        cell_specific_corrs_ho, gene_specific_corrs_ho,results_df_ho = training_utils.make_plots(tf.concat(true_list,0),
                                                            tf.concat(pred_list,0),
                                                            tf.concat(cell_type_list,0),
                                                            tf.concat(gene_list,0), 5000)
        print('cell_specific_correlation_ho: ' + str(cell_specific_corrs_ho))
        print('gene_specific_correlation_ho: ' + str(gene_specific_corrs_ho))

        wandb.log({'cell_specific_corrs_ho': cell_specific_corrs_ho,
                    'gene_specific_corrs_ho': gene_specific_corrs_ho},
                    step=1)
        results_df_ho.to_csv('rampage_preds_test_genes_ho.tsv',sep='\t',index=False,header=True)
# ---------------------------------------------------------------------------------------------------------------
        
        test_it,test_it_ho,tss_it,tss_it_ho,tss_it_ho_b = \
                training_utils.return_distributed_iterators(wandb.config.gcs_path, wandb.config.gcs_path_holdout,
                                                            GLOBAL_BATCH_SIZE, wandb.config.input_length,
                                                            wandb.config.max_shift, wandb.config.output_length_ATAC,
                                                            wandb.config.output_length, wandb.config.crop_size,
                                                            wandb.config.output_res, args.num_parallel, 48,
                                                            strategy, options,options_val, wandb.config.atac_mask_dropout,
                                                            wandb.config.atac_mask_dropout_val,
                                                            wandb.config.random_mask_size, wandb.config.log_atac,
                                                            wandb.config.use_atac, wandb.config.use_seq, wandb.config.seed,
                                                            wandb.config.val_data_seed, wandb.config.atac_corrupt_rate,
                                                            wandb.config.test_steps, False,
                                                            g, g_val,g_val_ho)

        pred_list = [] # list to store predictions
        true_list = [] # list to store true values
        cell_type_list = [] # list to store predictions
        gene_list = [] # list to store true values
        target_list = [] # list to store predictions
        output_list = [] # list to store true values
        for k in range(wandb.config.tss_steps):
            true, pred,gene,cell_type, target_rna, output_rna = strategy.run(val_step, args=(next(tss_it),))
            for x in strategy.experimental_local_results(true): # flatten the true values
                true_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(pred): # flatten the pred values
                pred_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(cell_type): # flatten the true values
                cell_type_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(gene): # flatten the pred values
                gene_list.append(tf.reshape(x, [-1]))

        cell_specific_corrs, gene_specific_corrs,results_df = training_utils.make_plots(tf.concat(true_list,0),
                                                            tf.concat(pred_list,0),
                                                            tf.concat(cell_type_list,0),
                                                            tf.concat(gene_list,0), 5000)


        print('cell_specific_correlation_test_nomotif: ' + str(cell_specific_corrs))
        print('gene_specific_correlation_test_nomotif: ' + str(gene_specific_corrs))
        results_df.to_csv('rampage_preds_test_genes_nomotif.tsv',sep='\t',index=False,header=True)

        pred_list = [] # list to store predictions
        true_list = [] # list to store true values
        cell_type_list = [] # list to store predictions
        gene_list = [] # list to store true values
        target_list = [] # list to store predictions
        output_list = [] # list to store true values
        for k in range(wandb.config.tss_steps_ho):
            true, pred,gene,cell_type, target_rna, output_rna = strategy.run(val_step_ho, args=(next(tss_it_ho),))
            for x in strategy.experimental_local_results(true): # flatten the true values
                true_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(pred): # flatten the pred values
                pred_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(cell_type): # flatten the true values
                cell_type_list.append(tf.reshape(x, [-1]))
            for x in strategy.experimental_local_results(gene): # flatten the pred values
                gene_list.append(tf.reshape(x, [-1]))


        cell_specific_corrs_ho, gene_specific_corrs_ho,results_df_ho = training_utils.make_plots(tf.concat(true_list,0),
                                                            tf.concat(pred_list,0),
                                                            tf.concat(cell_type_list,0),
                                                            tf.concat(gene_list,0), 5000)
        print('cell_specific_correlation_ho_nomotif: ' + str(cell_specific_corrs_ho))
        print('gene_specific_correlation_ho_nomotif: ' + str(gene_specific_corrs_ho))

        results_df_ho.to_csv('rampage_preds_test_genes_ho_nomotif.tsv',sep='\t',index=False,header=True)



if __name__ == '__main__':
    main()