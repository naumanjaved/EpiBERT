"""Training script for EpiBERT ATAC pre-training.

This is the original training loop with only cosmetic clean-ups:
• Added this module-level docstring for clarity.
• Removed a duplicate assignment of `date_string` (no functional impact).

All argument flags, wandb logging, TPU strategy logic and model initialisation
remain unchanged, so previously saved checkpoints and shell wrappers are still
compatible.
"""

import time
import os
import argparse
import wandb
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Environment configuration for TensorFlow
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'

## custom modules
from epibert.models import epibert_atac_pretrain as epibert # can toggle which model you want here
from epibert import optimizers
from epibert.scripts import training_utils_atac_pretrain as training_utils
from epibert import schedulers

from epibert.shared_training_utils import parse_bool_str

# Main function definition
def main():
    # set up argument parser
    parser = argparse.ArgumentParser(
        description='process input for epibert training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()

    if (parse_bool_str(args.load_init) and (args.checkpoint_path is not None)):
        input_ckpt = args.checkpoint_path.split('/')[-1]
        seed = args.seed
        run_id = input_ckpt.split('_')[-1]
        num_transformer_layers = input_ckpt.split('_')[-4].split('-')[-1]
        use_motif_activity=input_ckpt.split('_')[-3].split('-')[-1]
        filter_list_seq = input_ckpt.split('_')[-10:-4]
        filter_list_seq[0] = filter_list_seq[0].split('-')[1]
        filter_list_seq = ','.join(filter_list_seq)
        lr_base = '-'.join(input_ckpt.split('_')[-11].split('-')[1:])
    else:
        seed = args.seed
        run_id = args.run_id
        num_transformer_layers = args.num_transformer_layers
        use_motif_activity=args.use_motif_activity
        filter_list_seq = args.filter_list_seq
        lr_base = args.lr_base

    #def sweep_train(config_defaults=None):
    strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone) # initialize TPU
    mixed_precision.set_global_policy('mixed_bfloat16')
    g = tf.random.Generator.from_seed(seed) # training data random seed init
    g_val = tf.random.Generator.from_seed(args.val_data_seed) # validation data random seed init

    mod_run_name = '_'.join([args.model_save_basename,
                                str(args.input_length / 1000)[:4].rstrip('.') + 'k',
                            'LR-' + str(lr_base),
                            'C-' + filter_list_seq.replace(',','_'),
                            'T-' + str(num_transformer_layers),
                            'motif-' + str(use_motif_activity)])
    date_string = f'{datetime.now():%Y-%m-%d_%H:%M:%S%z}'.replace(' ', '_')

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
            'lr_base': float(lr_base),
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
            'tpu': args.tpu_name,
            'gcs_path': args.gcs_path,
            'gcs_path_holdout': args.gcs_path_holdout,
            'decay_steps': args.decay_steps,
            'val_examples_ho': args.val_examples_ho,
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
            'unmask_loss': parse_bool_str(args.unmask_loss),
            'weight_decay': float(args.weight_decay),
            'mask_noise' : parse_bool_str(args.mask_noise)
    }

    wandb.init(config=config,
                project= args.wandb_project,
                id=run_id,
                name = None if not parse_bool_str(args.load_init) else mod_run_name + "_" + str(args.seed),
                entity=args.wandb_user,
                resume="allow" if not (parse_bool_str(args.load_init) and (run_id is not None)) else "must")
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

        wandb.config.update({"train_steps": 1 + (34021 * 16 // (GLOBAL_BATCH_SIZE))},
                            allow_val_change=True)
        wandb.config.update({"val_steps_ho" : wandb.config.val_examples_ho // GLOBAL_BATCH_SIZE},
                            allow_val_change=True)
        wandb.config.update({"total_steps": 1 + (34021 * 16 // GLOBAL_BATCH_SIZE)},
                            allow_val_change=True)
        # create the dataset iterators, one for training, one for holdout validation  
        train_human_it, data_val_ho = \
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
                                                            wandb.config.val_steps_ho, wandb.config.use_motif_activity,
                                                            wandb.config.mask_noise,
                                                            g, g_val)
        
        #train_human_its_mult = train_human_its

        print('created dataset iterators')
        # initialize model
        model = epibert.epibert_atac_pretrain(kernel_transformation=wandb.config.kernel_transformation,
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
                                use_rot_emb=wandb.config.use_rot_emb)

        print('initialized model')

        # initialize optimizer with warmup and cosine decay
        init_learning_rate=1.0e-07
        optimizer = tf.keras.optimizers.Adam(learning_rate=init_learning_rate,
                                                epsilon=wandb.config.epsilon,
                                                #weight_decay=wandb.config.weight_decay,
                                                global_clipnorm=wandb.config.gradient_clip)
        optimizer.exclude_from_weight_decay(var_names = ['bias', 'batch_norm','layer_norm',
                                                        'BN', 'LN', 'LayerNorm','BatchNorm'])
        metric_dict = {} # initialize dictionary to store metrics

        batch_num = tf.Variable(0, name="batch_num")
        optimizer_step_track = tf.Variable(0, name="optimizer_step_track")
        best_val_loss = tf.Variable(1000.0, name="best_val_loss")
        step_to_start = tf.Variable(0, name="step_to_start")
        ckpt = tf.train.Checkpoint(batch_num=batch_num,
                                    optimizer=optimizer,
                                    model=model,
                                    best_val_loss=best_val_loss,
                                    step_to_start=step_to_start,
                                    optimizer_step_track=optimizer_step_track)
        if args.step_to_start is not None:
            step_to_start.assign(args.step_to_start)
        
        checkpoint_dir = os.path.join(wandb.config.model_save_dir, wandb.run.name)
        if wandb.config.load_init:
            checkpoint_dir = wandb.config.checkpoint_path
            wandb.run.name = wandb.config.checkpoint_path.split('/')[-1]

        manager = tf.train.CheckpointManager(ckpt,
                                                directory=checkpoint_dir,
                                                max_to_keep=30)

        # initialize functions for training and validation steps
        train_step, val_step, build_step, metric_dict = \
            training_utils.return_train_val_functions(
                model=model,
                optimizer=optimizer,
                strategy=strategy,
                metric_dict=metric_dict,
                num_replicas=NUM_REPLICAS,
                loss_type=wandb.config.loss_type,
                total_weight=wandb.config.total_weight_loss,
                unmask_loss=wandb.config.unmask_loss
            )

        val_losses = []
        if wandb.config.load_init: # if loading pretrained model, initialize the best val loss from previous run
            val_losses.append(best_val_loss.numpy())
        patience_counter = 0 # simple patience counter for early stopping
        stop_criteria = False
        best_epoch = 0 # track epoch with best validation loss 

        print('building model...')
        build_step(data_val_ho)
        total_params = 0
        for k in model.trainable_variables:
            var = k.values[0]
            total_params += tf.size(var)
        print('built model, total params: ' + str(total_params))

        wandb.config.update({"num_epochs_to_start": 0}, allow_val_change=True)
        if wandb.config.load_init:
            status = ckpt.restore(tf.train.latest_checkpoint(wandb.config.checkpoint_path))
            print('restored from checkpoint')
            print('restart training at epoch: ' + str(1+ batch_num.numpy()))
            print('restart at data batch: ' + str(batch_num.numpy()))
            wandb.config.update({"num_epochs_to_start": batch_num.numpy()}, 
                                allow_val_change=True)

        local_epoch = 0
        print(wandb.config)
        for epoch_idx in range(wandb.config.num_epochs):
            #epoch_i = (epoch_idx + wandb.config.num_epochs_to_start) % len(train_human_its_mult)
            #step_num = ((wandb.config.num_epochs_to_start + epoch_idx) * \
            #                wandb.config.train_steps * GLOBAL_BATCH_SIZE)
            step_num = step_to_start.numpy()
            if (epoch_idx == 0):
                if wandb.config.load_init:
                    if not wandb.config.reset_optimizer_state:
                        current_optimizer_step = optimizer_step_track.numpy()
                    else:
                        print('restart optimizer learning rate schedule')
                        current_optimizer_step = 0
                else:
                    current_optimizer_step = step_num//GLOBAL_BATCH_SIZE

            print('starting epoch_' + str(1 + wandb.config.num_epochs_to_start + epoch_idx) + \
                        ' at step: ' + str(step_num))
            start = time.time()

            for k in range(wandb.config.train_steps):
                lr = schedulers.cos_w_warmup(current_optimizer_step,
                                             wandb.config.lr_base,
                                             wandb.config.warmup_steps,
                                             wandb.config.decay_steps,
                                             wandb.config.decay_frac,
                                             wandb.config.return_constant_lr)
                if ((k == 0) and (epoch_idx == 0)):
                    print('starting lr at:' + str(optimizer.lr.values[0]))
                optimizer.lr.assign(lr)
                optimizer.learning_rate.assign(lr)
                strategy.run(train_step, args=(next(train_human_it),))
                current_optimizer_step += 1

            optimizer_step_track.assign(current_optimizer_step)
            print('lr at:' + str(optimizer.lr.values[0]))
            train_loss = metric_dict['train_loss'].result().numpy() * NUM_REPLICAS #this is the per example loss * NUM_REPLICAS # multiply by NUM_REPLICAS to get total loss
            print('train_loss: ' + str(train_loss))

            wandb.log({'train_loss': train_loss},
                        step=step_num)
            wandb.log({'ATAC_pearsons_tr': metric_dict['ATAC_PearsonR_tr'].result()['PearsonR'].numpy(),
                        'ATAC_R2_tr': metric_dict['ATAC_R2_tr'].result()['R2'].numpy()},
                        step=step_num)
            duration = (time.time() - start) / 60.
            print('completed epoch ' + str(1 + wandb.config.num_epochs_to_start + epoch_idx) + ' - duration(mins): ' + str(duration))

            step_to_start.assign_add(wandb.config.train_steps * GLOBAL_BATCH_SIZE)

            # main validation step:
            # - run the validation loop
            # - return the true and predicted values to allow for plotting and other metrics
            start = time.time()

            for k in range(wandb.config.val_steps_ho):
                strategy.run(val_step, args=(next(data_val_ho),))

            val_loss = metric_dict['val_loss'].result().numpy() * NUM_REPLICAS #this is the per example loss # multiply by NUM_REPLICAS to get total loss 
            print('val_loss: ' + str(val_loss))

            val_losses.append(val_loss)
            wandb.log({'val_loss': val_loss}, step=step_num)
            print('ATAC_pearsons: ' + str(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()))
            print('ATAC_R2: ' + str(metric_dict['ATAC_R2'].result()['R2'].numpy()))
            if wandb.config.unmask_loss:
                print('ATAC_pearsons_um: ' + str(metric_dict['ATAC_PearsonR_um'].result()['PearsonR'].numpy()))
                print('ATAC_R2_um: ' + str(metric_dict['ATAC_R2_um'].result()['R2'].numpy()))
            wandb.log({'ATAC_pearsons': metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy(),
                        'ATAC_R2': metric_dict['ATAC_R2'].result()['R2'].numpy()},
                        step=step_num)

            duration = (time.time() - start) / 60.
            print('completed epoch ' + str(epoch_idx + 1 + wandb.config.num_epochs_to_start) + ' validation - duration(mins): ' + str(duration))

            if len(val_losses) > 1:
                if val_loss < min(val_losses[:-1]):
                    best_val_loss.assign(val_loss)

            # Start early stopping checks:
            # - After epoch one if not loading from a checkpoint
            # - Immediately (epoch 0) if loading from a checkpoint, using the provided best loss
            if ((wandb.config.num_epochs_to_start+epoch_idx+1) > 0 and (not wandb.config.load_init)) or wandb.config.load_init:
                stop_criteria, patience_counter, best_epoch = \
                    training_utils.early_stopping(
                        current_val_loss=val_losses[-1],             # Last value from val_losses
                        logged_val_losses=val_losses,                # Full list of val_losses
                        best_epoch=best_epoch,                       # Best epoch so far
                        patience=wandb.config.patience,              # Patience for early stopping
                        patience_counter=patience_counter,           # Current patience counter
                        min_delta=wandb.config.min_delta,            # Minimum change for early stopping
                    )
                print('patience counter at: ' + str(patience_counter))
            ckpt.batch_num.assign_add(1)
            if ((epoch_idx+1) % args.savefreq) == 0:
                save_path = manager.save()
                print('saving model after: epoch ' + str(1 + wandb.config.num_epochs_to_start + epoch_idx))
                #print('corresponds to stop point: start at data batch ' + str(epoch_i))

            for key, item in metric_dict.items(): # reset metrics for new epoch
                item.reset_state()

            if stop_criteria:
                print('early stopping at: epoch ' + str(1 + wandb.config.num_epochs_to_start + epoch_idx))
                break

        print('best model was at: epoch ' + str(best_epoch))

# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()