#!/bin/bash -l

export TPU_LOAD_LIBRARY=0
export TPU_NAME=$1
export ZONE=$2

python3 train_model_atac_rampage_test.py \
            --tpu_name=$1 \
            --tpu_zone=$2 \
            --wandb_project="atac_rampage_paired_test" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_rampage_paired_test" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://$3/rampage_finetune/524k/paired_atac_rampage" \
            --gcs_path_holdout="gs://$3/rampage_finetune/524k/paired_atac_rampage_holdout" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=896 \
            --max_shift=4 \
            --batch_size=1 \
            --test_examples=92000 \
            --test_examples_ho=14720 \
            --test_TSS=105300 \
            --test_TSS_ho=16848 \
            --BN_momentum=0.90 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://$3/524k/rampage_finetune/models" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="8" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="4" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --rectify="True" \
            --filter_list_seq="512,640,640,768,896,1024" \
            --filter_list_atac="32,64" \
            --atac_mask_dropout=0.0 \
            --atac_mask_dropout_val=0.0 \
            --log_atac="False" \
            --random_mask_size="128" \
            --use_atac="True" \
            --final_point_scale="6" \
            --use_seq="True" \
            --atac_corrupt_rate="1000" \
            --use_motif_activity="True" \
            --total_weight_loss="0.15" \
            --use_rot_emb="True" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --decay_frac="0.10" \
            --gradient_clip="0.1" \
            --seed=9 \
            --val_data_seed=19 \
            --loss_type="poisson" \
            --model_save_basename="genformer" \
            --warmup_steps=5000 \
            --decay_steps=500000 \
            --load_init_FT="False" \
            --load_init="False" \
            --atac_scale="0.001" \
            --predict_atac="True"