#!/bin/bash -l

export TPU_LOAD_LIBRARY=0
export TPU_NAME=$1
export ZONE=$2

python3 train_model.py \
            --tpu_name="pod1" \
            --tpu_zone="europe-west4-a" \
            --wandb_project="enformer_atac_ft" \
            --wandb_user="${WANDB_USER:-default_user}" \
            --wandb_sweep_name="enformer_atac_ft" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_europe_west_copy/enformer_atac/tfrecords"\
            --num_epochs=50 \
            --warmup_frac=0.146 \
            --patience=30\
            --min_delta=0.00001 \
            --model_save_dir="gs://genformer_europe_west_copy/enformer_atac/models" \
            --model_save_basename="enformer_atac" \
            --lr_base1="5.0e-06" \
            --lr_base2="5.0e-04" \
            --gradient_clip="1.0" \
            --epsilon=1.0e-8 \
            --num_parallel=4 \
            --savefreq=1 \
            --train_examples=34021 \
            --val_examples=2213 \
            --num_targets=5 \
            --use_enformer_weights="True" \
            --enformer_checkpoint_path="sonnet_weights"
