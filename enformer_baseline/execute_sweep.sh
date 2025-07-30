#!/bin/bash -l

python3 train_model.py \
            --tpu_name="node-1" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_rampage_ft" \
            --wandb_user="${WANDB_USER:-default_user}" \
            --wandb_sweep_name="enformer_rampage_ft" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://enformer_baseline/human/tfrecords"\
            --gcs_path_TSS="gs://enformer_baseline/human_tss/tfrecords" \
            --num_epochs=50 \
            --warmup_frac=0.146 \
            --patience=30\
            --min_delta=0.00001 \
            --model_save_dir="gs://enformer_baseline/models" \
            --model_save_basename="enformer_baseline" \
            --lr_base1="5.0e-06" \
            --lr_base2="5.0e-04" \
            --gradient_clip="0.2" \
            --epsilon=1.0e-8 \
            --num_parallel=4 \
            --savefreq=1 \
            --train_examples=34021 \
            --val_examples=2213 \
            --val_examples_TSS=1721 \
            --num_targets=50 \
            --use_enformer_weights="True" \
            --enformer_checkpoint_path="sonnet_weights"
