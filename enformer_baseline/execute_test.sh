#!/bin/bash -l

python3 model_eval.py \
            --tpu_name="pod_uscentral" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_rampage_ft" \
            --wandb_user="${WANDB_USER:-default_user}" \
            --wandb_sweep_name="enformer_rampage_ft_test" \
            --gcs_path="gs://enformer_baseline/human/tfrecords" \
            --gcs_path_TSS="gs://enformer_baseline/human_tss/tfrecords" \
            --num_parallel=4 \
            --test_examples=1937 \
            --tss_examples=2032 \
            --num_targets=50 \
            --checkpoint_path="gs://enformer_baseline/models/enformer_baseline_2024-03-01_15:19:09_ENFORMER_LR1-1e-06_LR2-0.0001_GC-0.2_init-True_enformer_baseline_2024-03-01_15:19:00-28.data-00000-of-00001"