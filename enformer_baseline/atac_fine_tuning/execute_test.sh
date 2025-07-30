#!/bin/bash -l

python3 model_eval.py \
            --tpu_name="pod1" \
            --tpu_zone="europe-west4-a" \
            --wandb_project="enformer_rampage_atac_ft_test" \
            --wandb_user="${WANDB_USER:-default_user}" \
            --wandb_sweep_name="enformer_rampage_atac_ft_test" \
            --gcs_path="gs://genformer_europe_west_copy/enformer_atac/testing_all_cell_types/tfrecords" \
            --num_parallel=4 \
            --test_examples=1937 \
            --num_targets=50 \
            --checkpoint_path="gs://egenformer_europe_west_copy/enformer_atac/all_cell_types/models/enformer_atac_2024-10-29_04:31:31_ENFORMER_LR1-5e-06_LR2-0.0001_GC-1_init-True_enformer_atac_2024-10-29_04:31:08/39-37.data-00000-of-00001"