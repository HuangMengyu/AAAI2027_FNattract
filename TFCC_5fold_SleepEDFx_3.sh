#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1224 -p alvis
#SBATCH -t 10:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_multimodal
#SBATCH --array=1-5
#SBATCH --output=logs_3modality/SleepEDFx_3/newmethod_1vsall_SleepEDFx_3_%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

# python -m venv wandb_venv
source wandb_venv/bin/activate
# pip install wandb

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-FANTII_AdaptiveWeights}

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export WANDB_API_KEY='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'


export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:$PYTHONPATH"

# cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
python -u main_5fold.py \
    --dataset_name SleepEDFx_3 \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --warm_epochs 2 \
    --adaptive_warmup_threshold 0.035 \
    --three_mod_contrast 1vsall \
    --model_save_path checkpoints_3modality/newmethod_1vsall_SleepEDFx_3 \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    --use_fn_mask False \
    --adaptive_filter_thresholds False \
    --use_prior True \
    --prior_mode separate \
    --prior_model gmm \
    --prior_plot True \
    --temporal_binary_mode binary \
    --intra_binary_mode binary \
    --inter_binary_mode binary \
    --prior_gmm_metric cosine \
    --prior_center_cosine False \
    --use_intra_sample_for_temporal_filter False \
    --prior_save_dir prior_cache/SleepEDFx_3_1vsall \
    --prior_hard_neg_weight 1.0 \
    --prior_num_random_pairs 10000 \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter 200 \
    --prior_segment_len 40 

    
