#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1224 -p alvis
#SBATCH -t 3:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC_multimodal_bmm_ablation
#SBATCH --array=1-3
#SBATCH --output=logs/SleepEDFx/bmm_4000_4_warmup4_SleepEDFx_%A_%a.out

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
    --dataset_name SleepEDFx \
    --current_num_fold ${SLURM_ARRAY_TASK_ID} \
    --epochs 10 \
    --warm_epochs 4 \
    --adaptive_warmup_threshold 0.035 \
    --fn_filter_use_prior True \
    --model_save_path checkpoints/bmm_ablation/4000_4_warmup4_bmm_SleepEDFx \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    --use_prior True \
    --prior_mode separate \
    --prior_model bmm \
    --prior_plot False \
    --temporal_binary_mode binary \
    --intra_binary_mode binary \
    --inter_binary_mode binary \
    --prior_gmm_metric cosine \
    --prior_cancel_weighting False \
    --use_intra_sample_for_temporal_filter False \
    --prior_hard_neg_weight 1.0 \
    --prior_num_random_pairs 4000 \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter 200 \
    --prior_segment_len 4 \
    --fn_analysis True \
    --seeds 40

    
