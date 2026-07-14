#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=binary_prior_ablation_PAMAP2
#SBATCH --array=1-8

input_file='/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/binary_prior_ablation_PAMAP2.txt'
# Load parameters safely
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

# echo "cut_off=${cut_off} attract_filter=${attract_filter}"
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:$PYTHONPATH"

# cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

# mkdir -p $TMPDIR/TFCC_multimodal
# For Testing:
srun --output="logs_binary_prior/PAMAP2_maxiter200/warmup1_Binary${fn_filter_use_binary}_Prior${fn_filter_use_prior}_PAMAP2_%A_%a.out" \
python -u main_5fold.py \
    --dataset_name PAMAP2 \
    --current_num_fold ${current_num_fold} \
    --epochs 10 \
    --warm_epochs 1 \
    --adaptive_warmup_threshold 0.35 \
    --fn_filter_use_binary ${fn_filter_use_binary}\
    --fn_filter_use_prior ${fn_filter_use_prior}\
    --model_save_path checkpoints/binary_prior_ablation/warmup1_maxiter200_Binary${fn_filter_use_binary}_Prior${fn_filter_use_prior}_PAMAP2 \
    --batch_size 128 \
    --lr 1e-3 \
    --ssl True \
    --use_prior True \
    --prior_mode separate \
    --prior_model gmm \
    --prior_plot False \
    --temporal_binary_mode binary \
    --intra_binary_mode binary \
    --inter_binary_mode binary \
    --prior_gmm_metric cosine \
    --use_intra_sample_for_temporal_filter False \
    --prior_cancel_weighting False \
    --prior_hard_neg_weight 1.0 \
    --prior_num_random_pairs 4000 \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter 200 \
    --prior_segment_len 4
