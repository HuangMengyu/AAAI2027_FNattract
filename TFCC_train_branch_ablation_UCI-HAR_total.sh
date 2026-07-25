#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=branch_ablation_UCI-HAR_total
#SBATCH --array=1-30

input_file='/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/branch_ablation_SleepEDFx_UCI-HAR_total.txt'
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
# mkdir -p logs_branch_ablation/UCI-HAR_total
# Disabled: --use_intra_sample_for_temporal_filter is obsolete; temporal filter uses temporal classifiers and priors.
srun --output="logs_branch_ablation/UCI-HAR_total/warmup1_${filter_temporal}_${filter_intra}_${filter_inter}_UCI-HAR_total_%A_%a.out" \
python main_5fold.py \
    --dataset_name UCI-HAR_total \
    --current_num_fold ${current_num_fold} \
    --epochs 10 \
    --warm_epochs 1 \
    --adaptive_warmup_threshold 0.35 \
    --filter_temporal ${filter_temporal} \
    --filter_intra ${filter_intra} \
    --filter_inter ${filter_inter} \
    --model_save_path checkpoints/branch_ablation/warmup1_${filter_temporal}_${filter_intra}_${filter_inter}_UCI-HAR_total \
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
    --prior_cancel_weighting False \
    --prior_hard_neg_weight 1.0 \
    --prior_num_random_pairs 4000 \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter 200 \
    --prior_segment_len 4
