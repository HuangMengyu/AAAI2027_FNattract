#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=plot_scatter_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=logs_plot_scatter/PAMAP2_%A_%a.out

set -euo pipefail

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:${PYTHONPATH:-}"

echo "${CUDA_VISIBLE_DEVICES:-}"
echo "Running on node: ${SLURMD_NODENAME:-unknown}"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

SEED="${SEED:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_PLOT_PAIRS="${MAX_PLOT_PAIRS:-50000}"
PRIOR_PLOT="${PRIOR_PLOT:-True}"

extra_args=()
if [[ -n "${MAX_BATCHES}" ]]; then
    extra_args+=(--max_batches "${MAX_BATCHES}")
fi
if [[ -n "${MAX_PLOT_PAIRS}" ]]; then
    extra_args+=(--max_plot_pairs "${MAX_PLOT_PAIRS}")
fi

python -u scripts/plot_binary_gmm_pair_scatter.py \
    --dataset_name PAMAP2 \
    --current_num_fold "${SLURM_ARRAY_TASK_ID}" \
    --seed "${SEED}" \
    --checkpoint_path checkpoints_baseline_best/maxiter200_warmup1_PAMAP2 \
    --out_dir plots_scatter/PAMAP2 \
    --epochs 10 \
    --warm_epochs 1 \
    --batch_size 128 \
    --lr 1e-3 \
    --prior_mode separate \
    --prior_model gmm \
    --prior_plot "${PRIOR_PLOT}" \
    --prior_gmm_metric cosine \
    --prior_cancel_weighting False \
    --prior_hard_neg_weight 1.0 \
    --prior_num_random_pairs 4000 \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter 200 \
    --prior_segment_len 4 \
    --strict_load False \
    "${extra_args[@]}"
