#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=plot_scatter_PAMAP2
#SBATCH --array=1-4
#SBATCH --output=logs_plot_scatter/PAMAP2_%A_%a.out

module --quiet purge

PROJECT=/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027
IMAGE="$PROJECT/containers/alvis-reconstruction/experiment-arm64.sif"
WORKDIR="$PROJECT"

if [[ ! -r "$IMAGE" ]]; then
    echo "Container image is missing or unreadable: $IMAGE" >&2
    exit 1
fi
cd "$WORKDIR" || exit 1

# Retain Slurm's GPU assignment inside the clean container environment.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
export APPTAINERENV_PYTHONPATH="$PROJECT"

CONTAINER_ARGS=(
    --cleanenv
    --nv
    --bind /nobackup/proj
    # Support Python entry points that still contain the original dataset paths.
    --bind "/nobackup/proj/disk/naiss2025-22-1224/shared:/mimer/NOBACKUP/groups/naiss2025-22-1224"
    --bind "$PROJECT:/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027"
    --pwd "$WORKDIR"
)

set -euo pipefail


export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"
export PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"

echo "${CUDA_VISIBLE_DEVICES:-}"
echo "Running on node: ${SLURMD_NODENAME:-unknown}"


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

srun apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
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
