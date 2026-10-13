#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=plot_fn_analysis
#SBATCH --output=logs_plot_fn_analysis/SleepEDFx_%A.out

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

# #SBATCH -C NOGPU


# pip install kaleido==0.2.1

export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"


# For Testing:
srun apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python3 scripts/plot_fn_analysis.py \
  --with-prior /nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027/checkpoints/best_setup_4000_4_warmup4_SleepEDFx/0/fn_analysis/SleepEDFx_fold1_seed0/fn_attraction_epoch_summary.csv \
  --no-prior /nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027/checkpoints_fn_analysis/4000_4_warmup4_noprior_SleepEDFx/0/fn_analysis/SleepEDFx_fold1_seed0/fn_attraction_epoch_summary.csv \
  --title-prefix SleepEDFx \
  --out-dir plots_fn_analysis/SleepEDFx \
  --formats png \
  # --use_average


