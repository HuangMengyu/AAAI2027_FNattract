#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=TFCC_multimodal
#SBATCH --array=1-1
#SBATCH --output=log_analysis/%A.out

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

# pip install wandb

export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"
export PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUNBUFFERED="$PYTHONUNBUFFERED"

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export WANDB_API_KEY='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'
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"


export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"

srun apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python3 analyze_psd_prior_statistics.py \
  --dataset_name PAMAP2 \
  --fold 3 \
  --feature_type magnitude \
  --magnitude_method rms \
  --magnitude_transform log \
  --normalize_method none \
  --normalize_axis segment \
  --output_dir psd_prior_analysis/PAMAP2_fold3_none_magnitude


    # previous optimal setup
    # --cut_off_uni 1 \
    # --cut_off_multi 1 \
    # --attract_filter_uni 0.95 \
    # --attract_filter_multi 0.95 \
    # --gamma 0.3 \

    
