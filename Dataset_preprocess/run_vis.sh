#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=Dataset_preprocess
#SBATCH -t 00:30:00
#SBATCH --job-name=visualization
#SBATCH --output=logs_vis/%A.out

module --quiet purge

PROJECT=/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027
IMAGE="$PROJECT/containers/alvis-reconstruction/experiment-arm64.sif"
WORKDIR="$PROJECT/Dataset_preprocess"

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

# pip install kaleido==0.2.1

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# echo ${CUDA_VISIBLE_DEVICES}

# export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:$PYTHONPATH"

# # cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection

# For Testing:
srun apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python3 visualize_dataset.py \
  /nobackup/proj/disk/naiss2025-22-1224/shared/UCI-HAR_total \
  --subject 16 \
  --num-samples 6 \
  --sample-length 100 \
  --output figures/uci-har_total
