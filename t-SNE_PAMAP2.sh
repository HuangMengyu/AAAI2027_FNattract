#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=TFCC_t-SNE
#SBATCH --output=logs_t-SNE/PAMAP2_%A.out

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

# pip install umap-learn
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
python t-SNE.py \
  --dataset_name PAMAP2 \
  --fold 1 \
  --seed 0 \
  --apply_pca False \
  --baseline_checkpoint_path checkpoints_baseline_best/baseline_PAMAP2 \
  --method_checkpoint_path checkpoints/warmup_ablation/maxiter200_warmup1_PAMAP2 \
  --baseline_name baseline \
  --method_name new_method \
  --img_save_dir umap_outputs_nopca/PAMAP2_fold1_maxiter200_warmup1 \
  --feature_level transformer \
  --perplexity 30 \
  --reducer umap \
  --umap_n_neighbors 10 \
  --umap_min_dist 0.25 \


