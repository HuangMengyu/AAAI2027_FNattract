#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=TFCC-test
#SBATCH --array=1-2

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

input_file=/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027/binary_prior_ablation_test.txt
# Load parameters safely
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")


# echo "cut_off=${cut_off} attract_filter=${attract_filter}"
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"

export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"

mkdir -p "logs_binary_prior/test/PAMAP2_maxiter200"
srun --output="logs_binary_prior/test/PAMAP2_maxiter200/warmup1_Binary${fn_filter_use_binary}_Prior${fn_filter_use_prior}_PAMAP2_%A_%a.out" apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python test.py \
    --dataset_name PAMAP2 \
    --baseline_checkpoint_path checkpoints_baseline_best/baseline_PAMAP2 \
    --checkpoint_path checkpoints/binary_prior_ablation/warmup1_maxiter200_Binary${fn_filter_use_binary}_Prior${fn_filter_use_prior}_PAMAP2 \

