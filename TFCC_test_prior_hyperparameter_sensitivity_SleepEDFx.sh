#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=test_prior_sensitivity_SleepEDFx
#SBATCH --array=1-6

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

input_file='/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027/prior_hyperparameter_sensitivity_test.txt'
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")


echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"
export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"


log_dir='logs_prior_hyperparameter_sensitivity/test/SleepEDFx'
checkpoint_path="checkpoints/prior_hyperparameter_sensitivity/warmup4_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_SleepEDFx"
mkdir -p "$log_dir"

mkdir -p "${log_dir}"
srun --output="${log_dir}/warmup4_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_SleepEDFx_%A_%a.out" apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python -u test.py \
    --dataset_name SleepEDFx \
    --baseline_checkpoint_path checkpoints_baseline_best/baseline_SleepEDFx \
    --checkpoint_path "$checkpoint_path"
