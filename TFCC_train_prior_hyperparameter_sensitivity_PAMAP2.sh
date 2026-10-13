#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=prior_sensitivity_PAMAP2
#SBATCH --array=1-24

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

input_file='/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027/prior_hyperparameter_sensitivity_PAMAP2.txt'
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")


echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export APPTAINERENV_TORCH_DISTRIBUTED_DEBUG="$TORCH_DISTRIBUTED_DEBUG"
export PYTHONPATH="/nobackup/proj/disk/naiss2025-22-1224/shared/AAAI2027/AAAI2027:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"


log_dir='logs_prior_hyperparameter_sensitivity/PAMAP2'
checkpoint_path="checkpoints/prior_hyperparameter_sensitivity/warmup1_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_PAMAP2"
mkdir -p "$log_dir"

mkdir -p "${log_dir}"
srun --output="${log_dir}/warmup1_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_PAMAP2_%A_%a.out" apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
python -u main_5fold.py \
    --dataset_name PAMAP2 \
    --current_num_fold "${current_num_fold}" \
    --epochs 10 \
    --warm_epochs 1 \
    --adaptive_warmup_threshold 0.35 \
    --model_save_path "$checkpoint_path" \
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
    --prior_num_random_pairs "${prior_num_random_pairs}" \
    --prior_num_self_pairs 0 \
    --prior_delta_mode concat \
    --prior_fit_max_iter "${prior_fit_max_iter}" \
    --prior_segment_len "${prior_segment_len}"
