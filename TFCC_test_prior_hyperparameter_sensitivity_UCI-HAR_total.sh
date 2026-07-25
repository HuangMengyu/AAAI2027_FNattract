#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=test_prior_sensitivity_UCI-HAR
#SBATCH --array=6-6

input_file='/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/prior_hyperparameter_sensitivity_test.txt'
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:$PYTHONPATH"

cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

log_dir='logs_prior_hyperparameter_sensitivity/test/UCI-HAR_total'
checkpoint_path="checkpoints/prior_hyperparameter_sensitivity/warmup1_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_UCI-HAR_total"
mkdir -p "$log_dir"

srun --output="${log_dir}/warmup1_${sensitivity_setup}_pairs${prior_num_random_pairs}_iter${prior_fit_max_iter}_seg${prior_segment_len}_UCI-HAR_total_%A_%a.out" \
python -u test.py \
    --dataset_name UCI-HAR_total \
    --baseline_checkpoint_path checkpoints_baseline_best/baseline_UCI-HAR_total \
    --checkpoint_path "$checkpoint_path"
