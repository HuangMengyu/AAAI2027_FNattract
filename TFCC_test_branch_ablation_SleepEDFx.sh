#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-351 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC-test
#SBATCH --array=1-6

input_file=/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/branch_ablation_test.txt
# Load parameters safely
source <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$input_file")

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

# echo "cut_off=${cut_off} attract_filter=${attract_filter}"
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:$PYTHONPATH"
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

srun --output="logs_branch_ablation/test/SleepEDFx/${filter_temporal}_${filter_intra}_${filter_inter}_SleepEDFx_%A_%a.out" \
python test.py \
    --dataset_name SleepEDFx \
    --baseline_checkpoint_path checkpoints_baseline_best/baseline_SleepEDFx \
    --checkpoint_path checkpoints/branch_ablation/${filter_temporal}_${filter_intra}_${filter_inter}_SleepEDFx \

