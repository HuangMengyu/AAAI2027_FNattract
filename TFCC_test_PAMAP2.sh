#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=TFCC-test
#SBATCH --output=logs_4/test/PAMAP2_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0


echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"
export WANDB_API_KEY='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'

# cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM


python FANTII_SP_extend/test.py \
    --dataset_name PAMAP2 \
    --baseline_checkpoint_path /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM/FANTII_SP_extend/checkpoints/baseline_PAMAP2  \
    --checkpoint_path /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM/FANTII_SP_extend_prior/checkpoints/gmm_100iter_concat_cosine_adaptivewarmup0.35_usehardnegdownweight_PAMAP2  \
