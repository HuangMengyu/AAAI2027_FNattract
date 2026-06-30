#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1224 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=TFCC_multimodal
#SBATCH --array=1-1
#SBATCH --output=log_analysis/%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0

# python -m venv wandb_venv
source wandb_venv/bin/activate
# pip install wandb

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

echo ${CUDA_VISIBLE_DEVICES}
echo "Running on node: $SLURMD_NODENAME"

export WANDB_API_KEY='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'


export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM:$PYTHONPATH"

# cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/KDD2026_TriAFNM
python3 FANTII_SP_extend_prior/analyze_psd_prior_statistics.py \
  --dataset_name PAMAP2 \
  --fold 3 \
  --feature_type magnitude \
  --magnitude_method rms \
  --magnitude_transform log \
  --normalize_method none \
  --normalize_axis segment \
  --output_dir FANTII_SP_extend_prior/psd_prior_analysis/PAMAP2_fold3_none_magnitude


    # previous optimal setup
    # --cut_off_uni 1 \
    # --cut_off_multi 1 \
    # --attract_filter_uni 0.95 \
    # --attract_filter_multi 0.95 \
    # --gamma 0.3 \

    
