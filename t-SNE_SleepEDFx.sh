#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=TFCC_t-SNE
#SBATCH --output=logs_t-SNE/SleepEDFx_%A.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load plotly.py/5.16.0-GCCcore-12.3.0
module load einops/0.7.0-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a

python -m venv t-SNE_env
source t-SNE_env/bin/activate
# pip install kaleido==0.2.1

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo ${CUDA_VISIBLE_DEVICES}

export PYTHONPATH="/mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027:$PYTHONPATH"

# cd /cephyr/users/mengyuh/Alvis/ComparativeStudy/Codes/Codes/Collection
cd /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027

# For Testing:
python t-SNE.py \
  --dataset_name SleepEDFx \
  --fold 3 \
  --seed 0 \
  --apply_pca False \
  --baseline_checkpoint_path checkpoints_baseline_best/baseline_SleepEDFx \
  --method_checkpoint_path checkpoints_baseline_best/gmm_200iter_concat_cosine_adaptivewarmup0.35_SleepEDFx_current_best \
  --baseline_name baseline \
  --method_name new_method \
  --img_save_dir tsne_outputs_nopca/SleepEDFx_fold3 \
  --feature_level transformer

    