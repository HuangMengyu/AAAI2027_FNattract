#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=T4:1
# #SBATCH -C NOGPU
#SBATCH --job-name=plot_fn_analysis
#SBATCH --output=logs_plot_fn_analysis/SleepEDFx_%A.out

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
python3 scripts/plot_fn_analysis.py \
  --with-prior /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/checkpoints/best_setup_4000_4_warmup4_SleepEDFx/0/fn_analysis/SleepEDFx_fold1_seed0/fn_attraction_epoch_summary.csv \
  --no-prior /mimer/NOBACKUP/groups/naiss2025-22-1224/AAAI2027/checkpoints_fn_analysis/4000_4_warmup4_noprior_SleepEDFx/0/fn_analysis/SleepEDFx_fold1_seed0/fn_attraction_epoch_summary.csv \
  --title-prefix SleepEDFx \
  --out-dir plots_fn_analysis/SleepEDFx \
  --formats png \
  # --use_average

    