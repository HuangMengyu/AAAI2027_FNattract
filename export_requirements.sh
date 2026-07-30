#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-117 -p alvis
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --job-name=requirements
#SBATCH --output=logs_export_requirements/%A_%a.out

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.4.2-gfbf-2023a
module load einops/0.7.0-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a
    
# python3 -m pip freeze | grep -E '^(numpy|scipy|scikit-learn|tqdm|einops|torch|pandas|matplotlib|mne)==' > requirements.txt

# print some packages
# python3 - <<'PY' > requirements.txt
# packages = [
#     ("numpy", "numpy"),
#     ("scipy", "scipy"),
#     ("sklearn", "scikit-learn"),
#     ("tqdm", "tqdm"),
#     ("einops", "einops"),
#     ("torch", "torch"),
#     ("pandas", "pandas"),
#     ("matplotlib", "matplotlib"),
#     ("mne", "mne"),
# ]

# for import_name, package_name in packages:
#     try:
#         module = __import__(import_name)
#         version = getattr(module, "__version__", None)
#         if version is None and import_name == "sklearn":
#             import sklearn
#             version = sklearn.__version__
#         print(f"{package_name}=={version}" if version else package_name)
#     except Exception as exc:
#         print(f"# {package_name} not available: {exc}")
# PY

# print all packages
python3 - <<'PY' > requirements_all.txt
from importlib.metadata import distributions

items = []
for dist in distributions():
    name = dist.metadata.get("Name")
    version = dist.version
    if name and version:
        items.append((name.lower(), name, version))

for _, name, version in sorted(items):
    print(f"{name}=={version}")
PY

echo "Finished!"