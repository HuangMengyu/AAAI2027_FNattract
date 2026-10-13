#!/usr/bin/env bash
#SBATCH -A naiss2026-4-1354-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --chdir=.
#SBATCH -t 00:30:00
#SBATCH --job-name=requirements
#SBATCH --output=logs_export_requirements/%A_%a.out

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
srun apptainer exec "${CONTAINER_ARGS[@]}" "$IMAGE" \
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
