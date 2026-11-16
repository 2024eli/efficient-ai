#!/bin/bash
#SBATCH --job-name=qwen-icl2
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=icl2-%j.log
set -euo pipefail
# Submit from the experiment directory. Slurm copies scripts elsewhere, so use submit dir.
cd "$SLURM_SUBMIT_DIR"
if type module >/dev/null 2>&1; then module load anaconda3; fi
source "$(cat .venv-path)/bin/activate"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
nvidia-smi
case "${MODE:-smoke}" in
  smoke) extra=(--smoke) ;;
  full) extra=() ;;
  *) echo 'MODE must be smoke or full'; exit 2 ;;
esac
python -u run_icl.py --output "results/${MODE:-smoke}-${SLURM_JOB_ID}" "${extra[@]}"
# Automatic audit/summary; full analysis is brief CPU work after collection.
OPENBLAS_NUM_THREADS=1 python -u analyze.py "results/${MODE:-smoke}-${SLURM_JOB_ID}"
