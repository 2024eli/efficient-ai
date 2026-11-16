#!/bin/bash
#SBATCH --job-name=molora-real
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --output=molora-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source /vast/home/e/eli22/efficient-ai/.venv/bin/activate
export HF_HUB_OFFLINE=1
python -u serve_bench.py "${1:?Pass the previous LoRA run directory}" --out "real-${SLURM_JOB_ID}" "${@:2}"
