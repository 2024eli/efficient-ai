#!/bin/bash
#SBATCH --job-name=molora-schedule
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:10:00
#SBATCH --output=molora-schedule-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
source /vast/home/e/eli22/efficient-ai/.venv/bin/activate
python -u schedule_bench.py > "schedule-${SLURM_JOB_ID}.jsonl"