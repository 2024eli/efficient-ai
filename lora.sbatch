#!/bin/bash
#SBATCH --job-name=lora-energy
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=lora-energy-%j.log
set -euo pipefail
module load anaconda3
# Submit from the repo directory: sbatch runs in the submission directory.
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
nvidia-smi
case "${MODE:-smoke}" in
  # Smoke: 20 questions, 1 repeat, tiny fixed/forward subsets. Debugs loading, parsing, energy.
  smoke) extra=(--n 20 --repeats 1 --fixed-n 5 --forward-n 5 --merged) ;;
  full) extra=(--n 150 --repeats 2 --fixed-n 50 --forward-n 30 --merged) ;;
  *) echo 'MODE must be smoke or full'; exit 2 ;;
esac
python -u lora_energy.py --output "lora-results/${MODE:-smoke}-${SLURM_JOB_ID}" "${extra[@]}"
python -u lora_analyze.py "lora-results/${MODE:-smoke}-${SLURM_JOB_ID}"
