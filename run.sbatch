#!/bin/bash
#SBATCH --job-name=qwen-icl
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=qwen-icl-%j.log
set -euo pipefail
module load anaconda3
cd /vast/home/e/eli22/efficient-ai
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
nvidia-smi
python -u - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), 'architectures:', torch.cuda.get_arch_list())
x = torch.ones(16, device='cuda')
assert (x+x).sum().item() == 32
torch.cuda.synchronize()
PY
# MODE=smoke: both sizes and every shot condition, 3 questions, 1 repeat.
# MODE=full: 200 paired questions, 3 measurement repeats = 3600 generations.
case "${MODE:-smoke}" in
  smoke) count=3; repeats=1 ;;
  full) count=200; repeats=3 ;;
  *) echo 'MODE must be smoke or full'; exit 2 ;;
esac
python -u benchmark.py --n "$count" --repeats "$repeats" \
  --output "results/${MODE:-smoke}-${SLURM_JOB_ID}"
