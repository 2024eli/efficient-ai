#!/bin/bash
#SBATCH --job-name=routing-screen
#SBATCH --account=leebcc-systems-architecture-int
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=routing-screen-%j.log
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
case "${MODE:-smoke}" in
  smoke) extra=(--smoke) ;;
  collect) extra=() ;;
  *) echo 'MODE must be smoke or collect'; exit 2 ;;
esac
python -u collect.py --data screen-data.json --output "screen-results/${MODE:-smoke}-${SLURM_JOB_ID}" "${extra[@]}"
