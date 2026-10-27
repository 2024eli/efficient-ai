#!/bin/bash
# Run once from the repo directory before lora.sbatch: compute jobs are offline.
# Creates .venv here if missing (same pinned CUDA 12.8 torch as setup.sh), installs deps,
# downloads models/adapter/data into the default HF cache (~/.cache/huggingface), runs tests.
set -euo pipefail
module load anaconda3
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
else
  source .venv/bin/activate
fi
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
python - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
from lora_energy import ADAPTER, ADAPTER_REVISION
for model in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B']:
    snapshot_download(model, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
print(snapshot_download(ADAPTER, revision=ADAPTER_REVISION))
load_dataset('openai/gsm8k', 'main')
PY
python -m unittest -v test_lora.py
