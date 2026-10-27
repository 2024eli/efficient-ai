#!/bin/bash
# Run once (CPU/login policy permitting) before lora.sbatch: compute jobs are offline.
set -euo pipefail
module load anaconda3
cd /vast/home/e/eli22/efficient-ai
source .venv/bin/activate
python -m pip install -r requirements.txt
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
