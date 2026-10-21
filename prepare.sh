#!/bin/bash
set -euo pipefail
module load anaconda3
cd /vast/home/e/eli22/efficient-ai
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
from huggingface_hub import snapshot_download
for model in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B']:
    snapshot_download(model,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
PY
python prepare.py
