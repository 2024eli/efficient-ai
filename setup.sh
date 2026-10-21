#!/bin/bash
set -euo pipefail
module load anaconda3
cd /vast/home/e/eli22/efficient-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA 12.8 wheels include Blackwell support; don't reuse an unverified shared torch.
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -m unittest -v test_benchmark.py
# Download once before GPU allocation. Default HF cache needs ~25+ GB free.
python - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
for model in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B']:
    snapshot_download(model, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
load_dataset('openai/gsm8k', 'main')
PY
