#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if type module >/dev/null 2>&1; then module load anaconda3; fi
# Prefer the already-working private environment next to this experiment folder.
if [[ -z "${ICL_VENV:-}" ]]; then
  if [[ -f ../.venv/bin/activate ]]; then ICL_VENV="$(cd .. && pwd)/.venv";
  else ICL_VENV="$PWD/.venv"; fi
fi
if [[ ! -f "$ICL_VENV/bin/activate" ]]; then python -m venv "$ICL_VENV"; fi
source "$ICL_VENV/bin/activate"
if ! python -c 'import torch; assert torch.__version__.startswith("2.7.0") and torch.version.cuda == "12.8"' >/dev/null 2>&1; then
  python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
fi
python -m pip install -r ../requirements.txt
python -m pip check
python -c 'import torch,numpy,pyarrow,datasets; print("Imports OK",torch.__version__,numpy.__version__,pyarrow.__version__)'
python -m unittest -v test_icl.py
# Record only the environment path, not credentials. GPU jobs use the same environment.
printf '%s\n' "$ICL_VENV" > .venv-path
if [[ -f plan.json ]]; then
  echo 'Using existing frozen plan.json. Delete/archive intentionally to prepare a new study.'
else
  # Set EXCLUDE_DATA to the previous routing plan to avoid reusing previously inspected questions.
  if [[ -n "${EXCLUDE_DATA:-}" ]]; then python prepare.py --exclude-data "$EXCLUDE_DATA";
  else python prepare.py; fi
fi
