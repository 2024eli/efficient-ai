"""Shared strict numeric scoring and allocated-GPU NVML mapping."""
import re
from decimal import Decimal, InvalidOperation

def answer(text, gold=False):
    if gold:
        match = re.search(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    else:
        # Never silently use the last number in an unfinished reasoning trace.
        if '<think>' in text and '</think>' not in text:
            return None
        text = text.split('</think>')[-1] if '</think>' in text else text
        matches = list(re.finditer(r'(?im)^\s*FINAL:\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$', text))
        match = matches[-1] if matches else None
    try:
        return str(Decimal(match.group(1).replace(',', '')).normalize()) if match else None
    except InvalidOperation:
        return None

def gpu_handle(torch, nvml):
    # Match CUDA's device to NVML UUID; never assume Slurm GPU 0 is physical GPU 0.
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, 'uuid', ''))
    if not uuid:
        raise RuntimeError('Cannot map CUDA device to NVML UUID; energy attribution unavailable.')
    handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
    try:
        nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except nvml.NVMLError as exc:
        raise RuntimeError('GPU cumulative energy counter unavailable; do not substitute zero.') from exc
    return handle, uuid
