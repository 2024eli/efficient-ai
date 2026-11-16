"""Compare FIFO and adapter grouping on the same ready token vectors."""
import json
import random
import time

import pynvml as nvml
import torch
from torch import nn

from routing import RoutedLinear


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    width, rank, experts, batch_size = 2048, 16, 4, 32

    def linear(inputs, outputs):
        return nn.Linear(inputs, outputs, bias=False,
                         device=device, dtype=torch.bfloat16)

    gate = linear(width, experts)
    layer = RoutedLinear(
        linear(width, width), gate,
        [nn.Sequential(linear(width, rank), linear(rank, width))
         for _ in range(experts)],
        cutoff=0.0,
    ).eval()

    # Same 256 token vectors and eight batches in both conditions.
    x = torch.randn(8 * batch_size, width, device=device, dtype=torch.bfloat16)
    targets = gate(x).argmax(dim=-1)
    order = targets.argsort(stable=True)
    schedules = {
        "fifo": x.split(batch_size),
        "grouped": x[order].split(batch_size),
    }

    fifo_output = torch.cat([layer(batch) for batch in schedules["fifo"]])
    grouped_output = torch.cat([layer(batch) for batch in schedules["grouped"]])
    torch.testing.assert_close(fifo_output[order], grouped_output,
                               rtol=0.02, atol=0.02)

    # Count distinct experts selected in each batch outside the timing window.
    expert_calls = {
        name: sum(gate(batch).argmax(dim=-1).unique().numel() for batch in batches)
        for name, batches in schedules.items()
    }

    nvml.nvmlInit()
    try:
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        gpu = nvml.nvmlDeviceGetHandleByUUID(uuid)
        if nvml.nvmlDeviceGetMigMode(gpu)[0]:
            raise RuntimeError("Use a full GPU, not a MIG slice.")

        trials = [(repeat, name) for repeat in range(5) for name in schedules]
        random.Random(42).shuffle(trials)

        for repeat, name in trials:
            def run_window():
                for batch in schedules[name]:
                    layer(batch)

            for _ in range(10):
                run_window()
            torch.cuda.synchronize()

            for minimum_seconds in (2, 4, 8):
                start_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
                start = time.perf_counter()
                windows, elapsed = 0, 0.0
                while elapsed < minimum_seconds:
                    for _ in range(5):
                        run_window()
                    windows += 5
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start

                end_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
                if end_mj < start_mj:
                    raise RuntimeError("NVML energy counter decreased.")
                if end_mj > start_mj:
                    break
            else:
                raise RuntimeError("NVML energy counter did not advance after retries.")

            joules = (end_mj - start_mj) / 1000
            print(json.dumps({
                "repeat": repeat,
                "schedule": name,
                "batch_size": batch_size,
                "tokens_per_window": len(x),
                "expert_calls_per_window": expert_calls[name],
                "windows": windows,
                "gpu_joules": joules,
                "seconds": elapsed,
                "joules_per_token": joules / (windows * len(x)),
                "ms_per_window": 1000 * elapsed / windows,
            }), flush=True)
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()