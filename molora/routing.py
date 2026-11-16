"""Reference top-1 LoRA dispatch for frozen dense linear layers."""
import torch
from torch import nn


class RoutedLinear(nn.Module):
    """Gate scores must come from a trained router in quality evaluations.

    Each expert is a module computing a scaled LoRA delta, without the base projection.
    A confidence cutoff skips the delta entirely; the base always runs.
    """
    def __init__(self, base, gate, experts, cutoff=0.0):
        super().__init__()
        if not experts or not 0 <= cutoff <= 1:
            raise ValueError("Need at least one expert and a cutoff in [0, 1]")
        self.base = base
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.cutoff = cutoff

    def forward(self, x):
        y = self.base(x)
        tokens = x.reshape(-1, x.shape[-1])
        scores = self.gate(tokens).float().softmax(dim=-1)
        confidence, chosen = scores.max(dim=-1)
        if scores.shape[-1] != len(self.experts):
            raise ValueError("Gate must return one logit per expert")
        active = confidence >= self.cutoff
        result = y.reshape(-1, y.shape[-1])
        for i, expert in enumerate(self.experts):
            indices = torch.where(active & (chosen == i))[0]
            if indices.numel():
                delta = expert(tokens.index_select(0, indices))
                result = result.index_add(0, indices, delta.to(result.dtype))
        return result.reshape_as(y)


def choose_cutoff(validation, margin=0.02):
    """Pick lowest measured energy within margin of cutoff=0 on validation."""
    if not 0 <= margin <= 1:
        raise ValueError("margin must be in [0, 1]")
    baseline = next(row for row in validation if row["cutoff"] == 0)
    eligible = [r for r in validation if r["accuracy"] >= baseline["accuracy"] - margin]
    return min(eligible, key=lambda r: (r["joules_per_request"], r["cutoff"]))
