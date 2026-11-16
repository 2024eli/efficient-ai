Submit from this folder on Betty: sbatch run.sbatch. Uses existing CUDA
PyTorch and nvidia-ml-py. For batch-size-one decoding shape, change bench.py
invocation to include --tokens 1; defaults to 32 token vectors per layer pass.

This is a synthetic PyTorch dispatch probe, not the paper's kernel or a Qwen
inference run. Random weights cannot assess accuracy. Cutoff zero runs top-1
for every token; other cutoffs skip low-confidence tokens before executing
the expert. All conditions include the base, gate and dispatch in the board
energy window. torch.where on CUDA can synchronize the host, so do not
extrapolate this implementation's overhead to fused kernels.

For a model experiment, integrate the actual trained gate and frozen specialist
deltas at the projections where the baseline applies them. Gate confidence is a
heuristic, not a calibrated estimate that skipping preserves accuracy. Compare
cutoff zero and skip with the same gate, weights, workload and backend; select
the lowest-energy cutoff within the accuracy margin on validation, then freeze
and compare on fresh held-out requests, including a separately measured 8B
reference. Record skip fraction, answer length, accuracy, joules and latency.
