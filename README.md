# efficient-ai

Download requirements.txt to make venv. 

# Prelim testing how to run 

```bash
cd /vast/home/e/eli22/efficient-ai
bash setup.sh
sbatch --export=ALL,MODE=smoke run.sbatch
squeue -u eli22
```

Setup uses a private .venv, not the shared conda environment. It installs a pinned
PyTorch 2.7.0 CUDA 12.8 wheel and dependencies, runs CPU tests, and downloads both
models and GSM8K. The install/download can take time. If Betty policy disallows this
setup workload on login nodes, use an appropriate CPU allocation; never perform
model inference there. Check quota with parcc_quota.py first. Weights plus the
CUDA environment can consume tens of GB, especially with pip/Hugging Face caches.
If necessary use approved project storage for HF_HOME and PIP_CACHE_DIR, exporting
identical HF_HOME in setup.sh and run.sbatch. No project storage path is guessed.
Do not launch several setup processes at once.

After sbatch prints the job number, inspect qwen-icl-JOBID.log. A successful smoke
run produces results/smoke-JOBID/summary.csv with six rows and 18 predictions.

```bash
tail -f qwen-icl-JOBID.log
# Ctrl-C stops tail, not the submitted job.
```

After reviewing the smoke output:

```bash
sbatch --export=ALL,MODE=full run.sbatch
```

The full run produces 3600 observations: 200 questions x 2 models x 3 shot counts
x 3 repeated measurements. Four hours is an initial cap, not a runtime promise;
estimate runtime from the smoke job and adjust before submission. Logs and partial
predictions are flushed during execution. Summary is written only on completion;
partial runs are not automatically resumed. Jobs submitted with sbatch persist
when VS Code disconnects. This kit does not automatically submit the full run.
Use scancel JOBID to cancel a job if needed.

## Design and interpretation

- GSM8K main: deterministic random 200 test questions, identical across conditions.
  Fixed nested 0/2/4 examples sampled from train. Single example seed: prompt choice
  sensitivity remains future work. Warm-up uses a separate training item.
- BF16, batch size 1, SDPA, greedy decoding, max 512 generated tokens. No quantization.
  Thinking mode is explicitly disabled for both models. --thinking enables a separate
  experiment; do not pool its results with the default. A 512-token thinking budget
  may be insufficient; inspect token-limit rates and set a suitable budget.
- Concise worked examples, identical instruction, explicit FINAL number scoring.
  Missing FINAL markers count as incorrect and are separately flagged. Gold uses
  GSM8K #### answers. Inspect responses before interpreting accuracy differences.
- Warm-up excluded. Question/shot order shuffled within model; model order reverses
  across repeats. Models are reloaded each repeat. Three deterministic repeats
  measure performance variability, not three independent accuracy samples.
- CUDA synchronized generation latency includes CPU overhead inside generate,
  GPU prefill and decoding; excludes tokenization, transfer, model loading, downloads,
  and scoring. This is generation latency, not full client/server request latency.
- GPU energy is NVML cumulative board-energy counter delta (millijoules to joules)
  around each synchronized generation. No invented values or CPU power estimates.
  CUDA UUID identifies the assigned physical GPU. Counter failure stops the run.
  Do not use on MIG/shared GPUs; confirm no other process shares the assigned GPU.
  Full-GPU Slurm allocation does not reserve the entire multi-GPU node.
- Energy includes assigned-GPU baseline power during generation. It excludes CPU,
  RAM, networking, cooling, training and model-loading energy. Short-request counter
  granularity may add noise; preliminary measurements need validation before publication.
  Other GPUs on the same node may influence thermals/clocks. Smaller models can
  underutilize a B200: conclusions apply to this hardware and batch=1 workload.
- summary.csv reports accuracy, mean/median/p95 latency, mean GPU joules/question,
  GPU joules/correct answer, parse failures, length and truncation statistics.
  Zero correct answers gives a blank joules/correct field, not zero energy.
  Dollar cost is not inferred from energy: apply the project's actual GPU billing
  rate to allocation duration separately. These are baselines, not a test of MoLoRA.
- predictions.jsonl retains responses, scores, tokens and prompt hashes.
  manifest.json records dataset fingerprints/indices, settings, package versions,
  GPU UUID and resolved model commits; pip-freeze.txt records the environment.
  First runs resolve upstream revisions; later repeats pin the resolved model commit.
  Archive cache/snapshots for exact future reproduction. Do not treat repeat-level
  accuracy as independent; statistical comparisons should pair by test question.

## If something fails

- CUDA kernel/architecture error: share the log and driver version. The B200 requires
  a compatible Blackwell build; a successful import alone is insufficient.
- GPU UUID or NVML permission/counter error: share the error. No silent replacement
  with software-estimated energy is provided.
- Download/install error: share the last 30 log lines. Compute jobs use offline mode;
  setup must download all files first, in the same Hugging Face cache location.
- High parse failure/truncation: inspect raw responses and fix format/budget before
  spending time on a full run. Raising the budget changes the experimental condition.

References:
https://huggingface.co/Qwen/Qwen3-1.7B
https://pytorch.org/get-started/previous-versions/
https://huggingface.co/datasets/openai/gsm8k
https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
https://parcc.upenn.edu/training/slurm/
Resource requests use the current filter output supplied by the user:
1 B200, 28 CPU threads, 224 GB RAM, dgx QOS, leebcc-systems-architecture-int account.

# Week 1 Test

```bash
cd /vast/home/e/eli22/efficient-ai/routing-screen
bash prepare.sh
source ../.venv/bin/activate
python -m unittest -v test_screen.py
sbatch --export=ALL,MODE=smoke screen.sbatch
```

The smoke job measures 9 questions x 2 models = 18 requests: one per domain/split.
Check routing-screen-JOBID.log (in the directory where you submitted sbatch).

```bash
python audit.py screen-results/smoke-JOBID
```

Review format failures and truncation BEFORE the main collection. In particular,
the original 1.7B zero-shot smoke failed formatting. This kit clarifies the final-line
instruction and increases the shared limit to 768 tokens, but this has not yet been
validated on Betty. Do not interpret parse failures as mathematical errors.
If prompting/scoring changes, rerun both models with the same frozen protocol.
Never tune the prompt on final-test answers; use smoke TRAIN examples for changes,
then rebuild fresh held-out evaluation data if final-test outputs influenced changes.

When satisfied:

```bash
sbatch --export=ALL,MODE=collect screen.sbatch
```

After it completes:

```bash
source ../.venv/bin/activate
python audit.py screen-results/collect-JOBID
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze.py screen-results/collect-JOBID
```

analyze.py is a short CPU-only analysis, not GPU inference. Follow local policy
about login-node work; use a CPU allocation if required. sbatch jobs survive logout.
Four hours is a cap, not a runtime guarantee. The earlier smoke took ~1–3 seconds
for most responses: use the NEW smoke to estimate 1500 requests plus loading.
The analysis refuses partial runs. Partial JSONL is preserved; no auto-resume.
Do not prepare twice over the same screen-data.json: it refuses to overwrite the
frozen question set. Archive/rename it intentionally before preparing a new set.

## Fixed experiment

- Domains: GSM8K numeric math; ARC-Challenge science multiple choice; BBH boolean
  expressions symbolic logic. Boolean expressions are narrow and can saturate; they
  are a cheap diagnostic, not a broad reasoning claim.
- 250 unique questions/domain, split 100 router-train / 50 validation / 100 test.
  Total 750 paired questions, 1500 generations. GSM8K/ARC use their test source pool;
  BBH calls its sole source split train. These names do NOT define the router split:
  we freeze a new disjoint partition, deduplicating normalized question text first.
  This is a custom subset study, not a standard leaderboard evaluation. Pretraining
  contamination and semantic near-duplicates are not ruled out.
- No LLM weights are trained. Router training uses labeled base-model outcomes.
  The final 300 questions are held out from vocabulary fitting, router training,
  domain-policy selection and threshold selection. Stratification is by domain.
- Zero-shot, thinking OFF, BF16, SDPA, greedy, batch=1, common 768-token cap.
  No examples means this directly screens base-model routing, separate from the
  original shot study. Greedy generation has a fresh GenerationConfig to avoid
  inherited sampling warnings. Model revisions/data fingerprints are saved.
- Router: TF-IDF word unigrams/bigrams + regularized logistic regression, predicting
  small-model correctness from question text only. No reference answer, response,
  energy, test labels or model confidence is included in the features. Hyperparameters
  fixed in advance. A constant outcome training set produces a disclosed constant
  router rather than a crash or misleading classifier.
- Threshold: minimize validation GPU energy subject to accuracy >= validation 8B
  accuracy minus 2 percentage points. Change --tolerance BEFORE looking at test
  results. Always-large is available as a feasible fallback. This is an empirical
  validation rule, NOT a statistical guarantee that test accuracy is matched.
- Baselines: always-small, always-large, validation-selected domain-only policy,
  random routing at the learned policy's expected small fraction, and oracle.
  Oracle sees correctness/cost on test: maximize correctness, choose lower-energy
  model on ties. It is a hindsight ceiling, never a deployable baseline. Neither
  model may be correct, and oracle may outperform always-large in accuracy.

## What measurements mean

Both models are measured separately on each request, then replay chooses one row.
GPU UUID maps CUDA's allocated device to NVML; cumulative board energy is read
before and after tokenization, input transfer, generation and output decoding.
Synchronized latency includes those same request stages. It excludes router inference,
loading, downloads, answer scoring, and client/network/queue latency. Model loading
seconds are logged separately; loading energy is not measured. Warm-ups excluded.
This differs from the old kit's generation-only latency: do not pool the two.

Energy is GPU board energy only, including its baseline power during the interval.
No CPU/RAM/cooling, training or dollar cost is inferred. Short-interval NVML updates
and node thermal effects can add noise. Use the full assigned B200, not a MIG slice;
confirm no unrelated process shares that GPU. B200 batch=1 conclusions need not
transfer to smaller hardware or batched service. One default measurement pass is a
screening run; add --repeats 3 to collect.py for a follow-up (increase walltime).
Model order reverses by repeat, question order shuffles, and repeats are averaged
within question before router analysis. A single pass cannot estimate timing drift.

Router CPU latency is measured for individual requests (including text vectorization)
in three warmed passes with numerical thread pools limited to one thread. GPU runs
are sequential; this CPU timing need not run on identical hardware. CPU energy is
NOT available, so it is neither zero nor added to GPU joules. The energy headroom
column tells how much additional routing/service energy could erase replay savings.
Learned-route latency can be approximated by seconds + router_cpu_seconds_per_request,
but that remains a replay estimate, not measured service latency.

All routing energy/latency totals are SIMULATIONS using measured component requests.
They exclude simultaneous residency/idle costs of two GPUs, model loading/switching,
scheduling, contention, adapter switching and CPU router energy. Replaying from
separately loaded models does not demonstrate an efficient live serving system.
If promising, validate the chosen policy in a real serving setup next.

## Reports and decision

routing-summary.csv: six policies x balanced + three 80/10/10 domain-skew workloads.
It includes accuracy gap vs 8B, GPU joules/request and per correct answer, savings,
latency and parse/truncation rates. Weights change task proportions, not request
order. Grouped-versus-switching traffic cannot be answered by this replay study.

routing-report.json: frozen validation threshold, per-domain diagnostics, validation
curve, exploratory test curve, CPU latency, 95% paired domain-stratified bootstrap
intervals for balanced-test accuracy gap and energy savings. These intervals sample
questions conditional on the fitted router; they do not capture training-set choice,
measurement noise or new-domain shifts. Test curve is descriptive; choosing a new
threshold from it invalidates the held-out evaluation. Use new data for retuning.

test-routes.jsonl: deployed-policy choices using question-only scores.
measurements.jsonl: raw outputs, strict correctness, parse/truncation flags and costs.
router-parameters.json: coefficients/vocabulary for inspection (not a deployment loader).
manifest.json, data.json and pip-freeze.txt: settings, resolved commits, frozen pools.

Interpretation: first audit format errors. Then examine oracle energy/accuracy headroom,
actual learned-router savings, and whether domain-only routing is already competitive.
Do not declare comparable accuracy merely because a confidence interval overlaps zero.
For a strong noninferiority claim, the lower accuracy-gap bound must exceed the
predeclared negative tolerance, with a suitable sample-size plan. 100 test questions
per domain is preliminary and may be insufficient for a two-point claim.

Sources:
https://huggingface.co/datasets/openai/gsm8k
https://huggingface.co/datasets/allenai/ai2_arc
https://huggingface.co/datasets/maveriq/bigbenchhard
https://huggingface.co/Qwen/Qwen3-1.7B

Local checks run on a Mac, not Betty. Dependency versions used for those checks
may differ from the pinned Python 3.11 Betty environment; run test_screen.py remotely.
No remote changes, installations or submissions were performed by this chat.
