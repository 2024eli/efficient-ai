# Experiment 2 — does in-context learning justify its energy cost?

Prepared for Betty / eli22. This package is ready to transfer; it has NOT executed on
Betty and contains NO experimental findings. Local tests use synthetic measurements
only to verify arithmetic and file handling. Do not present those fixtures as results.

## Design

Question: do worked examples improve Qwen3-1.7B enough to justify their total GPU
energy, and can a few-shot 1.7B approach a freshly measured zero-shot Qwen3-8B?

- Small model: Qwen3-1.7B, 0/2/4/8/16 worked examples; no fine-tuning.
- Reference: Qwen3-8B, 0 shots, rerun on IDENTICAL questions and settings. This is a
  comparison against zero-shot 8B, not the best possible prompted 8B.
- GSM8K: 40 validation + 200 test questions. Validation selects a shot count; test
  estimates its performance. These are custom subsets, not full benchmark results.
- Demonstrations come ONLY from GSM8K train. Three disjoint random sets of 16; use
  nested prefixes for 2/4/8/16 within each set. Both question splits use all 3 sets.
- 0 shots and 8B are remeasured in each of the three rounds, giving timing repeats
  without pretending deterministic answers are independent accuracy observations.
- Total full collection: 240 questions x (5 small conditions + 1 large) x 3 rounds
  = 4,320 requests. Default smoke: 3 VALIDATION questions x 6 conditions = 18 requests.
- BF16, SDPA, batch=1, greedy (explicit call overrides), thinking OFF, common 768
  max generated tokens. Examples contain worked solutions; effect includes reasoning
  and formatting instruction, not just additional input length.
- Request order is shuffled within model/round. Model order alternates by round.
  Every shot condition gets a synthetic warm-up. No cross-request KV/prefix cache
  is reused; GPU weights remain resident within each model block. No prompt truncation.
- The default total input+output limit is 16,384 tokens (and no larger than the model
  limit). Exceeding it fails visibly, rather than silently removing examples.

## Transfer safely

Unzip and drag the WHOLE icl-experiment folder into your remote efficient-ai folder.
Keep this experiment in its own folder: analyze.py and requirements.txt names also
exist in the routing experiment. Do not overwrite those older files.

Expected remote location:
/vast/home/e/eli22/efficient-ai/icl-experiment

```bash
cd /vast/home/e/eli22/efficient-ai/icl-experiment
```

Check quota (models/environment may already be cached):

```bash
parcc_quota.py
```

To avoid reusing previously inspected routing questions, set EXCLUDE_DATA to that
study's saved data.json or screen-data.json when preparing. For example, if this exists:

```bash
EXCLUDE_DATA=/vast/home/e/eli22/efficient-ai/screen-results/collect-9251129/data.json bash setup.sh
```

Or if you kept only the original input:

```bash
EXCLUDE_DATA=/vast/home/e/eli22/efficient-ai/screen-data.json bash setup.sh
```

Use only a path that actually exists. If neither is available:

```bash
bash setup.sh
```

Without exclusions the experiment is internally held out from demonstrations, but
not guaranteed fresh relative to earlier experiments. That fact is recorded in the
plan. This does not rule out model pretraining contamination or semantic near-duplicates.

Setup loads anaconda3 when available, prefers ../.venv, installs pinned compatible
packages (including NumPy 1.26.4 and PyArrow 17), tests imports and CPU logic, downloads
models/dataset and freezes plan.json/model revisions. It records its venv in .venv-path.
Override with ICL_VENV=/absolute/path/to/venv if desired. It never installs into shared
conda base. Setup can change packages in your existing private venv; select a new
ICL_VENV if isolation matters. Respect Betty policy for installation/download work;
use a CPU allocation if required. No model inference runs during setup.

For a new venv, torch 2.7.0 CUDA 12.8 is installed from the official CUDA wheel index.
Model and dataset downloads reuse HF_HOME/default cache. Export the SAME HF_HOME
when preparing and submitting if you change it. Jobs use offline mode.

## Smoke first, then submit

Submit FROM the experiment directory (run.sbatch uses SLURM_SUBMIT_DIR):

```bash
sbatch --export=ALL,MODE=smoke run.sbatch
```

After completion, inspect icl2-JOBID.log. It should show 18 requests, completion,
and an audit with parse/truncation rates. Raw outputs are retained. Debug prompt/
format problems on validation examples only; preserve the final test set.

The job requests your previously validated allocation: account
leebcc-systems-architecture-int, dgx-b200, dgx QOS, 1 full B200, 28 CPU threads,
224 GB RAM. Cluster policies may change; current submission acceptance is the check.

Then:

```bash
sbatch --export=ALL,MODE=full run.sbatch
```

The job automatically analyzes a complete run. Six hours is a cap, not a promise.
Estimate runtime from the smoke and allow for prompt-length effects and model reloads.
All 4,320 requests plus load/warm-up work consume real GPU allocation time. Logs and
JSONL flush during execution; partial results survive interruption. Analysis refuses
missing/duplicate conditions. No automatic resume and no automatic full-run submission.
Slurm jobs survive VS Code disconnects. Cancel intentionally with scancel JOBID.

## Measurements: precise boundaries

Each request starts before chat formatting/tokenization and GPU transfer. Latency ends
after generation, CUDA synchronization, and output text decoding. GPU board energy
is the cumulative NVML counter delta spanning that interval. Excludes model loading,
warm-ups, scoring, downloads, scheduler delay, client networking and CPU/RAM energy.
Uses allocated CUDA UUID to identify NVML device; rejects MIG and invalid counters.
Includes assigned-GPU baseline power during execution. Inspect nvidia-smi.txt for
other processes. Full GPU allocation is not full-node exclusivity.

TTFT is measured at the FIRST GENERATED TOKEN-ID callback (not the prompt callback),
after a CUDA synchronize, relative to request start. It includes host input preparation,
transfer, prefill, first-token selection and callback overhead. This is offline token-ID
availability, NOT time to first visible text in a web client or vLLM server. Token-ID
copy/synchronization/streamer overhead is present in all conditions and can influence
timings. EOS can be the first generated token. Input preparation and remaining decode
latency are logged separately. These definitions differ from some serving benchmarks.

We do NOT claim to separately measure prefill energy and decode energy. TTFT and token
lengths help explain total costs, but don't isolate phase energy. Short-interval NVML
updates can add noise. A follow-up fixed-output-length/prefill-only probe would help
attribute computation cost, separate from end-to-end answer quality.

Scoring: exact decimal equivalence against the GSM8K #### answer. Only explicitly
marked FINAL answers are parsed; allow case differences, inline FINAL, literal
<answer> prefix and currency $. Missing/ambiguous answers count as wrong and are
flagged; no arbitrary last-number fallback. strict_parse_failure preserves the more
restrictive final-line diagnostic. Infinite values/repeating decimals are not guessed.
Every cap-truncated answer is flagged, even if its final number parses. If cap hits
are frequent, choose a larger common budget on validation and rerun all conditions.

## Analysis and figures

If rerunning analysis manually:

```bash
source "$(cat .venv-path)/bin/activate"
OPENBLAS_NUM_THREADS=1 python analyze.py results/full-JOBID
python plot.py results/full-JOBID
```

Or open results.ipynb in VS Code, select the recorded venv, edit RUN_DIR, Run All.
Plots read CSV files, not pasted numbers; export PNG/SVG/PDF under the run's figures/.
No GPU is needed for analysis or plotting. Use CPU allocation if login policy requires.

Files:
- measurements.jsonl: responses, correctness, total GPU J, total/first-token latency,
  input/output tokens, peak allocated memory, load time, example seed and prompt hash.
- manifest.json / plan.json / pip-freeze.txt / nvidia-smi.txt: provenance and frozen data.
- summary.csv: mean metrics by split/model/shots, pointwise bootstrap intervals.
- per_seed.csv: demonstration-set sensitivity; 0-shot repeats are timing replicates.
- paired_differences.csv: each small shot count versus small-0 and large-0, using the
  same resampled test question IDs. Positive accuracy delta is better; negative
  energy/latency delta is better; positive energy savings is better.
- analysis.json: shot counts selected on validation and their untouched test results.

Two predeclared selections minimize validation GPU energy while meeting accuracy:
(1) within 2 points of the BEST small-shot validation accuracy;
(2) within 2 points of zero-shot 8B validation accuracy.
No feasible small configuration gives null, NOT a claim of matching accuracy. No
shot count is picked using test performance. Tolerance is adjustable before final
evaluation; 40 validation questions are coarse for a 2-point margin.

Bootstrap: average the three rounds within each question/condition first, then
resample paired questions 2,000 times. Intervals are conditional on these 3 fixed
example sets, do not model the full distribution of possible demonstration sets,
and are pointwise (no multiple-comparison correction). Per-seed results show some
selection sensitivity; 3 seeds are not a comprehensive uncertainty estimate.
Do not treat 600 seed/question outcomes as 600 independent test questions.
For an accuracy-matching claim, examine the paired accuracy-gap LOWER confidence
bound against -0.02, not just means or overlap of separate intervals. 200 questions
may be inadequate for a strong two-point noninferiority claim. Label this a pilot.

## How to interpret the research result

Do not assume additional examples increase energy monotonically: longer prefill may
be offset by shorter/faster responses. Inspect TTFT, token lengths and total energy
alongside accuracy. Also check whether few-shot gains mainly fix formatting.
Do not claim a universal optimal shot count from this dataset/hardware/example pool.
If all small configurations miss the 8B accuracy target, that is an informative result.
No dollars, whole-system energy, vLLM concurrency, MoLoRA, adapters or learned adaptive
shot policy are evaluated. This is the measurement baseline before any adaptive ICL.

References:
https://huggingface.co/Qwen/Qwen3-1.7B
https://huggingface.co/datasets/openai/gsm8k
https://huggingface.co/docs/transformers/internal/generation_utils
