"""Experiment 1: GPU energy overhead vs accuracy benefit of one math LoRA on Qwen3-1.7B.

One PeftModel stays resident; the base condition runs inside disable_adapter(), so base and
LoRA share identical weights, prompt, tokenizer and generation settings. Phases:
  natural  generate to EOS under a common cap (end-to-end energy + accuracy)
  fixed    force exactly --fixed-tokens new tokens (isolates per-token adapter overhead)
  forward  repeated prefill-only passes on identical token sequences (compute overhead)
Qwen3-8B is then loaded alone and measured on the same questions (natural + fixed).
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from measurement_utils import answer, gpu_handle

BASE = 'Qwen/Qwen3-1.7B'
LARGE = 'Qwen/Qwen3-8B'
ADAPTER = 'ehzawad/qwen3-1.7b-gsm8k-grpo'
ADAPTER_REVISION = '83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8'
# The adapter card's inference prompt, used for EVERY condition so only the weights differ.
SYSTEM = ('You are a careful math reasoning assistant. Solve the problem step by step, but keep '
          'the solution concise. End with exactly one final answer in the form \\boxed{answer}.')
NUMBER = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def boxed_answer(text):
    """Numeric content of the LAST \\boxed{...}; None if absent or not a plain number."""
    if '<think>' in text and '</think>' not in text:
        return None
    # With thinking off the empty think block is in the prompt; the GRPO adapter still emits a
    # stray '</think>' mid-answer, so only strip reasoning when the output opened a think block.
    if '<think>' in text:
        text = text.split('</think>')[-1]
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 1, start + len('\\boxed{')
    while i < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        i += 1
    if depth:
        return None
    value = text[start + len('\\boxed{'):i - 1]
    # Tolerate unit words and currency/percent/thin-space markup; never guess between numbers.
    value = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '', value)
    value = re.sub(r'\\[$%,!;]|[$%]', '', value).strip().rstrip('.').strip()
    if not re.fullmatch(NUMBER, value):
        return None
    return str(Decimal(value.replace(',', '')).normalize())


def lenient_answer(text):
    """Secondary score only: \\boxed{} if parseable, else the last number on the last line."""
    strict = boxed_answer(text)
    if strict is not None or ('<think>' in text and '</think>' not in text):
        return strict
    lines = [line for line in text.split('</think>')[-1].splitlines() if line.strip()]
    numbers = re.findall(NUMBER, lines[-1]) if lines else []
    return str(Decimal(numbers[-1].replace(',', '')).normalize()) if numbers else None


def load_questions(n, seed):
    """Deterministic GSM8K-test subset; a smaller n is a prefix of a larger one."""
    from datasets import load_dataset
    data = load_dataset('openai/gsm8k', 'main')
    norm = lambda q: ' '.join(q.lower().split())
    train = {norm(r['question']) for r in data['train']}
    # Exact-text overlap with GSM8K train only. This cannot rule out the adapter having seen
    # GSM8K test items: its card does not state which split it was trained on.
    pool = [i for i, r in enumerate(data['test']) if norm(r['question']) not in train]
    random.Random(seed).shuffle(pool)
    if n > len(pool):
        raise ValueError(f'Requested {n} questions, only {len(pool)} available')
    items = []
    for i in pool[:n]:
        row = data['test'][i]
        items.append(dict(id=i, question=row['question'], target=answer(row['answer'], gold=True)))
        assert items[-1]['target'] is not None
    warm = [data['train'][i]['question'] for i in range(3)]
    info = dict(dataset_fingerprints={k: v._fingerprint for k, v in data.items()},
                test_ids=[r['id'] for r in items], train_overlap_excluded=len(data['test']) - len(pool))
    return items, warm, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=150)
    p.add_argument('--repeats', type=int, default=2, help='measurement repeats (noise estimate)')
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--thinking', action='store_true',
                   help='thinking mode for EVERY condition (the adapter appears to be trained this way)')
    p.add_argument('--fixed-tokens', type=int, default=256)
    p.add_argument('--fixed-n', type=int, default=50, help='questions in the fixed-length phase')
    p.add_argument('--forward-n', type=int, default=30, help='questions in the forward-pass phase')
    p.add_argument('--forward-passes', type=int, default=20, help='passes per forward measurement')
    p.add_argument('--merged', action='store_true',
                   help='also measure the adapter merged into a second copy of the base weights')
    p.add_argument('--skip-8b', action='store_true')
    p.add_argument('--adapter', default=ADAPTER)
    p.add_argument('--adapter-revision', default=ADAPTER_REVISION)
    p.add_argument('--idle-seconds', type=float, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.n, a.repeats, a.max_new_tokens, a.fixed_tokens, a.forward_passes) < 1:
        p.error('counts and token budgets must be positive')
    if a.fixed_n > a.n or a.forward_n > a.fixed_n:
        p.error('need forward-n <= fixed-n <= n (forward reuses fixed-phase sequences)')
    import torch
    import pynvml as nvml
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. Run this through Slurm on a GPU node.')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    nvml.nvmlInit()
    handle, uuid = gpu_handle(torch, nvml)
    items, warm, info = load_questions(a.n, a.seed)
    # Resolve the pinned adapter snapshot locally so the recorded commit is what gets loaded.
    adapter_path = snapshot_download(a.adapter, revision=a.adapter_revision)
    manifest = dict(args=vars(a), system_prompt=SYSTEM, gpu_uuid=uuid,
        gpu_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        job=os.environ.get('SLURM_JOB_ID'), adapter=a.adapter, adapter_commit=Path(adapter_path).name,
        adapter_config=json.loads((Path(adapter_path) / 'adapter_config.json').read_text()),
        model_commits={}, packages={n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'peft', 'datasets', 'nvidia-ml-py']}, **info)
    (out / 'questions.json').write_text(json.dumps(items, indent=2))
    (out / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()

    def measure(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        energy0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        joules = (nvml.nvmlDeviceGetTotalEnergyConsumption(handle) - energy0) / 1000
        if joules <= 0:
            raise RuntimeError('Energy counter did not advance; invalid measurement.')
        return result, seconds, joules, torch.cuda.max_memory_allocated() / 1e9

    # Idle board power over the same interval type; lets analysis report above-idle energy.
    _, idle_s, idle_j, _ = measure(lambda: time.sleep(a.idle_seconds))
    manifest['idle_watts_start'] = idle_j / idle_s
    save()

    def load(model_id):
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda:0').eval()
        torch.cuda.synchronize()
        manifest['model_commits'][model_id] = model.config._commit_hash
        manifest.setdefault('load_seconds', {})[model_id] = time.perf_counter() - start
        save()
        return model

    tok = AutoTokenizer.from_pretrained(BASE)
    def encode(question):
        text = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=a.thinking)
        return tok(text, return_tensors='pt').to('cuda:0')

    def config_for(model, fixed):
        tokens = a.fixed_tokens if fixed else a.max_new_tokens
        # min_new_tokens == max_new_tokens suppresses EOS: identical decode work per condition.
        return GenerationConfig(max_new_tokens=tokens, min_new_tokens=tokens if fixed else 0,
            do_sample=False, use_cache=True, eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tok.eos_token_id)

    f = (out / 'measurements.jsonl').open('w', buffering=1)
    fixed_sequences = {}

    def generate_row(phase, name, model, ctx, item, repeat, position):
        cfg = config_for(model, phase == 'fixed')
        def run():
            inputs = encode(item['question'])
            with torch.inference_mode(), ctx():
                tokens = model.generate(**inputs, generation_config=cfg)
            return inputs, tokens
        (inputs, tokens), seconds, joules, peak = measure(run)
        n_in = inputs['input_ids'].shape[1]
        generated = tokens[0, n_in:]
        text = tok.decode(generated, skip_special_tokens=True)
        n_out = len(generated)
        if phase == 'fixed':
            if n_out != a.fixed_tokens:
                raise RuntimeError(f'Fixed-length generation produced {n_out} tokens')
            if name == 'base' and repeat == 0:
                fixed_sequences[item['id']] = tokens.detach().clone()
        prediction = boxed_answer(text) if phase == 'natural' else None
        row = dict(phase=phase, condition=name, repeat=repeat, id=item['id'], position=position,
            correct=prediction is not None and prediction == item['target'] if phase == 'natural' else None,
            prediction=prediction, target=item['target'],
            parse_failure=prediction is None if phase == 'natural' else None,
            hit_token_limit=phase == 'natural' and n_out >= a.max_new_tokens,
            seconds=seconds, gpu_joules=joules, input_tokens=n_in, output_tokens=n_out,
            peak_allocated_gb=peak, response_sha256=hashlib.sha256(text.encode()).hexdigest(),
            response=text if phase == 'natural' else None)
        f.write(json.dumps(row) + '\n')
        print(f'{phase} {name} rep={repeat} id={item["id"]} correct={row["correct"]} '
              f'tokens={n_out} {seconds:.2f}s {joules:.1f}J', flush=True)

    def forward_row(name, model, ctx, item, repeat, position):
        ids = fixed_sequences[item['id']]
        def run():
            with torch.inference_mode(), ctx():
                for _ in range(a.forward_passes):
                    model(input_ids=ids, use_cache=False)
        _, seconds, joules, peak = measure(run)
        row = dict(phase='forward', condition=name, repeat=repeat, id=item['id'], position=position,
            passes=a.forward_passes, input_tokens=ids.shape[1], seconds=seconds, gpu_joules=joules,
            peak_allocated_gb=peak)
        f.write(json.dumps(row) + '\n')
        print(f'forward {name} rep={repeat} id={item["id"]} {seconds:.2f}s {joules:.1f}J', flush=True)

    def run_phases(conditions, phases):
        # Warm up every condition (both phase shapes) on TRAIN questions; excluded from results.
        for name, (model, ctx) in conditions.items():
            for q in warm:
                with torch.inference_mode(), ctx():
                    model.generate(**encode(q), generation_config=config_for(model, False))
                    model.generate(**encode(q), generation_config=config_for(model, True))
        subsets = dict(natural=items, fixed=items[:a.fixed_n], forward=items[:a.forward_n])
        for phase in phases:
            for repeat in range(a.repeats):
                rng = random.Random(f'{a.seed}-{phase}-{repeat}')
                order = subsets[phase].copy()
                rng.shuffle(order)
                for item in order:
                    # Conditions run back to back per question in random order: drift cancels.
                    names = list(conditions)
                    rng.shuffle(names)
                    for position, name in enumerate(names):
                        model, ctx = conditions[name]
                        if phase == 'forward':
                            forward_row(name, model, ctx, item, repeat, position)
                        else:
                            generate_row(phase, name, model, ctx, item, repeat, position)

    base = load(BASE)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to('cuda:0').eval()
    conditions = dict(base=(model, model.disable_adapter), lora=(model, contextlib.nullcontext))
    merged = None
    if a.merged:
        merged = PeftModel.from_pretrained(load(BASE), adapter_path).merge_and_unload().to('cuda:0').eval()
        conditions['lora_merged'] = (merged, contextlib.nullcontext)
    # Sanity: the adapter must actually change the logits, and disable_adapter must remove it.
    with torch.inference_mode():
        probe = encode(warm[0])
        with model.disable_adapter():
            base_logits = model(**probe).logits.float()
        lora_logits = model(**probe).logits.float()
        checks = dict(lora_vs_base_max_abs_logit_diff=(lora_logits - base_logits).abs().max().item())
        if merged is not None:
            checks['merged_vs_lora_max_abs_logit_diff'] = (merged(**probe).logits.float() - lora_logits).abs().max().item()
    if checks['lora_vs_base_max_abs_logit_diff'] == 0:
        raise RuntimeError('Adapter has no effect on logits; check loading.')
    manifest['adapter_checks'] = checks
    manifest['adapter_parameters'] = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    save()
    run_phases(conditions, ['natural', 'fixed', 'forward'])
    del model, base, merged, conditions
    gc.collect()
    torch.cuda.empty_cache()

    if not a.skip_8b:
        large = load(LARGE)
        run_phases({'8b': (large, contextlib.nullcontext)}, ['natural', 'fixed'])
        del large
        gc.collect()
        torch.cuda.empty_cache()

    _, idle_s, idle_j, _ = measure(lambda: time.sleep(a.idle_seconds))
    manifest['idle_watts_end'] = idle_j / idle_s
    manifest['complete'] = True
    save()
    f.close()
    nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
