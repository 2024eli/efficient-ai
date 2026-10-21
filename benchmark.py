"""GSM8K paired size/shot baseline. CUDA, batch=1, BF16, greedy decoding."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import statistics
import subprocess
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path


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


def messages(question, examples):
    result = [{'role': 'system', 'content': 'Solve the math problem. Show a concise solution, then end with a separate line FINAL: <number>.'}]
    for ex in examples:
        reasoning, final = ex['answer'].split('####')
        result.extend([{'role': 'user', 'content': ex['question']},
                       {'role': 'assistant', 'content': reasoning.strip() + '\nFINAL: ' + final.strip()}])
    result.append({'role': 'user', 'content': question})
    return result


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B'])
    p.add_argument('--shots', nargs='+', type=int, default=[0, 2, 4])
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.n < 1 or a.repeats < 1 or a.max_new_tokens < 1 or any(s < 0 for s in a.shots):
        p.error('n, repeats, token budget must be positive; shots must be nonnegative')
    import torch
    import pynvml as nvml
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. Run this through Slurm on a GPU node.')
    torch.manual_seed(a.seed)
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))))
    nvml.nvmlInit()
    handle, uuid = gpu_handle(torch, nvml)
    data = load_dataset('openai/gsm8k', 'main')
    if a.n > len(data['test']) or max(a.shots) > len(data['train']):
        p.error('Requested more questions/examples than available')
    rng = random.Random(a.seed)
    test_ids = rng.sample(range(len(data['test'])), a.n)
    example_ids = rng.sample(range(len(data['train'])), max(a.shots))
    examples = [data['train'][i] for i in example_ids]
    manifest = {'args': vars(a), 'test_ids': test_ids, 'example_ids': example_ids,
                'dataset_fingerprints': {k: v._fingerprint for k, v in data.items()},
                'gpu_uuid': uuid, 'gpu_name': torch.cuda.get_device_name(0),
                'python': platform.python_version(), 'cuda': torch.version.cuda,
                'packages': {name: importlib.metadata.version(name) for name in
                             ['torch', 'transformers', 'datasets', 'accelerate', 'nvidia-ml-py']},
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'model_revisions': {}}
    manifest_path = out / 'manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2))
    save_manifest()
    (out / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
    records = []
    with (out / 'predictions.jsonl').open('w', buffering=1) as f:
        # Reverse model order across repeats to reduce systematic order bias.
        for repeat in range(a.repeats):
            for model_id in (a.models if repeat % 2 == 0 else list(reversed(a.models))):
                print(f'Loading {model_id}, repeat {repeat}', flush=True)
                start = time.perf_counter()
                pinned_revision = manifest['model_revisions'].get(model_id)
                tok = AutoTokenizer.from_pretrained(model_id, revision=pinned_revision)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=pinned_revision, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa').to('cuda:0').eval()
                torch.cuda.synchronize()
                load_s = time.perf_counter() - start
                manifest['model_revisions'][model_id] = model.config._commit_hash
                save_manifest()
                def prepare(question, shots):
                    prompt = tok.apply_chat_template(messages(question, examples[:shots]), tokenize=False,
                            add_generation_prompt=True, enable_thinking=a.thinking)
                    return tok(prompt, return_tensors='pt').to('cuda:0'), prompt
                def generate(inputs):
                    with torch.inference_mode():
                        return model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                            do_sample=False, pad_token_id=tok.eos_token_id)
                order = [(idx, shots) for idx in test_ids for shots in a.shots]
                random.Random(a.seed + repeat).shuffle(order)
                # Warm up each prompt-length condition on a training question, not a test question.
                for shots in a.shots:
                    inputs, _ = prepare(data['train'][-1]['question'], shots)
                    warm = generate(inputs)
                    del warm, inputs
                torch.cuda.synchronize()
                for idx, shots in order:
                    item = data['test'][idx]
                    inputs, prompt = prepare(item['question'], shots)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    energy0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    start = time.perf_counter()
                    tokens = generate(inputs)
                    torch.cuda.synchronize()
                    latency = time.perf_counter() - start
                    energy1 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    generated = tokens[0, inputs['input_ids'].shape[1]:].tolist()
                    text = tok.decode(generated, skip_special_tokens=True)
                    pred, target = answer(text), answer(item['answer'], gold=True)
                    row = {'model': model_id, 'repeat': repeat, 'shots': shots,
                        'test_id': idx, 'correct': pred is not None and pred == target,
                        'prediction': pred, 'target': target, 'parse_failure': pred is None,
                        'generation_seconds': latency, 'gpu_joules': (energy1-energy0)/1000,
                        'input_tokens': inputs['input_ids'].shape[1], 'output_tokens': len(generated),
                        'hit_token_limit': len(generated) >= a.max_new_tokens,
                        'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9,
                        'model_load_seconds': load_s, 'thinking': a.thinking,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'response': text}
                    if row['gpu_joules'] <= 0:
                        raise RuntimeError('Energy counter did not advance; invalid measurement.')
                    f.write(json.dumps(row) + '\n')
                    records.append(row)
                    print(f'{model_id} rep={repeat} shots={shots} id={idx} correct={row["correct"]} latency={latency:.2f}s', flush=True)
                    del tokens, inputs
                del model, tok
                gc.collect()
                torch.cuda.empty_cache()
    summarize(records, out)
    nvml.nvmlShutdown()


def summarize(records, out):
    summary = []
    for model, shots in sorted({(r['model'], r['shots']) for r in records}):
        group = [r for r in records if r['model'] == model and r['shots'] == shots]
        seconds = sorted(r['generation_seconds'] for r in group)
        total_correct = sum(r['correct'] for r in group)
        total_j = sum(r['gpu_joules'] for r in group)
        summary.append({'model': model, 'shots': shots, 'observations': len(group),
            'unique_questions': len({r['test_id'] for r in group}),
            'accuracy': total_correct/len(group),
            'mean_seconds': statistics.mean(seconds), 'median_seconds': statistics.median(seconds),
            'p95_seconds': seconds[max(0, __import__('math').ceil(.95*len(seconds))-1)],
            'mean_gpu_joules': total_j/len(group),
            'gpu_joules_per_correct': total_j/total_correct if total_correct else None,
            'parse_failures': sum(r['parse_failure'] for r in group),
            'token_limit_hits': sum(r['hit_token_limit'] for r in group),
            'mean_input_tokens': statistics.mean(r['input_tokens'] for r in group),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in group)})
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
