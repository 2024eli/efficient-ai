"""Measure both models on all router splits. No routing decision uses test labels."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from measurement_utils import answer, gpu_handle

MODELS = ['Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B']


def parse_strict(text, item):
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.split('</think>')[-1]
    if item['domain'] == 'math':
        return answer(text)
    matches = re.findall(r'(?im)^\s*FINAL:\s*([A-Za-z0-9]+)\s*\.?\s*$', text)
    value = matches[-1] if matches else None
    if item['domain'] == 'logic':
        value = value.lower() if value else None
    elif value:
        value = value.upper()
    return value if value in item['allowed'] else None


def parse(text, item):
    # Normalize only an explicit FINAL line; never guess from reasoning numbers.
    text = re.sub(r'(?im)^(\s*FINAL:\s*)<answer>\s*', r'\1', text)
    if item['domain'] == 'science':
        # Accept "FINAL: B. explanation"; label must still be an allowed option.
        text = re.sub(r'(?im)^(\s*FINAL:\s*[A-Za-z0-9]+)\.\s+[^\n]+$', r'\1', text)
    return parse_strict(text, item)


def prompt_for(item):
    kind = {'math': 'a number without units', 'science': 'one option label',
            'logic': 'True or False'}[item['domain']]
    return [{'role': 'system', 'content': 'Solve the question with a concise explanation. '
             f'End with a separate line in exactly this format: FINAL: <answer>, where <answer> is {kind}. '
             'Replace <answer> with your answer and omit the angle brackets.'},
            {'role': 'user', 'content': item['question']}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='screen-data.json')
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('positive repeats required')
    import torch
    import pynvml as nvml
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    raw_data = Path(a.data).read_bytes()
    data = json.loads(raw_data)
    items = data['items']
    if a.smoke:
        items = [next(r for r in items if r['domain']==d and r['split']==s)
                 for d in ['math','science','logic'] for s in ['train','validation','test']]
    torch.set_num_threads(8)
    torch.manual_seed(42)
    nvml.nvmlInit()
    handle, uuid = gpu_handle(torch, nvml)
    manifest = dict(data_sha256=hashlib.sha256(raw_data).hexdigest(), args=vars(a),
        gpu_uuid=uuid, gpu_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        job=os.environ.get('SLURM_JOB_ID'), model_commits={},
        packages={n: importlib.metadata.version(n) for n in ['torch','transformers','datasets','nvidia-ml-py']})
    (out/'data.json').write_bytes(raw_data)
    (out/'pip-freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save()
    with (out/'measurements.jsonl').open('w',buffering=1) as f:
        for repeat in range(a.repeats):
            for model_id in (MODELS if repeat%2==0 else MODELS[::-1]):
                revision = manifest['model_commits'].get(model_id)
                tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
                start = time.perf_counter()
                model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
                torch.cuda.synchronize()
                load_seconds = time.perf_counter()-start
                manifest['model_commits'][model_id] = model.config._commit_hash
                save()
                config = GenerationConfig(max_new_tokens=768, do_sample=False, use_cache=True,
                    eos_token_id=model.generation_config.eos_token_id, pad_token_id=tok.eos_token_id)
                def encode(messages):
                    text = tok.apply_chat_template(messages,tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
                    return tok(text,return_tensors='pt').to('cuda:0')
                def generate(inputs):
                    with torch.inference_mode():
                        return model.generate(**inputs, generation_config=config,
                            do_sample=False, temperature=1.0, top_p=1.0, top_k=0)
                # Synthetic warm-up, excluded from measurements and router features.
                for q in ['What is 7 plus 8?', 'Evaluate True and False.', 'Which is a gas? A. oxygen B. iron']:
                    inputs = encode([{'role':'user','content':q}])
                    tokens = generate(inputs)
                    del tokens, inputs
                order = items.copy()
                random.Random(42+repeat).shuffle(order)
                for item in order:
                    torch.cuda.synchronize()
                    energy0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    start = time.perf_counter()
                    inputs = encode(prompt_for(item))
                    torch.cuda.reset_peak_memory_stats()
                    tokens = generate(inputs)
                    torch.cuda.synchronize()
                    text = tok.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                    seconds = time.perf_counter()-start
                    energy = (nvml.nvmlDeviceGetTotalEnergyConsumption(handle)-energy0)/1000
                    if energy <= 0:
                        raise RuntimeError('Invalid energy counter measurement')
                    prediction = parse(text,item)
                    output_tokens = tokens.shape[1]-inputs['input_ids'].shape[1]
                    row = dict(id=item['id'],domain=item['domain'],split=item['split'],model=model_id,
                        repeat=repeat,correct=prediction==item['target'],prediction=prediction,
                        parse_failure=prediction is None,strict_parse_failure=parse_strict(text,item) is None,hit_token_limit=output_tokens>=768,
                        seconds=seconds,gpu_joules=energy,input_tokens=inputs['input_ids'].shape[1],
                        output_tokens=output_tokens,load_seconds=load_seconds,response=text)
                    f.write(json.dumps(row)+'\n')
                    print(f'{model_id} rep={repeat} {item["domain"]}/{item["split"]} correct={row["correct"]} {seconds:.2f}s',flush=True)
                    del inputs,tokens
                del model,tok
                gc.collect()
                torch.cuda.empty_cache()
    nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
