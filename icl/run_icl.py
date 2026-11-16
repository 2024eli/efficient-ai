"""Paired GSM8K ICL experiment; measured requests, not routing replay."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from common import SMALL, LARGE, SHOTS, messages, parse, FirstTokenTimer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, default=Path('plan.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--context-limit', type=int, default=16384)
    a = p.parse_args()
    if min(a.max_new_tokens,a.context_limit) < 1:
        p.error('token limits must be positive')
    import torch
    import pynvml as nvml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    raw = a.plan.read_bytes()
    plan = json.loads(raw)
    a.output.mkdir(parents=True, exist_ok=False)
    items = plan['items']
    # Smoke uses validation questions only; no final-test output is used for prompt debugging.
    if a.smoke:
        items = [r for r in items if r['split']=='validation'][:3]
    sets = plan['demonstrations'][:1] if a.smoke else plan['demonstrations']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit through Slurm, never run on the login node.')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    nvml.nvmlInit()
    uuid = str(getattr(torch.cuda.get_device_properties(0), 'uuid', ''))
    if not uuid:
        raise RuntimeError('CUDA UUID missing; cannot safely map the assigned NVML GPU.')
    handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
    if nvml.nvmlDeviceGetMigMode(handle)[0] != 0:
        raise RuntimeError('Use a full GPU, not MIG, for attributable energy measurements.')
    nvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # fail early if unsupported
    (a.output/'plan.json').write_bytes(raw)
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = dict(plan_sha256=hashlib.sha256(raw).hexdigest(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        gpu_uuid=uuid, gpu_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        driver=nvml.nvmlSystemGetDriverVersion(), slurm_job=os.environ.get('SLURM_JOB_ID'),
        code_sha256={name:digest(Path(__file__).parent/name) for name in ['run_icl.py','common.py']},
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','datasets','numpy','pyarrow','nvidia-ml-py']},
        decoding=dict(do_sample=False,num_beams=1,temperature=1.,top_p=1.,top_k=50,
                      thinking=False,precision='bfloat16',batch_size=1,max_new_tokens=a.max_new_tokens))
    if isinstance(metadata['driver'],bytes):
        metadata['driver']=metadata['driver'].decode()
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    (a.output/'pip-freeze.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    (a.output/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    with (a.output/'measurements.jsonl').open('w',buffering=1) as f:
        for repetition, example_set in enumerate(sets):
            for model_id in ([SMALL,LARGE] if repetition%2==0 else [LARGE,SMALL]):
                revision = plan['models'][model_id]['revision']
                tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
                t = time.perf_counter()
                model = AutoModelForCausalLM.from_pretrained(model_id,revision=revision,
                    local_files_only=True,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa').to('cuda:0').eval()
                torch.cuda.synchronize()
                load_seconds = time.perf_counter()-t
                # Explicit model config and call kwargs prevent model-default sampling overrides.
                model.generation_config.do_sample=False
                model.generation_config.temperature=1.
                model.generation_config.top_p=1.
                model.generation_config.top_k=50
                shot_counts = SHOTS if model_id==SMALL else [0]
                limit = min(a.context_limit,model.config.max_position_embeddings)
                def encode(question, shots):
                    text = tokenizer.apply_chat_template(messages(question,example_set['items'][:shots]),
                        tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    inputs = tokenizer(text,return_tensors='pt')
                    if inputs['input_ids'].shape[1]+a.max_new_tokens > limit:
                        raise RuntimeError(f'Prompt exceeds context limit at {shots} shots; no silent truncation.')
                    return inputs.to('cuda:0'),text
                def generate(inputs, timer=None):
                    with torch.inference_mode():
                        return model.generate(**inputs,do_sample=False,num_beams=1,temperature=1.,
                            top_p=1.,top_k=50,use_cache=True,max_new_tokens=a.max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,streamer=timer)
                # Warm every shot condition on a synthetic question, excluded from metrics.
                for shots in shot_counts:
                    inputs,_ = encode('A box has 3 red balls and 4 blue balls. How many balls are in the box?',shots)
                    warm = generate(inputs)
                    del warm,inputs
                order = [(item,shots) for item in items for shots in shot_counts]
                random.Random(1000+repetition).shuffle(order)
                for item,shots in order:
                    torch.cuda.synchronize()
                    energy_start=nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    start=time.perf_counter()
                    inputs,prompt=encode(item['question'],shots)
                    prepared=time.perf_counter()
                    timer=FirstTokenTimer(torch.cuda.synchronize)
                    torch.cuda.reset_peak_memory_stats()
                    tokens=generate(inputs,timer)
                    torch.cuda.synchronize()
                    generation_end=time.perf_counter()
                    ids=tokens[0,inputs['input_ids'].shape[1]:].tolist()
                    response=tokenizer.decode(ids,skip_special_tokens=True)
                    end=time.perf_counter()
                    energy_end=nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    joules=(energy_end-energy_start)/1000
                    if joules<=0 or timer.first_time is None:
                        raise RuntimeError('Invalid energy or first-token measurement; stop rather than fabricate.')
                    prediction=parse(response)
                    eos=model.generation_config.eos_token_id
                    eos_ids=set(eos if isinstance(eos,list) else [eos])
                    capped=len(ids)>=a.max_new_tokens and (not ids or ids[-1] not in eos_ids)
                    row=dict(id=item['id'],split=item['split'],model=model_id,shots=shots,
                        repetition=repetition,example_seed=example_set['seed'],target=item['target'],
                        prediction=prediction,correct=prediction==item['target'],
                        parse_failure=prediction is None,strict_parse_failure=parse(response,False) is None,
                        hit_token_limit=capped,gpu_joules=joules,request_seconds=end-start,
                        input_preparation_seconds=prepared-start,ttft_seconds=timer.first_time-start,
                        generation_seconds=generation_end-prepared,
                        decode_after_first_seconds=generation_end-timer.first_time,
                        input_tokens=inputs['input_ids'].shape[1],output_tokens=len(ids),
                        peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                        model_load_seconds=load_seconds,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),response=response)
                    f.write(json.dumps(row)+'\n')
                    print(f'{model_id} seed={example_set["seed"]} {item["split"]} shots={shots} '
                          f'correct={row["correct"]} J={joules:.1f} TTFT={row["ttft_seconds"]:.3f}s',flush=True)
                    del tokens,inputs
                del model,tokenizer
                gc.collect()
                torch.cuda.empty_cache()
    nvml.nvmlShutdown()
    print('COMPLETE:',a.output,flush=True)


if __name__=='__main__':
    main()
