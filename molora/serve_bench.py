"""Timed single-worker real-request replay, with fixed per-request adapter assignments."""
import argparse
from collections import deque
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import time


def pick(ready, batch, lookahead):
    oldest = ready[0]
    selected = [oldest]
    selected += [r for r in list(ready)[1:lookahead] if r["adapter"] == oldest["adapter"]][:batch-1]
    used = {r["id"] for r in selected}
    selected += [r for r in ready if r["id"] not in used][:batch-len(selected)]
    used = {r["id"] for r in selected}
    return selected, deque(r for r in ready if r["id"] not in used)


class First:
    def __init__(self):
        self.seen_prompt = False
        self.at = None

    def put(self, _):
        if self.seen_prompt and self.at is None:
            self.at = time.perf_counter()
        self.seen_prompt = True

    def end(self):
        pass


def score(text, target):
    hits = re.findall(r"\\boxed\{\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\}", text)
    return bool(hits and Decimal(hits[-1].replace(",", "")) == Decimal(str(target)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prior_run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lookahead", type=int, default=32)
    p.add_argument("--interval", type=float, default=.05, help="seconds between arrivals")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cap", type=int, default=768)
    a = p.parse_args()
    if min(a.n,a.batch,a.lookahead,a.repeats,a.cap)<1 or not 0 <= a.interval < float("inf"):
        p.error("invalid workload")
    import torch
    import pynvml as nv
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = json.loads((a.prior_run/"questions.json").read_text())
    prior = json.loads((a.prior_run/"manifest.json").read_text())
    if a.n > len(source):
        p.error("not enough saved questions")
    random.Random(42).shuffle(source)
    trace = [dict(row, adapter="math" if i % 2 else "__base__", arrival=i*a.interval)
             for i,row in enumerate(source[:a.n])]
    if len({r["id"] for r in trace}) != a.n:
        raise ValueError("question IDs must be unique")
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/"trace.json").write_text(json.dumps(trace, indent=2))
    model_id = "Qwen/Qwen3-1.7B"
    rev = prior["model_commits"][model_id]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=rev,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda:0")
    model = PeftModel.from_pretrained(base, prior["adapter"], revision=prior["adapter_commit"],
        adapter_name="math", local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    (a.out/"settings.json").write_text(json.dumps(dict(
        args=vars(a), base_revision=rev, adapter=prior["adapter"], adapter_revision=prior["adapter_commit"],
        scoring="last explicit numeric boxed answer", thinking=False, trace_seed=42,
        torch=torch.__version__), default=str, indent=2))

    def generate(batch):
        prompts = [tok.apply_chat_template(
            [{"role":"system","content":prior["system_prompt"]},
             {"role":"user","content":r["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for r in batch]
        inputs = tok(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        first = First()
        with torch.inference_mode():
            tokens = model.generate(**inputs, adapter_names=[r["adapter"] for r in batch],
                do_sample=False, temperature=1., top_p=1., top_k=50, num_beams=1,
                use_cache=True, max_new_tokens=a.cap, pad_token_id=tok.pad_token_id, streamer=first)
        sequences = tokens[:, inputs.input_ids.shape[1]:].tolist()
        sequences = [s[:next((j+1 for j,t in enumerate(s) if t in eos), len(s))] for s in sequences]
        texts = tok.batch_decode(sequences, skip_special_tokens=True)
        return first.at, sequences, texts

    nv.nvmlInit()
    try:
        gpu = nv.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(0).uuid))
        if nv.nvmlDeviceGetMigMode(gpu)[0]:
            raise RuntimeError("A full GPU is needed")
        generate(trace[:min(2,a.batch)])
        outputs, summaries = {}, []
        with (a.out/"requests.jsonl").open("w", buffering=1) as f:
            for rep in range(a.repeats):
                for policy in (("fifo","grouped") if rep%2==0 else ("grouped","fifo")):
                    waiting, pending, rows = deque(trace), deque(), []
                    torch.cuda.synchronize()
                    energy0 = nv.nvmlDeviceGetTotalEnergyConsumption(gpu)
                    start = time.perf_counter()
                    while waiting or pending:
                        now = time.perf_counter()-start
                        while waiting and waiting[0]["arrival"] <= now:
                            pending.append(waiting.popleft())
                        if not pending:
                            time.sleep(max(0,waiting[0]["arrival"]-now))
                            continue
                        batch, pending = pick(pending,a.batch,a.lookahead if policy=="grouped" else 1)
                        dispatch = time.perf_counter()-start
                        first, tokens, texts = generate(batch)
                        done = time.perf_counter()-start
                        for r, ids, text in zip(batch,tokens,texts):
                            row = dict(id=r["id"], adapter=r["adapter"], repeat=rep, policy=policy,
                                batch_size=len(batch), lora_batch=any(q["adapter"]=="math" for q in batch),
                                queue_seconds=dispatch-r["arrival"], ttft_seconds=first-start-r["arrival"],
                                latency_seconds=done-r["arrival"], output_tokens=len(ids),
                                capped=len(ids)==a.cap and ids[-1] not in eos, correct=score(text,r["target"]),
                                response=text)
                            rows.append(row)
                            outputs[rep,policy,r["id"]] = ids
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-start
                    energy = (nv.nvmlDeviceGetTotalEnergyConsumption(gpu)-energy0)/1000
                    if energy <= 0:
                        raise RuntimeError("Energy counter did not advance")
                    def p95(name):
                        return sorted(r[name] for r in rows)[(95*len(rows)+99)//100-1]
                    summary = dict(repeat=rep,policy=policy,requests=len(rows),
                        joules_per_request=energy/len(rows),requests_per_second=len(rows)/seconds,
                        accuracy=sum(r["correct"] for r in rows)/len(rows),
                        cap_hits=sum(r["capped"] for r in rows),
                        p95_queue_seconds=p95("queue_seconds"),p95_ttft_seconds=p95("ttft_seconds"),
                        p95_latency_seconds=p95("latency_seconds"))
                    summaries.append(summary)
                    for row in rows:
                        f.write(json.dumps(row)+"\n")
                    print(json.dumps(summary),flush=True)
        changes = [sum(outputs[rep,"fifo",r["id"]] != outputs[rep,"grouped",r["id"]] for r in trace)
                   for rep in range(a.repeats)]
        (a.out/"summary.json").write_text(json.dumps(dict(runs=summaries,changed_token_sequences=changes),indent=2))
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
