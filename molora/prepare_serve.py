"""Prepare frozen inputs for serve_bench.py; no GPU inference."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import random
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("serve-inputs"))
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--adapter", default="ehzawad/qwen3-1.7b-gsm8k-grpo")
    p.add_argument("--adapter-revision", default="83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8")
    p.add_argument("--base-revision", default="main")
    p.add_argument("--cache-only", action="store_true")
    a = p.parse_args()
    if a.out.exists() or a.n < 1:
        p.error("provide a new output directory and a positive question count")

    from datasets import DownloadConfig, load_dataset
    from huggingface_hub import snapshot_download

    base = "Qwen/Qwen3-1.7B"
    def snapshot(repo, revision, patterns):
        return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns,
                                      local_files_only=a.cache_only))

    base_path = snapshot(base, a.base_revision, ["*.json", "*.safetensors", "*.txt", "*.jinja"])
    adapter_path = snapshot(a.adapter, a.adapter_revision, ["adapter_config.json", "adapter_model.safetensors"])
    if not list(base_path.glob("*.safetensors")) or not (adapter_path/"adapter_model.safetensors").is_file():
        raise ValueError("Model or adapter weights missing from cache; rerun without --cache-only")
    data = load_dataset("openai/gsm8k", "main", download_config=DownloadConfig(local_files_only=a.cache_only))
    def key(question):
        return " ".join(question.casefold().split())
    seen = {key(r["question"]) for r in data["train"]}
    rows = []
    for i, r in enumerate(data["test"]):
        k = key(r["question"])
        if k in seen:
            continue
        seen.add(k)
        value = r["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        if "####" not in r["answer"] or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
            raise ValueError(f"Unexpected answer format at test index {i}")
        rows.append(dict(id=i, question=r["question"], target=str(Decimal(value))))
    if len(rows) < a.n:
        p.error("not enough distinct test questions")
    questions = random.Random(42).sample(rows, a.n)
    manifest = dict(model_commits={base:base_path.name}, adapter=a.adapter,
        adapter_commit=adapter_path.name, dataset="openai/gsm8k",
        dataset_fingerprints={k:v._fingerprint for k,v in data.items()}, seed=42,
        system_prompt=r"You are a careful math reasoning assistant. Solve the problem step by step, but keep the solution concise. End with exactly one final answer in the form \boxed{answer}.")
    a.out.mkdir(parents=True, exist_ok=False)
    for name, data in [("questions.json", questions), ("manifest.json", manifest)]:
        (a.out/name).write_text(json.dumps(data, indent=2)+"\n")
    print(f"Wrote {len(questions)} questions and checkpoint revisions to {a.out.resolve()}")


if __name__ == "__main__":
    main()
