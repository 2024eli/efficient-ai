"""Freeze examples/questions/model revisions; download before allocating a GPU."""
import argparse
import json
import random
from pathlib import Path
from common import key, gold, SMALL, LARGE, SHOTS


def build_plan(train, test, n_test, n_validation, exclude=None):
    excluded = set(exclude or [])
    # Exclude normalized train/test duplicates; also deduplicate within each pool.
    train_unique = {key(r['question']): dict(r) for r in train}
    pool = {key(r['question']): dict(r) for r in test
            if key(r['question']) not in excluded and key(r['question']) not in train_unique}
    if n_test+n_validation > len(pool):
        raise ValueError('Insufficient unique, non-excluded evaluation questions')
    ordered = sorted(pool)
    random.Random(20261027).shuffle(ordered)
    items = []
    for i, ident in enumerate(ordered[:n_test+n_validation]):
        row = pool[ident]
        items.append(dict(id=ident, question=row['question'], target=gold(row['answer']),
                          split='validation' if i < n_validation else 'test'))
    used = set()
    examples = []
    # Three disjoint example sets. Within a set, 2/4/8/16 shots are nested prefixes.
    for seed in [11, 22, 33]:
        available = sorted(set(train_unique)-used)
        ids = random.Random(seed).sample(available, 16)
        used.update(ids)
        examples.append(dict(seed=seed, ids=ids, items=[train_unique[k] for k in ids]))
    return dict(shots=SHOTS, items=items, demonstrations=examples,
                excluded_questions=len(excluded), n_test=n_test, n_validation=n_validation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-test', type=int, default=200)
    p.add_argument('--n-validation', type=int, default=40)
    p.add_argument('--exclude-data', type=Path,
                   help='Previous routing data.json/screen-data.json; exclude its questions')
    p.add_argument('--output', type=Path, default=Path('plan.json'))
    a = p.parse_args()
    if a.output.exists():
        p.error('Plan already exists. Archive it deliberately rather than overwriting.')
    if min(a.n_test, a.n_validation) < 1:
        p.error('positive split sizes required')
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    data = load_dataset('openai/gsm8k', 'main')
    excluded = []
    if a.exclude_data:
        old = json.loads(a.exclude_data.read_text())
        excluded = [key(r['question']) for r in old['items']]
    plan = build_plan(data['train'], data['test'], a.n_test, a.n_validation, excluded)
    plan['dataset_fingerprints'] = {s: d._fingerprint for s, d in data.items()}
    plan['models'] = {}
    for model in [SMALL, LARGE]:
        snapshot = snapshot_download(model, allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
        plan['models'][model] = dict(revision=Path(snapshot).name)
    plan['excluded_source'] = str(a.exclude_data) if a.exclude_data else None
    with a.output.open('x') as f:
        json.dump(plan, f, indent=2)
    print(f'Frozen {len(plan["items"])} questions; 3 disjoint sets of 16 training examples.')
    print('Excluded previous data:', a.exclude_data or 'NONE; do not claim a fresh test set relative to prior studies')


if __name__ == '__main__':
    main()
