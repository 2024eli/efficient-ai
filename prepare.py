"""Build fixed disjoint router-train/validation/test pools; no LLM fine-tuning."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def split_rows(rows, sizes, seed):
    unique = {}
    for row in rows:
        key = hashlib.sha256(' '.join(row['question'].lower().split()).encode()).hexdigest()
        if key in unique and unique[key]['target'] != row['target']:
            raise ValueError('Conflicting duplicate question')
        unique[key] = dict(row, id=row['domain'] + ':' + key)
    pool = sorted(unique.values(), key=lambda r: r['id'])
    if sum(sizes) > len(pool):
        raise ValueError(f'Requested {sum(sizes)} unique questions, only {len(pool)} available')
    random.Random(seed).shuffle(pool)
    result, offset = [], 0
    for split, size in zip(['train', 'validation', 'test'], sizes):
        result.extend(dict(row, split=split) for row in pool[offset:offset+size])
        offset += size
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='screen-data.json')
    p.add_argument('--sizes', nargs=3, type=int, default=[100, 50, 100])
    a = p.parse_args()
    if min(a.sizes) < 1:
        p.error('all split sizes must be positive')
    from datasets import load_dataset
    from measurement_utils import answer
    specs = [('math', 'openai/gsm8k', 'main', 'test'),
             ('science', 'allenai/ai2_arc', 'ARC-Challenge', 'test'),
             ('logic', 'maveriq/bigbenchhard', 'boolean_expressions', 'train')]
    result, sources = [], []
    for domain, name, config, split in specs:
        dataset = load_dataset(name, config, split=split)
        sources.append(dict(domain=domain, dataset=name, config=config, source_split=split,
                            fingerprint=dataset._fingerprint))
        rows = []
        for item in dataset:
            if domain == 'math':
                question, target = item['question'], answer(item['answer'], gold=True)
                allowed = None
            elif domain == 'science':
                labels = item['choices']['label']
                question = item['question'] + '\n' + '\n'.join(
                    f'{label}. {text}' for label, text in zip(labels, item['choices']['text']))
                target, allowed = item['answerKey'], labels
            else:
                question, target, allowed = item['input'], item['target'].lower(), ['true', 'false']
            assert target is not None
            rows.append(dict(domain=domain, question=question, target=target, allowed=allowed))
        result.extend(split_rows(rows, a.sizes, 42))
    path = Path(a.output)
    with path.open('x') as f:
        json.dump(dict(seed=42, sizes=a.sizes, sources=sources, items=result), f, indent=2)
    print(f'Saved {len(result)} questions to {path}')


if __name__ == '__main__':
    main()
