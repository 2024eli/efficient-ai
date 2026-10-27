"""CPU-only analysis of a lora_energy.py run: Tables A/B, LoRA overhead, oracle-router headroom."""
import argparse
import collections
import csv
import json
import random
import statistics
from pathlib import Path

SMALL = ['base', 'lora']


def per_question(rows):
    """Average repeats within (condition, id). Correctness uses repeat 0; flips are reported."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['condition'], r['id'])].append(r)
    result = {}
    for key, group in groups.items():
        group.sort(key=lambda r: r['repeat'])
        result[key] = dict(gpu_joules=statistics.mean(r['gpu_joules'] for r in group),
            seconds=statistics.mean(r['seconds'] for r in group),
            output_tokens=statistics.mean(r.get('output_tokens', 0) for r in group),
            correct=group[0].get('correct'), correct_flips=len({r.get('correct') for r in group}) > 1,
            text_changed=len({r.get('response_sha256') for r in group}) > 1)
    return result


def overhead(q, a, b, ids, field='gpu_joules', samples=2000, seed=0):
    """Ratio-of-means overhead of a vs b in percent, with a paired bootstrap 95% CI over questions."""
    def ratio(sample):
        return (sum(q[(a, i)][field] for i in sample) / sum(q[(b, i)][field] for i in sample) - 1) * 100
    rng = random.Random(seed)
    boots = sorted(ratio([rng.choice(ids) for _ in ids]) for _ in range(samples))
    return dict(percent=ratio(ids), ci95=[boots[int(.025 * samples)], boots[int(.975 * samples) - 1]])


def repeat_noise(rows):
    """Median |relative difference| of the same (condition, question) across repeats: the noise floor."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['condition'], r['id'])].append(r['gpu_joules'])
    by_condition = collections.defaultdict(list)
    for (condition, _), values in groups.items():
        if len(values) > 1:
            by_condition[condition].append((max(values) - min(values)) / statistics.mean(values) * 100)
    return {c: statistics.median(v) for c, v in by_condition.items()}


def summary_table(q, conditions, ids):
    table = []
    for c in conditions:
        group = [q[(c, i)] for i in ids]
        joules = sum(r['gpu_joules'] for r in group)
        tokens = sum(r['output_tokens'] for r in group)
        correct = sum(bool(r['correct']) for r in group)
        table.append(dict(condition=c, questions=len(group), accuracy=correct / len(group),
            mean_gpu_joules=joules / len(group), mean_latency_ms=1000 * statistics.mean(r['seconds'] for r in group),
            joules_per_output_token=joules / tokens if tokens else None,
            mean_output_tokens=tokens / len(group), joules_per_correct=joules / correct if correct else None))
    return table


def oracle_choice(options):
    """Hindsight ceiling: maximize correctness, then minimize energy."""
    return min(options, key=lambda o: (not o['correct'], o['gpu_joules']))


def min_energy_for_accuracy(q, ids, target_correct):
    """Exact minimum-energy base/LoRA assignment reaching target_correct (two options per question)."""
    joules, correct, upgrades = 0., 0, []
    for i in ids:
        cheap, other = sorted([q[('base', i)], q[('lora', i)]], key=lambda r: r['gpu_joules'])
        joules += cheap['gpu_joules']
        correct += bool(cheap['correct'])
        if other['correct'] and not cheap['correct']:
            upgrades.append(other['gpu_joules'] - cheap['gpu_joules'])
    for extra in sorted(upgrades):
        if correct >= target_correct:
            break
        joules, correct = joules + extra, correct + 1
    return (joules / len(ids), correct / len(ids)) if correct >= target_correct else (None, None)


def policies(q, ids, conditions):
    rows = []
    def add(name, choices):
        rows.append(dict(policy=name, accuracy=statistics.mean(bool(c['correct']) for c in choices),
                         mean_gpu_joules=statistics.mean(c['gpu_joules'] for c in choices)))
    for c in conditions:
        add(f'always-{c}', [q[(c, i)] for i in ids])
    add('oracle base/lora', [oracle_choice([q[(c, i)] for c in SMALL]) for i in ids])
    if '8b' in conditions:
        add('oracle base/lora/8b', [oracle_choice([q[(c, i)] for c in SMALL + ['8b']]) for i in ids])
    lora_correct = sum(bool(q[('lora', i)]['correct']) for i in ids)
    joules, accuracy = min_energy_for_accuracy(q, ids, lora_correct)
    rows.append(dict(policy='oracle min-energy matching always-lora accuracy', accuracy=accuracy, mean_gpu_joules=joules))
    reference = next(r['mean_gpu_joules'] for r in rows if r['policy'] == 'always-lora')
    for r in rows:
        r['energy_saving_vs_always_lora_pct'] = (1 - r['mean_gpu_joules'] / reference) * 100 if r['mean_gpu_joules'] else None
    return rows


def benefit_table(q, ids):
    labels = {(True, True): 'both correct', (False, True): 'base wrong, LoRA correct',
              (True, False): 'base correct, LoRA wrong', (False, False): 'both wrong'}
    groups = collections.defaultdict(list)
    for i in ids:
        groups[labels[(bool(q[('base', i)]['correct']), bool(q[('lora', i)]['correct']))]].append(i)
    table = []
    for label in labels.values():
        members = groups[label]
        mean = lambda c, f: statistics.mean(q[(c, i)][f] for i in members) if members else None
        table.append(dict(group=label, questions=len(members), fraction=len(members) / len(ids),
            base_mean_joules=mean('base', 'gpu_joules'), lora_mean_joules=mean('lora', 'gpu_joules'),
            base_mean_tokens=mean('base', 'output_tokens'), lora_mean_tokens=mean('lora', 'output_tokens')))
    return table, groups


def write_csv(path, table):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def markdown(table):
    keys = list(table[0])
    fmt = lambda v: f'{v:.4g}' if isinstance(v, float) else ('' if v is None else str(v))
    lines = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    lines += ['| ' + ' | '.join(fmt(r[k]) for k in keys) + ' |' for r in table]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    a = p.parse_args()
    run = Path(a.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise SystemExit('Run is incomplete; refusing to analyze partial measurements.')
    rows = [json.loads(line) for line in (run / 'measurements.jsonl').read_text().splitlines()]
    phases = {ph: [r for r in rows if r['phase'] == ph] for ph in ['natural', 'fixed', 'forward']}
    report = dict(run=str(run), adapter=manifest.get('adapter'), adapter_commit=manifest.get('adapter_commit'),
                  idle_watts=[manifest.get('idle_watts_start'), manifest.get('idle_watts_end')])

    natural = per_question(phases['natural'])
    conditions = [c for c in ['base', 'lora', 'lora_merged', '8b'] if (c, phases['natural'][0]['id']) in natural]
    ids = sorted({i for c, i in natural if c == 'base'})
    table_a = summary_table(natural, conditions, ids)
    table_b, groups = benefit_table(natural, ids)
    policy = policies(natural, ids, conditions)
    report['natural'] = dict(table_a=table_a, table_b=table_b, policies=policy,
        energy_overhead_lora_vs_base=overhead(natural, 'lora', 'base', ids),
        latency_overhead_lora_vs_base=overhead(natural, 'lora', 'base', ids, 'seconds'),
        output_token_change_lora_vs_base=overhead(natural, 'lora', 'base', ids, 'output_tokens'),
        repeat_noise_median_pct=repeat_noise(phases['natural']),
        correctness_flips_across_repeats={c: sum(natural[(c, i)]['correct_flips'] for i in ids) for c in conditions},
        parse_failures={c: sum(r['parse_failure'] for r in phases['natural'] if r['condition'] == c) for c in conditions},
        token_limit_hits={c: sum(r['hit_token_limit'] for r in phases['natural'] if r['condition'] == c) for c in conditions})
    if 'lora_merged' in conditions:
        report['natural']['energy_overhead_merged_vs_base'] = overhead(natural, 'lora_merged', 'base', ids)

    for phase in ['fixed', 'forward']:
        if not phases[phase]:
            continue
        q = per_question(phases[phase])
        phase_ids = sorted({i for c, i in q if c == 'base'})
        present = [c for c in conditions if (c, phase_ids[0]) in q]
        units = phases[phase][0]['passes'] if phase == 'forward' else manifest['args']['fixed_tokens']
        table = [dict(condition=c, questions=len(phase_ids),
            mean_gpu_joules=statistics.mean(q[(c, i)]['gpu_joules'] for i in phase_ids),
            mean_latency_ms=1000 * statistics.mean(q[(c, i)]['seconds'] for i in phase_ids),
            **{f'joules_per_{"pass" if phase == "forward" else "token"}':
               statistics.mean(q[(c, i)]['gpu_joules'] for i in phase_ids) / units}) for c in present]
        entry = dict(table=table, repeat_noise_median_pct=repeat_noise(phases[phase]),
                     energy_overhead_lora_vs_base=overhead(q, 'lora', 'base', phase_ids),
                     latency_overhead_lora_vs_base=overhead(q, 'lora', 'base', phase_ids, 'seconds'))
        if 'lora_merged' in present:
            entry['energy_overhead_merged_vs_base'] = overhead(q, 'lora_merged', 'base', phase_ids)
        report[phase] = entry

    with (run / 'per-question.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'group'] + [f'{c}_{k}' for c in conditions for k in ['correct', 'gpu_joules', 'seconds', 'output_tokens']])
        group_of = {i: g for g, members in groups.items() for i in members}
        for i in ids:
            writer.writerow([i, group_of[i]] + [natural[(c, i)][k] for c in conditions for k in ['correct', 'gpu_joules', 'seconds', 'output_tokens']])
    write_csv(run / 'table-a.csv', table_a)
    write_csv(run / 'table-b.csv', table_b)
    write_csv(run / 'policies.csv', policy)
    (run / 'lora-report.json').write_text(json.dumps(report, indent=2))

    n = report['natural']
    print('## Table A: natural generation\n' + markdown(table_a))
    print('\n## Table B: which questions benefit\n' + markdown(table_b))
    print('\n## Oracle routing headroom (hindsight ceilings, not deployable)\n' + markdown(policy))
    for phase in ['fixed', 'forward']:
        if phase in report:
            print(f'\n## {phase} phase\n' + markdown(report[phase]['table']))
    print('\n## LoRA overhead vs base (ratio of means, paired bootstrap 95% CI)')
    for label, entry in [('natural energy', n['energy_overhead_lora_vs_base']),
                         ('natural output tokens', n['output_token_change_lora_vs_base']),
                         ('fixed-length energy', report.get('fixed', {}).get('energy_overhead_lora_vs_base')),
                         ('forward-pass energy', report.get('forward', {}).get('energy_overhead_lora_vs_base'))]:
        if entry:
            print(f'{label}: {entry["percent"]:+.2f}% [{entry["ci95"][0]:+.2f}, {entry["ci95"][1]:+.2f}]')
    print('repeat noise (median % spread):', {ph: report[ph]['repeat_noise_median_pct'] for ph in ['natural', 'fixed', 'forward'] if ph in report})
    print('parse failures:', n['parse_failures'], 'token-limit hits:', n['token_limit_hits'])
    print(f'\nWrote {run}/lora-report.json, table-a.csv, table-b.csv, policies.csv, per-question.csv')


if __name__ == '__main__':
    main()
