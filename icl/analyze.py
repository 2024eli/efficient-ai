"""Paired question-level analysis; select shot count using validation only."""
import argparse
import collections
import csv
import json
from pathlib import Path
from common import SMALL,LARGE,SHOTS

METRICS=['correct','gpu_joules','request_seconds','ttft_seconds','input_tokens',
         'output_tokens','parse_failure','strict_parse_failure','hit_token_limit']


def select_shots(rows, target_accuracy, tolerance=.02):
    feasible=[r for r in rows if r['correct']>=target_accuracy-tolerance]
    return min(feasible,key=lambda r:(r['gpu_joules'],r['shots']))['shots'] if feasible else None


def load_complete(run):
    manifest=json.loads((run/'manifest.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    expected_ids={r['id'] for r in plan['items']}
    if manifest['args']['smoke']:
        expected_ids={r['id'] for r in [r for r in plan['items'] if r['split']=='validation'][:3]}
    seeds=[r['seed'] for r in plan['demonstrations']]
    if manifest['args']['smoke']:
        seeds=seeds[:1]
    expected={(ident,seed,model,shot) for ident in expected_ids for seed in seeds
              for model,shots in [(SMALL,SHOTS),(LARGE,[0])] for shot in shots}
    rows=[];seen=set()
    for line in (run/'measurements.jsonl').read_text().splitlines():
        row=json.loads(line)
        k=(row['id'],row['example_seed'],row['model'],row['shots'])
        if k in seen or k not in expected:
            raise ValueError('Duplicate or unexpected measurement '+str(k))
        seen.add(k);rows.append(row)
    if seen!=expected:
        raise ValueError(f'Incomplete run: {len(seen)}/{len(expected)} requests. Do not analyze partial conditions.')
    return manifest,plan,rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--tolerance',type=float,default=.02)
    a=p.parse_args()
    if a.bootstrap<1 or not 0<=a.tolerance<=1:
        p.error('invalid bootstrap count or tolerance')
    import numpy as np
    manifest,plan,rows=load_complete(a.run)
    groups=collections.defaultdict(list)
    for r in rows:
        groups[(r['split'],r['model'],r['shots'])].append(r)
    seed_rows=[]
    for (split,model,shots),group in sorted(groups.items()):
        for seed in sorted({r['example_seed'] for r in group}):
            subset=[r for r in group if r['example_seed']==seed]
            seed_rows.append(dict(split=split,model=model,shots=shots,seed=seed,
                **{k:float(np.mean([r[k] for r in subset])) for k in METRICS}))
    if manifest['args']['smoke']:
        for r in seed_rows:
            print(r)
        for r in rows:
            if r['parse_failure'] or r['hit_token_limit']:
                print('AUDIT',r['model'],r['shots'],r['id'],r['response'])
        print('Smoke complete. Review audit before full collection. No shot selection from smoke.')
        return
    summary=[];arrays={};ids_by_split={}
    rng=np.random.default_rng(123)
    for split in ['validation','test']:
        ids=sorted({r['id'] for r in rows if r['split']==split})
        ids_by_split[split]=ids
        for model,shots in [(SMALL,s) for s in SHOTS]+[(LARGE,0)]:
            group=groups[(split,model,shots)]
            by_id=collections.defaultdict(list)
            for r in group:
                by_id[r['id']].append(r)
            # Average seeds/repeated 0-shot measurements BEFORE resampling questions.
            values=np.array([[np.mean([r[k] for r in by_id[i]]) for k in METRICS] for i in ids])
            arrays[(split,model,shots)]=values
            boot=values[rng.integers(0,len(ids),size=(a.bootstrap,len(ids)))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            result=dict(split=split,model=model,shots=shots,n_questions=len(ids),n_requests=len(group))
            for j,k in enumerate(METRICS):
                result[k]=float(values[:,j].mean())
                result[k+'_lo']=float(lo[j]);result[k+'_hi']=float(hi[j])
            result['gpu_joules_per_correct']=result['gpu_joules']/result['correct'] if result['correct'] else None
            summary.append(result)
    differences=[]
    for model,shots in [(SMALL,s) for s in SHOTS]:
        values=arrays[('test',model,shots)]
        for ref in [SMALL,LARGE]:
            baseline=arrays[('test',ref,0)]
            delta=values-baseline
            draws=rng.integers(0,len(delta),size=(a.bootstrap,len(delta)))
            boot=delta[draws].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            entry=dict(model=model,shots=shots,reference=ref)
            for j,k in enumerate(METRICS):
                entry[k+'_difference']=float(delta[:,j].mean())
                entry[k+'_lo']=float(lo[j]);entry[k+'_hi']=float(hi[j])
            # Ratios of total energy, retaining question pairing in bootstrap samples.
            savings=1-values[:,1].mean()/baseline[:,1].mean()
            savings_boot=1-values[draws,1].mean(axis=1)/baseline[draws,1].mean(axis=1)
            entry['gpu_energy_savings_fraction']=float(savings)
            entry['gpu_energy_savings_lo'],entry['gpu_energy_savings_hi']=map(float,np.quantile(savings_boot,[.025,.975]))
            differences.append(entry)
    validation=[r for r in summary if r['split']=='validation' and r['model']==SMALL]
    large_val=next(r for r in summary if r['split']=='validation' and r['model']==LARGE)
    selection=dict(tolerance=a.tolerance,
        near_best_small=select_shots(validation,max(r['correct'] for r in validation),a.tolerance),
        near_large=select_shots(validation,large_val['correct'],a.tolerance))
    report=dict(selection=selection,bootstrap=a.bootstrap,
        note='Pointwise paired-question bootstrap, conditional on three fixed example sets and this hardware run. No selection on test; no multiple-comparison correction.',
        examples='Three disjoint sets; nested shot prefixes. Seeds are not independent test questions.',
        selected_test_results={})
    for purpose,shot in [('near_best_small',selection['near_best_small']),('near_large',selection['near_large'])]:
        if shot is None:
            report['selected_test_results'][purpose]=None
        else:
            report['selected_test_results'][purpose]=dict(
                summary=next(r for r in summary if r['split']=='test' and r['model']==SMALL and r['shots']==shot),
                vs_large=next(r for r in differences if r['shots']==shot and r['reference']==LARGE))
    for filename,data in [('summary.csv',summary),('per_seed.csv',seed_rows),('paired_differences.csv',differences)]:
        with (a.run/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (a.run/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Wrote summary.csv, per_seed.csv, paired_differences.csv, analysis.json')


if __name__=='__main__':
    main()
