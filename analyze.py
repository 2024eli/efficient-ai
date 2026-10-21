"""Held-out router evaluation; all energy and latency routing totals are replay estimates."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

SMALL, LARGE = 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-8B'
DOMAINS = ['math','science','logic']


def select_threshold(candidates, tolerance):
    # candidates must contain validation statistics only.
    baseline = next(r['accuracy'] for r in candidates if r['threshold']==2.0)
    feasible = [r for r in candidates if r['accuracy'] >= baseline-tolerance]
    return min(feasible,key=lambda r:(r['gpu_joules'], -r['accuracy']))['threshold']


def oracle_choice(small, large):
    # Accuracy-maximizing hindsight ceiling, cheaper choice for ties.
    return min([small,large],key=lambda r:(-r['accuracy'],r['gpu_joules']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--tolerance',type=float,default=.02)
    a=p.parse_args()
    if not 0<=a.tolerance<=1:
        p.error('tolerance must lie in [0,1]')
    out=Path(a.run)
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['args']['smoke']:
        p.error('Smoke data is for format/hardware validation, not router training.')
    data=json.loads((out/'data.json').read_text())
    items={r['id']:r for r in data['items']}
    groups={}
    for line in (out/'measurements.jsonl').read_text().splitlines():
        r=json.loads(line)
        groups.setdefault((r['id'],r['model']),[]).append(r)
    expected=manifest['args']['repeats']
    pairs={}
    for key,item in items.items():
        pair={}
        for model in [SMALL,LARGE]:
            rows=groups.get((key,model),[])
            if len(rows)!=expected or {r['repeat'] for r in rows}!=set(range(expected)):
                raise ValueError(f'Incomplete or duplicate measurements: {key} {model}')
            pair[model]={k:float(np.mean([r[k] for r in rows])) for k in
                         ['gpu_joules','seconds','correct','parse_failure','hit_token_limit']}
            pair[model]['accuracy']=pair[model].pop('correct')
        pairs[key]=pair
    def ids(split):
        return sorted(k for k,r in items.items() if r['split']==split)
    train,validation,test=ids('train'),ids('validation'),ids('test')
    # Questions ONLY: no answers, model outputs, measured costs, or test labels as features.
    vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=20000,sublinear_tf=True)
    x=vectorizer.fit_transform([items[k]['question'] for k in train])
    y=np.array([pairs[k][SMALL]['accuracy']>=.5 for k in train],dtype=int)
    with threadpool_limits(limits=1):
        if len(set(y))==2:
            model=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(x,y)
            if model.n_iter_[0]>=1000:
                raise RuntimeError('Router did not converge')
            def probability(texts):
                return model.predict_proba(vectorizer.transform(texts))[:,1]
        else:
            model=None
            def probability(texts):
                return np.full(len(texts),float(y[0]))
        scores={k:float(v) for k,v in zip(validation+test,probability([items[k]['question'] for k in validation+test]))}
        overhead=[]
        probability([items[test[0]]['question']])
        for _ in range(3):
            for k in test:
                start=time.perf_counter()
                probability([items[k]['question']])
                overhead.append(time.perf_counter()-start)
    overhead_mean=float(np.mean(overhead))
    def rows_for(keys,threshold):
        return [pairs[k][SMALL if scores[k]>=threshold else LARGE] for k in keys]
    def metrics(keys,rows,weights=None):
        weights=np.ones(len(keys)) if weights is None else np.array(weights)
        weights=weights/weights.sum()
        result={k:float(np.dot(weights,[r[k] for r in rows])) for k in
                ['accuracy','gpu_joules','seconds','parse_failure','hit_token_limit']}
        result['gpu_joules_per_correct']=result['gpu_joules']/result['accuracy'] if result['accuracy'] else None
        return result
    thresholds=[-1.0]+list(np.linspace(0,1,21))+[2.0]
    validation_curve=[dict(threshold=t,**metrics(validation,rows_for(validation,t))) for t in thresholds]
    chosen=select_threshold(validation_curve,a.tolerance)
    selected_test=rows_for(test,chosen)
    policies={
        'always_small':[pairs[k][SMALL] for k in test],
        'always_large':[pairs[k][LARGE] for k in test],
        'learned_router':selected_test,
        'oracle_hindsight':[oracle_choice(pairs[k][SMALL],pairs[k][LARGE]) for k in test]}
    # Domain-only baseline uses validation labels, not test labels. Domain known at request time.
    domain_policy={}
    for d in DOMAINS:
        keys=[k for k in validation if items[k]['domain']==d]
        small=metrics(keys,[pairs[k][SMALL] for k in keys])
        large=metrics(keys,[pairs[k][LARGE] for k in keys])
        domain_policy[d]=SMALL if small['accuracy']>=large['accuracy']-a.tolerance and small['gpu_joules']<large['gpu_joules'] else LARGE
    policies['domain_validation_router']=[pairs[k][domain_policy[items[k]['domain']]] for k in test]
    rng=np.random.default_rng(42)
    fraction=float(np.mean([scores[k]>=chosen for k in test]))
    policies['random_same_expected_small_fraction']=[pairs[k][SMALL if rng.random()<fraction else LARGE] for k in test]
    workloads={'balanced':{d:1/3 for d in DOMAINS}}
    workloads.update({f'{d}_heavy':{k:(.8 if k==d else .1) for k in DOMAINS} for d in DOMAINS})
    report=[]
    for workload,shares in workloads.items():
        weights=[shares[items[k]['domain']]/sum(items[j]['domain']==items[k]['domain'] for j in test) for k in test]
        baseline=metrics(test,policies['always_large'],weights)
        for name,rows in policies.items():
            result=metrics(test,rows,weights)
            result.update(workload=workload,policy=name,
                accuracy_gap_vs_large=result['accuracy']-baseline['accuracy'],
                gpu_energy_savings_fraction=1-result['gpu_joules']/baseline['gpu_joules'],
                router_cpu_seconds_per_request=overhead_mean if name=='learned_router' else None,
                allowable_unmeasured_router_joules_per_request=baseline['gpu_joules']-result['gpu_joules'])
            report.append(result)
    with (out/'routing-summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    # Paired, domain-stratified bootstrap; repeats averaged before resampling questions.
    bootstrap=[]
    rng=np.random.default_rng(43)
    for _ in range(2000):
        sample=[]
        for domain in DOMAINS:
            indices=[i for i,k in enumerate(test) if items[k]['domain']==domain]
            sample.extend(rng.choice(indices,len(indices),replace=True))
        gap=np.mean([selected_test[i]['accuracy']-pairs[test[i]][LARGE]['accuracy'] for i in sample])
        ej=np.mean([selected_test[i]['gpu_joules'] for i in sample])
        base=np.mean([pairs[test[i]][LARGE]['gpu_joules'] for i in sample])
        bootstrap.append([gap,1-ej/base])
    ci=np.quantile(bootstrap,[.025,.975],axis=0).T.tolist()
    # Save text router coefficients for inspection; no unsafe pickle needed.
    router=dict(vocabulary={k:int(v) for k,v in vectorizer.vocabulary_.items()},idf=vectorizer.idf_.tolist(),
        coefficients=model.coef_.tolist() if model is not None else None,
        intercept=model.intercept_.tolist() if model is not None else None)
    (out/'router-parameters.json').write_text(json.dumps(router))
    diagnostic=dict(selected_threshold=chosen,tolerance=a.tolerance,
        validation_curve=validation_curve,domain_policy=domain_policy,
        constant_router=model is None,test_small_fraction=fraction,
        router_cpu_seconds=dict(mean=overhead_mean,p50=float(np.median(overhead)),p95=float(np.quantile(overhead,.95))),
        balanced_test_bootstrap_95=dict(accuracy_gap_vs_large=ci[0],gpu_energy_savings_fraction=ci[1]),
        test_by_domain={d:{name:metrics([k for k in test if items[k]['domain']==d],
            [row for k,row in zip(test,rows) if items[k]['domain']==d]) for name,rows in policies.items()} for d in DOMAINS},
        test_threshold_curve_exploratory_only=[dict(threshold=t,**metrics(test,rows_for(test,t))) for t in thresholds],
        note='Replay estimates exclude router CPU energy, co-resident GPU idle costs, model switching/loading and live scheduling; not deployment measurements.')
    (out/'routing-report.json').write_text(json.dumps(diagnostic,indent=2))
    with (out/'test-routes.jsonl').open('w') as f:
        for k in test:
            f.write(json.dumps(dict(id=k,domain=items[k]['domain'],small_probability=scores[k],
                choice=SMALL if scores[k]>=chosen else LARGE))+'\n')
    print('Validation-selected threshold:',chosen,'; test small fraction:',fraction)
    print('Balanced test 95% paired bootstrap intervals:',diagnostic['balanced_test_bootstrap_95'])
    print('Reports:',out/'routing-summary.csv',out/'routing-report.json')


if __name__=='__main__':
    main()
