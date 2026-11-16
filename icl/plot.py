"""Read analyze.py outputs and export slide-ready figures; no hardcoded outcomes."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import SMALL,LARGE


def make_plots(run):
    with (run/'summary.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='test']
    with (run/'per_seed.csv').open() as f:
        seeds=[r for r in csv.DictReader(f) if r['split']=='test' and r['model']==SMALL]
    small=sorted([r for r in rows if r['model']==SMALL],key=lambda r:int(r['shots']))
    large=next(r for r in rows if r['model']==LARGE)
    out=run/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    x=np.arange(len(small));labels=[r['shots'] for r in small]
    def values(metric):
        return np.array([float(r[metric]) for r in small])
    def save(fig,name):
        fig.text(.07,.025,'Paired GSM8K test questions; 3 example sets | B200, BF16, batch=1 | GPU board energy only',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.96))
        for ext in ['png','svg','pdf']:
            fig.savefig(out/f'{name}.{ext}',dpi=300,facecolor='white')
        plt.show()
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,metric,title,unit,scale in [
        (axes[0],'correct','Accuracy','Correct (%)',100),
        (axes[1],'gpu_joules','Total request GPU energy','Joules / question',1),
        (axes[2], 'ttft_seconds', 'Time to first generated token', 'Milliseconds', 1000)]:
        y=values(metric)*scale;lo=values(metric+'_lo')*scale;hi=values(metric+'_hi')*scale
        ax.plot(x,y,'o-',color='#00897B',label='1.7B: mean of 3 example sets')
        ax.fill_between(x,lo,hi,color='#00897B',alpha=.15,label='95% question-bootstrap interval')
        baseline=float(large[metric])*scale
        ax.axhline(baseline,color='#315A9E',ls='--',label='8B zero-shot')
        ax.set_xticks(x,labels);ax.set_xlabel('In-context examples')
        ax.set_ylabel(unit);ax.set_title(title);ax.grid(alpha=.15)
        if metric=='correct': ax.set_ylim(0,100)
        else: ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8,loc='best')
    save(fig,'01_accuracy_energy_ttft')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,title in [(axes[0],'input_tokens','Prompt length'),(axes[1],'output_tokens','Response length')]:
        ax.plot(x,values(metric),'o-',color='#00897B')
        ax.set_xticks(x,labels);ax.set_xlabel('In-context examples');ax.set_ylabel('Tokens')
        ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    save(fig,'02_token_lengths')
    fig,ax=plt.subplots(figsize=(8,5))
    for seed in sorted({r['seed'] for r in seeds}):
        subset=sorted([r for r in seeds if r['seed']==seed],key=lambda r:int(r['shots']))
        ax.plot(x,[100*float(r['correct']) for r in subset],'o-',label='Example seed '+seed)
    ax.set_xticks(x,labels);ax.set_xlabel('In-context examples');ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0,100);ax.set_title('Sensitivity to demonstration selection')
    ax.legend();ax.grid(alpha=.15);save(fig,'03_example_seed_sensitivity')
    print('Saved figures:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    make_plots(p.parse_args().run)
