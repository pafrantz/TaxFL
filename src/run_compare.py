import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, taxfl_experiment_v2 as v2
seeds=[42,123,456,789,1024]
def ci(v):
    a=np.array(v,float); m=a.mean()
    return m,(1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0)
rows={}
for method in ['fedavg','fedprox']:
    L={'auc':[],'auprc':[],'f1':[]}; Fd={'auc':[],'auprc':[],'f1':[]}
    for s in seeds:
        r=v2.run_ablation(seed=s, method=method, verbose=False)
        L['auc'].append(r['local_auc']); L['auprc'].append(r['local_auprc']); L['f1'].append(r['local_f1'])
        Fd['auc'].append(r['fed_auc']); Fd['auprc'].append(r['fed_auprc']); Fd['f1'].append(r['fed_f1'])
    rows[method]=(L,Fd)
    print('seed-by-seed dAUC (%s):'%method, [round(Fd['auc'][i]-L['auc'][i],3) for i in range(len(seeds))], flush=True)

print('\n=== CURRENT CODE: local vs federated, jurisdiction C (5 seeds, mean +/- 95%% CI) ===')
print('%-22s %16s %16s %16s'%('','AUC','AUPRC','F1'))
for method in ['fedavg','fedprox']:
    L,Fd=rows[method]
    for nm,D in [('local-only',L),(method.upper(),Fd)]:
        a=ci(D['auc']); ap=ci(D['auprc']); f=ci(D['f1'])
        print('%-22s %8.3f+/-%.3f %8.3f+/-%.3f %8.3f+/-%.3f'%(nm,a[0],a[1],ap[0],ap[1],f[0],f[1]))
