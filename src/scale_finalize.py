import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, os, time, numpy as np
import taxfl_experiment_v3_gpu as v3
from taxfl_experiment import set_seed, GraphSAGE, local_train, evaluate

CFG = {5:(800,400,100), 10:(2000,400,150)}  # (nw, ne, n_ev)
def ci(v):
    a=np.array(v,float); m=a.mean()
    return m,(1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0)

def run(n_jur, seeds=(42,123,456)):
    nw,ne,nev = CFG[n_jur]; R={}
    for method in ['local','fedavg','fedprox']:
        f1s,aucs,aps=[],[],[]
        for s in seeds:
            set_seed(s); ds=v3.make_scale_data(n_jur, nw, ne, nev, s)
            if method=='local':
                m=GraphSAGE(10,64,2,2)
                for _ in range(40): local_train(m, ds[-1], 10)
                auc,f1,ap=evaluate(m, ds[-1])
            else:
                auc,f1,ap=v3.fed_run_scale(ds, method=method, seed=s)
            f1s.append(f1); aucs.append(auc); aps.append(ap)
        R[method]={'AUC':ci(aucs),'AUPRC':ci(aps),'F1':ci(f1s)}
    return R

if __name__=='__main__':
    n=int(sys.argv[1]); t=time.time()
    R=run(n)
    path='scale_results.json'
    allr=json.load(open(path)) if os.path.exists(path) else {}
    allr[str(n)]=R; json.dump(allr,open(path,'w'),indent=2)
    print('n_jur=%d done in %.0fs'%(n,time.time()-t))
    for meth in R:
        print('%-8s AUC %.3f±%.3f | AUPRC %.3f±%.3f | F1 %.3f±%.3f'%(
            meth,*R[meth]['AUC'],*R[meth]['AUPRC'],*R[meth]['F1']))
