import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Where federation HELPS: bilateral RFB+Bancos (vertical intra-country).
local(RFB), local(Bancos) vs federated RFB+Bancos (FedAvg/FedProx/SCAFFOLD),
evaluated on each silo's held-out test. SGD, 5 seeds, 95% CI."""
import numpy as np, torch
from taxfl_experiment import set_seed, GraphSAGE, evaluate
from taxfl_experiment_v3_gpu import make_intra_country_data
from taxfl_scaffold import sgd_local, fedavg_w, ci

DEV='cpu'
def get_two(seed):
    sd,_=make_intra_country_data(seed=seed)
    return sd['RFB'], sd['Bancos']

def train_local(data, rounds=40, le=10, lr=0.1):
    set_seed(0); m=GraphSAGE(10,64,2,2).to(DEV)
    gw={n:p.detach().clone() for n,p in m.named_parameters()}
    for _ in range(rounds): sgd_local(m,data,le,lr,gw=gw)
    return m

def train_fed(ds, method, rounds=40, le=10, lr=0.1, mu=0.01):
    set_seed(0); g=GraphSAGE(10,64,2,2).to(DEV)
    gw={n:p.detach().clone() for n,p in g.named_parameters()}
    cg={n:torch.zeros_like(p) for n,p in g.named_parameters()}
    cl=[{n:torch.zeros_like(p) for n,p in g.named_parameters()} for _ in ds]
    sizes=[len(d.x) for d in ds]
    for _ in range(rounds):
        ws=[]; agg_dc={n:torch.zeros_like(p) for n,p in g.named_parameters()}
        for k,data in enumerate(ds):
            m=GraphSAGE(10,64,2,2).to(DEV)
            with torch.no_grad():
                for n,p in m.named_parameters(): p.copy_(gw[n])
            if method=='scaffold':
                sgd_local(m,data,le,lr,gw=gw,cg=cg,cl=cl[k])
                with torch.no_grad():
                    yk={n:p.detach().clone() for n,p in m.named_parameters()}
                    for n in gw:
                        nc=cl[k][n]-cg[n]+(gw[n]-yk[n])/(le*lr)
                        agg_dc[n]+=(nc-cl[k][n])/len(ds); cl[k][n]=nc
            else:
                sgd_local(m,data,le,lr,gw=gw,mu=(mu if method=='fedprox' else 0.0))
            ws.append({n:p.detach().clone() for n,p in m.named_parameters()})
        new=fedavg_w(ws,sizes)
        with torch.no_grad():
            for n,p in g.named_parameters(): p.copy_(new[n]); gw[n]=new[n].clone()
            if method=='scaffold':
                for n in cg: cg[n]+=agg_dc[n]
    return g

seeds=[42,123,456,789,1024]
targets={'RFB':0,'Bancos':1}
R={f'{m}@{t}':{'auc':[],'auprc':[],'f1':[]}
   for t in targets for m in ['local','FedAvg','FedProx','SCAFFOLD']}
for s in seeds:
    rfb,ban=get_two(s); ds=[rfb,ban]
    locs={'RFB':train_local(rfb),'Bancos':train_local(ban)}
    feds={'FedAvg':train_fed(ds,'fedavg'),'FedProx':train_fed(ds,'fedprox'),
          'SCAFFOLD':train_fed(ds,'scaffold')}
    for t,ti in targets.items():
        a,f,ap=evaluate(locs[t], ds[ti]); R[f'local@{t}']['auc'].append(a); R[f'local@{t}']['f1'].append(f); R[f'local@{t}']['auprc'].append(ap)
        for mn,g in feds.items():
            a,f,ap=evaluate(g, ds[ti]); R[f'{mn}@{t}']['auc'].append(a); R[f'{mn}@{t}']['f1'].append(f); R[f'{mn}@{t}']['auprc'].append(ap)
    print('seed %d done'%s, flush=True)

print('\n=== Bilateral RFB+Bancos (5 seeds, mean +/- 95%% CI) ===')
print('%-16s%16s%16s%16s'%('config','AUC','AUPRC','F1'))
for t in targets:
    print('--- evaluated on %s test ---'%t)
    for m in ['local','FedAvg','FedProx','SCAFFOLD']:
        k=f'{m}@{t}'; a=ci(R[k]['auc']); ap=ci(R[k]['auprc']); f=ci(R[k]['f1'])
        print('%-16s%8.3f+/-%.3f%8.3f+/-%.3f%8.3f+/-%.3f'%(m,a[0],a[1],ap[0],ap[1],f[0],f[1]))
