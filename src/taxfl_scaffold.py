import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""SCAFFOLD added to the REAL TaxFL pipeline (v1 data + model + evaluate).
Apples-to-apples SGD comparison: local / FedAvg / FedProx / SCAFFOLD on the
3-jurisdiction cross-border setup, evaluated on jurisdiction C's held-out test."""
import numpy as np, torch, torch.nn.functional as F, copy
from taxfl_experiment import (set_seed, generate_jurisdiction_data,
    inject_evasion_chains, to_pyg, GraphSAGE, evaluate, class_weight_tensor)

DEV='cpu'
def build(seed, n_ent=300, n_wal=400, n_legit=1000, n_ev=60):
    jurs=[]
    for jid in range(3):
        f,l,s,d=generate_jurisdiction_data(n_ent,n_wal,n_legit,jid,seed)
        jurs.append({'feats':f,'labels':l,'src':list(s),'dst':list(d),
                     'n_entities':n_ent,'n_wallets':n_wal})
    jurs,_=inject_evasion_chains(jurs,n_ev,seed)
    return [to_pyg(j['feats'],j['labels'],j['src'],j['dst'],seed,DEV) for j in jurs]

def sgd_local(model, data, epochs, lr, gw=None, mu=0.0, cg=None, cl=None):
    w=class_weight_tensor(data.y[data.train_mask], data.x.device)
    model.train()
    for _ in range(epochs):
        for p in model.parameters(): p.grad=None
        out=model(data.x,data.edge_index)
        loss=F.cross_entropy(out[data.train_mask],data.y[data.train_mask],weight=w)
        if mu>0 and gw is not None:
            loss=loss+(mu/2)*sum(((p-gw[n])**2).sum() for n,p in model.named_parameters())
        loss.backward()
        with torch.no_grad():
            for n,p in model.named_parameters():
                g=p.grad+5e-4*p
                if cg is not None: g=g-cl[n]+cg[n]   # SCAFFOLD correction
                p-=lr*g

def fedavg_w(ws, sizes):
    tot=sum(sizes); avg={}
    for k in ws[0]: avg[k]=sum(w[k]*(s/tot) for w,s in zip(ws,sizes))
    return avg

def run_method(method, datasets, rounds=40, le=10, lr=0.1, mu=0.01):
    set_seed(0); in_ch=datasets[0].x.shape[1]
    g=GraphSAGE(in_ch,64,2,2).to(DEV)
    gw={n:p.detach().clone() for n,p in g.named_parameters()}
    cg={n:torch.zeros_like(p) for n,p in g.named_parameters()}
    cl=[{n:torch.zeros_like(p) for n,p in g.named_parameters()} for _ in datasets]
    sizes=[len(d.x) for d in datasets]
    for _ in range(rounds):
        ws=[]; dcs=[{n:torch.zeros_like(p) for n,p in g.named_parameters()}]
        agg_dc={n:torch.zeros_like(p) for n,p in g.named_parameters()}
        for k,data in enumerate(datasets):
            m=GraphSAGE(in_ch,64,2,2).to(DEV)
            with torch.no_grad():
                for n,p in m.named_parameters(): p.copy_(gw[n])
            if method=='scaffold':
                sgd_local(m,data,le,lr,gw=gw,cg=cg,cl=cl[k])
                with torch.no_grad():
                    yk={n:p.detach().clone() for n,p in m.named_parameters()}
                    for n in gw:
                        nc=cl[k][n]-cg[n]+(gw[n]-yk[n])/(le*lr)
                        agg_dc[n]+=(nc-cl[k][n])/len(datasets); cl[k][n]=nc
            else:
                sgd_local(m,data,le,lr,gw=gw,mu=(mu if method=='fedprox' else 0.0))
            ws.append({n:p.detach().clone() for n,p in m.named_parameters()})
        new=fedavg_w(ws,sizes)
        with torch.no_grad():
            for n,p in g.named_parameters(): p.copy_(new[n]); gw[n]=new[n].clone()
            if method=='scaffold':
                for n in cg: cg[n]+=agg_dc[n]
    return evaluate(g, datasets[2])

def run_local(datasets, rounds=40, le=10, lr=0.1):
    set_seed(0); in_ch=datasets[0].x.shape[1]
    m=GraphSAGE(in_ch,64,2,2).to(DEV)
    gw={n:p.detach().clone() for n,p in m.named_parameters()}
    for _ in range(rounds): sgd_local(m,datasets[2],le,lr,gw=gw)  # gw unused (mu=0)
    return evaluate(m, datasets[2])

def ci(v):
    a=np.array(v,float); m=a.mean()
    return m,(1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0)

if __name__=='__main__':
    seeds=[42,123,456,789,1024]
    R={k:{'auc':[],'auprc':[],'f1':[]} for k in ['local','FedAvg','FedProx','SCAFFOLD']}
    for s in seeds:
        ds=build(s)
        for nm,fn in [('local',lambda d:run_local(d)),
                      ('FedAvg',lambda d:run_method('fedavg',d)),
                      ('FedProx',lambda d:run_method('fedprox',d)),
                      ('SCAFFOLD',lambda d:run_method('scaffold',d))]:
            auc,f1,auprc=fn(ds)
            R[nm]['auc'].append(auc); R[nm]['f1'].append(f1); R[nm]['auprc'].append(auprc)
        print('seed %d done'%s, flush=True)
    print('\n=== SGD comparison, 3-jur cross-border, jur C test (5 seeds, mean +/- 95%% CI) ===')
    print('%-12s%18s%18s%18s'%('method','AUC','AUPRC','F1'))
    for k in R:
        a=ci(R[k]['auc']); ap=ci(R[k]['auprc']); f=ci(R[k]['f1'])
        print('%-12s%10.3f+/-%.3f%10.3f+/-%.3f%10.3f+/-%.3f'%(k,a[0],a[1],ap[0],ap[1],f[0],f[1]))
