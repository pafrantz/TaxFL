"""Independent corrected re-run of the SCALE experiment (Appendix B.7.1, 3 jur),
using Pedro's EXACT make_scale_data generator + a clean global train/test split.
Optimizer Adam(lr=0.01, wd=5e-4) to match his local_train; eval on held-out test
of the last jurisdiction (his evaluation target ds[-1]). Reports mean +/- 95% CI."""
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

DEV = 'cpu'

# ---- Pedro's exact generator (copied verbatim from v3_gpu, returns raw arrays) ----
def make_scale_raw(n_jur, n_wallets_per, n_entities_per, n_evasion, seed):
    rng = np.random.RandomState(seed); out=[]
    for jid in range(n_jur):
        ne, nw = n_entities_per, n_wallets_per; n = ne+nw
        feats = np.zeros((n,10), dtype=np.float32)
        feats[:ne,0]=rng.lognormal(10,1.5,ne); feats[:ne,1]=rng.uniform(0.1,0.35,ne)
        feats[:ne,2]=rng.poisson(12,ne).astype(float); feats[:ne,3]=rng.exponential(0.3,ne)
        feats[ne:,4]=rng.lognormal(8,2,nw); feats[ne:,5]=rng.poisson(50,nw).astype(float)
        feats[ne:,6]=rng.lognormal(6,1,nw); feats[ne:,7]=rng.uniform(0,1,nw)
        feats[ne:,8]=rng.exponential(0.1,nw); feats[ne:,9]=rng.poisson(2,nw).astype(float)
        feats /= (feats.std(axis=0)+1e-8); labels=np.zeros(n,dtype=np.int64)
        src,dst=[],[]
        for _ in range(2000):
            e=rng.randint(0,ne); w=ne+rng.randint(0,nw); src+=[e,w]; dst+=[w,e]
        for _ in range(n_evasion//n_jur):
            ie=rng.randint(0,ne); iw=ne+rng.randint(0,nw)
            feats[ie,0]*=0.2; feats[iw,4]*=8; feats[iw,8]+=0.7
            labels[ie]=1; labels[iw]=1; src+=[ie,iw]; dst+=[iw,ie]
        out.append((feats,labels,np.array(src),np.array(dst)))
    return out

class SAGE(torch.nn.Module):
    def __init__(s,ind,h=64,drop=0.3):
        super().__init__(); s.c1=SAGEConv(ind,h,aggr='mean'); s.c2=SAGEConv(h,2,aggr='mean'); s.d=drop
    def forward(s,x,ei):
        x=F.relu(s.c1(x,ei)); x=F.dropout(x,p=s.d,training=s.training); return s.c2(x,ei)

def to_tensors(feats,labels,src,dst,seed):
    x=torch.tensor(feats,dtype=torch.float); y=torch.tensor(labels,dtype=torch.long)
    ei=torch.tensor(np.vstack([src,dst]),dtype=torch.long)
    rng=np.random.default_rng(seed); n=len(labels)
    tr,te=[],[]
    for c in (0,1):
        idx=np.where(labels==c)[0]; rng.shuffle(idx); cut=int(0.3*len(idx))
        te+=idx[:cut].tolist(); tr+=idx[cut:].tolist()
    return dict(x=x,y=y,ei=ei,tr=torch.tensor(sorted(tr)),te=torch.tensor(sorted(te)))

def cw(y,idx):
    yt=y[idx]; npos=max(int((yt==1).sum()),1); nneg=max(int((yt==0).sum()),1)
    w=torch.tensor([1/nneg,1/npos],dtype=torch.float); return w/w.sum()*2

def local_steps(model,d,epochs,lr=0.01,gw=None,mu=0.0):
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4); model.train()
    w=cw(d['y'],d['tr'])
    for _ in range(epochs):
        opt.zero_grad(); out=model(d['x'],d['ei'])
        loss=F.cross_entropy(out[d['tr']],d['y'][d['tr']],weight=w)
        if mu>0 and gw is not None:
            loss=loss+(mu/2)*sum(((p-g)**2).sum() for p,g in zip(model.parameters(),gw))
        loss.backward(); opt.step()

@torch.no_grad()
def ev(model,d):
    model.eval(); p=F.softmax(model(d['x'],d['ei']),1)[:,1][d['te']].cpu().numpy()
    yt=d['y'][d['te']].cpu().numpy(); pred=(p>=0.5).astype(int)
    return (roc_auc_score(yt,p) if len(np.unique(yt))>1 else np.nan,
            average_precision_score(yt,p) if yt.sum()>0 else np.nan,
            f1_score(yt,pred,zero_division=0))

def get_w(m): return [p.detach().clone() for p in m.parameters()]
def set_w(m,ws):
    with torch.no_grad():
        for p,q in zip(m.parameters(),ws): p.copy_(q)

def federate(silos,method,rounds=40,le=10,mu=0.01):
    g=SAGE(10).to(DEV); sizes=np.array([len(s['tr']) for s in silos],float); wt=sizes/sizes.sum()
    gw=get_w(g)
    for _ in range(rounds):
        agg=[torch.zeros_like(p) for p in gw]
        for k,s in enumerate(silos):
            m=SAGE(10).to(DEV); set_w(m,gw)
            local_steps(m,s,le,gw=gw,mu=(mu if method=='fedprox' else 0.0))
            for i,p in enumerate(m.parameters()): agg[i]+=wt[k]*p.detach()
        gw=agg; set_w(g,gw)
    return g

def ci(v):
    a=np.array(v,float); a=a[~np.isnan(a)]; m=a.mean()
    return m,(1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.0)

seeds=[42,123,456,789,1011]
res={k:{'auc':[],'auprc':[],'f1':[]} for k in ['local(last jur)','FedAvg','FedProx']}
for s in seeds:
    torch.manual_seed(s); np.random.seed(s)
    raw=make_scale_raw(3,400,400,60,s)
    silos=[to_tensors(*r,seed=s+j) for j,r in enumerate(raw)]
    last=silos[-1]
    m=SAGE(10).to(DEV)
    for _ in range(40): local_steps(m,last,10)   # local trains only on last jur train split
    a,ap,f=ev(m,last)
    res['local(last jur)']['auc'].append(a); res['local(last jur)']['auprc'].append(ap); res['local(last jur)']['f1'].append(f)
    for meth in ['fedavg','fedprox']:
        g=federate(silos,meth); a,ap,f=ev(g,last)
        key='FedAvg' if meth=='fedavg' else 'FedProx'
        res[key]['auc'].append(a); res[key]['auprc'].append(ap); res[key]['f1'].append(f)
    print(f"seed {s} done",flush=True)

print("\n=== B.7.1 Small (3 jur), CORRECTED global split | eval on last-jur held-out test ===")
print(f"{'method':<16}{'AUC':>16}{'AUPRC':>16}{'F1':>16}")
for k in res:
    a=ci(res[k]['auc']); ap=ci(res[k]['auprc']); f=ci(res[k]['f1'])
    print(f"{k:<16}{a[0]:>8.3f}±{a[1]:.3f}{ap[0]:>8.3f}±{ap[1]:.3f}{f[0]:>8.3f}±{f[1]:.3f}")
