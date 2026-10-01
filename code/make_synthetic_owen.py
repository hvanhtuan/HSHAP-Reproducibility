import itertools, math, numpy as np, json
from itertools import combinations
rng=np.random.default_rng(20261001)
sizes=[3,5,4]; blocks=[]; k=0
for b in sizes: blocks.append(list(range(k,k+b))); k+=b
p=k; G=len(blocks)
c=rng.uniform(0.5,1.5,p); w=np.array([1.0,0.8,0.6])
def make_v(kappa):
    cache={}
    def v(S):
        S=frozenset(S)
        if S in cache: return cache[S]
        t=[sum(c[j] for j in B if j in S)/sum(c[j] for j in B) for B in blocks]
        val=sum(w[g]*t[g]**1.5 for g in range(G))+kappa*(t[0]**3*t[1]+t[1]**3*t[2]*0.5+t[2]**2*t[0])
        cache[S]=val; return val
    return v
def shapley(v,players,vv=None):
    n=len(players); out={}
    for j in players:
        others=[x for x in players if x!=j]; s=0
        for r in range(n):
            wt=math.factorial(r)*math.factorial(n-r-1)/math.factorial(n)
            for T in combinations(others,r):
                s+=wt*(v(set(T)|{j})-v(set(T)))
        out[j]=s
    return out
def owen(v):
    psi={}; Phi={}
    Q=list(range(G))
    for g in Q:
        others=[h for h in Q if h!=g]; s=0
        for r in range(G):
            wR=math.factorial(r)*math.factorial(G-r-1)/math.factorial(G)
            for R in combinations(others,r):
                QR=set(j for h in R for j in blocks[h])
                s+=wR*(v(QR|set(blocks[g]))-v(QR))
        Phi[g]=s
        B=blocks[g]; b=len(B)
        for j in B:
            s=0; ot=[x for x in B if x!=j]
            for r in range(G):
                wR=math.factorial(r)*math.factorial(G-r-1)/math.factorial(G)
                for R in combinations(others,r):
                    QR=set(x for h in R for x in blocks[h])
                    for m in range(b):
                        wT=math.factorial(m)*math.factorial(b-m-1)/math.factorial(b)
                        for T in combinations(ot,m):
                            s+=wR*wT*(v(QR|set(T)|{j})-v(QR|set(T)))
            psi[j]=s
    return psi,Phi
def inv(a,b):
    n=len(a); return sum(1 for i in range(n) for j in range(i+1,n) if (a[i]-a[j])*(b[i]-b[j])<0)
res=[]
for kappa in [0,0.5,1,2,4]:
    v=make_v(kappa); phi=shapley(v,list(range(p))); psi,Phi=owen(v)
    vN=v(set(range(p)))
    eff1=max(abs(sum(psi[j] for j in B)-Phi[g]) for g,B in enumerate(blocks)); eff2=abs(sum(Phi.values())-vN)
    ph=np.array([phi[j] for j in range(p)]); ps=np.array([psi[j] for j in range(p)])
    D=np.abs(ps-ph).sum()/np.abs(ph).sum()
    sg=[sum(phi[j] for j in B) for B in blocks]
    dg=[Phi[g]-sg[g] for g in range(G)]
    maxd=max(abs(x) for x in dg)/max(abs(x) for x in Phi.values())
    grp_inv=inv([Phi[g] for g in range(G)],sg)
    feat_inv=inv(ps,ph)
    res.append(dict(kappa=kappa,D=D,maxdg=maxd,feat_inv=feat_inv,grp_inv=grp_inv,eff=max(eff1,eff2),vN=vN,Phi=[Phi[g] for g in range(G)],sg=sg))
    print(kappa, round(D,4), round(maxd,4), feat_inv, grp_inv, "%.1e"%max(eff1,eff2), [round(Phi[g],3) for g in range(G)], [round(x,3) for x in sg])
# count coalitions
cnt=set()
def evalset():
    S=set()
    Q=list(range(G))
    for g in Q:
        others=[h for h in Q if h!=g]
        for r in range(G):
            for R in combinations(others,r):
                QR=frozenset(x for h in R for x in blocks[h]); S.add(QR); S.add(QR|frozenset(blocks[g]))
                for m in range(len(blocks[g])+1):
                    for T in combinations(blocks[g],m): S.add(QR|frozenset(T))
    return S
print("coalitions",len(evalset()))
# MC algorithm 2, kappa=2
v=make_v(2); psi,Phi=owen(v); psi=np.array([psi[j] for j in range(p)])
# range of marginal contributions
mc=[]
def perm_draw():
    order=[]; gp=rng.permutation(G)
    for g in gp: order+=list(rng.permutation(blocks[g]))
    return order
def est(R):
    acc=np.zeros(p)
    for _ in range(R):
        o=perm_draw(); S=set(); prev=v(S)
        for j in o:
            S.add(j); cur=v(S); acc[j]+=cur-prev; prev=cur
    return acc/R
vN=v(set(range(p)))
out=[]
for R in [100,1000,5000]:
    errs=[np.abs(est(R)-psi).max() for _ in range(20)]
    eps=vN*math.sqrt(math.log(2*p/0.05)/(2*R))
    out.append((R,float(np.mean(errs)),float(np.max(errs)),eps)); print(R,np.mean(errs),np.max(errs),eps)
# --- check of Proposition 3(b): degree-2 game (pairwise interactions only) versus degree-3 game, partition (3,5,4) ---
qd, cb = [], []
for sd in range(1, 6):
    r2 = np.random.default_rng(100 + sd); a = r2.normal(size=p); b = r2.normal(size=(p, p)); c3 = r2.normal(size=(p, p, p))
    vq = lambda S: sum(a[j] for j in S) + sum(b[j, k] for j in S for k in S if j < k)
    vc = lambda S: vq(S) + sum(c3[i, j, k] for i in S for j in S for k in S if i < j < k)
    for v_, store in [(vq, qd), (vc, cb)]:
        ph = shapley(v_, list(range(p))); ps, _ = owen(v_)
        store.append(max(abs(ps[j] - ph[j]) for j in range(p)))
print("quadratic max|psi-phi| per seed", ["%.1e" % x for x in qd]); print("cubic max|psi-phi| per seed", ["%.2f" % x for x in cb])
json.dump(dict(res=res,mc=out,vN=vN,quadratic=qd,cubic=cb),open("synthetic_owen_results.json","w"),default=float)
