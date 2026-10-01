"""KB4c — proxy-variable screening by controlled intervention on real data (replaces the simulated Tier 3 of the proposal).
A* = elev (elevated building) is removed from the model; true candidate P_r = A* with a fraction r of values permuted; negative control = full permutation."""
import sys, json, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from nfip_lib import *
from hce import hce_solve, node_stats
from hshap import owen_sample

pre = json.load(open("../results/prereg_H1.json"))
d = load(); T, node = build_tree(d)
enc = pickle.load(open("../results/kb2_test_enc.pkl", "rb")); z = enc["Z"]["simul"]; xi = enc["xi"]; kk = enc["k"]["simul"]
TR = ((d.year >= 2012) & (d.year <= 2018)).values; TE = ((d.year >= 2021) & (d.year <= 2023)).values
root_z = {}
for t in range(2012, 2025):
    src = (d.year.values + LAG) <= t
    Ev, Sv = node_stats(T, node[src], d.E.values[src], d.S.values[src]); root_z[t] = hce_solve(T, Ev, Sv, [0, kk, kk, kk], 1.0, Sv.sum() / Ev.sum())[0]
RS = [0.1, 0.3, 0.5, 0.7]; NNEG = 4; NX = 60; RMC = 60
# NREP and the output file can be passed on the command line; the default is the pre-registered configuration (10 repetitions).
# Each repetition uses its own seeds (5000+rep, SEED+rep), so the first 10 repetitions of an extended run
# reproduce the registered results exactly.
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUTP = sys.argv[2] if len(sys.argv) > 2 else "../results/kb4c_proxy.json"
base_cols = [c for c in CAT if c != "elev"] + NUM
PLAY = ["geo", "zone", "crs", "occ", "pfirm", "floors", "cyear", "prim", "ded", "cov", "logcov"] + [f"P{r}" for r in RS] + [f"N{c}" for c in range(NNEG)]
GROUPS = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], list(range(11, 11 + len(RS) + NNEG))]
p = len(PLAY)
elev = (d.elev.values == "1").astype(float)
res = {"A_star": "elev", "corr_elev_rate_train": float(np.corrcoef(elev[TR], d.rate.values[TR])[0, 1]), "reps": []}
t00 = time.time()
for rep in range(NREP):
    g = np.random.default_rng(5000 + rep)
    cand = {}
    for r in RS:
        v = elev.copy(); m = g.random(len(v)) < r; v[m] = g.permutation(v[m]); cand[f"P{r}"] = v
    for c in range(NNEG):
        cand[f"N{c}"] = g.permutation(elev)
    X = d[base_cols].copy()
    for c in [c for c in CAT if c != "elev"]:
        X[c] = X[c].astype("category")
    X["z"] = z
    for k, v in cand.items():
        X[k] = v
    m = lgb.train(dict(LGB_BASE, objective="tweedie", tweedie_variance_power=xi, seed=SEED + rep), lgb.Dataset(X[TR], d.rate.values[TR], weight=d.E.values[TR]),
                  pre["rounds"]["EAGB (simultaneous HCE)"])
    ref = {c: (d.loc[TR].groupby(c).E.sum().idxmax() if c in CAT else float(np.median(d.loc[TR, c]))) for c in base_cols}
    for k in cand:
        ref[k] = 0.0 if np.mean(cand[k][TR]) < .5 else 1.0
    pts = g.choice(np.where(TE)[0], NX, replace=False)
    cats = {c: X[c].cat.categories for c in X.columns if str(X[c].dtype) == "category"}
    def nu_batch(i, masks):
        rows = {}
        keep = ((np.array(masks)[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)
        for j, nm in enumerate(PLAY):
            if nm == "geo":
                rows["z"] = np.where(keep[:, j], z[i], root_z[int(d.year.values[i])])
            else:
                col = X[nm].values[i] if nm not in cand else cand[nm][i]
                rows[nm] = np.where(keep[:, j], col, ref[nm])
        F = pd.DataFrame(rows)
        for c, cc in cats.items():
            F[c] = pd.Categorical(F[c].astype(str), categories=cc)
        return np.log(m.predict(F[X.columns]))
    imp_o = np.zeros(p); imp_s = np.zeros(p)
    for i in pts:
        # collect all masks needed in the R_mc permutations for both methods, then call the model once
        orders_o, orders_s = [], []
        for _ in range(RMC):
            o = []
            for gg in g.permutation(len(GROUPS)):
                o += list(g.permutation(GROUPS[gg]))
            orders_o.append(o); orders_s.append(list(g.permutation(p)))
        masks = {0}
        for o in orders_o + orders_s:
            mm = 0
            for j in o:
                mm |= 1 << j; masks.add(mm)
        masks = sorted(masks); val = dict(zip(masks, nu_batch(i, masks)))
        for orders, imp in [(orders_o, imp_o), (orders_s, imp_s)]:
            contrib = np.zeros(p)
            for o in orders:
                mm = 0; u = val[0]
                for j in o:
                    mm |= 1 << j; u2 = val[mm]; contrib[j] += u2 - u; u = u2
            imp += np.abs(contrib / RMC) / NX
    lab = np.r_[np.ones(len(RS)), np.zeros(NNEG)]; ci = np.arange(11, p)
    rec = {"rep": rep}
    for nm, imp in [("owen", imp_o), ("shapley", imp_s)]:
        sc = imp[ci]
        rec[nm] = dict(auc=float(roc_auc_score(lab, sc)), detect_rate=float(np.mean(sc[:len(RS)] > sc[len(RS):].max())),
                       detect_by_r={str(r): bool(sc[k] > sc[len(RS):].max()) for k, r in enumerate(RS)}, imp_candidates=sc.round(6).tolist())
    rec["corr_Pr_elev"] = {str(r): float(np.corrcoef(cand[f"P{r}"], elev)[0, 1]) for r in RS}
    res["reps"].append(rec); print(rep, rec["owen"]["auc"], rec["shapley"]["auc"], flush=True)
for nm in ["owen", "shapley"]:
    a = np.array([r[nm]["auc"] for r in res["reps"]]); dr = np.array([r[nm]["detect_rate"] for r in res["reps"]])
    res[nm + "_summary"] = dict(auc_mean=float(a.mean()), auc_sd=float(a.std(ddof=1)), detect_mean=float(dr.mean()),
                                detect_by_r={str(r): float(np.mean([x[nm]["detect_by_r"][str(r)] for x in res["reps"]])) for r in RS})
diff = np.array([r["owen"]["auc"] - r["shapley"]["auc"] for r in res["reps"]])
bs = [np.mean(diff[np.random.default_rng(i).integers(0, len(diff), len(diff))]) for i in range(4000)]
res["auc_diff_owen_minus_shap"] = dict(mean=float(diff.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], lo95_1s=float(np.quantile(bs, .05)))
res["sec_total"] = time.time() - t00
res["NREP"] = NREP
json.dump(res, open(OUTP, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "reps"}, indent=1))
