"""KB4c/KB5 (test of H2): H-SHAP versus SHAP on the locked EAGB model (test mode).
Contents: efficiency I3, fidelity, frequency–severity decomposition (TK6 on the real model), sensitivity to the masking operator, stability under two noise sources,
cost (GT4) and the proxy-screening protocol by controlled intervention."""
import json, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from nfip_lib import *
from hce import hce_solve, node_stats
from hshap import shapley_from_table, owen_from_table, owen_sample

rng = np.random.default_rng(SEED + 45)
pre = json.load(open("../results/prereg_H1.json"))
d = load(); T, node = build_tree(d)
enc = pickle.load(open("../results/kb2_test_enc.pkl", "rb")); z = enc["Z"]["simul"]; xi = enc["xi"]; kk = enc["k"]["simul"]
idx = json.load(open("../results/models/test_index.json"))
M = {nm: lgb.Booster(model_file="../results/models/" + f) for nm, f in idx.items()}
main, mf, ms = M["EAGB (simultaneous HCE)"], M["EAGB-PG freq"], M["EAGB-PG sev"]
TR = (d.year >= 2012) & (d.year <= 2018); TE = (d.year >= 2021) & (d.year <= 2023)
# encoder value at the root for each year (used when geo is masked)
root_z = {}
for t in range(2012, 2025):
    src = (d.year.values + LAG) <= t
    Ev, Sv = node_stats(T, node[src], d.E.values[src], d.S.values[src])
    root_z[t] = hce_solve(T, Ev, Sv, [0, kk, kk, kk], 1.0, Sv.sum() / Ev.sum())[0]
PLAY = ["geo", "zone", "crs", "occ", "elev", "pfirm", "floors", "cyear", "prim", "ded", "cov", "logcov"]
GROUPS = [[0, 1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11]]
p = len(PLAY)
Xtr = features(d[TR], z[TR.values])
cats = {c: Xtr[c].cat.categories for c in CAT}


def ref_row():
    r = {}
    w = d.E[TR]
    for c in CAT:
        r[c] = d.loc[TR].groupby(c).E.sum().idxmax()
    for c in NUM:
        o = np.argsort(d.loc[TR, c].values); cw = np.cumsum(w.values[o]) / w.sum(); r[c] = d.loc[TR, c].values[o][np.searchsorted(cw, .5)]
    return r
REF = ref_row()


def masked_frame(rows, masks, ref=REF):
    """rows: DataFrame of points (columns CAT+NUM+z+year); masks: bitmask array. Returns a DataFrame (len(rows)*len(masks))."""
    n, m = len(rows), len(masks)
    out = {}
    keep = ((masks[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)  # (m,p)
    for j, nm in enumerate(PLAY):
        if nm == "geo":
            zr = np.array([root_z[int(t)] for t in rows.year.values])
            out["z"] = np.where(keep[None, :, j], rows.z.values[:, None], zr[:, None]).ravel()
        else:
            rv = ref[nm]
            col = np.asarray(rows[nm], dtype=object)
            v = np.where(keep[None, :, j], col[:, None].astype(object), np.full((n, m), rv, dtype=object)).ravel()
            out[nm] = v
    X = pd.DataFrame(out)
    for c in CAT:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    for c in NUM:
        X[c] = X[c].astype(float)
    return X[CAT + NUM + ["z"]]


def table(model, rows, log=True, ref=REF, masks=None):
    masks = np.arange(1 << p) if masks is None else masks
    pr = model.predict(masked_frame(rows, masks, ref)).reshape(len(rows), len(masks))
    v = np.log(pr) if log else pr
    return v - v[:, [0]]


te_idx = np.where(TE.values)[0]
pts = d.iloc[rng.choice(te_idx, 300, replace=False)].copy(); pts["z"] = z[pts.index]
R = {"n_points": len(pts), "players": PLAY, "groups": [[PLAY[j] for j in g] for g in GROUPS], "reference": {k: (str(v) if k in CAT else float(v)) for k, v in REF.items()}}
# ---------- 1. efficiency, fidelity, cost
t0 = time.time(); nu = table(main, pts); t_table = time.time() - t0
t0 = time.time(); phi = shapley_from_table(nu, p); t_sh = time.time() - t0
t0 = time.time(); psi, Phi = owen_from_table(nu, p, GROUPS); t_ow = time.time() - t0
R["I3"] = dict(max_group=float(max(np.abs(psi[:, g].sum(1) - Phi[:, i]).max() for i, g in enumerate(GROUPS))),
               max_total=float(np.abs(Phi.sum(1) - nu[:, -1]).max()), max_total_shapley=float(np.abs(phi.sum(1) - nu[:, -1]).max()))
# additive fidelity on random coalitions
fid = {"owen": [], "shapley": []}
for i in range(len(pts)):
    T_ = rng.integers(1, 1 << p, 200)
    keep = ((T_[:, None] >> np.arange(p)[None, :]) & 1)
    y = nu[i, T_]
    for nm, a in [("owen", psi[i]), ("shapley", phi[i])]:
        yh = keep @ a; fid[nm].append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
# group-level fidelity: coalitions are unions of groups
gm = [sum(1 << j for j in g) for g in GROUPS]
gfid = {"owen": [], "shapley": []}
for i in range(len(pts)):
    for Rm in range(1, 8):
        mask = sum(gm[g] for g in range(3) if (Rm >> g) & 1)
        gfid["owen"].append(nu[i, mask] - sum(Phi[i, g] for g in range(3) if (Rm >> g) & 1))
        gfid["shapley"].append(nu[i, mask] - sum(phi[i, GROUPS[g]].sum() for g in range(3) if (Rm >> g) & 1))
R["fidelity"] = {nm: dict(R2_mean=float(np.mean(v)), R2_median=float(np.median(v))) for nm, v in fid.items()}
R["group_fidelity_mae"] = {nm: float(np.mean(np.abs(v))) for nm, v in gfid.items()}
# cost: number of distinct coalitions needed by Owen
need = set()
for g, Bg in enumerate(GROUPS):
    others = [h for h in range(3) if h != g]
    for r in range(3):
        from itertools import combinations
        for Rr in combinations(others, r):
            QR = sum(gm[h] for h in Rr)
            for Tm in range(1 << len(Bg)):
                mask = QR | sum(1 << Bg[b] for b in range(len(Bg)) if (Tm >> b) & 1)
                need.add(mask)
need = np.array(sorted(need))
t0 = time.time(); nu_o = table(main, pts, masks=need); t_owen_calls = time.time() - t0
R["cost"] = dict(owen_distinct_coalitions=int(len(need)), owen_calls_bound=int(3 * 2 ** (2 + 5)), shapley_coalitions=int(1 << p),
                 sec_model_calls_all_coalitions=t_table, sec_model_calls_owen_only=t_owen_calls, sec_shapley_from_table=t_sh, sec_owen_from_table=t_ow,
                 n_points=len(pts))
# ---------- 2. frequency–severity decomposition on the real model (EAGB-PG)
sub = pts.iloc[:100]
nl, nm_ = table(mf, sub), table(ms, sub)
class _Prod:
    def predict(self, X): return mf.predict(X) * ms.predict(X)
npi = table(_Prod(), sub)
pl, Pl = owen_from_table(nl, p, GROUPS); pm, Pm = owen_from_table(nm_, p, GROUPS); pp, Pp = owen_from_table(npi, p, GROUPS)
rl, rm_, rp = table(mf, sub, log=False), table(ms, sub, log=False), table(_Prod(), sub, log=False)
ql, _ = owen_from_table(rl, p, GROUPS); qm, _ = owen_from_table(rm_, p, GROUPS); qp, _ = owen_from_table(rp, p, GROUPS)
R["freq_sev"] = dict(log_scale_max_resid=float(np.abs(pp - pl - pm).max()), log_scale_group_max_resid=float(np.abs(Pp - Pl - Pm).max()),
                     raw_scale_share_nonadditive=float(np.mean(np.abs(qp - ql - qm) > 1e-9 * (1 + np.abs(qp)))),
                     raw_scale_median_rel_resid=float(np.median(np.abs(qp - ql - qm).sum(1) / np.abs(qp).sum(1))),
                     share_freq_group=[float(np.mean(np.abs(Pl[:, g]) / (np.abs(Pl[:, g]) + np.abs(Pm[:, g]) + 1e-12))) for g in range(3)])
# ---------- 3. sensitivity to the masking operator (alternative reference and marginal expectation)
sub = pts.iloc[:60]
bg_idx = rng.choice(np.where(TR.values)[0], 32, replace=False)
bg = d.iloc[bg_idx].copy(); bg["z"] = z[bg_idx]
def table_marginal(model, rows, bgr):
    acc = 0
    for b in range(len(bgr)):
        r = {c: bgr[c].values[b] for c in CAT + NUM}
        v = np.log(model.predict(masked_frame(rows, np.arange(1 << p), r)).reshape(len(rows), -1))
        acc = acc + v
    acc = acc / len(bgr)
    return acc - acc[:, [0]]
t0 = time.time(); num_ = table_marginal(main, sub, bg); t_marg = time.time() - t0
phm = shapley_from_table(num_, p); psm, Phm = owen_from_table(num_, p, GROUPS)
alt = dict(REF); alt["zone"] = "X"; alt["elev"] = "1"
nua = table(main, sub, ref=alt); pha = shapley_from_table(nua, p); psa, Pha = owen_from_table(nua, p, GROUPS)
def rankcorr(a, b):
    return float(np.mean([spearmanr(a[i], b[i]).correlation for i in range(len(a))]))
R["masker_sensitivity"] = dict(
    marginal_vs_ref_owen_feature=rankcorr(psi[:60], psm), marginal_vs_ref_shapley_feature=rankcorr(phi[:60], phm),
    altref_vs_ref_owen_feature=rankcorr(psi[:60], psa), altref_vs_ref_shapley_feature=rankcorr(phi[:60], pha),
    global_rank_owen_marg=float(spearmanr(np.abs(psi[:60]).mean(0), np.abs(psm).mean(0)).correlation),
    global_rank_shap_marg=float(spearmanr(np.abs(phi[:60]).mean(0), np.abs(phm).mean(0)).correlation),
    sec_marginal_60pts=t_marg)
# ---------- 4. stability under two noise sources
def grp_sh(ph): return np.stack([ph[:, g].sum(1) for g in GROUPS], 1)
sub = pts.iloc[:100]
# (a) model noise: county-cluster bootstrap on the training set, same locked number of rounds
Xall = features(d, z); Xt = Xall[TR.values]; yt = d.rate.values[TR.values]; wt = d.E.values[TR.values]; cl = d.county.values[TR.values]
uc, inv = np.unique(cl, return_inverse=True)
repsA = {"owen_group": [], "shap_group": [], "owen_feat": [], "shap_feat": []}
t0 = time.time()
for r in range(8):
    cnt = np.bincount(np.random.default_rng(1000 + r).integers(0, len(uc), len(uc)), minlength=len(uc))
    wboot = wt * cnt[inv]
    mb = lgb.train(dict(LGB_BASE, objective="tweedie", tweedie_variance_power=xi, seed=SEED + r), lgb.Dataset(Xt[wboot > 0], yt[wboot > 0], weight=wboot[wboot > 0]), pre["rounds"]["EAGB (simultaneous HCE)"])
    nb = table(mb, sub); ph = shapley_from_table(nb, p); ps, Ps = owen_from_table(nb, p, GROUPS)
    repsA["owen_group"].append(Ps); repsA["shap_group"].append(grp_sh(ph)); repsA["owen_feat"].append(ps); repsA["shap_feat"].append(ph)
R["sec_stabilityA"] = time.time() - t0
# (b) background noise: keep the model, 8 different background sets for the marginal-expectation operator
sub_b = pts.iloc[:40]
repsB = {"owen_group": [], "shap_group": [], "owen_feat": [], "shap_feat": []}
for r in range(8):
    bi = np.random.default_rng(2000 + r).choice(np.where(TR.values)[0], 16, replace=False); bgr = d.iloc[bi].copy()
    nb = table_marginal(main, sub_b, bgr); ph = shapley_from_table(nb, p); ps, Ps = owen_from_table(nb, p, GROUPS)
    repsB["owen_group"].append(Ps); repsB["shap_group"].append(grp_sh(ph)); repsB["owen_feat"].append(ps); repsB["shap_feat"].append(ph)


def stab(reps):
    out = {}
    for key in ["owen_group", "shap_group"]:
        A = np.stack(reps[key])  # (R, n, G)
        disp = A.std(0).sum(1) / (np.abs(A).mean(0).sum(1) + 1e-12)  # normalised dispersion per point
        out[key + "_disp_per_point"] = disp
        # share of group-rank flips between pairs of repetitions
        Rn = A.shape[0]; flips = []
        for a in range(Rn):
            for b in range(a + 1, Rn):
                ra, rb = np.argsort(np.argsort(-np.abs(A[a]), 1), 1), np.argsort(np.argsort(-np.abs(A[b]), 1), 1)
                flips.append(np.mean(np.any(ra != rb, 1)))
        out[key + "_rank_flip"] = float(np.mean(flips))
    for key in ["owen_feat", "shap_feat"]:
        A = np.stack(reps[key]); imp = np.abs(A).mean(1)  # (R, p) global importance
        cs = [spearmanr(imp[a], imp[b]).correlation for a in range(len(imp)) for b in range(a + 1, len(imp))]
        out[key + "_global_rank_spearman"] = float(np.mean(cs))
    ratio = out["owen_group_disp_per_point"] / out["shap_group_disp_per_point"]
    bs = [np.mean(ratio[np.random.default_rng(i).integers(0, len(ratio), len(ratio))]) for i in range(2000)]
    out["disp_ratio_mean"] = float(np.mean(ratio)); out["disp_ratio_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    out["disp_ratio_hi95_1s"] = float(np.quantile(bs, .95))
    out["owen_group_disp_mean"] = float(np.mean(out.pop("owen_group_disp_per_point"))); out["shap_group_disp_mean"] = float(np.mean(out.pop("shap_group_disp_per_point")))
    return out
R["stability_model_noise"] = stab(repsA); R["stability_background_noise"] = stab(repsB)
# global importance (main model)
R["global_importance"] = {"owen_feature": dict(zip(PLAY, np.abs(psi).mean(0).round(5).tolist())), "shap_feature": dict(zip(PLAY, np.abs(phi).mean(0).round(5).tolist())),
                          "owen_group": np.abs(Phi).mean(0).round(5).tolist(), "shap_group_sum": np.abs(grp_sh(phi)).mean(0).round(5).tolist()}
np.savez_compressed("../results/kb45_attr.npz", psi=psi, phi=phi, Phi=Phi, nu_full=nu[:, -1])
json.dump(R, open("../results/kb45_hshap.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(R, indent=1, ensure_ascii=False, default=float)[:6000])
