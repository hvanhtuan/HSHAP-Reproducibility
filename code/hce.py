"""Reference code for HCE (hierarchical credibility encoding; Definition 3.3, Algorithms 3.1 and 3.2, Definition 3.4 of the associated study).
The tree is stored in arrays: parent[v] (root = -1) and depth[v]; node 0 is the root.
"""
import numpy as np
from fractions import Fraction


class Tree:
    def __init__(self, paths):
        """paths: list of label tuples from level 1 to the leaf, e.g. (state, county, tract).
        Returns a tree rooted at 'ROOT' in which every prefix is a node."""
        self.index = {(): 0}
        parent = [-1]; depth = [0]; label = [()]
        for p in paths:
            for l in range(1, len(p) + 1):
                key = tuple(p[:l])
                if key not in self.index:
                    self.index[key] = len(parent)
                    parent.append(self.index[key[:-1]]); depth.append(l); label.append(key)
        self.parent = np.array(parent); self.depth = np.array(depth); self.label = label
        self.n = len(parent); self.L = int(self.depth.max())
        # pre-order: sorting by depth suffices because a parent always has smaller depth than its child
        self.pre = np.argsort(self.depth, kind="stable")
        self.post = self.pre[::-1]

    def node_of(self, path):
        """Deepest node present in the tree (falls back to the nearest ancestor if absent)."""
        p = tuple(path)
        while p not in self.index:
            p = p[:-1]
        return self.index[p]

    def ancestors_at(self, v, level):
        while self.depth[v] > level:
            v = self.parent[v]
        return v


def node_stats(tree, node_ids, E, S):
    Ev = np.bincount(node_ids, weights=E, minlength=tree.n)
    Sv = np.bincount(node_ids, weights=S, minlength=tree.n)
    return Ev, Sv


def hce_solve(tree, Ev, Sv, k, k0, mu0, fixed_root=False, exact=False):
    """Algorithm 3.1 HCE-SOLVE. k: per-level weights, k[l] for the edge into a node of depth l (l>=1).
    fixed_root=True: HCE-FIXED-ROOT variant (Corollary 3.1). exact=True: rational arithmetic (Fraction)."""
    par, dep = tree.parent, tree.depth
    if exact:
        a = [Fraction(x) for x in Ev]; b = [Fraction(x) for x in Sv]
        kk = [Fraction(x) for x in k]; k0 = Fraction(k0); mu0 = Fraction(mu0)
    else:
        a = np.array(Ev, dtype=float).copy(); b = np.array(Sv, dtype=float).copy(); kk = np.asarray(k, float)
    if not fixed_root:
        if not (k0 > 0 or np.sum(np.asarray(Ev, float)) > 0):
            raise ValueError("Assumption A4 violated: no node has positive exposure and k0=0")
        a[0] = a[0] + k0; b[0] = b[0] + k0 * mu0
    for v in tree.post:
        if v == 0:
            continue
        u = par[v]; kv = kk[dep[v]]
        den = a[v] + kv
        a[u] = a[u] + kv * a[v] / den
        b[u] = b[u] + kv * b[v] / den
    theta = [None] * tree.n if exact else np.empty(tree.n)
    theta[0] = mu0 if fixed_root else b[0] / a[0]
    for v in tree.pre:
        if v == 0:
            continue
        kv = kk[dep[v]]
        theta[v] = (b[v] + kv * theta[par[v]]) / (a[v] + kv)
    return theta


def hce_objective(tree, Ev, Sv, theta, k, k0, mu0, S2E=None):
    """Objective J(θ) of Definition 3.3, omitting the constant Σ S_i^2/E_i if S2E=None."""
    th = np.asarray(theta, dtype=object if isinstance(theta[0], Fraction) else float)
    J = 0
    for v in range(tree.n):
        J += Ev[v] * th[v] ** 2 - 2 * Sv[v] * th[v]
        if v > 0:
            J += k[tree.depth[v]] * (th[v] - th[tree.parent[v]]) ** 2
    J += k0 * (th[0] - mu0) ** 2
    if S2E is not None:
        J += S2E
    return J


def hce_gradient(tree, Ev, Sv, theta, k, k0, mu0):
    th = np.asarray(theta, float); g = 2 * (np.asarray(Ev) * th - np.asarray(Sv))
    for v in range(1, tree.n):
        u = tree.parent[v]; d = 2 * k[tree.depth[v]] * (th[v] - th[u])
        g[v] += d; g[u] -= d
    g[0] += 2 * k0 * (th[0] - mu0)
    return g


def hce_dense(tree, Ev, Sv, k, k0, mu0):
    """Independent reference solver: builds the matrix H and solves directly (used for TK1)."""
    n = tree.n; H = np.diag(np.asarray(Ev, float)); b = np.asarray(Sv, float).copy()
    for v in range(1, n):
        u = tree.parent[v]; kv = k[tree.depth[v]]
        H[v, v] += kv; H[u, u] += kv; H[u, v] -= kv; H[v, u] -= kv
    H[0, 0] += k0; b[0] += k0 * mu0
    return np.linalg.solve(H, b), H


def hce_sequential(tree, Ev, Sv, k, k0, mu0):
    """Definition 3.4: sequential top-down shrinkage using cumulative subtree statistics."""
    Et = np.array(Ev, float).copy(); St = np.array(Sv, float).copy()
    for v in tree.post:
        if v:
            Et[tree.parent[v]] += Et[v]; St[tree.parent[v]] += St[v]
    th = np.empty(tree.n); th[0] = (St[0] + k0 * mu0) / (Et[0] + k0)
    for v in tree.pre:
        if v:
            kv = k[tree.depth[v]]
            th[v] = (St[v] + kv * th[tree.parent[v]]) / (Et[v] + kv)
    return th
