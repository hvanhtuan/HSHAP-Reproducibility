"""H-SHAP (Definition 3.9 — two-tier Owen value), exact Shapley and a sampled version (Algorithms 3.5 and 3.6).
The value function is given as a table nu[..., mask], where mask is a bitmask over the p players (nu[..., 0] = 0 after normalisation)."""
import numpy as np
from math import factorial
from itertools import combinations


def _subsets(bits):
    """Enumerate all subsets (bitmasks) of a list of bits."""
    out = [0]
    for b in bits:
        out += [m | (1 << b) for m in out]
    return out


def shapley_from_table(nu, p):
    """nu: array (..., 2^p). Returns φ (..., p)."""
    w = np.array([factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)])
    pc = np.array([bin(m).count("1") for m in range(1 << p)])
    phi = np.zeros(nu.shape[:-1] + (p,))
    allm = np.arange(1 << p)
    for j in range(p):
        T = allm[(allm >> j) & 1 == 0]
        phi[..., j] = ((nu[..., T | (1 << j)] - nu[..., T]) * w[pc[T]]).sum(-1)
    return phi


def owen_from_table(nu, p, groups):
    """groups: list of lists of feature indices (a partition). Returns (ψ (..., p), Φ (..., G))."""
    G = len(groups)
    gmask = [sum(1 << j for j in g) for g in groups]
    psi = np.zeros(nu.shape[:-1] + (p,)); Phi = np.zeros(nu.shape[:-1] + (G,))
    for g, Bg in enumerate(groups):
        others = [h for h in range(G) if h != g]
        bg = len(Bg)
        for r in range(G):
            for R in combinations(others, r):
                cR = factorial(r) * factorial(G - r - 1) / factorial(G)
                QR = sum(gmask[h] for h in R)
                Phi[..., g] += cR * (nu[..., QR | gmask[g]] - nu[..., QR])
                for j in Bg:
                    rest = [x for x in Bg if x != j]
                    for t in range(bg):
                        cT = factorial(t) * factorial(bg - t - 1) / factorial(bg)
                        for T in combinations(rest, t):
                            Tm = sum(1 << x for x in T)
                            psi[..., j] += cR * cT * (nu[..., QR | Tm | (1 << j)] - nu[..., QR | Tm])
    return psi, Phi


def owen_sample(nu_fn, p, groups, R_mc, rng):
    """Algorithm 3.6: permutations respecting the block structure. nu_fn(mask)->float. Returns ψ̂, Φ̂, s^2."""
    sm = np.zeros(p); sq = np.zeros(p)
    for _ in range(R_mc):
        order = []
        for g in rng.permutation(len(groups)):
            order += list(rng.permutation(groups[g]))
        m = 0; u = nu_fn(0)
        for j in order:
            m |= 1 << j; u2 = nu_fn(m); d = u2 - u
            sm[j] += d; sq[j] += d * d; u = u2
    psi = sm / R_mc; s2 = (sq - R_mc * psi ** 2) / (R_mc - 1)
    Phi = np.array([psi[g].sum() for g in groups])
    return psi, Phi, s2


def masked_rows(x, ref, p):
    """Generate 2^p rows: features in mask keep the value of x, the others take the reference value ref.
    x, ref: object arrays of length p. Returns an array (2^p, p)."""
    M = np.arange(1 << p)
    keep = ((M[:, None] >> np.arange(p)[None, :]) & 1).astype(bool)
    X = np.where(keep, np.asarray(x, dtype=object)[None, :], np.asarray(ref, dtype=object)[None, :])
    return X
