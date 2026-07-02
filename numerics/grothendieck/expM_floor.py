"""expM_floor.py — is distOpFlat EXACTLY 1 for codim>=2 (lower bound target), and k-independent?

expL gave the upper bound distOpFlat <= 1 for codim>=2 (the constant orthogonal competitor, an
exact hand construction). To turn c(k)=1 into a theorem we need the matching LOWER bound
distOpFlat >= 1. Before proving it, validate the target two ways:

  (1) k-independence (this is task (b)). The orthogonal competitor W W^T does not involve the flux
      angle, so sup_e ||flux_k(e) - W W^T|| = 1 for EVERY k -> distOpFlat <= 1 for all k at codim>=2.
      Confirmed here directly for k=1,2,3.

  (2) Is 1 a true floor? The decisive probe: WARM-START the Riemannian descent FROM the orthogonal
      competitor and let it try to go below 1. If it cannot (stays at ~1), the orthogonal competitor
      is (at least a strong local) optimum and distOpFlat = 1 is the right target. We also run a
      cold heavy search as a cross-check. If ANY run dips below 1, the c(k)=1 story is wrong.

Codim>=2 cases: (d=4,r=2) and (d=6,r=3). Reuses coarse-geometry-numerics.py. Writes floor_probe.json.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(4, 2), (6, 3)]        # codim 2 and 3
K_LIST = (1, 2, 3)
M_LIST = (4, 6, 8, 12)
op = lambda A: float(np.linalg.norm(A, 2))


def W_frame(d, r):
    """The orthogonal competitor frame: top r coords, avoiding the rotating 2-plane {0,1}."""
    W = np.zeros((d, r))
    for j, c in enumerate(range(d - r, d)):
        W[c, j] = 1.0
    return W


def orthogonal_ub(m, k, d, r):
    S = cgn.std_frame(d, r); WWt = W_frame(d, r) @ W_frame(d, r).T
    phh, phv = cgn.flux_angles(m, k)
    worst = 0.0
    for x in range(m):
        for y in range(m):
            for phi in (phh[x, y], phv[x, y]):
                worst = max(worst, op(S @ cgn.rot_core(phi, r) @ S.T - WWt))
    return worst


def descent(m, k, d, r, Vs, steps=600):
    """Riemannian descent on the flat-competitor frames from a given init Vs; returns best sup_e."""
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf = np.stack([S @ cgn.rot_core(phi, r) @ S.T for (_, _, phi) in E])
    best = np.inf
    for t in range(steps):
        beta = 5.0 + 75.0 * t / (steps - 1)
        lr = 0.5 * (1.0 - 0.9 * t / (steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M = Tf - np.einsum('eij,ekj->eik', Vd, Vsr)
        U_, Sg, Vt_ = np.linalg.svd(M)
        sig = Sg[:, 0]; u = U_[:, :, 0]; w = Vt_[:, 0, :]
        best = min(best, float(sig.max()))
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        Vsrtw = np.einsum('edr,ed->er', Vsr, w)
        Vdtu = np.einsum('edr,ed->er', Vd, u)
        gVd = -(p[:, None, None]) * np.einsum('ed,er->edr', u, Vsrtw)
        gVs = -(p[:, None, None]) * np.einsum('ed,er->edr', w, Vdtu)
        grad = np.zeros_like(Vs)
        np.add.at(grad, dst, gVd)
        np.add.at(grad, src, gVs)
        Vs = cgn._polar(Vs - lr * grad)
    return best


def main():
    out = {}
    any_below = False
    print(f"{'d':>3} {'r':>3} {'codim':>5} {'k':>3} {'m':>4} {'orth UB':>8} "
          f"{'warm-from-W':>12} {'cold(6 rs)':>11} {'<1?':>4}")
    for (d, r) in CASES:
        Wb = np.broadcast_to(W_frame(d, r), (1, d, r))
        for k in K_LIST:
            for m in M_LIST:
                ub = orthogonal_ub(m, k, d, r)
                Vs0 = np.broadcast_to(W_frame(d, r), (m * m, d, r)).copy()
                warm = descent(m, k, d, r, Vs0)
                cold = np.inf
                for s in range(6):
                    Vr = cgn._polar(np.random.default_rng(s).standard_normal((m * m, d, r)))
                    cold = min(cold, descent(m, k, d, r, Vr))
                below = min(warm, cold) < 0.999
                any_below = any_below or below
                out[f"d{d}r{r}_k{k}_m{m}"] = dict(orth_ub=round(ub, 4),
                                                  warm=round(warm, 4), cold=round(cold, 4))
                print(f"{d:>3} {r:>3} {d - r:>5} {k:>3} {m:>4} {ub:>8.4f} "
                      f"{warm:>12.4f} {cold:>11.4f} {str(below):>4}", flush=True)
    print("\nverdict:",
          "SOME competitor beat 1 -> distOpFlat < 1, c(k)=1 story WRONG" if any_below else
          "nothing beats 1 (warm-start from W cannot descend below it) -> distOpFlat = 1 is the "
          "right target; floor is k-INDEPENDENT (orth UB = 1 for all k).")
    with open(os.path.join(HERE, "floor_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote floor_probe.json")


if __name__ == "__main__":
    main()
