"""expQ_pythag.py — the Pythagorean / Pi-restricted bound: is min_competitor max_e ||(flux-T')P_Pi|| >= 1?

The seam (intermediate beta) is where the far bound (delta>=beta) and near bound (delta>=rho_m-beta^2)
both fall below 1. The Pythagorean split, for a unit u in the rotating plane Pi:
    ||(flux(e)-T'(e))u||^2 = ||R(phi_e)u - P_Pi T'(e)u||^2 + ||P_{Pi^perp} T'(e)u||^2,
so sup over u in Pi gives the Pi-RESTRICTED operator distance
    PB := max_e || (flux(e)-T'(e)) P_Pi ||_2   (top singular value of the d x 2 restriction).
Restricting inputs only lowers a norm, so  delta >= PB  RIGOROUSLY.  Hence if
    min over flat competitors of PB  >= 1,
then delta >= 1 and the codim>=2 lower bound is CLOSED (the seam too).

This script MINIMIZES PB over the flat-competitor frames (projected gradient, softmax over edges,
top-singular-value gradient on the d x 2 restriction), multi-restart + orthogonal init, and reports
min PB vs 1. Decisive: min PB >= 1 closes (a); min PB < 1 means the Pi-restricted bound is not
enough (the closing competitor beats flux on Pi-inputs but pays off-Pi). Reuses cgn. Writes pythag.json.
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


def min_PB(m, k, d, r, steps=500, seed=0, V_init=None):
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf2 = np.stack([(S @ cgn.rot_core(phi, r) @ S.T)[:, :2] for (_, _, phi) in E])  # (E,d,2)
    rng = np.random.default_rng(seed)
    if V_init == "W":
        W = np.zeros((d, r));
        for j, c in enumerate(range(d - r, d)): W[c, j] = 1.0
        Vs = np.broadcast_to(W, (m * m, d, r)).copy()
    elif V_init == "S":
        Vs = np.broadcast_to(S, (m * m, d, r)).copy()
    else:
        Vs = cgn._polar(rng.standard_normal((m * m, d, r)))
    best = np.inf
    for t in range(steps):
        beta = 5 + 70 * t / (steps - 1); lr = 0.4 * (1 - 0.9 * t / (steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M2 = Tf2 - np.einsum('eij,ekj->eik', Vd, Vsr[:, :2, :])    # (E,d,2)
        U_, Sg, Vt_ = np.linalg.svd(M2, full_matrices=False)        # U_(E,d,2), Sg(E,2), Vt(E,2,2)
        sig = Sg[:, 0]; u = U_[:, :, 0]; v = Vt_[:, 0, :]           # (E,), (E,d), (E,2)
        best = min(best, float(sig.max()))
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        vVsr2 = np.einsum('ek,ekj->ej', v, Vsr[:, :2, :])           # (E,r) = v^T Vsr2
        uVd = np.einsum('ed,edj->ej', u, Vd)                        # (E,r) = u^T Vd
        gVd = -(p[:, None, None]) * np.einsum('ed,ej->edj', u, vVsr2)         # (E,d,r)
        gVsr = np.zeros_like(Vsr)
        gVsr[:, :2, :] = -(p[:, None, None]) * np.einsum('ek,ej->ekj', v, uVd)  # rows 0,1
        grad = np.zeros_like(Vs)
        np.add.at(grad, dst, gVd)
        np.add.at(grad, src, gVsr)
        Vs = cgn._polar(Vs - lr * grad)
    return best


def main():
    out = {}
    print(f"{'d':>3} {'r':>3} {'m':>3} {'min PB (Pi-restricted distance)':>32} {'>=1?':>5}")
    allok = True
    for (d, r) in [(4, 2), (6, 3)]:
        for m in (4, 6, 8):
            vals = [min_PB(m, 1, d, r, V_init="S"), min_PB(m, 1, d, r, V_init="W")]
            for s in range(5):
                vals.append(min_PB(m, 1, d, r, seed=10 + s))
            mp = min(vals)
            ok = mp >= 0.999; allok = allok and ok
            out[f"d{d}r{r}_m{m}"] = round(mp, 4)
            print(f"{d:>3} {r:>3} {m:>3} {mp:>32.4f} {str(ok):>5}")
        print()
    print("min PB >= 1 in all cases:", allok,
          "\n=> if so, delta >= PB >= 1 RIGOROUSLY closes the codim>=2 lower bound (incl. the seam)."
          if allok else
          "\n=> some min PB < 1: the Pi-restricted bound alone does NOT close it; closing competitor pays off-Pi.")
    with open(os.path.join(HERE, "pythag.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote pythag.json")


if __name__ == "__main__":
    main()
