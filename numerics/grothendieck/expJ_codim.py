"""expJ_codim.py — Q5: does the separation survive higher codimension (d=4, r=2)?

Every lossy experiment so far fixed d=3, r=2 (codimension d-r = 1). The conjecture ranges over
2<=r<=d. This runs the decisive quantities at codim 2 (d=4, r=2) against the d=3 control:

  coreGap   = sigma_min on the surviving core (G2; should stay 1 -- rotations are core isometries)
  leak      = ||(1-P)[U,V]P||  (G4; ideal should be 0, wobble O(eps))
  distOpFlat= best flat competitor (Exp G; the conjectured quantity -- does the floor survive?)

The sharp worry, connecting to expI: more codimension = MORE off-support escape room for a flat
competitor, so distOpFlat could DROP (the cheap channel widens). Q5 asks whether it stays bounded
away from 0 and m-uniform anyway. Reuses coarse-geometry-numerics.py. Writes codim_sweep.json.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
from scipy.linalg import svd

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

HERE = os.path.dirname(os.path.abspath(__file__))
M_LIST = (4, 6, 8, 12, 16)
K = 1
EPS = 1e-2
SQRT2 = np.sqrt(2.0)
op = lambda A: float(np.linalg.norm(A, 2))


def coregap_and_leak(m, d, r):
    """G2 coreGap = sigma_min(core block of U); G4 leak ||(1-P)[U,V]P|| ideal + wobble."""
    Th, Tv = cgn.flux_edge_maps(m, K, d, r)
    U = cgn.big_U(m, Th, d); V = cgn.big_V(m, Tv, d)
    rtot = m * m * r
    _, sig = cgn.polar_unitary_core(U, rtot)
    coregap = float(sig[rtot - 1])
    P = cgn.core_proj(m, d, r); Q = np.eye(P.shape[0]) - P
    leak_ideal = op(Q @ (U @ V - V @ U) @ P)
    rng = np.random.default_rng(7)
    Thw, Tvw = cgn.flux_edge_maps_tilted(m, K, EPS, d, r, rng=rng)
    Uw = cgn.big_U(m, Thw, d); Vw = cgn.big_V(m, Tvw, d)
    leak_wob = op(Q @ (Uw @ Vw - Vw @ Uw) @ P)
    return coregap, leak_ideal, leak_wob / EPS


def best_distflat(m, d, r, steps=400, restarts=5):
    best = cgn._distflat_minimize(m, K, d, r, steps, 5.0, 60.0, 0.5, 0, V_init="S")
    for rs in range(restarts):
        best = min(best, cgn._distflat_minimize(m, K, d, r, steps, 5.0, 60.0, 0.5, seed=30 + rs))
    return float(best)


def main():
    out = {}
    for (d, r) in [(3, 2), (4, 2)]:
        label = f"d={d},r={r} (codim {d - r})"
        print("=" * 78)
        print(f"{label}{'  [CONTROL]' if d == 3 else '  [TEST: codimension 2]'}")
        print("=" * 78)
        print(f"{'m':>4} {'coreGap':>9} {'leak ideal':>11} {'leak/eps(wob)':>13} "
              f"{'best distOpFlat':>16} {'/sqrt2':>8}")
        rows = {}
        for m in M_LIST:
            cg, li, lw = coregap_and_leak(m, d, r)
            bd = best_distflat(m, d, r)
            rows[str(m)] = dict(coreGap=round(cg, 4), leak_ideal=round(li, 6),
                                leak_over_eps=round(lw, 4), distOpFlat=round(bd, 4))
            print(f"{m:>4} {cg:>9.4f} {li:>11.2e} {lw:>13.4f} {bd:>16.4f} {bd / SQRT2:>8.3f}",
                  flush=True)
        out[f"d{d}r{r}"] = rows
        bds = [v["distOpFlat"] for v in rows.values()]
        print(f"    -> distOpFlat: inf_m = {min(bds):.4f}, sup_m = {max(bds):.4f}, "
              f"{'rises (no decay)' if bds[-1] >= bds[0] else 'CHECK: drops with m'}\n")
    # compare codim 1 vs 2
    d3 = [out['d3r2'][str(m)]['distOpFlat'] for m in M_LIST]
    d4 = [out['d4r2'][str(m)]['distOpFlat'] for m in M_LIST]
    print("codim 1 vs 2  distOpFlat per m:", dict(zip(M_LIST, zip(d3, d4))))
    print("verdict:",
          "codim-2 floor holds and is m-uniform (Q5 survives)"
          if min(d4) > 0.5 and d4[-1] >= d4[0] - 0.05 else
          "codim-2 distOpFlat behaves differently -- inspect")
    with open(os.path.join(HERE, "codim_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote codim_sweep.json")


if __name__ == "__main__":
    main()
