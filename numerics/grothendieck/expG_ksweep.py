"""expG_ksweep.py — direct distOpFlat (Exp G) at k = 1, 2, 3 (the head-on c(k) test).

`00-orientation-and-open-frontier.md` §8b flags two "Next" steps to decide whether the
binding separation c(k) genuinely decays with charge: (1) a cleaner geodesic homotopy
(`expC_geodesic.py`), and (2) a k-resolved Exp G. This is (2).

Exp G minimizes the CONJECTURED quantity itself,
    distOpFlat(flux) = inf_{T' flat} sup_e ||flux(e) - T'(e)||_2,
by Riemannian descent over the product of site-frame Stiefel manifolds (the engine lives in
`coarse-geometry-numerics.py`, `_distflat_minimize`). It is an UPPER bound on distOpFlat, so
it can only falsify: if the best competitor -> 0 as m grows, Plateau is false for that k.
Paper 5 ran k=1 only (best competitor rises to sqrt2 from below, no decay). Here we run
k=1 (control, should reproduce the paper) and k=2,3 with extra steps/restarts for robustness.

Reads the engine via importlib (the filename has hyphens). Writes distOpFlat_ksweep.json.
  python3 expG_ksweep.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

HERE = os.path.dirname(os.path.abspath(__file__))
M_LIST = (4, 6, 8, 12, 16, 24, 32)
K_LIST = (1, 2, 3)
D, R = 3, 2
STEPS = 400          # paper used 300; bump for robustness given the time budget
RESTARTS = 5         # paper used 3 (+ S-init); more restarts = tighter upper bound
SQRT2 = np.sqrt(2.0)


def best_distflat(m, k):
    """Best (smallest) sup_e||flux-T'|| over the S-init run + RESTARTS random restarts.
    Returns (V=S baseline, best competitor)."""
    base = cgn._distflat_minimize(m, k, D, R, steps=1, beta0=1, beta1=1, lr0=0,
                                  seed=0, V_init="S")            # objective at V=S
    best = cgn._distflat_minimize(m, k, D, R, STEPS, 5.0, 60.0, 0.5, 0, V_init="S")
    for rs in range(RESTARTS):
        best = min(best, cgn._distflat_minimize(m, k, D, R, STEPS, 5.0, 60.0, 0.5,
                                                seed=10 + rs))
    return float(base), float(best)


def main():
    print(f"steps={STEPS} restarts={RESTARTS}+S-init   d={D} r={R}")
    print(f"{'k':>3} {'m':>4} {'V=S base':>10} {'best distOpFlat':>16} {'/sqrt2':>8} "
          f"{'sec':>6}")
    out = {}
    for k in K_LIST:
        bests = {}
        for m in M_LIST:
            t0 = time.time()
            base, best = best_distflat(m, k)
            bests[str(m)] = round(best, 4)
            print(f"{k:>3} {m:>4} {base:>10.4f} {best:>16.4f} {best / SQRT2:>8.3f} "
                  f"{time.time() - t0:>6.1f}", flush=True)
        finite = list(bests.values())
        # the binding floor is the SMALLEST best-competitor distance over m (inf_m)
        inf_m = min(finite)
        out[k] = {"per_m": bests, "inf_over_m": round(inf_m, 4),
                  "max_over_m": round(max(finite), 4)}
        print(f"    -> k={k}: inf_m best = {inf_m:.4f} ({inf_m / SQRT2:.3f} sqrt2); "
              f"sup_m best = {max(finite):.4f}; decay with m? "
              f"{'NO (rises/flat)' if bests[str(M_LIST[-1])] >= bests[str(M_LIST[0])] else 'CHECK'}\n",
              flush=True)
    with open(os.path.join(HERE, "distOpFlat_ksweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote distOpFlat_ksweep.json")


if __name__ == "__main__":
    main()
