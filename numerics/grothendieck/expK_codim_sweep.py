"""expK_codim_sweep.py — Q5 trend: does the distOpFlat floor slide to 0 as codimension grows?

expJ found (codim 1 -> 2, r=2): coreGap=1 and ideal leak=0 are codimension-invariant, but the
small-m distOpFlat floor DROPS (m=4: 1.244 -> 1.082) as the extra ambient dimensions give flat
competitors more off-support escape room. Two points can't tell whether inf_m -> a positive limit
or -> 0. This sweeps r=2, d=3,4,5,6 (codimension d-r = 1,2,3,4) to read the trend of the binding
small-m floor, and confirms the m-transience (-> sqrt2 at large m) survives at each codim.

distOpFlat is an UPPER bound (Riemannian descent over the flat-competitor site frames), so it can
only falsify; a floor that stays bounded away from 0 is consistent with (not proof of) Plateau.
coreGap is checked at a few m to confirm G2 stays 1. Reuses coarse-geometry-numerics.py.
Writes codim_trend.json.
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
M_LIST = (4, 6, 8, 12, 16)
M_COREGAP = (4, 8)          # coreGap is m-uniform; a couple of m suffice (avoids big SVDs)
R = 2
D_LIST = (3, 4, 5, 6)
K = 1
SQRT2 = np.sqrt(2.0)


def coregap(m, d, r):
    Th, _ = cgn.flux_edge_maps(m, K, d, r)
    U = cgn.big_U(m, Th, d)
    _, sig = cgn.polar_unitary_core(U, m * m * r)
    return float(sig[m * m * r - 1])


def best_distflat(m, d, r, steps=400, restarts=6):
    best = cgn._distflat_minimize(m, K, d, r, steps, 5.0, 60.0, 0.5, 0, V_init="S")
    for rs in range(restarts):
        best = min(best, cgn._distflat_minimize(m, K, d, r, steps, 5.0, 60.0, 0.5, seed=40 + rs))
    return float(best)


def main():
    out = {}
    print(f"{'d':>3} {'codim':>5} {'coreGap':>9} | "
          + " ".join(f"m={m:<5}" for m in M_LIST) + "  inf_m  (/sqrt2)")
    for d in D_LIST:
        cg = min(coregap(m, d, R) for m in M_COREGAP)        # worst-case over checked m
        bds = {m: best_distflat(m, d, R) for m in M_LIST}
        inf_m = min(bds.values())
        out[f"d{d}r{R}"] = dict(codim=d - R, coreGap=round(cg, 4),
                                distOpFlat={str(m): round(v, 4) for m, v in bds.items()},
                                inf_over_m=round(inf_m, 4), m4=round(bds[4], 4))
        row = " ".join(f"{bds[m]:<7.4f}" for m in M_LIST)
        print(f"{d:>3} {d - R:>5} {cg:>9.4f} | {row}  {inf_m:.4f} ({inf_m / SQRT2:.3f})",
              flush=True)

    print("\n=== binding small-m floor vs codimension (the Q5 trend) ===")
    print(f"{'codim':>6} {'distOpFlat(m=4)':>16} {'inf_m':>9}")
    m4s = []
    for d in D_LIST:
        r = out[f"d{d}r{R}"]
        m4s.append(r["m4"])
        print(f"{d - R:>6} {r['m4']:>16.4f} {r['inf_over_m']:>9.4f}")
    # crude trend read: successive drops in the m=4 floor
    drops = [round(m4s[i] - m4s[i + 1], 4) for i in range(len(m4s) - 1)]
    print(f"successive m=4 drops (codim 1->2->3->4): {drops}")
    print("read:",
          "drops are SHRINKING -> floor likely tends to a positive limit (Q5 ok)"
          if all(drops[i] >= drops[i + 1] - 1e-3 for i in range(len(drops) - 1)) and m4s[-1] > 0.5
          else "drops not clearly shrinking / floor low -> codim-dependence of c(k) is real, watch")
    with open(os.path.join(HERE, "codim_trend.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote codim_trend.json")


if __name__ == "__main__":
    main()
