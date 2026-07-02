"""expAZ_wraploop.py — does (C-H)|_{-k} hold via a SINGLE WRAP LOOP (row/col)? A possible big simplification.

expAY revealed: the Karp witness can be a (1,1) diagonal, but the best single WRAP loop (one row or column)
still has sum_C log(2R_e) <= 0 -- (C-H) holds via a wrap loop, NOT requiring the staircase. The earlier
"S2 fails" was about the TOTAL S_h = sum over ALL rows (the average row); the MIN individual wrap loop is the
right, weaker object. CLAIM TO TEST:

   (W) for an all-ray Chern=-k line, min over the m rows + m cols of  sum_loop log(2R_e)  <=  0.

If (W) holds robustly (even pushed to the -k coherence ceiling), then (C-H)|_{-k} reduces to a 1-D magnetic-
RING statement per row/column -- vastly more tractable than the staircase. Crucially the Landau-gauge COLUMNS
carry UNIFORM flux theta_x=2pi k x/m^2 (constant in y), so a column is a clean 1-D magnetic ring.

This script pushes many -k lines to the coherence ceiling (the hard regime) and reports min_wrap = min single
wrap-loop sum log(2R), with WHICH loop (row/col, index) achieves it, and the worst (largest) min_wrap over all
seeds (want <= 0). Reuses expU/expAE/expAK/expAR/expAN. Writes wraploop.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expU = _load("expU", "expU_shadow.py"); expAE = _load("expAE", "expAE_pq.py")
expAK = _load("expAK", "expAK_loops.py"); expAR = _load("expAR", "expAR_sector.py")
expAN = _load("expAN", "expAN_wilson.py")


def wrap_sums(u, src, dst, phi, m):
    d = expAN.edge_data(u, src, dst, phi); L = d['log2R']; R = d['R']
    if not np.all(R > 0):
        return None
    def eidx(x, y, h): return 2*((x % m)*m + (y % m)) + (0 if h else 1)
    rows = np.array([sum(L[eidx(x, y, True)] for x in range(m)) for y in range(m)])   # horizontal wrap loops
    cols = np.array([sum(L[eidx(x, y, False)] for y in range(m)) for x in range(m)])   # vertical wrap loops
    # (W'): per-loop MEAN R (the LINEAR target; mean R<=1/2 => prod 2R<=1 by AM-GM)
    rowsMR = np.array([np.mean([R[eidx(x, y, True)] for x in range(m)]) for y in range(m)])
    colsMR = np.array([np.mean([R[eidx(x, y, False)] for y in range(m)]) for x in range(m)])
    min_loop_meanR = float(min(rowsMR.min(), colsMR.min()))
    allw = np.concatenate([rows, cols])
    i = int(np.argmin(allw))
    which = (f"row y={i}" if i < m else f"col x={i-m}")
    coh = min(R[0::2].mean(), R[1::2].mean())
    return float(allw.min()), which, float(coh), min_loop_meanR


def main():
    print("(W): min over rows+cols of sum log(2R) <= 0.   (W'): min over rows+cols of MEAN R <= 1/2 (LINEAR).\n")
    out = {"runs": []}
    global_worst = -np.inf; global_worst_mr = -np.inf
    for (m, k) in [(6, 2), (5, 2), (4, 2), (6, 1), (5, 1)]:
        worst = -np.inf; worst_mr = -np.inf; n = 0; ceilcoh = -np.inf; worst_detail = None
        for sd in range(16):
            np.random.seed(7000 + 31*sd + 100*k + m)
            u0, src, dst, phi = expAK.find_all_ray_line(m, k, tries=12)
            if u0 is None:
                continue
            ch0 = expAE.chern_of_u(m, u0)
            if ch0 is None or abs(ch0+k) > 0.25:
                continue
            _, up, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2200)
            for u in (u0, up):
                if u is None:
                    continue
                ch = expAE.chern_of_u(m, u)
                if ch is None or abs(ch+k) > 0.25:
                    continue
                r = wrap_sums(u, src, dst, phi, m)
                if r is None:
                    continue
                mn, which, coh, min_loop_meanR = r
                n += 1; ceilcoh = max(ceilcoh, coh)
                worst_mr = max(worst_mr, min_loop_meanR)               # want <= 1/2
                if mn > worst:
                    worst = mn; worst_detail = dict(min_wrap=round(mn, 3), which=which, coh=round(coh, 3),
                                                    chern=round(float(ch), 2), min_loop_meanR=round(min_loop_meanR, 3))
        if n == 0:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line"); continue
        global_worst = max(global_worst, worst); global_worst_mr = max(global_worst_mr, worst_mr)
        wd = worst_detail
        okW = worst <= 1e-9; okWp = worst_mr <= 0.5 + 1e-9
        print(f"m={m} k={k}: {n} lines, ceil coh {ceilcoh:.3f} | WORST min_wrap={worst:+.3f} -> (W){'OK' if okW else 'X'}"
              f" | WORST min-loop-meanR={worst_mr:.3f} -> (W'){'OK<=1/2' if okWp else ' X >1/2'}  ({wd['which']})")
        out["runs"].append(dict(m=m, k=k, n=n, max_coh=round(ceilcoh, 3), worst_min_wrap=round(worst, 4),
                                worst_min_loop_meanR=round(worst_mr, 4), W_holds=bool(okW), Wp_holds=bool(okWp),
                                worst_detail=wd))
    out["global_worst_min_wrap"] = round(float(global_worst), 4)
    out["global_worst_min_loop_meanR"] = round(float(global_worst_mr), 4)
    holds = global_worst <= 1e-9; holdsP = global_worst_mr <= 0.5 + 1e-9
    print(f"\nGLOBAL worst min_wrap = {global_worst:+.4f} -> (W) {'HOLDS' if holds else 'FAILS'}")
    print(f"GLOBAL worst min-loop-meanR = {global_worst_mr:.4f} -> (W') [linear] {'HOLDS (<=1/2)' if holdsP else 'FAILS (>1/2)'}")
    print("VERDICT:",
          "(W') HOLDS -- some single row/column has MEAN R <= 1/2: (C-H)|_-k reduces to a LINEAR 1-D magnetic-\n"
          "  ring bound (mean R<=1/2 => prod 2R<=1 by AM-GM). The cleanest target yet; directly the r=2 import." if holdsP else
          "(W') FAILS (>1/2) but (W) " + ("HOLDS" if holds else "FAILS") + " -- the wrap loop works via SPREAD\n"
          "  (prod 2R<=1 with mean R>1/2), not the linear mean bound. (W) is the right (non-linear) 1-D target.")
    with open(os.path.join(HERE, "wraploop.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote wraploop.json")


if __name__ == "__main__":
    main()
