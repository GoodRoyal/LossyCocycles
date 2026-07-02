"""expAC_mscale.py — DECISIVE m-scaling of the even-k isotropic (star)-margin: does the codim>=2 bound
hold (tight, margin->0^-) or FAIL (margin->0^+) for even k as m->infty?

Entry 45: even-k isotropic margin = -0.05 at m=8. But the bound distOpFlat>=1 is TIGHT (c=1, the unwound
competitor sits at PB=1), so a winding competitor approaching PB=1 from above (margin->0^-) is the EXPECTED
signature of a bound that HOLDS. The bound FAILS only if margin crosses to strictly POSITIVE (PB strictly
<1). The naive smooth-gauge heuristic (psi_e=O(k/m)->0, overlaps->1, margin->+0.7) predicts FAILURE; the
m=8 numerics (-0.05) are inconclusive on the sign in the limit. This sweeps m to settle it.

Two measurements per (k,m):
  (A) isotropic (star)-margin via the fast spinor+rho ascent (expAB.ascend), best over restarts & degrees,
      + the REAL PB of alpha=rho e(zeta) for the flux-k family (cross-check margin<0 <=> PB>=1).
  (B) FULL-shadow min PB (expU.minimize_pb, no isotropy restriction) as ground truth on the bound.
Verdict: margin(m) and (PB(m)-1) trends. ->0^- with PB>=1 == bound HOLDS tight; PB<1 anywhere == FAILS.
Reuses expAB, expU. Writes mscale.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
expAB = _load("expAB", "expAB_evenk.py")


def best_iso_margin(m, k, restarts=6):
    """best (star)-margin over restarts & degree sectors; return (best_margin, real_PB_at_best, deg_at_best)."""
    best = -np.inf; bestpb = None; bestdeg = None
    for sd in range(restarts):
        rng = np.random.default_rng(7000 + 31 * k + sd)
        Z0 = rng.standard_normal((m * m, 2)) + 1j * rng.standard_normal((m * m, 2))
        Z0 /= np.linalg.norm(Z0, axis=1, keepdims=True)
        rho0 = rng.uniform(0.5, np.sqrt(2), m * m)
        mg, Zf, rhof = expAB.ascend(m, k, Z0, rho0, steps=1500)
        if mg > best:
            best = mg; bestpb = expAB.real_pb(m, k, Zf, rhof); bestdeg = expAB.zeta_chern(m, Zf)
    return best, bestpb, bestdeg


def full_min_pb(m, k, r=3, restarts=4):
    """ground-truth full-shadow min PB (no isotropy restriction)."""
    best = np.inf
    for sd in range(restarts):
        A0 = expU.proj_cap(np.random.default_rng(5000 + 13 * k + sd).standard_normal((m * m, 2, r)))
        mp, _ = expU.minimize_pb(m, k, r, A0, steps=1200)
        best = min(best, mp)
    return best


def main():
    out = {}
    print("EVEN-k (k=2) m-scaling.  margin<0 & PB>=1 => bound HOLDS (tight); margin>0 / PB<1 => FAILS.\n")
    print(f"{'k':>3} {'m':>4} {'iso (star)-margin':>18} {'real PB(iso)':>13} {'deg':>5} {'full minPB':>11}")
    for k in (2, 1):                                   # k=2 is the target; k=1 as the (odd, proven) control
        for m in (6, 8, 12, 16):
            mg, pb, deg = best_iso_margin(m, k)
            fpb = full_min_pb(m, k, restarts=3) if m <= 12 else None   # full minimizer is slow; cap at m=12
            ds = f"{deg:+.1f}" if deg is not None else "und"
            fs = f"{fpb:.4f}" if fpb is not None else "  (skip)"
            print(f"{k:>3} {m:>4} {mg:>+18.4f} {pb:>13.4f} {ds:>5} {fs:>11}")
            out[f"k{k}_m{m}"] = dict(iso_margin=round(float(mg), 4), iso_pb=round(float(pb), 4),
                                    deg=(round(float(deg), 2) if deg is not None else None),
                                    full_min_pb=(round(float(fpb), 4) if fpb is not None else None))
        print()
    # verdict on the even-k trend
    mk2 = [(int(kk.split('_m')[1]), v["iso_margin"], v["iso_pb"]) for kk, v in out.items() if kk.startswith("k2_m")]
    mk2.sort()
    print("k=2 trend (m, margin, PB):", [(m, mg, pb) for m, mg, pb in mk2])
    pos = any(mg > 1e-3 or pb < 0.999 for _, mg, pb in mk2)
    if pos:
        print("VERDICT: even-k margin crosses POSITIVE / PB<1 -> codim>=2 bound FAILS for even k. MAJOR.")
    else:
        trend = "rising to 0^-" if mk2[-1][1] > mk2[0][1] else "flat/!rising"
        print(f"VERDICT: even-k margin stays <0 and PB>=1 (margin {trend}); bound HOLDS, tight at c=1. "
              f"largest m margin={mk2[-1][1]:+.4f}, PB={mk2[-1][2]:.4f}.")
    with open(os.path.join(HERE, "mscale.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote mscale.json")


if __name__ == "__main__":
    main()
