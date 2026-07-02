"""expBC_variance_bound.py -- the next target: all-ray Chern -k ==> some ring has var0 >= c (c m-uniform).

Entry 64 found the uniform-mode VARIANCE var0 = <0|H^2|0> - <0|H|0>^2 is the m-uniform, discriminating carrier
of (C-H)|_{-k} (where mean R / (W') degrades). Target now: show the Chern FORCES var0 large on SOME ring,
m-uniformly. Contrapositive: all rings low-variance (coherent) => Chern 0. This script pins the mechanism on
genuine all-ray Chern -k lines pushed to the coherence ceiling (the minimal-variance, hardest in-sector states):

  (1) THE CONSTANT c.  c(m) := min over lines of ( max over rings of var0 ).  The target wants c(m) >= c_0 > 0
      m-uniformly (NOT slowly decaying like the (W') margin did, 0.285->0.143).
  (2) CONCENTRATION.  Is var0 carried by O(1) rings? Participation ratio P = (sum v)^2 / (sum v^2) in [1, 2m];
      P ~ O(1) => concentrated on a few rings (the variance-level analogue of Entry 60's single wrap loop);
      P ~ 2m  => spread (then a max-ring bound would be 1/m-thin -- bad). Also report the SUM rule sum_rings var0.
  (3) MECHANISM.  Pooled correlation of a ring's var0 with: |Berry holonomy| |Theta-flux|, flux frustration,
      and magnitude deficit (1 - Wilson|p|). Which one drives the variance? (Entry 63: flux alone only 50%.)

All quantities read from expBA's per-ring dict (var0, Theta, frustr, Wilson, meanR). Writes variance_bound.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBA = _load("expBA", "expBA_ringspectrum.py")


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("expBC -- target: all-ray Chern -k => some ring has var0 >= c (m-uniform). Constant / concentration / mechanism.\n")
    out = {"per_mk": []}
    c_of_m = {}
    for (m, k) in [(5, 2), (6, 2), (7, 2)]:
        lines = expBA.collect_lines(m, k)
        if not lines:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line"); continue
        maxring_var = []      # per line: max over rings of var0  (-> c = min of these)
        sumvar = []           # per line: sum over rings of var0  (sum rule)
        partratio = []        # per line: participation ratio of the var0 distribution
        # pooled per-ring data for mechanism correlations
        pv = []; pBerry = []; pfrustr = []; pdef = []
        worst_line_dist = None; worst_maxring = np.inf
        for ch, rings in lines:
            v = np.array([r["var0"] for r in rings])
            mr = float(v.max()); maxring_var.append(mr); sumvar.append(float(v.sum()))
            P = float((v.sum()**2) / max((v**2).sum(), 1e-30)); partratio.append(P)
            if mr < worst_maxring:            # the line that MINIMIZES max-ring var0 = the stressing case for c
                worst_maxring = mr; worst_line_dist = np.sort(v)[::-1]
            for r in rings:
                pv.append(r["var0"])
                pBerry.append(abs(r["Theta"]))            # |combined holonomy| (Berry+flux); proxy for connection energy
                pfrustr.append(r["frustr"])
                pdef.append(1.0 - r["Wilson"])            # magnitude deficit 1 - prod|p|
        c = float(np.min(maxring_var)); c_of_m[m] = c
        Pmean = float(np.mean(partratio)); n_rings = 2*m
        print(f"=== m={m} k={k}: {len(lines)} lines, {n_rings} rings/line ===")
        print(f"  (1) c(m)=min_line max_ring var0 = {c:.3f}   [target: >= c0>0 m-uniform]   "
              f"(median max-ring var0 = {np.median(maxring_var):.3f})")
        print(f"  (2) participation ratio P (mean) = {Pmean:.2f} / {n_rings}  "
              f"=> {'CONCENTRATED on ~%.1f rings' % Pmean if Pmean < 0.5*n_rings else 'SPREAD'};  "
              f"sum_rings var0 (mean) = {np.mean(sumvar):.2f}")
        print(f"      stressing-line var0 spectrum (sorted): {np.round(worst_line_dist, 3).tolist()}")
        print(f"  (3) mechanism corr(var0, .): |holonomy|={pearson(pv,pBerry):+.2f}  "
              f"frustr={pearson(pv,pfrustr):+.2f}  magdeficit(1-|W|)={pearson(pv,pdef):+.2f}")
        out["per_mk"].append(dict(m=m, k=k, n_lines=len(lines), c=round(c, 4),
                                  median_maxring_var0=round(float(np.median(maxring_var)), 4),
                                  participation_mean=round(Pmean, 3), n_rings=n_rings,
                                  sum_var0_mean=round(float(np.mean(sumvar)), 3),
                                  corr_holonomy=round(pearson(pv, pBerry), 3),
                                  corr_frustr=round(pearson(pv, pfrustr), 3),
                                  corr_magdeficit=round(pearson(pv, pdef), 3),
                                  stressing_var0_spectrum=np.round(worst_line_dist, 4).tolist()))
        print()

    print("--- m-uniformity of the target constant c(m) ---")
    seq = "  ".join(f"m={mm}:c={c_of_m[mm]:.3f}" for mm in sorted(c_of_m))
    if len(c_of_m) >= 2:
        ms = sorted(c_of_m); trend = c_of_m[ms[-1]] - c_of_m[ms[0]]
        verdict = ("STABLE/positive => var0 looks m-uniform (target plausible)" if abs(trend) < 0.15 and min(c_of_m.values()) > 0.2
                   else "DRIFTS -- need more m to judge")
    else:
        verdict = "only one m available"
    print(f"  {seq}    [{verdict}]")
    out["c_of_m"] = {int(mm): round(c_of_m[mm], 4) for mm in c_of_m}
    with open(os.path.join(HERE, "variance_bound.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote variance_bound.json")


if __name__ == "__main__":
    main()
