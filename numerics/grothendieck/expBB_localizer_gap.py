"""expBB_localizer_gap.py -- swap the uniform TEST vector for the true ground state: is the localizer GAP a
sharper, m-uniform certificate of (W) than the uniform-mode quadratic form? (Entry 64; follows expBA/Entry 63.)

Entry 63 wrote (W') as the uniform-mode quadratic form  2*meanR = <0|H_ring|0> <= 1  (|0> the uniform plane
wave; H_ring[s+1,s]=z_e=e^{i phi_e}p_e). The uniform mode is a VARIATIONAL TEST vector. Here we diagonalize
H_ring fully and compare candidate spectral certificates, on genuine all-ray Chern -k lines at growing m, and
-- the part the uniform-mode test cannot do -- check each candidate's DISCRIMINATION (worst/frustrated ring vs
best/coherent ring on the SAME line) and its m-UNIFORMITY (does the margin survive as m grows? Entry 58: the
all-ray slack g(m) decays toward 0, so the (W') margin is suspected to vanish).

Candidate certificates per ring (all from spec(H_ring) = ev):
  C0  uniform-mode  u = 2*meanR = <0|H|0>           [(W'), current]  bad-ring test: u <= 1
  C1  true ground   lmin = min ev                                     (variational LOWER bound on u)
  C2  band top      lmax = max ev ; deficit 2 - lmax                  (coherent ring has lmax=2)
  C3  localizer gap g = min |ev|  (closest eigenvalue to 0)           (the Loring gap notion)
  C4  spectral spread / Dirichlet energy:  var0 = <0|H^2|0> - <0|H|0>^2  (Entry 62: psi-variance carries it)
  C5  loop energy  E = sum (-log 2R_e)  (>=0 on a bad ring)           [(W), nonlinear]
We report, per (m,k), the WORST ring (min meanR) and the BEST ring (max meanR) on the same line, so each
candidate's separation is visible, and track the worst-over-lines margin vs m. Reuses expBA. Writes localizer_gap.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBA = _load("expBA", "expBA_ringspectrum.py")

# candidate certificates, all read directly from expBA's per-ring dict (no recomputation):
#   C0 uniform 2*meanR (=unif_energy) ; bad ring: <=1     C1 true ground lmin (variational lower bound on C0)
#   C2 band-top deficit 2-lmax (coherent=0)               C3 localizer gap = min|ev|
#   C4 variance var0 = <0|H^2|0>-<0|H|0>^2 (Entry 62)     C5 loop energy -sum log 2R (>=0 bad)
def _descr(ring):
    return dict(meanR=ring["meanR"], C0_unif=ring["unif_energy"], C1_lmin=ring["lmin"],
                C2_lmax=ring["lmax"], C2_deficit=2.0 - ring["lmax"], C3_gap=ring["gap"],
                C4_var0=ring["var0"], C5_energy=ring["energy"])


def main():
    print("expBB -- uniform test vector  ->  true ground state / localizer gap. Sharper? m-uniform? Discriminating?\n")
    out = {"per_mk": []}
    margins = {c: {} for c in ("C0_unif", "C2_deficit", "C3_gap", "C4_var0")}  # track worst margin vs m

    for (m, k) in [(4, 2), (5, 2), (6, 2)]:
        lines = expBA.collect_lines(m, k)
        if not lines:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line"); continue
        # for each line: worst ring (min meanR) and best ring (max meanR), recompute full descriptors
        worst_recs = []; best_recs = []
        for ch, rings in lines:
            mrs = np.array([r["meanR"] for r in rings])
            iw = int(np.argmin(mrs)); ib = int(np.argmax(mrs))
            worst_recs.append(_descr(rings[iw]))
            best_recs.append(_descr(rings[ib]))
        # aggregate: the conjecture-stressing case = the line whose WORST ring is least obstructed
        def agg(recs, key, worst="max"):
            vals = [r[key] for r in recs]
            return (max(vals) if worst == "max" else min(vals))
        # C0: bad ring has C0<=1 ; stressing line = max over lines of (min-ring C0) = max of worst_recs C0
        c0_stress = agg(worst_recs, "C0_unif", "max"); c0_margin = 1.0 - c0_stress
        # C2 deficit: bad ring has large deficit; stressing = min over lines of worst-ring deficit
        c2_stress = agg(worst_recs, "C2_deficit", "min"); c2_margin = c2_stress  # want > 0, m-stable
        # C3 gap: how close to 0 is the obstructed ring's nearest eigenvalue
        c3_worst = agg(worst_recs, "C3_gap", "max")
        # C4 var0: bad ring has large variance; stressing = min worst-ring var0
        c4_stress = agg(worst_recs, "C4_var0", "min")
        # discrimination: worst-ring vs best-ring means of each candidate
        def mean_key(recs, key): return float(np.mean([r[key] for r in recs]))
        disc = {c: (mean_key(worst_recs, c), mean_key(best_recs, c)) for c in
                ("C0_unif", "C1_lmin", "C2_deficit", "C3_gap", "C4_var0", "C5_energy")}
        margins["C0_unif"][m] = c0_margin; margins["C2_deficit"][m] = c2_margin
        margins["C3_gap"][m] = c3_worst;  margins["C4_var0"][m] = c4_stress

        print(f"=== m={m} k={k}: {len(lines)} lines ===")
        print(f"  C0 uniform 2meanR : stressing(worst-ring,max) = {c0_stress:.3f}  margin to 1 = {c0_margin:+.3f}")
        print(f"  C2 band deficit 2-lmax: stressing(min) = {c2_stress:.3f}  (>0 & m-stable wanted)")
        print(f"  C3 localizer gap min|ev| (worst ring, max) = {c3_worst:.3f}")
        print(f"  C4 variance var0 : stressing(min) = {c4_stress:.3f}")
        print(f"  DISCRIMINATION (worst-ring mean | best-ring mean):")
        for c in ("C0_unif", "C1_lmin", "C2_deficit", "C3_gap", "C4_var0", "C5_energy"):
            w, b = disc[c]; sep = "BLIND" if abs(w-b) < 0.05 else f"sep={w-b:+.2f}"
            print(f"      {c:11s}: worst={w:+.3f}  best={b:+.3f}   [{sep}]")
        out["per_mk"].append(dict(m=m, k=k, n_lines=len(lines),
                                  C0_margin=round(c0_margin, 4), C2_deficit_stress=round(c2_stress, 4),
                                  C3_gap_worst=round(c3_worst, 4), C4_var0_stress=round(c4_stress, 4),
                                  discrimination={c: [round(disc[c][0], 4), round(disc[c][1], 4)]
                                                  for c in disc}))
        print()

    # m-uniformity verdict
    print("--- m-uniformity (does the margin survive growing m?) ---")
    for c, label, want in [("C0_unif", "C0 uniform-mode margin to 1", "shrinks => NOT m-uniform"),
                           ("C2_deficit", "C2 band-deficit stressing", "stable>0 => m-uniform candidate"),
                           ("C4_var0", "C4 variance stressing", "stable>0 => m-uniform candidate")]:
        seq = [f"m={mm}:{margins[c][mm]:+.3f}" for mm in sorted(margins[c])]
        print(f"  {label:32s}: {'  '.join(seq)}   ({want})")
    out["margins_vs_m"] = {c: {int(mm): round(margins[c][mm], 4) for mm in margins[c]} for c in margins}
    with open(os.path.join(HERE, "localizer_gap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote localizer_gap.json")


if __name__ == "__main__":
    main()
