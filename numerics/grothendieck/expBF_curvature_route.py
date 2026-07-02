"""expBF_curvature_route.py -- de-risk the analytic crux's candidate route. (Entry 67.)

The crux (Entry 66): prove  Sum_rings var0 >= a*m  (a>0, m-uniform) under all-ray + Chern -k.
The digest's CANDIDATE route: lower-bound Sum var0 by a Berry-curvature energy  Sum_plaq F^2  (`~ Chern^2`).

THE SCALING WORRY (analytic, before any numerics):
  Cauchy-Schwarz on the m^2 plaquettes gives   Sum F^2 >= (Sum F)^2 / m^2 = (2*pi*k)^2 / m^2,
  which DECAYS like 1/m^2, whereas Sum var0 ~ a*m GROWS. So "~ Chern^2" alone is off by ~m^3 and
  cannot deliver the m-uniform constant. The route can only survive if the ACTUAL curvature on these
  lines is spatially EXTENSIVE -- O(1) per plaquette, sign-disordered, summing to 2*pi*k by cancellation
  -- not spread thin. That is the all-ray hypothesis's job, not the Chern's. This script measures which.

Measures, on all-ray Chern -k ceiling lines (m=5,6; reuses expBD's line generation + sum rule):
  (A) BAND Berry curvature  F_plaq = angle(prod of 4 link variables)  (Sum F_plaq = 2*pi*(-k), exact).
  (B) COMBINED-CONNECTION curvature  G_plaq = plaquette holonomy of psi = phi + arg p  (Entry 59 object).
  For each: total energy Sum F^2 / Sum G^2; the spread vs concentration of |F_plaq| (participation ratio
  P = (Sum F^2)^2 / Sum F^4, and max|F|); the Cauchy-Schwarz floor (2*pi*k)^2/m^2; and the ratios
  Sum var0 / Sum F^2, Sum var0 / Sum G^2.  Cross-m (5 vs 6): does Sum F^2 (resp Sum G^2) grow like Sum var0?

Verdict logic:
  - If Sum F^2 ~ Cauchy-Schwarz floor (curvature SPREAD, P ~ m^2) -> band-curvature route is DEAD (wrong scaling).
  - If Sum F^2 is O(1)-per-plaquette EXTENSIVE (P/m^2 bounded below, max|F|=O(1)) and tracks Sum var0
    -> route ALIVE, and the content is "all-ray forces extensive curvature", to be proved next.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBD = _load("expBD", "expBD_sumrule.py")
expAE = expBD.expAE; expAK = expBD.expAK; expAR = expBD.expAR; expAN = expBD.expAN; expBA = expBD.expBA


def band_curvature(u, m):
    """Per-plaquette band Berry curvature F_plaq (m x m), via Fukui-Hatsugai-Suzuki link variables.
    Sum F_plaq = 2*pi*chern.  (b = normalized u reshaped to (m,m,r).)"""
    al = u.reshape(m, m, -1)
    nrm = np.linalg.norm(al, axis=2, keepdims=True)
    if nrm.min() < 1e-9:
        return None
    b = al / nrm
    F = np.empty((m, m))
    for x in range(m):
        for y in range(m):
            u1 = np.vdot(b[x, y],               b[(x+1) % m, y])
            u2 = np.vdot(b[(x+1) % m, y],        b[(x+1) % m, (y+1) % m])
            u3 = np.vdot(b[(x+1) % m, (y+1) % m], b[x, (y+1) % m])
            u4 = np.vdot(b[x, (y+1) % m],        b[x, y])
            F[x, y] = np.angle(u1*u2*u3*u4)
    return F


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def combined_curvature(psi, m):
    """Per-plaquette holonomy of the combined connection psi (one value per edge), wrapped to (-pi,pi]."""
    G = np.empty((m, m))
    for x in range(m):
        for y in range(m):
            s = (psi[eidx(x, y, True, m)] + psi[eidx(x+1, y, False, m)]
                 - psi[eidx(x, y+1, True, m)] - psi[eidx(x, y, False, m)])
            G[x, y] = (s + np.pi) % (2*np.pi) - np.pi
    return G


def participation(vals2):
    """P = (sum v)^2 / sum v^2 for v = F_plaq^2 >=0; P in [1, #plaq]. P ~ #plaq => spread; P ~ 1 => concentrated."""
    v = vals2.ravel()
    s1 = float(v.sum()); s2 = float((v**2).sum())
    return (s1*s1/s2) if s2 > 0 else 0.0


def collect_lines(m, k, want=6):
    """Mirror expBD's in-sector all-ray ceiling-line generation, but gather several lines."""
    out = []
    for sd in range(40):
        if len(out) >= want:
            break
        np.random.seed(7000 + 31*sd + 100*k + m)
        u0, s2, d2, ph2 = expAK.find_all_ray_line(m, k, tries=24)
        if u0 is None:
            continue
        ch0 = expAE.chern_of_u(m, u0)
        if ch0 is None or abs(ch0 + k) > 0.25:
            continue
        _, up, _, _ = expAR.coherence_ascend_in_sector(u0, s2, d2, ph2, m, k, steps=2200)
        for u in (u0, up):
            if u is None:
                continue
            ch = expAE.chern_of_u(m, u)
            if ch is None or abs(ch + k) > 0.25:
                continue
            d = expAN.edge_data(u, s2, d2, ph2)
            if not np.all(d['R'] > np.abs(d['q']) + 1e-12):
                continue
            out.append((u, s2, d2, ph2, d))
    return out


def main():
    print("expBF -- can the Berry-curvature-energy route give the right m-scaling for Sum_rings var0?\n")
    print("  worry: Cauchy-Schwarz floor Sum F^2 >= (2*pi*k)^2/m^2 DECAYS ~1/m^2; Sum var0 ~ a*m GROWS.\n")
    out = {"per_mk": []}
    agg = {}
    for (m, k) in [(5, 2), (6, 2)]:
        lines = collect_lines(m, k)
        if not lines:
            print(f"m={m} k={k}: no all-ray Chern -{k} line"); continue
        cs_floor = (2*np.pi*k)**2 / m**2
        rows = []
        for (u, s2, d2, ph2, d) in lines:
            pieces = expBD.sumrule_pieces(u, s2, d2, ph2, m)
            if pieces is None:
                continue
            sv = pieces["sv_direct"]
            F = band_curvature(u, m)
            if F is None:
                continue
            G = combined_curvature(d['psi'], m)
            SF2 = float(np.sum(F**2)); SG2 = float(np.sum(G**2))
            rows.append(dict(
                sumvar0=sv, SF2=SF2, SG2=SG2,
                chern_check=float(F.sum()/(2*np.pi)),
                maxF=float(np.max(np.abs(F))), maxG=float(np.max(np.abs(G))),
                P_F=participation(F**2), P_G=participation(G**2),
                ratio_var_F=sv/SF2 if SF2 > 0 else np.inf,
                ratio_var_G=sv/SG2 if SG2 > 0 else np.inf))
        if not rows:
            print(f"m={m} k={k}: no usable line"); continue
        def med(key): return float(np.median([r[key] for r in rows]))
        nplaq = m*m
        rec = dict(m=m, k=k, n_lines=len(rows), nplaq=nplaq,
                   cs_floor=round(cs_floor, 4),
                   sumvar0=round(med("sumvar0"), 4),
                   SF2_band=round(med("SF2"), 4), SG2_combined=round(med("SG2"), 4),
                   maxF=round(med("maxF"), 4), maxG=round(med("maxG"), 4),
                   P_F=round(med("P_F"), 3), P_G=round(med("P_G"), 3),
                   ratio_var_F=round(med("ratio_var_F"), 3), ratio_var_G=round(med("ratio_var_G"), 3))
        agg[m] = rec
        print(f"=== m={m} k={k}: {len(rows)} in-sector all-ray ceiling lines (medians) ===")
        print(f"  Sum var0           = {rec['sumvar0']:.3f}   (target ~ a*m, a~0.7-0.8)")
        print(f"  (A) BAND  Sum F^2  = {rec['SF2_band']:.3f}   CS floor (2pi k)^2/m^2 = {rec['cs_floor']:.3f}"
              f"   max|F_plaq| = {rec['maxF']:.3f}   P_F = {rec['P_F']:.2f} / {nplaq} plaq")
        print(f"  (B) COMB  Sum G^2  = {rec['SG2_combined']:.3f}   max|G_plaq| = {rec['maxG']:.3f}"
              f"   P_G = {rec['P_G']:.2f} / {nplaq} plaq")
        print(f"      Sum var0 / Sum F^2 = {rec['ratio_var_F']:.3f}    Sum var0 / Sum G^2 = {rec['ratio_var_G']:.3f}")
        print(f"      curvature spread?  P_F/nplaq = {rec['P_F']/nplaq:.2f}  (->1 spread/EXTENSIVE, ->0 concentrated)\n")
        out["per_mk"].append(rec)
    # cross-m scaling verdict
    if 5 in agg and 6 in agg:
        def g(q): return agg[6][q]/agg[5][q] if agg[5][q] else float('nan')
        print("=== cross-m scaling (ratio m=6 / m=5;  linear-in-m would be 6/5=1.20) ===")
        print(f"  Sum var0 : {g('sumvar0'):.2f}     Sum F^2(band): {g('SF2_band'):.2f}"
              f"     Sum G^2(comb): {g('SG2_combined'):.2f}")
        print(f"  CS floor (2pi k)^2/m^2 scales as (5/6)^2 = {(5/6)**2:.2f}  [DECAYS]")
        verdict = ("BAND route plausible: Sum F^2 grows with m like Sum var0 (extensive curvature)."
                   if g('SF2_band') > 1.05 else
                   "BAND route SUSPECT: Sum F^2 does NOT grow like Sum var0 -> wrong m-scaling (as feared).")
        out["scaling"] = dict(var0=round(g('sumvar0'), 3), SF2=round(g('SF2_band'), 3),
                              SG2=round(g('SG2_combined'), 3), cs=round((5/6)**2, 3), verdict=verdict)
        print(f"\n  VERDICT: {verdict}")
    with open(os.path.join(HERE, "curvature_route.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote curvature_route.json")


if __name__ == "__main__":
    main()
