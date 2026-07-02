"""expBG_density_attack.py -- attack the intensive density bound  2*sig_p + 2*sig_C - 4*sig_R >= a0 > 0.  (Entry 68.)

Entry 67 reframed the crux: prove  a = Sum var0 / m = 2*sig_p + 2*sig_C - 4*sig_R >= a0,  sig_p=Sp2/m^2,
sig_C=C2/m^2, sig_R=(Sum_rings meanR_ring^2)/m, all m-stable. The only nontrivial piece is sig_R (the negative
mean-R^2 density); upper-bounding it topologically proves the target.

Two routes tested here:

(1) CAUCHY-SCHWARZ RELAXATION (drop the ring variance). Per ring, meanR_ring^2 <= <R_e^2>_ring (Jensen), and
    rings partition edges, so  T := Sum_rings meanR_ring^2 <= (1/m) Sum_all R_e^2.  Hence
        Sum var0 >= B_CS := (2/m)[ Sp2 + C2 - 2*Sum R_e^2 ]  =  (2/m)[ Sum(I_e^2 - R_e^2) + C2 ],
    a STATIC per-edge + NNN bound (no ring structure left), with I_e=Im(z_e), R_e=Re(z_e). If B_CS >= a0*m with
    a0>0 (m-stable), the crux reduces to a clean static inequality. Measures B_CS, its density B_CS/m =
    2sig_p+2sig_C-4sig_{R2} (sig_{R2}=(1/m^2)Sum R^2), the discarded slack (4/m)Sum_rings Var_ring(R), and
    <R^2>/<|p|^2> (the phase-disorder ratio; ~1/2 => B_CS ~ 2sig_C > 0).

(2) FLATNESS-DEFECT COUNTING (Entry-65: variance is SPREAD, not concentrated). var0_ring = Var_s(w_s),
    w_s = conj(z_{e_s}) + z_{e_{s-1}}; var0_ring=0 iff the ring is flat (w_s const). Measures the per-ring var0
    distribution over all 2m rings and the fraction with var0 >= delta, for delta in {0.1,0.2,0.3}, vs m. Target
    form: #{rings: var0>=delta} >= rho*m (rho,delta m-uniform) => Sum var0 >= rho*delta*m (a counting proof).

Reuses expBF/expBD. Writes density_attack.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBF = _load("expBF", "expBF_curvature_route.py")
expBD = expBF.expBD; expAN = expBF.expAN; expBA = expBF.expBA


def ring_edge_lists(m):
    return expBD.ring_edge_lists(m)


def analyse_line(u, src, dst, phi, m):
    d = expAN.edge_data(u, src, dst, phi)
    p, q = d['p'], d['q']
    z = np.exp(1j*phi) * p
    R = z.real; I = z.imag
    rings = ring_edge_lists(m)
    # exact pieces
    Sp2 = float(np.sum(np.abs(p)**2))
    C2 = 0.0; T = 0.0; ringvarR = 0.0
    per_ring_var0 = []
    for e in rings:
        ze = z[e]; mm = len(e)
        C2 += float(np.sum([(ze[s]*ze[(s-1) % mm]).real for s in range(mm)]))
        mr = float(np.mean(R[e]))
        T += mr*mr
        ringvarR += float(np.mean(R[e]**2) - mr*mr)        # Var_ring(R) >= 0, the discarded C-S slack
        per_ring_var0.append(expBA.ring_operator(ze)["var0"])
    sumR2 = float(np.sum(R**2)); sumI2 = float(np.sum(I**2))
    sum_var0 = float(np.sum(per_ring_var0))
    B_CS = (2.0/m)*(Sp2 + C2 - 2.0*sumR2)                  # Cauchy-Schwarz lower bound on sum_var0
    # NEAREST-NEIGHBOUR form (verified identity):  B_CS = (1/m) Sum_{consec pairs}[(dImz)^2 - (dRez)^2]
    #   = (1/m)[E_L(Im z) - E_L(Re z)] = -(1/m) Re Sum(dz)^2 ,  E_L = ring-Laplacian Dirichlet energy.
    nn = 0.0; per_ring_D = []
    for e in rings:
        ze = z[e]; mm = len(e); Dr = 0.0
        for s in range(mm):
            dz = ze[s] - ze[(s-1) % mm]
            Dr += dz.imag**2 - dz.real**2
        per_ring_D.append(Dr); nn += Dr
    nn_form = nn/m
    return dict(
        m=m, sum_var0=sum_var0, B_CS=B_CS,
        a=sum_var0/m, B_CS_density=B_CS/m,
        sig_p=Sp2/m**2, sig_C=C2/m**2, sig_R=T/m, sig_R2=sumR2/m**2,
        slack=(4.0/m)*ringvarR,                            # sum_var0 - B_CS  (must be >=0, the dropped part)
        R2_over_p2=sumR2/Sp2,                              # <R^2>/<|p|^2>; ~1/2 => phase-disordered
        nn_form=nn_form, nn_err=abs(B_CS-nn_form),         # verify B_CS == nearest-neighbour form
        per_ring_D=np.array(per_ring_D),                   # per-ring E_L(I)-E_L(R); aggregate = B_CS*m
        per_ring_var0=np.array(per_ring_var0))


def main():
    print("expBG -- density bound  a = 2sig_p + 2sig_C - 4sig_R >= a0 ?   two routes.\n")
    out = {"per_mk": []}
    agg = {}
    deltas = [0.1, 0.2, 0.3]
    for (m, k) in [(5, 2), (6, 2)]:
        lines = expBF.collect_lines(m, k, want=8)
        recs = [analyse_line(u, s2, d2, ph2, m) for (u, s2, d2, ph2, d) in lines]
        if not recs:
            print(f"m={m}: none"); continue
        med = lambda key: float(np.median([r[key] for r in recs]))
        # flatness-defect: pool per-ring var0 across lines, fraction >= delta
        allv = np.concatenate([r["per_ring_var0"] for r in recs])
        frac = {d: float(np.mean(allv >= d)) for d in deltas}
        allD = np.concatenate([r["per_ring_D"] for r in recs])
        rec = dict(m=m, k=k, n_lines=len(recs),
                   a=round(med("a"), 4), B_CS_density=round(med("B_CS_density"), 4),
                   sum_var0=round(med("sum_var0"), 3), B_CS=round(med("B_CS"), 3),
                   nn_err=float(np.max([r["nn_err"] for r in recs])),
                   D_pos_frac=round(float(np.mean(allD > 0)), 3), D_mean=round(float(allD.mean()), 4),
                   D_min=round(float(allD.min()), 4),
                   sig_p=round(med("sig_p"), 4), sig_C=round(med("sig_C"), 4),
                   sig_R=round(med("sig_R"), 4), sig_R2=round(med("sig_R2"), 4),
                   slack=round(med("slack"), 4), R2_over_p2=round(med("R2_over_p2"), 4),
                   frac_ge=frac, nring=2*m)
        agg[m] = rec; out["per_mk"].append(rec)
        print(f"=== m={m} k={k}: {len(recs)} lines (medians), {2*m} rings/line ===")
        print(f"  a = Sum var0/m          = {rec['a']:.3f}   (target a >= a0 > 0)")
        print(f"  ROUTE 1 (Cauchy-Schwarz relaxation):")
        print(f"    B_CS density 2sp+2sC-4sR2 = {rec['B_CS_density']:.3f}   "
              f"(>0 ?  if yes, crux -> STATIC per-edge+NNN inequality)")
        print(f"    NN identity check: B_CS == (1/m)Sum[(dImz)^2-(dRez)^2], max err = {rec['nn_err']:.1e}; "
              f"per-ring D: mean={rec['D_mean']:+.3f} frac>0={rec['D_pos_frac']:.2f} min={rec['D_min']:+.3f}")
        print(f"    sig_p={rec['sig_p']:.3f}  sig_C={rec['sig_C']:.3f}  sig_R={rec['sig_R']:.3f}  "
              f"sig_R2={rec['sig_R2']:.3f}   <R^2>/<|p|^2>={rec['R2_over_p2']:.3f}")
        print(f"    discarded slack (4/m)Sum Var_ring(R) = {rec['slack']:.3f}  "
              f"(= a - B_CS_density; how lossy the relaxation is)")
        print(f"  ROUTE 2 (flatness-defect counting): fraction of rings with var0 >= delta:")
        print(f"    " + "   ".join(f"delta={d}: {frac[d]:.2f}" for d in deltas))
        print()
    if 5 in agg and 6 in agg:
        g = lambda q: agg[6][q]/agg[5][q] if agg[5][q] else float('nan')
        print("=== cross-m (ratio 6/5; m-stable density => ~1.0) ===")
        print(f"  a: {g('a'):.2f}   B_CS_density: {g('B_CS_density'):.2f}   sig_R: {g('sig_R'):.2f}   "
              f"slack: {g('slack'):.2f}")
        b5, b6 = agg[5]['B_CS_density'], agg[6]['B_CS_density']
        if b5 > 0.02 and b6 > 0.02:
            v = (f"ROUTE 1 ALIVE: B_CS density > 0 and m-stable ({b5:.2f},{b6:.2f}) -> the Cauchy-Schwarz "
                 f"relaxation already proves a0>0; crux reduces to the STATIC bound Sum(I^2-R^2)+C2 >= a0'*m.")
        else:
            v = (f"ROUTE 1 TOO LOSSY: B_CS density ~0 or <0 ({b5:.2f},{b6:.2f}) -> dropping the ring variance "
                 f"kills it; must keep Var_ring(R) (Poincare/counting, ROUTE 2).")
        out["verdict_route1"] = v
        print(f"\n  VERDICT: {v}")
    with open(os.path.join(HERE, "density_attack.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print("\nwrote density_attack.json")


if __name__ == "__main__":
    main()
