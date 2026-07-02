"""expAO_linattack.py — stress-test the LINEAR (H)-sufficient condition.

From expAN: 2R_e = 2 - ||u_t - e^{i phi}u_s||^2, and via AM-GM the (H) bound follows from
   (S3)  min( mean_h R_e , mean_v R_e ) <= 1/2     (mean over all m^2 edges of that direction),
which is LINEAR in the line projectors (a magnetic tight-binding energy). On harvested lines S3 held but
with thin margin (m=5: mean_h R=0.54>1/2 in ONE direction, saved by the other). So DIRECTLY ATTACK it:

   maximize  J(u) = min( mean_h R , mean_v R )   over unit line fields,
bucket by Chern, and ask: does any ALL-RAY Chern!=0 line reach J > 1/2 (breaking S3)?

If S3 survives a determined attack -> the linear bound is the clean provable target for (H).
If S3 breaks (some all-ray Chern!=0 line has both direction-means > 1/2) -> fall back to the log-sum S2
(min(S_h,S_v)<=0), which has larger margins; report S2 on the S3-breaking line too.
Reuses expU/expAE/expAN. Writes linattack.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
expAE = _load("expAE", "expAE_pq.py")
expAN = _load("expAN", "expAN_wilson.py")


def means_and_grad(u, src, dst, phi, m, soft_beta):
    """J = soft-min(mean_h R, mean_v R); returns J and Wirtinger ascent gradient.
    mean_dir R = (1/m^2) sum_{e in dir} Re[e^{i phi_e}<u_t,u_s>]."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)
    eph = np.exp(1j*phi)
    R = (eph*p).real
    Rh = R[0::2]; Rv = R[1::2]
    mh = Rh.mean(); mv = Rv.mean()
    # soft-min weights between the two direction-means
    a = np.array([mh, mv]); mn = a.min()
    w2 = np.exp(-soft_beta*(a-mn)); w2 /= w2.sum()       # weight on [h, v]
    # grad of Re[e^{i phi}<u_t,u_s>]: d/dconj(u_s)= 1/2 e^{-i phi} u_t ; d/dconj(u_t)= 1/2 e^{i phi} u_s
    # weight each edge by w2[dir]/m^2
    we = np.empty(len(src)); we[0::2] = w2[0]/(m*m); we[1::2] = w2[1]/(m*m)
    gs = we[:, None]*np.conj(eph)[:, None]*ut
    gt = we[:, None]*eph[:, None]*us
    g = np.zeros_like(u); np.add.at(g, src, gs); np.add.at(g, dst, gt)
    return float(mn), float(mh), float(mv), g


def attack(u0, src, dst, phi, m, steps=2500, lr=0.05):
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = -np.inf; bestu = u.copy()
    for t in range(1, steps+1):
        beta = 30 + 270*(t/steps)
        J, mh, mv, g = means_and_grad(u, src, dst, phi, m, beta)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        u = expAE.project_unit(u + lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps))
        if J > best: best = J; bestu = u.copy()
    return best, bestu


def main():
    print("Attack the LINEAR (H)-sufficient condition S3: maximize min(mean_h R, mean_v R) over unit lines.")
    print("Question: can an ALL-RAY Chern!=0 line reach min-mean-R > 1/2 (break S3)?\n")
    out = {"runs": []}
    worst_allray = -np.inf  # largest min-mean-R among all-ray Chern!=0 lines (want <= 1/2)
    for m, k in ((6, 2), (5, 2), (6, 1), (7, 2), (8, 2)):
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
        best_overall = -np.inf; best_rec = None
        worst_halfplane = -np.inf; hp_rec = None
        for sd in range(24):
            rng = np.random.default_rng(7000+53*k+sd)
            u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
            J, uf = attack(u0, src, dst, phi, m)
            ch = expAE.chern_of_u(m, uf)
            rec, _ = expAN.analyze_line(uf, src, dst, phi, m, k)
            allray = rec["all_ray"]; chr_ok = (ch is not None and abs(ch) > 0.5)
            # min_e R over the two directions
            d = expAN.edge_data(uf, src, dst, phi); minR = float(d['R'].min())
            halfplane = minR > 1e-9
            if J > best_overall:
                best_overall = J
                best_rec = dict(seed=sd, chern=(round(ch, 2) if ch is not None else None),
                                min_mean_R=round(J, 4), meanRh=rec["meanRh"], meanRv=rec["meanRv"],
                                all_ray=allray, min_e_R=round(minR, 4), half_plane=halfplane,
                                min_S=rec["min_S"], S2_holds=rec["S2_holds"], S3_holds=rec["S3_holds"])
            if allray and chr_ok:
                worst_allray = max(worst_allray, J)
            if halfplane and chr_ok and J > worst_halfplane:
                worst_halfplane = J
                hp_rec = dict(seed=sd, chern=round(ch, 2), min_mean_R=round(J, 4),
                              min_e_R=round(minR, 4), all_ray=allray,
                              meanRh=rec["meanRh"], meanRv=rec["meanRv"], S3_holds=rec["S3_holds"])
        out["runs"].append(dict(m=m, k=k, best=best_rec,
                                 worst_halfplane=(round(float(worst_halfplane), 4) if np.isfinite(worst_halfplane) else None),
                                 hp_rec=hp_rec))
        b = best_rec
        print(f"=== m={m} k={k}: best min-mean-R = {b['min_mean_R']:+.4f}  (Chern={b['chern']}, "
              f"all_ray={b['all_ray']}, half_plane={b['half_plane']}, min_e R={b['min_e_R']})")
        print(f"      mean_h R={b['meanRh']:+.4f} mean_v R={b['meanRv']:+.4f} | "
              f"S3(<=1/2)={b['S3_holds']}  S2 min(S_h,S_v)={b['min_S']:+.3f}(<=0?{b['S2_holds']})")
        if hp_rec is not None:
            print(f"      max min-mean-R among HALF-PLANE(R>0) Chern!=0 = {worst_halfplane:+.4f}  "
                  f"(Chern={hp_rec['chern']}, all_ray={hp_rec['all_ray']}, min_e R={hp_rec['min_e_R']}, "
                  f"S3={hp_rec['S3_holds']})")
        else:
            print(f"      (no half-plane Chern!=0 line found this m,k)")
    out["worst_allray_min_mean_R"] = (round(float(worst_allray), 4) if np.isfinite(worst_allray) else None)
    print(f"\nWORST (largest) min-mean-R over ALL-RAY Chern!=0 lines = {out['worst_allray_min_mean_R']}")
    survived = np.isfinite(worst_allray) and worst_allray <= 0.5 + 1e-6
    print("VERDICT:",
          "S3 SURVIVES: no all-ray Chern!=0 line reaches min-mean-R>1/2 -> the LINEAR bound suffices for\n"
          "  (H) and is the clean provable target (magnetic tight-binding energy <= 1/2 in one direction)."
          if survived else
          "S3 BROKEN: found an all-ray Chern!=0 line with both direction-means > 1/2 -> AM-GM is too lossy;\n"
          "  must use the log-sum S2 (check S2_holds above on that line) or (H) directly." if np.isfinite(worst_allray)
          else "attack produced no all-ray Chern!=0 line (flattening forces empty edges) -- consistent with the\n"
          "  empty-edge regime; S3 is only TESTED where all-ray lines exist (small m).")
    with open(os.path.join(HERE, "linattack.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote linattack.json")


if __name__ == "__main__":
    main()
