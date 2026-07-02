"""expAP_coupled.py — attack the COUPLED target (Entry 54's open core), staying inside all-ray.

Target (Entry 54): all-ray (R_e>|q_e| on every edge) AND Chern!=0  =>  min(mean_h R, mean_v R) <= 1/2.
Equivalently mean_h R + mean_v R = 2 - D/(2m^2), D = sum_all ||u_t - e^{i phi}u_s||^2 (magnetic Dirichlet
energy); the question is whether all-ray + Chern!=0 forces enough frustration that some direction-mean <= 1/2.

The unconstrained attack (expAO) escaped all-ray (made empty edges). HERE we PENALIZE to stay all-ray:
   maximize  J(u) = softmin(mean_h R, mean_v R) - lambda * sum_e relu(|q_e| - R_e + margin),
so the optimum is pushed to high mean-R WHILE keeping R_e - |q_e| >= margin (all-ray). Then bucket by Chern
and report, among genuinely all-ray Chern!=0 lines, the largest min-mean-R achieved. If it stays <= 1/2 across
m and seeds -> strong evidence the coupled bound is true and TIGHT; expose which edges bind (R_e ~ |q_e|).

Reuses expU/expAE/expAN. Writes coupled.json.
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
expAI = _load("expAI", "expAI_cycle.py")


def full_feasibility(u, src, dst, phi, N):
    """The DECISIVE test on a candidate all-ray line: compute the REAL theta_e (bilinear-corrected) and
    Karp mu*. PB>=1 (bound holds) <=> mu* >= 0. mu*<0 would FALSIFY the r>2 conjecture. Also report the
    Hermitian theta^(0)=-log(2R) Karp mu0* (that's what (H)/(S2) test) to see if the bilinear is what saves it."""
    thetas, kinds, Rep, aq = expAI.edge_thetas(u, src, dst, phi)
    allray = kinds.count('ray') == len(kinds)
    mu = expAI.karp_max_mean_cycle(N, src, dst, thetas) if allray else None
    # Hermitian-only theta0
    theta0 = -np.log(np.maximum(2*Rep, 1e-12))
    mu0 = expAI.karp_max_mean_cycle(N, src, dst, theta0) if allray else None
    return allray, (float(mu) if mu is not None else None), (float(mu0) if mu0 is not None else None)


def obj_and_grad(u, src, dst, phi, m, soft_beta, lam, margin):
    """J = softmin(mean_h R, mean_v R) - lam*sum relu(|q_e|-R_e+margin); Wirtinger ascent gradient."""
    us, ut = u[src], u[dst]
    eph = np.exp(1j*phi)
    p = np.einsum('ej,ej->e', ut.conj(), us)             # <u_t,u_s>
    q = np.einsum('ej,ej->e', ut, us)                    # (u_t,u_s) bilinear
    R = (eph*p).real
    absq = np.abs(q)
    Rh = R[0::2]; Rv = R[1::2]; mh = Rh.mean(); mv = Rv.mean()
    a = np.array([mh, mv]); mn = a.min()
    w2 = np.exp(-soft_beta*(a-mn)); w2 /= w2.sum()
    # --- gradient of softmin(mean_h, mean_v) ---
    we = np.empty(len(src)); we[0::2] = w2[0]/(m*m); we[1::2] = w2[1]/(m*m)
    gs = we[:, None]*np.conj(eph)[:, None]*ut            # d R / dconj(u_s) = 1/2 e^{-i phi}u_t (x2)
    gt = we[:, None]*eph[:, None]*us
    g = np.zeros_like(u); np.add.at(g, src, gs); np.add.at(g, dst, gt)
    # --- penalty grad: minimize sum relu(|q|-R+margin) => ascent gradient -lam * d(relu)/... ---
    viol = (absq - R + margin) > 0
    if viol.any():
        # d|q|/dconj(u_s) = q*conj(u_t)/(2|q|)*2 = q conj(u_t)/|q|; d|q|/dconj(u_t)= q conj(u_s)/|q|
        # dR/dconj(u_s)= e^{-i phi}u_t ; dR/dconj(u_t)= e^{i phi}u_s   (the x2'd Wirtinger gradients)
        absq_safe = np.maximum(absq, 1e-12)
        dq_s = (q/absq_safe)[:, None]*ut.conj()
        dq_t = (q/absq_safe)[:, None]*us.conj()
        dR_s = np.conj(eph)[:, None]*ut
        dR_t = eph[:, None]*us
        vmask = viol.astype(float)[:, None]
        ps = -lam*vmask*(dq_s - dR_s)
        pt = -lam*vmask*(dq_t - dR_t)
        np.add.at(g, src, ps); np.add.at(g, dst, pt)
    pen = float(np.sum(np.maximum(absq - R + margin, 0.0)))
    return float(mn - lam*pen), float(mn), float(mh), float(mv), g, R, absq


def attack(u0, src, dst, phi, m, steps=3000, lr=0.04, lam=2.0, margin=0.02):
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = -np.inf; bestu = u.copy()
    for t in range(1, steps+1):
        beta = 40 + 260*(t/steps)
        lam_t = lam*(0.3 + 0.7*t/steps)                  # ramp penalty so it settles all-ray late
        J, mn, mh, mv, g, R, absq = obj_and_grad(u, src, dst, phi, m, beta, lam_t, margin)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        u = expAE.project_unit(u + lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps))
        # track best feasible (all-ray) min-mean-R
        allray = bool(np.all(R > absq + 1e-9))
        if allray and mn > best: best = mn; bestu = u.copy()
    return best, bestu


def main():
    print("COUPLED attack (Entry 54): maximize min(mean_h R, mean_v R) WHILE staying all-ray (R_e>|q_e|).")
    print("Target: all-ray + Chern!=0 => min-mean-R <= 1/2.  Can a Chern!=0 all-ray line beat 1/2?\n")
    out = {"runs": []}
    global_worst = -np.inf
    for m, k in ((5, 2), (6, 2), (7, 2), (8, 2), (6, 1), (4, 2)):
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
        best = -np.inf; brec = None; n_ar = 0
        for sd in range(20):
            rng = np.random.default_rng(9000+71*k+sd)
            u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
            J, uf = attack(u0, src, dst, phi, m)
            if not np.isfinite(J):       # never reached all-ray
                continue
            ch = expAE.chern_of_u(m, uf)
            rec, _ = expAN.analyze_line(uf, src, dst, phi, m, k)
            if not rec["all_ray"]:
                continue
            chr_ok = (ch is not None and abs(ch) > 0.5)
            if chr_ok:
                n_ar += 1
                d = expAN.edge_data(uf, src, dst, phi)
                slack = float((d['R'] - np.abs(d['q'])).min())   # how tight all-ray binds
                if rec["min_meanR"] > best:
                    best = rec["min_meanR"]
                    # DECISIVE: real Karp mu* (bound holds <=> mu*>=0) vs Hermitian mu0* ((H)/(S2) proxy)
                    ar2, mu, mu0 = full_feasibility(uf, src, dst, phi, N)
                    brec = dict(seed=sd, chern=round(ch, 2), min_mean_R=round(rec["min_meanR"], 4),
                                meanRh=rec["meanRh"], meanRv=rec["meanRv"],
                                allray_slack=round(slack, 4), min_S=rec["min_S"],
                                mu_star=(round(mu, 4) if mu is not None else None),
                                mu0_hermitian=(round(mu0, 4) if mu0 is not None else None))
        out["runs"].append(dict(m=m, k=k, n_allray_chern=n_ar, best=brec))
        if brec is None:
            print(f"=== m={m} k={k}: no all-ray Chern!=0 line reached (penalty couldn't hold all-ray "
                  f"with high mean-R) -> empty-edge regime dominates")
            continue
        global_worst = max(global_worst, best)
        print(f"=== m={m} k={k}: {n_ar} all-ray Chern!=0 lines; BEST min-mean-R = {brec['min_mean_R']:+.4f} "
              f"(Chern={brec['chern']})")
        print(f"      mean_h R={brec['meanRh']:+.4f} mean_v R={brec['meanRv']:+.4f}  "
              f"all-ray slack min(R-|q|)={brec['allray_slack']:+.4f}  | S2 min(S_h,S_v)={brec['min_S']:+.3f}")
        print(f"      => coupled bound min-mean-R<=1/2 : {'HOLDS' if best <= 0.5+1e-6 else '** EXCEEDS 1/2 (S3/S2 too lossy) **'}")
        print(f"      *** DECISIVE: real Karp mu*={brec['mu_star']}  (Hermitian mu0*={brec['mu0_hermitian']})  "
              f"-> bound {'HOLDS (PB>=1)' if (brec['mu_star'] is not None and brec['mu_star']>=-1e-6) else '** FAILS: PB<1, CONJECTURE FALSE **'}")
    out["global_worst_min_mean_R"] = (round(float(global_worst), 4) if np.isfinite(global_worst) else None)
    print(f"\nGLOBAL worst (largest) all-ray Chern!=0 min-mean-R = {out['global_worst_min_mean_R']}")
    print("VERDICT:",
          "coupled bound SURVIVES: even pushed to max coherence while held all-ray, every Chern!=0 line keeps\n"
          "  min-mean-R <= 1/2 -> the LINEAR coupled target is true & tight; all-ray binds (small slack)."
          if (np.isfinite(global_worst) and global_worst <= 0.5+1e-6) else
          "coupled bound EXCEEDED 1/2: an all-ray Chern!=0 line beat 1/2 -> the AM-GM (S3) step is too lossy;\n"
          "  must use the log-sum S2 (check min_S above) or (H) directly." if np.isfinite(global_worst) else
          "penalty never held an all-ray Chern!=0 line at high mean-R -> consistent w/ all-ray<->coherence\n"
          "  incompatibility; coupled bound vacuously safe in this search.")
    with open(os.path.join(HERE, "coupled.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote coupled.json")


if __name__ == "__main__":
    main()
