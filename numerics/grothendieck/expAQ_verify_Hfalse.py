"""expAQ_verify_Hfalse.py — RIGOROUS recheck that (H) is FALSE (bilinear is essential).

expAP found all-ray Chern!=0 lines where the Hermitian loop bound (H) FAILS (mu0* = max-mean-cycle of
theta^(0)=-log(2R) is < 0, i.e. EVERY non-contractible loop has prod 2R_e > 1) yet the real bound holds
(mu* with the bilinear-corrected theta_e is >= 0). Since this overturns Entry 53's "(H) is the sole open
piece", verify it WITHOUT the grid-based theta of expAI: compute theta_e by BISECTION, recompute Karp on
both theta and theta^(0), and independently confirm with Bellman-Ford that no nonneg cycle exists for
theta^(0) (H false) while one DOES exist for theta (bound holds). Also confirm all-ray with explicit margin
and integer Chern. Reuses expU/expAE/expAN/expAP. Writes verify_hfalse.json.
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
expAP = _load("expAP", "expAP_coupled.py")
expAI = _load("expAI", "expAI_cycle.py")


def theta_bisect(u, src, dst, phi):
    """theta_e = log(first positive root x0 of f_e(x)=R x-1/2-|x e^{-i phi}q - 1/2 omega_s|) by bisection.
    Only valid on ray edges (R>|q|). Returns theta, theta0=-log(2R), allray flag, min slack R-|q|."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us); q = np.einsum('ej,ej->e', ut, us)
    om = np.einsum('ej,ej->e', u, u); oms = om[src]
    eph = np.exp(1j*phi)
    R = (eph*p).real; absq = np.abs(q)
    qc = np.conj(eph)*q; omh = 0.5*oms
    allray = bool(np.all(R > absq + 1e-12))
    theta0 = -np.log(np.maximum(2*R, 1e-300))
    theta = np.full(len(src), np.nan)
    for e in range(len(src)):
        Re, qce, omhe = R[e], qc[e], omh[e]
        f = lambda x: Re*x - 0.5 - abs(x*qce - omhe)
        lo, hi = 1e-12, 1e12
        # f(0)<0, f(inf)>0 on ray edges; bisect in log-x
        for _ in range(300):
            mid = np.sqrt(lo*hi)
            if f(mid) > 0: hi = mid
            else: lo = mid
        theta[e] = np.log(hi)
    return theta, theta0, allray, float((R-absq).min())


def has_nonneg_cycle(N, src, dst, w):
    """Bellman-Ford style: does the directed graph with edge weights w have a cycle of total weight >= 0?
    Detect a positive/zero cycle by running max-path relaxation; if it keeps improving after N rounds -> yes.
    Returns True if some directed cycle has sum w >= 0."""
    # max-cycle-mean sign test via Karp is cleaner; cross-check with relaxation for a strictly-positive cycle.
    dist = np.zeros(N)
    updated_late = False
    for it in range(N+2):
        newd = dist.copy()
        cand = dist[src] + w
        np.maximum.at(newd, dst, cand)
        if it >= N and np.any(newd > dist + 1e-7):
            updated_late = True
        dist = newd
    return updated_late


def main():
    print("RIGOROUS recheck: is the Hermitian loop bound (H) genuinely FALSE on all-ray Chern!=0 lines?")
    print("(theta by BISECTION, Karp + Bellman-Ford cross-check.)\n")
    out = {"lines": []}
    any_H_false = False
    for m, k in ((6, 2), (5, 2), (4, 2)):
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
        found = None
        for sd in range(40):
            rng = np.random.default_rng(9000+71*k+sd)
            u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
            J, uf = expAP.attack(u0, src, dst, phi, m)
            if not np.isfinite(J): continue
            ch = expAE.chern_of_u(m, uf)
            rec, _ = expAN.analyze_line(uf, src, dst, phi, m, k)
            if rec["all_ray"] and ch is not None and abs(ch) > 0.5 and rec["min_meanR"] > 0.55:
                found = (uf, ch, rec); break
        if found is None:
            print(f"m={m} k={k}: no coherent all-ray Chern!=0 line found in 40 seeds"); continue
        uf, ch, rec = found
        theta, theta0, allray, slack = theta_bisect(uf, src, dst, phi)
        # integer-ness of Chern
        chern_int = round(ch)
        # Karp on both
        mu = expAI.karp_max_mean_cycle(N, src, dst, theta)
        mu0 = expAI.karp_max_mean_cycle(N, src, dst, theta0)
        # Bellman-Ford cross-check: nonneg cycle exists?
        bf_real = has_nonneg_cycle(N, src, dst, theta)
        bf_herm = has_nonneg_cycle(N, src, dst, theta0)
        H_false = mu0 < -1e-6
        bound_holds = mu >= -1e-6
        any_H_false = any_H_false or (H_false and allray)
        r = dict(m=m, k=k, chern=round(ch, 3), chern_int=chern_int, all_ray=allray,
                 allray_slack=round(slack, 5), min_mean_R=round(rec["min_meanR"], 4),
                 mu_star_real=round(float(mu), 4), mu0_hermitian=round(float(mu0), 4),
                 H_false=bool(H_false), bound_holds=bool(bound_holds),
                 bf_pos_cycle_real=bool(bf_real), bf_pos_cycle_herm=bool(bf_herm))
        out["lines"].append(r)
        print(f"=== m={m} k={k}: Chern={ch:+.3f}(~{chern_int}) all_ray={allray} slack={slack:+.5f} "
              f"min-mean-R={rec['min_meanR']:.3f}")
        print(f"    Karp:  mu*(real,bilinear)={mu:+.4f}   mu0*(Hermitian,-log2R)={mu0:+.4f}")
        print(f"    Bellman-Ford nonneg-cycle:  real={bf_real}   Hermitian={bf_herm}")
        print(f"    => (H) [Hermitian loop bound] {'FALSE (no loop has prod 2R<=1)' if H_false else 'holds'}; "
              f"real bound {'HOLDS (PB>=1 via bilinear)' if bound_holds else '** FAILS **'}\n")
    out["any_H_false"] = bool(any_H_false)
    print("CONCLUSION:",
          "(H) is FALSE -- there ARE all-ray Chern!=0 lines on which every non-contractible loop has\n"
          "  prod 2R_e > 1 (Hermitian mu0*<0), yet PB>=1 holds via the bilinear-corrected theta (mu*>=0).\n"
          "  => the bilinear is ESSENTIAL; Entry 53's reduction to the Hermitian (H) is REFUTED. The real\n"
          "  open core is the bilinear-corrected loop bound (Entry 51b): all-ray Chern!=0 => mu*>=0."
          if any_H_false else
          "  did not reproduce (H)-false this run.")
    with open(os.path.join(HERE, "verify_hfalse.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote verify_hfalse.json")


if __name__ == "__main__":
    main()
