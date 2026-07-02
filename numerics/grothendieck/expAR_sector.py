"""expAR_sector.py — does (H) survive in the BINDING sector Chern = -k?  Entry 55's open caveat.

Entry 55 refuted (H) using Chern +1/+2 lines, but could NOT reach a coherent all-ray line in the binding
sector Chern = -k = -2 (the coherence objective fights the -2 topology from random inits). Here we test
(H)|_{Chern=-k} the RIGHT way: CONTINUATION from a genuine all-ray Chern=-2 line.

Mechanism that makes this a fair test: coherence ascent RAISES overlaps |p_e| (away from 0), which STABILIZES
the Chern number (a plaquette's Berry phase only jumps when an overlap vanishes). So ascending coherence
*from* a -2 all-ray line should stay at -2, unlike random high-coherence inits (low-|Chern| basins). We:
  1. harvest an all-ray Chern=-2 line (expAK.find_all_ray_line),
  2. coherence-ascend min(mean_h R, mean_v R) with the all-ray penalty (expAP.obj_and_grad), small steps,
  3. monitor Chern every step; keep the most coherent line that is STILL all-ray AND Chern=-2,
  4. on that line compute Hermitian mu0* (=> (H)|_{-k} fails iff mu0*<0) and real mu* (bound holds iff >=0).
If min(mean R) climbs > 1/2 in-sector with mu0*<0 -> (H)|_{-k} ALSO FALSE (bilinear essential even in binding
sector). If the -2 topology caps coherence at min(mean R)<=1/2 / keeps mu0*>=0 -> (H)|_{-k} may hold (tight).
Reuses expU/expAE/expAK/expAN/expAP/expAQ/expAI. Writes sector.json.
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
expAK = _load("expAK", "expAK_loops.py")
expAN = _load("expAN", "expAN_wilson.py")
expAP = _load("expAP", "expAP_coupled.py")
expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
expAI = _load("expAI", "expAI_cycle.py")


def coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=6000, lr=0.02, lam=3.0, margin=0.015):
    """LEASHED ascent: maximize min(mean_h R, mean_v R) with all-ray penalty, but BACKTRACK any step that
    leaves the Chern=-k sector (reject + reset momentum). Finds the true coherence ceiling INSIDE the sector.
    Keeps the best all-ray Chern=-k line seen."""
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = -np.inf; bestu = None; n_in_sector = 0; n_reject = 0
    last_good = u.copy()
    for t in range(1, steps+1):
        beta = 50 + 250*(t/steps)
        lam_t = lam*(0.5 + 0.5*t/steps)
        J, mn, mh, mv, g, R, absq = expAP.obj_and_grad(u, src, dst, phi, m, beta, lam_t, margin)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        ucand = expAE.project_unit(u + lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps))
        # leash: check sector of the candidate
        ch = expAE.chern_of_u(m, ucand)
        in_sector = (ch is not None and abs(ch+k) < 0.25)
        if in_sector:
            u = ucand; last_good = ucand.copy()
            # recompute R,absq for the accepted point to log all-ray accurately
            usf, utf = u[src], u[dst]
            pf = np.einsum('ej,ej->e', utf.conj(), usf); qf = np.einsum('ej,ej->e', utf, usf)
            Rf = (np.exp(1j*phi)*pf).real; aqf = np.abs(qf)
            allray = bool(np.all(Rf > aqf + 1e-9))
            mhf = Rf[0::2].mean(); mvf = Rf[1::2].mean()
            if allray:
                n_in_sector += 1; mnmean = min(mhf, mvf)
                if mnmean > best: best = mnmean; bestu = u.copy()
        else:
            # reject: revert to last in-sector point, damp momentum (kick perpendicular slightly)
            n_reject += 1; u = last_good.copy()
            mt *= 0.3; vt *= 0.3
    return best, bestu, n_in_sector, n_reject


def main():
    print("Test (H)|_{Chern=-k}: CONTINUATION from all-ray Chern=-2 lines, push coherence, stay in -2 sector.\n")
    out = {"runs": []}
    worst_in_sector = -np.inf   # largest in-sector all-ray min-mean-R (want to see if it can exceed 1/2)
    H_k_false = False
    for m, k in ((6, 2), (5, 2)):
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
        best_rec = None; best_mean = -np.inf
        for trial in range(6):
            # fresh harvest of an all-ray Chern=-k start (varies the seed inside find_all_ray_line via global RNG)
            np.random.seed(1234 + 7*trial)
            u0, s2, d2, p2 = expAK.find_all_ray_line(m, k, tries=30)
            if u0 is None:
                continue
            ch0 = expAE.chern_of_u(m, u0)
            if ch0 is None or abs(ch0 + k) > 0.25:
                continue
            rec0, _ = expAN.analyze_line(u0, src, dst, phi, m, k)
            mean0 = rec0["min_meanR"]
            best, bestu, n_sec, n_rej = coherence_ascend_in_sector(u0, src, dst, phi, m, k)
            if bestu is None:
                continue
            # diagnose the most-coherent in-sector all-ray line
            th, th0, ar, slack = expAQ.theta_bisect(bestu, src, dst, phi)
            mu = expAI.karp_max_mean_cycle(N, src, dst, th)
            mu0 = expAI.karp_max_mean_cycle(N, src, dst, th0)
            chf = expAE.chern_of_u(m, bestu)
            recf, _ = expAN.analyze_line(bestu, src, dst, phi, m, k)
            if best > best_mean:
                best_mean = best
                best_rec = dict(trial=trial, chern_start=round(float(ch0), 2), chern_final=round(float(chf), 3),
                                start_min_mean_R=round(float(mean0), 4), pushed_min_mean_R=round(float(best), 4),
                                all_ray=ar, allray_slack=round(slack, 5), n_in_sector_samples=n_sec,
                                mu_star_real=round(float(mu), 4), mu0_hermitian=round(float(mu0), 4),
                                H_k_false=bool(mu0 < -1e-6 and ar), bound_holds=bool(mu >= -1e-6))
        out["runs"].append(dict(m=m, k=k, best=best_rec))
        if best_rec is None:
            print(f"=== m={m} k={k}: could not harvest/keep an all-ray Chern=-{k} line -- skipped"); continue
        b = best_rec
        worst_in_sector = max(worst_in_sector, b["pushed_min_mean_R"])
        H_k_false = H_k_false or b["H_k_false"]
        print(f"=== m={m} k={k}: start Chern={b['chern_start']} (min-mean-R {b['start_min_mean_R']:.3f})"
              f" --push--> Chern={b['chern_final']:+.3f}, min-mean-R={b['pushed_min_mean_R']:.4f}")
        print(f"      all_ray={b['all_ray']} slack={b['allray_slack']:+.5f} in-sector samples={b['n_in_sector_samples']}")
        print(f"      Karp: mu*(real)={b['mu_star_real']:+.4f}  mu0*(Hermitian)={b['mu0_hermitian']:+.4f}")
        print(f"      => (H)|_-k {'** FALSE (mu0*<0 in binding sector) **' if b['H_k_false'] else 'holds (mu0*>=0)'};"
              f" bound {'holds' if b['bound_holds'] else 'FAILS'}")
        print(f"      => min-mean-R in -k sector {'EXCEEDS 1/2' if b['pushed_min_mean_R']>0.5 else 'stays <=1/2'}\n")
    out["worst_in_sector_min_mean_R"] = (round(float(worst_in_sector), 4) if np.isfinite(worst_in_sector) else None)
    out["H_k_false"] = bool(H_k_false)
    print("VERDICT:",
          "(H)|_{Chern=-k} ALSO FALSE -- even continuing from a -2 line, coherence ascent reaches an all-ray\n"
          "  -2 line with every loop prod 2R>1 (mu0*<0), bound saved only by the bilinear (mu*>=0). The\n"
          "  Hermitian route is dead in the binding sector too; pursue the bilinear-corrected mu*>=0."
          if H_k_false else
          f"(H)|_-k SURVIVED this continuation: most-coherent in-sector all-ray line has min-mean-R="
          f"{out['worst_in_sector_min_mean_R']} and mu0*>=0 (the -2 topology caps coherence / keeps the\n"
          "  Hermitian loop bound). Weak positive evidence that (H) restricted to Chern=-k may hold.")
    with open(os.path.join(HERE, "sector.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote sector.json")


if __name__ == "__main__":
    main()
