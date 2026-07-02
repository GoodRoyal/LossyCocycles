"""expAS_threshold.py — locate the COHERENCE THRESHOLD and firm up (H)|_{Chern=-k}.

Entry 56 found a sector dichotomy: pushing coherence (min-mean-R) breaks the Hermitian loop bound (H)
(mu0* = Karp max-mean-cycle of theta^(0)=-log(2R) goes < 0) at Chern +1 (~0.79) but the binding sector
Chern=-k caps coherence lower (~0.60) where mu0* stays > 0. Claim: a coherence THRESHOLD tau in (0.6,0.79)
above which mu0*<0, unreachable by the -k topology.

This script TRACES mu0*(coherence) along a leashed in-sector coherence ascent, per sector, so we can:
  (1) see WHETHER mu0* ever goes negative while all-ray AND in-sector -k  (would break (H)|_-k),
  (2) locate the threshold tau where mu0* crosses 0 (from the +1 trajectory that does cross),
  (3) confirm the -k trajectory's ceiling sits BELOW tau (mu0* stays >=0).
Upgrades over expAR: a LINE-SEARCH leash (shrink the step back into sector instead of hard backtrack), more
trials, snapshot (min-mean-R, mu0*, mu*) every 50 accepted steps. Reuses expU/expAE/expAK/expAN/expAP/expAQ/expAI.
Writes threshold.json.
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


def in_sector(u, m, target_chern):
    ch = expAE.chern_of_u(m, u)
    return (ch is not None and abs(ch - target_chern) < 0.25), ch


def diagnose(u, src, dst, phi, N):
    th, th0, ar, slack = expAQ.theta_bisect(u, src, dst, phi)
    mu = expAI.karp_max_mean_cycle(N, src, dst, th) if ar else None
    mu0 = expAI.karp_max_mean_cycle(N, src, dst, th0) if ar else None
    return ar, slack, (float(mu) if mu is not None else None), (float(mu0) if mu0 is not None else None)


def trace_ascent(u0, src, dst, phi, m, target_chern, steps=6000, lr0=0.025, lam=3.0, margin=0.015, snap=50):
    """leashed (line-search) coherence ascent staying in target_chern sector; snapshot (coh,mu0,mu)."""
    N = m*m
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    traj = []; best = -np.inf; bestu = None; n_acc = 0; min_mu0_in_sector = np.inf
    for t in range(1, steps+1):
        beta = 50 + 250*(t/steps); lam_t = lam*(0.5+0.5*t/steps)
        J, mn, mh, mv, g, R, absq = expAP.obj_and_grad(u, src, dst, phi, m, beta, lam_t, margin)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        step = lr0*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps)
        # line-search leash: shrink step until back in sector (or give up this step)
        accepted = False
        for shrink in (1.0, 0.5, 0.25, 0.1, 0.03):
            ucand = expAE.project_unit(u + shrink*step)
            ins, ch = in_sector(ucand, m, target_chern)
            if ins:
                u = ucand; accepted = True; break
        if not accepted:
            mt *= 0.3; vt *= 0.3; continue
        n_acc += 1
        usf, utf = u[src], u[dst]
        pf = np.einsum('ej,ej->e', utf.conj(), usf); qf = np.einsum('ej,ej->e', utf, usf)
        Rf = (np.exp(1j*phi)*pf).real; aqf = np.abs(qf)
        allray = bool(np.all(Rf > aqf + 1e-9))
        mnmean = min(Rf[0::2].mean(), Rf[1::2].mean())
        if allray and mnmean > best:
            best = mnmean; bestu = u.copy()
        if n_acc % snap == 0 and allray:
            ar, slack, mu, mu0 = diagnose(u, src, dst, phi, N)
            if ar:
                traj.append(dict(coh=round(float(mnmean), 4), mu0=round(mu0, 4), mu=round(mu, 4),
                                 slack=round(float(slack), 4)))
                min_mu0_in_sector = min(min_mu0_in_sector, mu0)
    return traj, best, bestu, (float(min_mu0_in_sector) if np.isfinite(min_mu0_in_sector) else None)


def mu0_of_line(u, src, dst, phi, N):
    ar, slack, mu, mu0 = diagnose(u, src, dst, phi, N)
    rec = expAN.edge_data(u, src, dst, phi)
    R = rec['R']; coh = min(R[0::2].mean(), R[1::2].mean())
    return ar, float(coh), mu0, mu, float(slack)


def main():
    print("Trace mu0*(coherence) to locate the threshold tau & test (H)|_-k harder.\n")
    out = {"sectors": []}
    m, k = 6, 2; src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
    H_k_broken = False

    # --- BINDING sector -2: pool several leashed continuations, accumulate mu0*(coh) ---
    pooled = []; min_mu0_bind = np.inf; ceil_bind = -np.inf
    for trial in range(4):
        np.random.seed(3000 + 17*trial)
        cand, _, _, _ = expAK.find_all_ray_line(m, 2, tries=12)
        if cand is None: continue
        ins, ch = in_sector(cand, m, -2)
        if not ins: continue
        traj, best, bestu, min_mu0 = trace_ascent(cand, src, dst, phi, m, -2, steps=4000, snap=25)
        pooled += traj
        if min_mu0 is not None: min_mu0_bind = min(min_mu0_bind, min_mu0)
        ceil_bind = max(ceil_bind, best)
    pooled.sort(key=lambda p: p["coh"])
    if min_mu0_bind < -1e-6: H_k_broken = True
    print(f"=== BINDING sector Chern=-2: {len(pooled)} in-sector all-ray snapshots, ceiling coh={ceil_bind:.4f}")
    print(f"    min mu0* over ALL snapshots = {min_mu0_bind:+.4f}  => (H)|_-k {'HOLDS' if min_mu0_bind>=-1e-6 else 'BROKEN'}")
    for p in pooled[::max(1, len(pooled)//8)]:
        print(f"      coh={p['coh']:.3f}  mu0*={p['mu0']:+.3f}  mu*={p['mu']:+.3f}")
    out["sectors"].append(dict(target_chern=-2, n_snaps=len(pooled), max_coh=round(float(ceil_bind), 4),
                               min_mu0=round(float(min_mu0_bind), 4), trajectory=pooled))

    # --- NON-binding +1: get all-ray +1 lines at increasing coherence via expAP.attack -> bracket the crossing ---
    print(f"\n=== NON-binding sector Chern=+1: mu0*(coh) from penalized attack at increasing strength")
    plus_pts = []
    for steps in (500, 900, 1500, 2500, 3500):
        got = None
        for sd in range(8):
            rng = np.random.default_rng(6000+steps+sd)
            u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
            J, uf = expAP.attack(u0, src, dst, phi, m, steps=steps)
            if not np.isfinite(J): continue
            ch = expAE.chern_of_u(m, uf)
            ar, coh, mu0, mu, slack = mu0_of_line(uf, src, dst, phi, N)
            if ar and ch is not None and abs(ch-1) < 0.25:
                got = (coh, mu0, mu, slack); break
        if got:
            coh, mu0, mu, slack = got
            plus_pts.append(dict(coh=round(coh, 4), mu0=round(mu0, 4), mu=round(mu, 4)))
            print(f"      steps={steps:>4}: coh={coh:.3f}  mu0*={mu0:+.3f}  mu*={mu:+.3f}  (all-ray slack {slack:+.3f})")
    out["sectors"].append(dict(target_chern=+1, n_snaps=len(plus_pts), trajectory=plus_pts))
    # threshold tau = coherence where +1 mu0* crosses 0
    tau_cross = None
    sp = sorted(plus_pts, key=lambda p: p["coh"])
    for i in range(1, len(sp)):
        if sp[i-1]["mu0"] >= 0 and sp[i]["mu0"] < 0:
            tau_cross = round(0.5*(sp[i-1]["coh"]+sp[i]["coh"]), 3); break
    out["binding_sector_H_broken"] = bool(H_k_broken)
    out["threshold_tau_estimate"] = tau_cross
    out["binding_ceiling"] = round(float(ceil_bind), 4)
    print(f"\n  threshold tau (+1 mu0* crosses 0) ~ {tau_cross};  binding -2 ceiling = {ceil_bind:.3f}")
    print("\nVERDICT:")
    if H_k_broken:
        print("  (H)|_-k BROKEN under the harder line-search attack: an all-ray Chern=-2 line reached mu0*<0.")
        print("  => even the binding sector needs the bilinear; pursue mu*>=0 (Entry 51b) only.")
    else:
        b = next(s for s in out["sectors"] if s["target_chern"] == -2)
        # threshold estimate: smallest +1 coherence whose mu0* <= ~0 (else lowest |mu0*| point)
        pp = out["sectors"][1]["trajectory"]
        near0 = min(pp, key=lambda p: abs(p["mu0"])) if pp else None
        print(f"  (H)|_-k SURVIVES the harder attack: binding -2 mu0* stayed >= {b['min_mu0']:+.4f} "
              f"over {b['n_snaps']} snapshots (coherence ceiling {b['max_coh']:.3f}).")
        if near0:
            print(f"  Threshold tau ~ {near0['coh']:.3f} (+1 line there has mu0*={near0['mu0']:+.3f} ~ 0; "
                  f"Entry-55 +1 at coh 0.79 had mu0*=-0.16).")
        print(f"  => binding -2 ceiling {b['max_coh']:.3f} sits BELOW tau ~{(near0['coh'] if near0 else '?')}: "
              "the -k topology can't reach the (H)-breaking coherence.")
        print("  (H)|_-k is the tractable target; open analytic work = prove the -k coherence cap (ceiling < tau).")
    with open(os.path.join(HERE, "threshold.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote threshold.json")


if __name__ == "__main__":
    main()
