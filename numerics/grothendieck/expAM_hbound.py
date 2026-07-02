"""expAM_hbound.py — Entry 53b: stress-test the Hermitian loop bound (H) — try to VIOLATE it.

After the monotonicity lemma (Entry 53: theta_e >= theta_e^(0) = -log(2 R_e), R_e=Re[e^{i phi}<u_t,u_s>]),
the even-k r>2 bound REDUCES to the purely Hermitian statement:
   (H)  for an all-ray Chern!=0 line (R_e>|q_e| on all edges), some non-contractible loop C has
        sum_{e in C} log(2 R_e) <= 0   (equivalently max-loop theta^(0)-sum >= 0).
The mechanism FOR (H): keeping R_e>0 with Chern=-k forces the line to WIND (track the flux), driving
|p_e|<1 and hence R_e down -> some loop product prod 2R_e <= 1.  To trust the reduction, try to BREAK (H):
flatten a Chern=-2 all-ray line (push every R_e up toward 1) and see if all wrap-loops can be made to have
prod 2R_e > 1 (max-loop theta^(0) < 0).  If (H) survives a determined flattening attack -> strong.

Objective: MAXIMIZE min_e R_e over unit line fields (flatten), with analytic Wirtinger gradient; bucket by
Chern; for the flattest all-ray Chern=-2 lines, report min_e R_e, all-ray status, and max-loop theta^(0).
Also report the binding mechanism: |p_e| vs the winding. Reuses expU/expAE/expAK/expAL. Writes hbound.json.
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
expAL = _load("expAL", "expAL_monotone.py")


def grad_flatten(u, src, dst, phi, beta):
    """ascent gradient of soft-min over edges of R_e = Re[e^{i phi_e}<u_t,u_s>]."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)
    R = (np.exp(1j*phi)*p).real
    mn = R.min(); w = np.exp(-beta*(R-mn)); w /= w.sum()
    eph = np.exp(1j*phi)
    # d/dconj(u_s) Re[e^{i phi} <u_t,u_s>] = 1/2 e^{-i phi} u_t ; d/dconj(u_t) = 1/2 e^{i phi} u_s
    gs = w[:, None]*np.conj(eph)[:, None]*ut
    gt = w[:, None]*eph[:, None]*us
    g = np.zeros_like(u); np.add.at(g, src, gs); np.add.at(g, dst, gt)
    return g, float(mn)


def flatten(u0, src, dst, phi, steps=1500, lr=0.05):
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = -np.inf; bestu = u.copy()
    for t in range(1, steps+1):
        beta = 20 + 380*(t/steps)
        g, _ = grad_flatten(u, src, dst, phi, beta)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        u = expAE.project_unit(u + lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps))
        mn = grad_flatten(u, src, dst, phi, 1e3)[1]
        if mn > best: best = mn; bestu = u.copy()
    return best, bestu


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def loop_theta0(u, src, dst, phi, m):
    theta0, thetas, kinds, R, aq = expAL.theta0_and_theta(u, src, dst, phi)
    allray = kinds.count('ray') == len(kinds)
    H0 = [float(sum(theta0[eidx(x, y, True, m)] for x in range(m))) for y in range(m)]
    V0 = [float(sum(theta0[eidx(x, y, False, m)] for y in range(m))) for x in range(m)]
    return allray, max(max(H0), max(V0)), R, aq


def main():
    print("Entry 53b: try to VIOLATE the Hermitian loop bound (H) by flattening a Chern=-2 line.\n")
    m, k = 6, 2
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
    out = {"attempts": []}
    worst_H = np.inf      # most negative max-loop theta0 found among all-ray Chern=-2 lines (want to stay >=0)
    n_allray_minusk = 0
    for sd in range(30):
        rng = np.random.default_rng(2000+sd)
        u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
        mnR, uf = flatten(u0, src, dst, phi)
        ch = expAE.chern_of_u(m, uf)
        allray, maxloop0, R, aq = loop_theta0(uf, src, dst, phi, m)
        rec = dict(seed=sd, chern=(round(ch,2) if ch is not None else None), min_R=round(float(mnR),4),
                   all_ray=allray, max_loop_theta0=round(float(maxloop0),4),
                   median_absp=round(float(np.median(np.abs(np.einsum('ej,ej->e',
                        uf[dst].conj(), uf[src])))),3))
        out["attempts"].append(rec)
        if ch is not None and abs(ch+k) < 0.4 and allray:
            n_allray_minusk += 1
            worst_H = min(worst_H, maxloop0)
    # show the flattest Chern=-2 all-ray attempts
    print(f"{'sd':>3} {'Chern':>6} {'min R_e':>8} {'all-ray':>8} {'max-loop θ0':>12} {'med|p_e|':>9}")
    for r in sorted(out["attempts"], key=lambda z: -(z["min_R"])):
        flag = "  <-- all-ray Chern=-2" if (r["chern"] is not None and abs(r["chern"]+k)<0.4 and r["all_ray"]) else ""
        print(f"{r['seed']:>3} {str(r['chern']):>6} {r['min_R']:>8.4f} {str(r['all_ray']):>8} "
              f"{r['max_loop_theta0']:>12.4f} {r['median_absp']:>9.3f}{flag}")
    out["n_allray_minusk"] = n_allray_minusk
    out["worst_max_loop_theta0"] = (round(float(worst_H),4) if n_allray_minusk else None)
    print(f"\nall-ray Chern=-2 lines found: {n_allray_minusk}; "
          f"worst (min) max-loop theta^(0) among them = {out['worst_max_loop_theta0']}")
    print("VERDICT:",
          "(H) HOLDS on every all-ray Chern=-2 line found (max-loop theta^(0) stayed >=0) even under a\n"
          "  determined flattening attack -> the reduction even-k => (H) is robust; flattening can't beat the\n"
          "  half-plane winding that forces |p_e|<1." if (n_allray_minusk and worst_H >= -1e-6) else
          "(H) was VIOLATED (found all-ray Chern=-2 line with all wrap-loops prod 2R_e>1) -> bilinear\n"
          "  correction is ESSENTIAL; monotonicity necessary but not sufficient." if n_allray_minusk else
          "flattening produced NO all-ray Chern=-2 line (flattening forces empty edges) -> consistent with\n"
          "  the mechanism: you cannot flatten a winding line without breaking the half-plane.")
    with open(os.path.join(HERE, "hbound.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote hbound.json")


if __name__ == "__main__":
    main()
