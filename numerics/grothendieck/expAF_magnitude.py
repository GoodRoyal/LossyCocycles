"""expAF_magnitude.py — Entry 49: even-k binding case. Can MAGNITUDES rescue a Chern=-k line into PB<1?

Entry 48 showed: in the even-k (-2) sector there EXIST unit line-direction fields [u_s] with |p_e|>|q_e|
on every edge (the necessary moduli condition). The full per-edge condition is g_e>|G_e| with, writing
nu_s=sqrt(n_s) (so alpha_s=nu_s u_s) and the cap nu_s^2(1+|omega_s|)<=2:
    g_e = nu_t nu_s Re[e^{i phi_e} p_e] - 1/2 nu_s^2
    |G_e| = nu_s | nu_t e^{-i phi_e} q_e - 1/2 nu_s omega_s |
    g_e - |G_e| = nu_s * [ nu_t Re[e^{i phi_e} p_e] - 1/2 nu_s - | nu_t e^{-i phi_e} q_e - 1/2 nu_s omega_s | ].
So PB<1 <=> the BRACKET > 0 on every edge.  This script FIXES the line directions u_s (topology / Chern
fixed) and OPTIMIZES the magnitudes nu_s under the cap to maximize min_e(bracket).  If even the best
magnitude profile leaves min_e(bracket) <= 0 for every Chern=-k line we try, the even-k obstruction is
robust to magnitude choice: the cap cannot rescue a wound line -> strong evidence PB>=1 for even k.

It separates the two ingredients of g_e>|G_e|: the LINE (phase/winding, p_e,q_e,omega_s fixed) vs the
MAGNITUDE PROFILE (nu_s, the cap).  Lines come from expAE's margin-ascent (-k sector) + random -k lines.
Analytic gradient in nu (FD-checked). Reuses expAE/expU. Writes magnitude.json.
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


def edge_data(u, src, dst, phi):
    """fixed per-edge complex data from unit line field u: p_e=<u_t,u_s>, q_e=(u_t,u_s), omega_s=(u_s,u_s)."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)
    q = np.einsum('ej,ej->e', ut, us)
    om = np.einsum('ej,ej->e', u.conj()*0 + u, u)   # (u_s,u_s) per SITE
    rep = np.exp(1j*phi) * p                         # e^{i phi} p_e
    req = np.exp(-1j*phi) * q                        # e^{-i phi} q_e
    return rep, req, om


def brackets(nu, src, dst, rep, req, om):
    """bracket_e = nu_t Re[rep_e] - 1/2 nu_s - | nu_t req_e - 1/2 nu_s omega_s |  (sign of g_e-|G_e|)."""
    nus, nut = nu[src], nu[dst]
    oms = om[src]
    lin = nut * rep.real - 0.5 * nus
    Gc = nut * req - 0.5 * nus * oms
    return lin - np.abs(Gc), nus, nut, Gc


def softmin_mag(nu, src, dst, rep, req, om, beta):
    br, *_ = brackets(nu, src, dst, rep, req, om)
    mn = br.min(); w = np.exp(-beta*(br-mn)); w /= w.sum()
    F = mn - (1.0/beta)*np.log(np.exp(-beta*(br-mn)).sum())
    return F, w, br


def grad_mag(nu, src, dst, rep, req, om, beta):
    """d softmin(bracket) / d nu_s, analytic.  bracket_e depends on nu_s (src) and nu_t (dst)."""
    F, w, br = softmin_mag(nu, src, dst, rep, req, om, beta)
    nus, nut = nu[src], nu[dst]; oms = om[src]
    Gc = nut*req - 0.5*nus*oms
    aG = np.abs(Gc); aG = np.maximum(aG, 1e-12)
    # d bracket/d nu_t = Re[rep] - Re[ conj(Gc) * req ] / |Gc|
    dbr_dnut = rep.real - (Gc.conj()*req).real/aG
    # d bracket/d nu_s = -1/2 - Re[ conj(Gc) * (-1/2 oms) ]/|Gc| = -1/2 + (1/2) Re[conj(Gc) oms]/|Gc|
    dbr_dnus = -0.5 + 0.5*(Gc.conj()*oms).real/aG
    g = np.zeros_like(nu)
    np.add.at(g, src, w*dbr_dnus); np.add.at(g, dst, w*dbr_dnut)
    return g, F, br


def cap_clip(nu, om):
    cap = np.sqrt(2.0/(1.0+np.abs(om)))
    return np.clip(nu, 1e-4, cap)


def optimize_magnitudes(u, src, dst, phi, steps=2500, restarts=3):
    rep, req, om = edge_data(u, src, dst, phi)
    best = -np.inf
    for sd in range(restarts):
        rng = np.random.default_rng(900+sd)
        cap = np.sqrt(2.0/(1.0+np.abs(om)))
        nu = cap_clip(rng.uniform(0.3, 1.0, u.shape[0])*cap, om)
        mt = np.zeros_like(nu); vt = np.zeros_like(nu); b1,b2,eps=0.9,0.999,1e-8
        for t in range(1, steps+1):
            beta = 20 + 380*(t/steps)
            g, F, br = grad_mag(nu, src, dst, rep, req, om, beta)
            mt=b1*mt+(1-b1)*g; vt=b2*vt+(1-b2)*g*g
            nu = cap_clip(nu + 0.02*mt/(1-b1**t)/(np.sqrt(vt/(1-b2**t))+eps), om)
            br_now = brackets(nu, src, dst, rep, req, om)[0].min()
            best = max(best, br_now)
    return best


def fd_check(u, src, dst, phi):
    rep, req, om = edge_data(u, src, dst, phi)
    rng = np.random.default_rng(3)
    om_abs = np.abs(om); cap = np.sqrt(2/(1+om_abs))
    nu = cap_clip(rng.uniform(0.3,0.9,u.shape[0])*cap, om)
    g,_,_ = grad_mag(nu, src, dst, rep, req, om, 25.0)
    i = 4; eps=1e-6
    def F(nn): return softmin_mag(nn, src, dst, rep, req, om, 25.0)[0]
    nup=nu.copy(); nup[i]+=eps; num=nu.copy(); num[i]-=eps
    fd=(F(nup)-F(num))/(2*eps)
    print(f"mag-grad check: analytic g[{i}]={g[i]:+.5f}  finite-diff={fd:+.5f}")


def get_minusk_lines(m, k, src, dst, seeds=60, n_lines=20):
    """collect unit line fields in the Chern=-k sector via expAE margin-ascent (drives to even Chern;
    keep those landing exactly in -k). Many seeds: margin-ascent scatters across even sectors."""
    lines = []
    for sd in range(seeds):
        rng = np.random.default_rng(7000+sd)
        u0 = expAE.project_unit(rng.standard_normal((m*m,3))+1j*rng.standard_normal((m*m,3)))
        _, uf = expAE.ascend(u0, src, dst, m, steps=1000)
        c = expAE.chern_of_u(m, uf)
        if c is not None and abs(c-(-k))<0.4:
            lines.append(("margin-asc", uf))
        if len(lines) >= n_lines: break
    return lines


def main():
    out = {}
    print("Entry 49: can MAGNITUDES (under the cap) rescue a Chern=-k line into PB<1 (bracket>0 all edges)?\n")
    src0,dst0 = expAE.edge_idx(6,2)
    u_chk = expAE.project_unit(np.random.default_rng(0).standard_normal((36,3))+1j*np.random.default_rng(1).standard_normal((36,3)))
    fd_check(u_chk, src0, dst0, np.array([e[2] for e in expU.edges(6,2)]))
    print()
    for (k, m) in ((2, 8), (2, 6)):        # k=2 = binding even case; -2 lines are harvestable here
        src, dst = expAE.edge_idx(m, k)
        phi = np.array([e[2] for e in expU.edges(m, k)])
        lines = get_minusk_lines(m, k, src, dst, seeds=60, n_lines=20)
        best_bracket = -np.inf; nlines = len(lines); per = []
        for tag, u in lines:
            bb = optimize_magnitudes(u, src, dst, phi, steps=1800, restarts=3)
            per.append(round(float(bb), 5)); best_bracket = max(best_bracket, bb)
        per.sort(reverse=True)
        rec = dict(n_minusk_lines=nlines, top_brackets=per[:8],
                   best_min_bracket=(round(float(best_bracket),5) if nlines else None))
        out[f"k{k}_m{m}"] = rec
        status = ("no -k lines found" if nlines == 0 else
                  ("MAGNITUDES RESCUE: bracket>0 achievable -> PB<1 !!" if best_bracket > 1e-4 else
                   f"magnitudes cannot rescue any of {nlines} lines: min-bracket pinned <=0 (PB>=1, tight)"))
        print(f"k={k} m={m}: {nlines} Chern=-{k} lines; best (over lines+magnitudes) min-edge bracket "
              f"= {best_bracket:+.5f}  -> {status}")
        if per: print(f"          top-8 per-line best brackets: {per[:8]}")
    print()
    rescue = any((v["best_min_bracket"] is not None and v["best_min_bracket"] > 1e-4) for v in out.values())
    print("VERDICT:",
          "*** magnitudes rescue a wound line -> PB<1 reachable, bound FAILS ***" if rescue else
          "for every Chern=-k line tried, no magnitude profile under the cap makes bracket>0 everywhere: "
          "the even-k obstruction is robust to magnitude choice (topology+phase, not the cap).")
    with open(os.path.join(HERE, "magnitude.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote magnitude.json")


if __name__ == "__main__":
    main()
