"""expAG_holonomy.py — Entry 50: the even-k line-field holonomy problem, made explicit and TESTED.

Goal: attack 'Chern[u]!=0 => some edge has bracket_e<=0' (= the open r>2 lower bound).

Reduction (to TEST here, not assume): with alpha_s=nu_s u_s, bracket_e>0 (PB_e<1) implies, after squaring
LHS=nu_t Re[e^{i phi}p_e]-1/2 nu_s>0 and dividing by nu_s nu_t>0,
    a_e r_e + c_s / r_e > b_e ,        r_e := nu_t/nu_s = e^{Delta y_e}  (a COBOUNDARY: prod over loops=1)
with a_e=(Re[e^{i phi}p_e])^2-|q_e|^2,  b_e=Re[e^{i phi}p_e]-Re[e^{-i phi}q_e conj(omega_s)],
     c_s=(1/4)(1-|omega_s|^2)>=0  (source-site only).
AM-GM normal form: a_e r + c_s/r = 2 sqrt(a_e c_s) cosh(Delta y_e - delta_e*),  delta_e*=1/2 log(c_s/a_e),
so (when a_e>0) the condition is cosh(Delta y_e - delta_e*) > beta_e := b_e/(2 sqrt(a_e c_s)).
 -> beta_e<=1: edge imposes NO constraint on the magnitude gradient.
 -> beta_e>1 : Delta y_e must DODGE the interval (delta_e*-xi_e, delta_e*+xi_e), xi_e=arccosh(beta_e).
The magnitude competitor picks a node potential y (so Delta y is exact); feasibility = can y dodge every
active interval while summing to 0 on loops. The Chern obstruction (if any) must live in THIS.

This script, on harvested Chern=-2 lines:
 (1) VALIDATE the reduction: bracket_e>0  <=>  (a_e r_e+c_s/r_e>b_e  AND  half-plane LHS>0), for random nu;
 (2) map the structure: sign of a_e, fraction of active edges (beta_e>1), and the loop-sum of the FORCED
     gradient on active edges vs the coboundary constraint (the candidate obstruction);
 (3) re-solve the magnitude minimax in (Delta y) space and confirm it pins at bracket~0^- (matches expAF).
Reuses expAE/expAF/expU. Writes holonomy.json.
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
expAF = _load("expAF", "expAF_magnitude.py")


def line_coeffs(u, src, dst, phi):
    """per-edge a_e,b_e and per-site c_s from a unit line field u (directions only)."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)      # <u_t,u_s>
    q = np.einsum('ej,ej->e', ut, us)             # (u_t,u_s)
    om = np.einsum('ej,ej->e', u, u)              # (u_s,u_s) per SITE
    Rep = (np.exp(1j*phi)*p).real                 # Re[e^{i phi} p_e]
    oms = om[src]
    a = Rep**2 - np.abs(q)**2
    b = Rep - (np.exp(-1j*phi)*q*np.conj(oms)).real
    c = 0.25*(1 - np.abs(om)**2)                  # per site
    return a, b, c, Rep, p, q, om


def validate_reduction(u, src, dst, phi, rng, ntest=200):
    a, b, c, Rep, p, q, om = line_coeffs(u, src, dst, phi)
    cs = c[src]
    rep_full, req_full, om_full = expAF.edge_data(u, src, dst, phi)
    maxbad = 0
    for _ in range(ntest):
        N = u.shape[0]
        nu = rng.uniform(0.2, 1.4, N)
        br = expAF.brackets(nu, src, dst, rep_full, req_full, om_full)[0]   # ground truth bracket
        nus, nut = nu[src], nu[dst]; r = nut/nus
        quad = a*r + cs/r - b                       # >0 part of reduction
        halfplane = (nut*Rep - 0.5*nus)             # LHS>0
        pred = (quad > 0) & (halfplane > 0)
        truth = br > 0
        maxbad = max(maxbad, int(np.sum(pred != truth)))
    return maxbad


def minimax_in_y(u, src, dst, phi, m, steps=4000, restarts=4):
    """maximize min_e bracket over node potentials y (nu_s=exp(y_s)) under the cap; matches expAF."""
    rep, req, om = expAF.edge_data(u, src, dst, phi)
    cap = np.sqrt(2.0/(1.0+np.abs(om)))
    best = -np.inf
    for sd in range(restarts):
        rng = np.random.default_rng(1234+sd)
        y = np.log(np.clip(rng.uniform(0.3,1.0,u.shape[0])*cap, 1e-3, cap))
        mt = np.zeros_like(y); vt = np.zeros_like(y); b1,b2,eps=0.9,0.999,1e-8
        for t in range(1, steps+1):
            beta = 30 + 600*(t/steps)
            nu = np.exp(y)
            g_nu, F, br = expAF.grad_mag(nu, src, dst, rep, req, om, beta)
            g = g_nu * nu                               # chain rule d/dy = nu * d/dnu
            mt=b1*mt+(1-b1)*g; vt=b2*vt+(1-b2)*g*g
            y = y + 0.03*mt/(1-b1**t)/(np.sqrt(vt/(1-b2**t))+eps)
            y = np.minimum(y, np.log(cap))              # enforce cap
            br_now = expAF.brackets(np.exp(y), src, dst, rep, req, om)[0].min()
            best = max(best, br_now)
    return best


def structure(u, src, dst, phi, m):
    a, b, c, Rep, p, q, om = line_coeffs(u, src, dst, phi)
    cs = c[src]
    pos_a = float(np.mean(a > 0))
    # active edges: beta_e>1 requires a_e>0 and c_s>0
    good = (a > 1e-12) & (cs > 1e-12)
    beta = np.full_like(a, -1.0)
    beta[good] = b[good]/(2*np.sqrt(a[good]*cs[good]))
    active = good & (beta > 1.0)
    frac_active = float(np.mean(active))
    # for active edges, the FORBIDDEN interval center delta* and half-width xi
    delta_star = np.zeros_like(a); xi = np.zeros_like(a)
    delta_star[good] = 0.5*np.log(cs[good]/a[good])
    xi[active] = np.arccosh(np.clip(beta[active], 1.0, None))
    return dict(frac_a_pos=round(pos_a,3), frac_active=round(frac_active,3),
                n_active=int(active.sum()), n_edges=int(a.size),
                beta_active_med=(round(float(np.median(beta[active])),3) if active.any() else None),
                xi_active_med=(round(float(np.median(xi[active])),3) if active.any() else None),
                deltastar_active_absmed=(round(float(np.median(np.abs(delta_star[active]))),3) if active.any() else None))


def main():
    out = {}
    m, k = 8, 2
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)])
    rng = np.random.default_rng(0)
    print(f"Entry 50: even-k line-field holonomy, m={m} k={k}.  Harvest Chern=-2 lines, dissect the reduction.\n")
    # harvest a few -2 lines
    lines = expAF.get_minusk_lines(m, k, src, dst, seeds=40, n_lines=6)
    print(f"harvested {len(lines)} Chern=-2 lines.\n")
    recs = []
    for i, (tag, u) in enumerate(lines):
        bad = validate_reduction(u, src, dst, phi, rng)
        st = structure(u, src, dst, phi, m)
        mm = minimax_in_y(u, src, dst, phi, m)
        st["reduction_mismatches"] = bad
        st["minimax_bracket"] = round(float(mm), 5)
        recs.append(st)
        print(f"line {i}: reduction mismatches={bad:>2} (want 0) | a>0 frac={st['frac_a_pos']} | "
              f"active(beta>1) frac={st['frac_active']} ({st['n_active']}/{st['n_edges']}) | "
              f"minimax bracket={st['minimax_bracket']:+.5f}")
        print(f"        active edges: median beta={st['beta_active_med']} xi={st['xi_active_med']} "
              f"|delta*|={st['deltastar_active_absmed']}")
    out[f"k{k}_m{m}"] = recs
    allbad = sum(r["reduction_mismatches"] for r in recs)
    print(f"\nREDUCTION VALIDATION: total mismatches across lines = {allbad} (must be 0).")
    print("If 0: bracket_e>0 <=> a_e r_e+c_s/r_e>b_e (+half-plane), r_e=nu_t/nu_s a coboundary. The magnitude")
    print("minimax is exactly: pick node potential y to dodge the active forbidden intervals; the open claim")
    print("is that Chern!=0 makes this infeasible (some edge forced into its interval).")
    with open(os.path.join(HERE, "holonomy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote holonomy.json")


if __name__ == "__main__":
    main()
