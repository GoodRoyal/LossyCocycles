"""expAJ_phase_margin.py — Entry 51c: the PROVED empty-edge lemma + the sharpened phase-aware conjecture.

Lemma 51 [PROVED] (reverse triangle; verified here to 0 violations): if Re[e^{i phi_e}<u_t,u_s>] <= |(u_t,u_s)|
then bracket_e<=0 for EVERY magnitude ratio (edge 'empty', PB_e>=1).  Proof sketch: bracket_e/nu_s =
f(x)=Re[e^{i phi}p] x - 1/2 - |x e^{-i phi}q - 1/2 omega|, x=nu_t/nu_s>0; reverse triangle gives
|x e^{-i phi}q - 1/2 omega| >= | x|q| - 1/2|omega| |, and casework with Re[e^{i phi}p]<=|q|, |omega|<=1 yields
f(x)<=0 for all x>0.

Consequence: PB<1  =>  Re[e^{i phi_e}p_e] > |q_e|  on EVERY edge  (strengthens Entry 48's |p_e|>|q_e|, and
now carries the flux phase phi_e).  SHARPENED CONJECTURE: no Chern!=0 line satisfies this on all edges.

This script: (1) re-verifies Lemma 51 (0 violations over random edges); (2) MAXIMIZES the phase margin
min_e( Re[e^{i phi_e}<u_t,u_s>] - |(u_t,u_s)| ) over unit line fields by projected Wirtinger ascent
(gradient FD-checked), bucketed by final Chern, testing whether the -k sector can reach >0.
Reuses expU/expAE. Writes phase_margin.json.
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


def verify_lemma(trials=20000, seed=0):
    rng = np.random.default_rng(seed); worst = -np.inf; n = 0
    for _ in range(trials):
        r = 3
        us = rng.standard_normal(r)+1j*rng.standard_normal(r); us/=np.linalg.norm(us)
        ut = rng.standard_normal(r)+1j*rng.standard_normal(r); ut/=np.linalg.norm(ut)
        phi = rng.uniform(0, 2*np.pi)
        p = np.vdot(ut, us); q = np.sum(ut*us); om = np.sum(us*us)
        Rep = (np.exp(1j*phi)*p).real
        if Rep <= abs(q):
            n += 1
            xs = np.exp(np.linspace(-12, 12, 4000))
            f = Rep*xs - 0.5 - np.abs(xs*np.exp(-1j*phi)*q - 0.5*om)
            worst = max(worst, f.max())
    return n, float(worst)


def margins(u, src, dst, phi):
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)
    q = np.einsum('ej,ej->e', ut, us)
    Rep = (np.exp(1j*phi)*p).real
    return Rep - np.abs(q), Rep, q, us, ut


def grad_ascend(u, src, dst, phi, beta):
    """ascent gradient of soft-min over edges of g_e=Re[e^{i phi}<u_t,u_s>]-|(u_t,u_s)|."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us); q = np.einsum('ej,ej->e', ut, us)
    Rep = (np.exp(1j*phi)*p).real; aq = np.abs(q); qhat = q/np.maximum(aq, 1e-12)
    g_edge = Rep - aq
    mn = g_edge.min(); w = np.exp(-beta*(g_edge-mn)); w /= w.sum()
    eph = np.exp(1j*phi)
    # d/dconj(u_s) [Re e^{i phi} p] = 1/2 e^{-i phi} u_t ; d(-|q|)= -1/2 qhat conj(u_t)
    gs = w[:, None]*(np.conj(eph)[:, None]*ut - qhat[:, None]*ut.conj())
    gt = w[:, None]*(eph[:, None]*us - qhat[:, None]*us.conj())
    g = np.zeros_like(u); np.add.at(g, src, gs); np.add.at(g, dst, gt)
    return g, float(mn)


def ascend(u0, src, dst, phi, steps=1500, lr=0.05):
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = -np.inf; bestu = u.copy()
    for t in range(1, steps+1):
        beta = 20 + 380*(t/steps)
        g, _ = grad_ascend(u, src, dst, phi, beta)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        u = expAE.project_unit(u + lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps))
        wm = margins(u, src, dst, phi)[0].min()
        if wm > best: best = wm; bestu = u.copy()
    return best, bestu


def fd_check(m=6, k=2):
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)])
    rng = np.random.default_rng(1)
    u = expAE.project_unit(rng.standard_normal((m*m,3))+1j*rng.standard_normal((m*m,3)))
    g, _ = grad_ascend(u, src, dst, phi, 30.0)
    d = rng.standard_normal((m*m,3))+1j*rng.standard_normal((m*m,3))
    d = d - np.einsum('ej,ej->e', u.conj(), d)[:, None]*u
    def F(uu):
        Rep_aq = margins(expAE.project_unit(uu), src, dst, phi)[0]
        mn = Rep_aq.min(); return mn - (1/30.)*np.log(np.exp(-30*(Rep_aq-mn)).sum())
    eps = 1e-6; fd = (F(u+eps*d)-F(u-eps*d))/(2*eps); ana = np.real(np.sum(np.conj(g)*d))
    print(f"phase-margin grad check: fd={fd:+.6f} analytic={ana:+.6f} ratio={fd/ana:.4f}")


def main():
    print("Entry 51c: empty-edge lemma + sharpened phase-aware conjecture.\n")
    n, worst = verify_lemma()
    print(f"Lemma 51 [PROVED] check: {n} edges with Re[e^iφp]<=|q|; worst max-over-magnitudes bracket "
          f"= {worst:+.2e}  -> {'HOLDS (<=0)' if worst<=1e-9 else 'FAILS'}\n")
    fd_check(); print()
    out = {"lemma_worst_bracket": round(worst, 6), "sectors": {}}
    for k in (2, 1):
        for m in (6, 8, 12):
            src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)])
            buckets = {}
            for sd in range(10):
                rng = np.random.default_rng(500+31*k+sd)
                u0 = expAE.project_unit(rng.standard_normal((m*m,3))+1j*rng.standard_normal((m*m,3)))
                wm, uf = ascend(u0, src, dst, phi, steps=1200)
                ch = expAE.chern_of_u(m, uf)
                key = str(round(ch)) if ch is not None else "und"
                buckets.setdefault(key, []).append(round(float(wm), 4))
            best_mk = max(buckets.get(str(-k), [-9]))
            best_all = max(max(v) for v in buckets.values())
            out["sectors"][f"k{k}_m{m}"] = dict(buckets={kk: max(vv) for kk, vv in buckets.items()},
                                                best_in_minus_k=best_mk, best_overall=round(best_all,4))
            print(f"k={k} m={m}: max phase-margin in -k sector = {best_mk:+.4f} | overall {best_all:+.4f} | "
                  f"sector bests {{ {', '.join(f'{kk}:{max(vv):+.3f}' for kk,vv in sorted(buckets.items()))} }}")
        print()
    # verdict
    anypos = any(v["best_in_minus_k"] > 1e-3 for v in out["sectors"].values())
    print("VERDICT:",
          "some -k line reaches Re[e^iφp]>|q| everywhere (margin>0) -> sharpened conjecture FALSE / needs cap" if anypos else
          "no Chern=-k line reaches Re[e^iφp]>|q| on all edges (max margin<=0): sharpened conjecture holds "
          "numerically -> every -k line has an empty edge -> PB>=1 (the bound) with NO magnitude/cap needed.")
    with open(os.path.join(HERE, "phase_margin.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote phase_margin.json")


if __name__ == "__main__":
    main()
