"""expO_support_bound.py — verify the rigorous far-side lower bound  delta >= max_s sin(theta_max).

Claim (rotating-plane support bound) [PROVED, elementary]:
For any flat competitor T'(e) = V(dst)V(src)^T,
    delta := sup_e ||flux(e) - T'(e)||_2  >=  max_s sin theta_max( rot-plane , col V(s) ),
where rot-plane = span{e0,e1} (the 2-plane flux rotates) and theta_max is the largest principal
angle between it and the site frame's column space.

Proof: for the edge with destination s and a unit u in the rot-plane, flux(e)u = R(phi_e)u is a
unit vector IN the rot-plane and T'(e)u in col V(s), so ||flux(e)u - T'(e)u|| >= dist(R(phi)u,
col V(s)); R(phi) maps the rot-plane onto itself, so sup over u gives sin theta_max(rot-plane,
col V(s)). Take the worst site (every site is some edge's destination). QED.

This is TIGHT on the binding side: orthogonal support => sin theta_max = 1 => delta >= 1 (the
orthogonal competitor of expL). It is the rigorous "far" half of the codim>=2 lower bound. This
script checks delta >= max_s sin theta_max over a battery of competitors (orthogonal, in-core,
random, tilted, and an optimizer-best), and reports tightness.

cos theta_max(rot-plane, col V(s)) = sigma_min( V(s)^T E ),  E = [e0 e1] (d x 2);
sin theta_max = sqrt(1 - sigma_min^2). Reuses coarse-geometry-numerics.py. Writes support_bound.json.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

HERE = os.path.dirname(os.path.abspath(__file__))
op = lambda A: float(np.linalg.norm(A, 2))


def delta_of(m, k, d, r, Vs):
    """sup_e ||flux(e) - V(dst)V(src)^T||, Vs shape (m*m, d, r)."""
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    worst = 0.0
    for (s_, t_, phi) in E:
        flux = S @ cgn.rot_core(phi, r) @ S.T
        Tp = Vs[t_] @ Vs[s_].T
        worst = max(worst, op(flux - Tp))
    return worst


def max_sin_theta(Vs, d):
    """max_s sin theta_max(rot-plane span{e0,e1}, col V(s))."""
    E2 = np.eye(d)[:, :2]                       # the rotating 2-plane basis
    worst = 0.0
    for s in range(Vs.shape[0]):
        smin = np.linalg.svd(Vs[s].T @ E2, compute_uv=False).min()   # cos theta_max
        worst = max(worst, np.sqrt(max(0.0, 1.0 - smin**2)))
    return float(worst)


def free_min_frames(m, k, d, r, steps=400, seed=0, V_init=None):
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf = np.stack([S @ cgn.rot_core(phi, r) @ S.T for (_, _, phi) in E])
    rng = np.random.default_rng(seed)
    Vs = (np.broadcast_to(S, (m * m, d, r)).copy() if V_init == "S"
          else cgn._polar(rng.standard_normal((m * m, d, r))))
    best = np.inf; bestV = Vs.copy()
    for t in range(steps):
        beta = 5.0 + 60.0 * t / (steps - 1); lr = 0.5 * (1 - 0.9 * t / (steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M = Tf - np.einsum('eij,ekj->eik', Vd, Vsr)
        U_, Sg, Vt_ = np.linalg.svd(M)
        sig = Sg[:, 0]; u = U_[:, :, 0]; w = Vt_[:, 0, :]
        if sig.max() < best:
            best = float(sig.max()); bestV = Vs.copy()
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        gVd = -(p[:, None, None]) * np.einsum('ed,er->edr', u, np.einsum('edr,ed->er', Vsr, w))
        gVs = -(p[:, None, None]) * np.einsum('ed,er->edr', w, np.einsum('edr,ed->er', Vd, u))
        grad = np.zeros_like(Vs); np.add.at(grad, dst, gVd); np.add.at(grad, src, gVs)
        Vs = cgn._polar(Vs - lr * grad)
    return bestV


def main():
    d, r, k = 4, 2, 1
    out = {}; all_ok = True
    print(f"d={d} r={r} (codim {d - r}); checking  delta >= max_s sin(theta_max)\n")
    print(f"{'m':>3} {'competitor':>14} {'delta':>8} {'max sinθ':>9} {'delta>=bound?':>13} {'tight?':>7}")
    for m in (4, 6, 8):
        S = cgn.std_frame(d, r)
        W = np.zeros((d, r));  W[2, 0] = W[3, 1] = 1.0           # orthogonal frame
        comps = {
            "orthogonal": np.broadcast_to(W, (m * m, d, r)).copy(),
            "in-core(S)": np.broadcast_to(S, (m * m, d, r)).copy(),
            "random": cgn._polar(np.random.default_rng(1).standard_normal((m * m, d, r))),
            "tilted.5": cgn._polar((np.broadcast_to(0.6 * S + 0.4 * W, (m * m, d, r))).copy()),
            "optimizer": free_min_frames(m, k, d, r, V_init="S"),
        }
        for name, Vs in comps.items():
            dl = delta_of(m, k, d, r, Vs)
            bd = max_sin_theta(Vs, d)
            ok = dl >= bd - 1e-6; all_ok = all_ok and ok
            tight = abs(dl - bd) < 1e-3
            out[f"m{m}_{name}"] = dict(delta=round(dl, 4), bound=round(bd, 4), ok=bool(ok))
            print(f"{m:>3} {name:>14} {dl:>8.4f} {bd:>9.4f} {str(ok):>13} {str(tight):>7}")
        print()
    print("ALL satisfy delta >= max_s sin(theta_max):", all_ok)
    print("(orthogonal competitor is the tight case: delta = bound = 1 -> the far-side bound is sharp.)")
    with open(os.path.join(HERE, "support_bound.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote support_bound.json")


if __name__ == "__main__":
    main()
