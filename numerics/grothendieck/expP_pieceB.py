"""expP_pieceB.py — careful run at piece B (the near-regime lower bound). Verify the Pi-compression.

Piece B wants: when beta := max_s sin theta_max(Pi, col V(s)) is small (supports nearly contain the
rotating plane Pi=span{e0,e1}), delta := sup_e||flux(e)-T'(e)|| >= rho_m - c*beta, rho_m the U(1)
(in-core) distOpFlat radius. The reduction (compress everything onto Pi):

  h_s := P_Pi V(s)  (the frame's Pi-shadow),
  Ttil(e) := P_Pi T'(e) P_Pi = h_dst h_src^T          [structure: a flat product on Pi]
  h_s h_s^T = I_Pi - E_s,  ||E_s|| = sin^2 theta_max(Pi,col V(s)) <= beta^2   [near-co-isometric]
  delta >= sup_e || R(phi_e) - Ttil(e) ||             [compression is contractive; flux|_Pi = R(phi)]

This script verifies all three [PROVED] claims numerically over a battery of competitors, and
reports (beta, delta, comp = sup_e||R-Ttil||, frame defect max_s||h_s h_s^T - I||, beta^2) plus
rho_m for reference. The honest open step (NOT verified here as a bound, only measured): turning
sup_e||R - h_dst h_src^T|| into >= rho_m - O(beta^2). Note Ttil/T-hat are 2x2 CONTRACTIONS, not
rotations, so this does NOT trivially reduce to the scalar U(1) unitary bound -- the contraction
structure persists (an honest finding). Reuses coarse-geometry-numerics.py. Writes pieceB.json.
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


def rho_m_incore(m, k=1, steps=800, restarts=20):
    """U(1) in-core distOpFlat: min over site phases of max_e 2|sin((phi_e - dtheta_e)/2)|."""
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E]); phi = np.array([e[2] for e in E])
    n = m * m

    def run(theta, seed):
        best = np.inf
        for t in range(steps):
            beta = 5 + 75 * t / (steps - 1); lr = 0.5 * (1 - 0.9 * t / (steps - 1))
            psi = phi - (theta[dst] - theta[src]); f = 2 * np.abs(np.sin(psi / 2))
            best = min(best, f.max())
            mx = f.max(); p = np.exp(beta * (f - mx)); p /= p.sum()
            g = p * np.sign(np.sin(psi / 2)) * np.cos(psi / 2)
            grad = np.zeros(n); np.add.at(grad, dst, -g); np.add.at(grad, src, g)
            theta = theta - lr * grad
        return best
    best = run(np.zeros(n), 0)
    for rs in range(restarts):
        best = min(best, run(2 * np.pi * np.random.default_rng(rs).random(n), rs))
    return float(best)


def analyze(m, k, d, r, Vs):
    S = cgn.std_frame(d, r)
    Ppi = np.zeros((d, d)); Ppi[0, 0] = Ppi[1, 1] = 1.0      # projection onto Pi = span{e0,e1}
    E = cgn._edge_list(m, k)
    # beta and frame defect
    E2 = np.eye(d)[:, :2]
    beta = 0.0; defect = 0.0
    for s in range(Vs.shape[0]):
        smin = np.linalg.svd(Vs[s].T @ E2, compute_uv=False).min()
        beta = max(beta, np.sqrt(max(0.0, 1 - smin**2)))
        hhT = (Ppi @ Vs[s] @ Vs[s].T @ Ppi)[:2, :2]
        defect = max(defect, op(hhT - np.eye(2)))
    # delta, and the compressed distance sup_e ||R - Ttil|| ; verify structure Ttil = h_dst h_src^T
    delta = 0.0; comp = 0.0; struct = 0.0
    for (s_, t_, phi) in E:
        flux = S @ cgn.rot_core(phi, r) @ S.T
        Tp = Vs[t_] @ Vs[s_].T
        delta = max(delta, op(flux - Tp))
        Ttil = (Ppi @ Tp @ Ppi)[:2, :2]
        R = cgn.rot_core(phi, 2)[:2, :2] if r >= 2 else np.eye(2)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        comp = max(comp, op(R - Ttil))
        hd = (Ppi @ Vs[t_])[:2, :]; hs = (Ppi @ Vs[s_])[:2, :]
        struct = max(struct, op(Ttil - hd @ hs.T))

    # r=2 straightening: u_s = (h_s h_s^T)^{-1/2} h_s is in O(2); T_hat(e)=u_dst u_src^T a flat O(2)
    # competitor.  Measure sup_e ||Ttil - T_hat|| (near-regime error) and check T_hat in O(2).
    # Defined only for beta<1 (h_s h_s^T invertible); beta=1 is the far regime (piece A).
    straight = ortho = float("nan")
    if r == 2 and defect < 0.999:
        def inv_sqrt(A):
            w, Q = np.linalg.eigh(A); return Q @ np.diag(w**-0.5) @ Q.T
        U = [inv_sqrt(((Ppi @ Vs[s])[:2, :]) @ ((Ppi @ Vs[s])[:2, :]).T) @ (Ppi @ Vs[s])[:2, :]
             for s in range(Vs.shape[0])]
        straight = ortho = 0.0
        for (s_, t_, phi) in E:
            Ttil = (Ppi @ (Vs[t_] @ Vs[s_].T) @ Ppi)[:2, :2]
            That = U[t_] @ U[s_].T
            straight = max(straight, op(Ttil - That))
            ortho = max(ortho, op(That @ That.T - np.eye(2)))
    return dict(beta=beta, defect=defect, beta2=beta**2, delta=delta, comp=comp, struct=struct,
                straight=straight, ortho=ortho)


def main():
    d, r, k = 4, 2, 1
    out = {}
    for m in (4, 6):
        rho = rho_m_incore(m, k)
        S = cgn.std_frame(d, r); W = np.zeros((d, r)); W[2, 0] = W[3, 1] = 1.0
        comps = {"orthogonal": np.broadcast_to(W, (m * m, d, r)).copy(),
                 "in-core": np.broadcast_to(S, (m * m, d, r)).copy(),
                 "random": cgn._polar(np.random.default_rng(2).standard_normal((m * m, d, r)))}
        for a_deg in (10, 30, 60):                                   # constant tilts S -> W
            a = np.deg2rad(a_deg)
            comps[f"tilt{a_deg}"] = cgn._polar(
                np.broadcast_to(np.cos(a) * S + np.sin(a) * W, (m * m, d, r)).copy())
        print("=" * 96)
        print(f"m={m}  rho_m(U(1) in-core) = {rho:.4f}   "
              f"[verify: struct~0, comp<=delta, defect<=beta^2; explore delta vs rho_m - c*beta]")
        print("=" * 96)
        print(f"{'competitor':>12} {'beta':>7} {'delta':>8} {'comp':>8} "
              f"{'straight(||Ttil-That||)':>23} {'beta^2':>8} {'That∈O2?':>9} "
              f"{'rho-2b^2':>9}")
        for name, Vs in comps.items():
            a = analyze(m, k, d, r, Vs)
            a["rho_m"] = rho
            out[f"m{m}_{name}"] = {q: round(v, 5) for q, v in a.items()}
            print(f"{name:>12} {a['beta']:>7.4f} {a['delta']:>8.4f} {a['comp']:>8.4f} "
                  f"{a['straight']:>23.4f} {a['beta2']:>8.4f} {a['ortho']:>9.1e} "
                  f"{rho - 2 * a['beta2']:>9.4f}")
        print()
    with open(os.path.join(HERE, "pieceB.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote pieceB.json")


if __name__ == "__main__":
    main()
