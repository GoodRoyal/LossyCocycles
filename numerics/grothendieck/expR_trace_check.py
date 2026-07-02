"""expR_trace_check.py — verify the trace identity behind the Pythagorean proof of PB >= 1.

Claim (per edge e, summing the Pythagorean identity over an orthonormal basis {e0,e1} of Pi):
    sum_{i} ||(flux(e)-T'(e)) u_i||^2  =  4 - tr(E_src) - 2 tr(R(phi_e)^T Ttil(e)),
where Ttil(e) = A_dst A_src^T (2x2 compression, A_s = V(s)[:2,:]), and E_src = I - A_src A_src^T.
Hence  PB(e)^2 >= (1/2) sum_i = 2 - tr(E_src)/2 - tr(R(phi_e)^T Ttil(e)),  so PB < 1 forces
    tr(R(phi_e)^T Ttil(e)) > 1 - tr(E_src)/2     for every edge e.                       (TR)
For r=2 Ttil ~ a rotation R(psi_e) with psi a coboundary (straightening, expP), so (TR) reads
2 cos(phi_e - psi_e) > 1 - O(beta^2), i.e. |phi_e - psi_e| < ~pi/3 for all e -- contradicting that
a coboundary psi cannot match flux's winding (the U(1) min-max angle is >= ~80deg > 60deg,
m-uniform, = the rho_m import).  This script checks the identity to machine precision and reports
max_e tr(R^T Ttil) vs (1 - tr(E)/2) for the orthogonal competitor (where it should be tight).

Reuses cgn. Writes nothing (verification only)."""

from __future__ import annotations

import importlib.util
import os
import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)
op = lambda A: float(np.linalg.norm(A, 2))


def check(m, k, d, r, Vs, label):
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    max_resid = 0.0; PB = 0.0; worst_TR = -np.inf
    for (s_, t_, phi) in E:
        flux = S @ cgn.rot_core(phi, r) @ S.T
        Tp = Vs[t_] @ Vs[s_].T
        D2 = (flux - Tp)[:, :2]                      # (flux-T') restricted to Pi inputs (d x 2)
        lhs = float(np.sum(D2**2))                   # sum_i ||(flux-T')u_i||^2 = ||D2||_F^2
        A_dst = Vs[t_][:2, :]; A_src = Vs[s_][:2, :]    # (2 x r)
        Ttil = A_dst @ A_src.T                       # (2 x 2)
        trE = 2.0 - np.trace(A_src @ A_src.T)
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        rhs = 4.0 - trE - 2.0 * np.trace(R.T @ Ttil)
        max_resid = max(max_resid, abs(lhs - rhs))
        PB = max(PB, op(D2))
        worst_TR = max(worst_TR, np.trace(R.T @ Ttil) - (1.0 - trE / 2.0))   # >0 would be needed for PB<1
    print(f"  {label:>11}: identity max|lhs-rhs| = {max_resid:.2e};  PB = {PB:.4f};  "
          f"max_e [tr(R^T Ttil) - (1-trE/2)] = {worst_TR:+.4f}  "
          f"(needs >0 on EVERY edge for PB<1; <=0 somewhere => PB>=1)")


def main():
    d, r, k = 4, 2, 1
    print("Verifying  sum_i||(flux-T')u_i||^2 = 4 - tr(E_src) - 2 tr(R^T Ttil)  (per edge), "
          "and the (TR) margin.\n")
    for m in (4, 6):
        S = cgn.std_frame(d, r); W = np.zeros((d, r)); W[2, 0] = W[3, 1] = 1.0
        comps = {"orthogonal": np.broadcast_to(W, (m * m, d, r)).copy(),
                 "in-core": np.broadcast_to(S, (m * m, d, r)).copy(),
                 "random": cgn._polar(np.random.default_rng(3).standard_normal((m * m, d, r)))}
        print(f"m={m}:")
        for name, Vs in comps.items():
            check(m, k, d, r, Vs, name)
        print()


if __name__ == "__main__":
    main()
