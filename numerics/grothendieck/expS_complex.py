"""expS_complex.py — verify the complex reformulation (TR') and probe the r>2 winding obstruction.

The trace condition (TR) tr(R(phi_e)^T Ttil(e)) > 1 - tr(E_src)/2 reformulates, with the
complexified Pi-shadow  alpha_s := V(s)[0,:] + i V(s)[1,:] in C^r, as
    (TR')  Re[ e^{i phi_e} <alpha_dst, alpha_src> ] > (1/2) ||alpha_src||^2 ,
using  tr(R(phi)^T Ttil) = Re[e^{i phi} <alpha_dst,alpha_src>]  and  ||alpha_s||^2 = 2 - tr(E_s).
This unifies all r (no r=2 / O(2) straightening needed).

For r=2: alpha_s in C^2, and the O(2) straightening makes the connection arg<alpha_dst,alpha_src>
a COBOUNDARY (Chern 0), while (TR') forces it to wind with the flux (Chern -k) -> contradiction.
For r>2: the Pi-shadow line [alpha_s] in CP^{r-1} can carry NONZERO Chern, so the r=2 argument
does NOT extend; yet min PB = 1 numerically (expQ). This script (1) verifies (TR') to machine
precision, and (2) computes the Pi-shadow line-bundle Chern (Fukui-Hatsugai lattice formula) for
several competitors, to characterise the (still open) r>2 obstruction. Reuses cgn."""

from __future__ import annotations

import importlib.util
import os
import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)


def alpha_field(Vs):
    """alpha_s = V(s)[0,:] + i V(s)[1,:] in C^r, one per site."""
    return Vs[:, 0, :] + 1j * Vs[:, 1, :]      # (n, r)


def verify_TR(m, k, d, r, Vs, label):
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    al = alpha_field(Vs)
    max_resid_tr = max_resid_nrm = 0.0
    for (s_, t_, phi) in E:
        A_dst = Vs[t_][:2, :]; A_src = Vs[s_][:2, :]
        Ttil = A_dst @ A_src.T
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        lhs = np.trace(R.T @ Ttil)
        rhs = np.real(np.exp(1j * phi) * np.vdot(al[t_], al[s_]))    # Re[e^{i phi}<a_dst,a_src>]
        max_resid_tr = max(max_resid_tr, abs(lhs - rhs))
    for s in range(Vs.shape[0]):
        trE = 2.0 - np.trace(Vs[s][:2, :] @ Vs[s][:2, :].T)
        max_resid_nrm = max(max_resid_nrm, abs(np.vdot(al[s], al[s]).real - (2 - trE)))
    print(f"  {label:>11}: |tr(R^T Ttil) - Re[e^{{iφ}}<a,a>]|max = {max_resid_tr:.1e}; "
          f"| ||a||^2 - (2-trE) |max = {max_resid_nrm:.1e}")


def shadow_chern(m, Vs):
    """Fukui-Hatsugai lattice Chern of the Pi-shadow line [alpha_s] in CP^{r-1} over the torus."""
    al = alpha_field(Vs).reshape(m, m, -1)             # (m,m,r)
    nrm = np.linalg.norm(al, axis=2, keepdims=True)
    if nrm.min() < 1e-9:
        return None                                     # shadow vanishes somewhere -> undefined
    b = al / nrm
    F = 0.0
    for x in range(m):
        for y in range(m):
            u1 = np.vdot(b[x, y], b[(x + 1) % m, y])
            u2 = np.vdot(b[(x + 1) % m, y], b[(x + 1) % m, (y + 1) % m])
            u3 = np.vdot(b[(x + 1) % m, (y + 1) % m], b[x, (y + 1) % m])
            u4 = np.vdot(b[x, (y + 1) % m], b[x, y])
            F += np.angle(u1 * u2 * u3 * u4)
    return F / (2 * np.pi)


def main():
    d, r, k = 4, 2, 1
    print("(1) Verify the complex reformulation (TR') [should be ~1e-15]:\n")
    for m in (4, 6):
        S = cgn.std_frame(d, r); W = np.zeros((d, r)); W[2, 0] = W[3, 1] = 1.0
        comps = {"in-core": np.broadcast_to(S, (m * m, d, r)).copy(),
                 "random": cgn._polar(np.random.default_rng(4).standard_normal((m * m, d, r)))}
        print(f"m={m}:")
        for name, Vs in comps.items():
            verify_TR(m, k, d, r, Vs, name)
        print()

    print("(2) Pi-shadow line-bundle Chern (probing the r>2 obstruction); flux Chern = k =", k)
    for (d, r) in [(4, 2), (6, 3)]:
        for m in (6,):
            S = cgn.std_frame(d, r)
            comps = {"in-core": np.broadcast_to(S, (m * m, d, r)).copy(),
                     "random": cgn._polar(np.random.default_rng(5).standard_normal((m * m, d, r)))}
            for name, Vs in comps.items():
                c = shadow_chern(m, Vs)
                cs = f"{c:+.3f}" if c is not None else "undef (shadow vanishes)"
                print(f"  d={d} r={r} m={m} {name:>8}: Pi-shadow Chern = {cs}")
    print("\n(honest) min PB = 1 settles r>2 numerically (expQ); the r=2 proof uses Chern-0 via the")
    print("O(2)/coboundary straightening, which does NOT extend to r>2 -- the r>2 obstruction is open.")


if __name__ == "__main__":
    main()
