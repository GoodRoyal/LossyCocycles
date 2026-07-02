"""expT_winding.py — does a WINDING (nonzero Pi-shadow Chern) competitor beat PB=1 for r>2?

The r>2 worry: PB<1 forces the Pi-shadow line [alpha_s] in CP^{r-1} to carry Chern -k, which (unlike
r=2) is topologically allowed. If gradient descent for PB were trapped in the trivial (Chern 0)
sector, min PB = 1 (expQ) might miss a winding competitor with PB<1 -> the codim>=2 bound would FAIL
for r>2. This script settles it: minimize PB from inits with KNOWN nonzero shadow Chern, and report
the final PB and final shadow Chern. (Chern can change during descent only by the shadow vanishing
somewhere, so we report both endpoints.)

Also verifies the clean reformulation  (TR') <=> ||alpha_dst - e^{i phi_e} alpha_src|| < ||alpha_dst||.

alpha_s = V(s)[0,:] + i V(s)[1,:] in C^r. Shadow Chern via Fukui-Hatsugai. Reuses cgn.
Writes winding.json."""

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


def shadow_chern(m, Vs):
    al = (Vs[:, 0, :] + 1j * Vs[:, 1, :]).reshape(m, m, -1)
    nrm = np.linalg.norm(al, axis=2, keepdims=True)
    if nrm.min() < 1e-7:
        return None
    b = al / nrm
    F = 0.0
    for x in range(m):
        for y in range(m):
            u1 = np.vdot(b[x, y], b[(x + 1) % m, y]); u2 = np.vdot(b[(x + 1) % m, y], b[(x + 1) % m, (y + 1) % m])
            u3 = np.vdot(b[(x + 1) % m, (y + 1) % m], b[x, (y + 1) % m]); u4 = np.vdot(b[x, (y + 1) % m], b[x, y])
            F += np.angle(u1 * u2 * u3 * u4)
    return F / (2 * np.pi)


def pb_min(m, k, d, r, Vs, steps=600):
    """minimize PB = max_e ||(flux-T')P_Pi|| from init Vs; return (final PB, Vs_final)."""
    S = cgn.std_frame(d, r); E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf2 = np.stack([(S @ cgn.rot_core(phi, r) @ S.T)[:, :2] for (_, _, phi) in E])
    Vs = Vs.copy(); best = np.inf; bestV = Vs.copy()
    for t in range(steps):
        beta = 5 + 70 * t / (steps - 1); lr = 0.4 * (1 - 0.9 * t / (steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M2 = Tf2 - np.einsum('eij,ekj->eik', Vd, Vsr[:, :2, :])
        U_, Sg, Vt_ = np.linalg.svd(M2, full_matrices=False)
        sig = Sg[:, 0]; u = U_[:, :, 0]; v = Vt_[:, 0, :]
        if sig.max() < best:
            best = float(sig.max()); bestV = Vs.copy()
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        vVsr2 = np.einsum('ek,ekj->ej', v, Vsr[:, :2, :]); uVd = np.einsum('ed,edj->ej', u, Vd)
        gVd = -(p[:, None, None]) * np.einsum('ed,ej->edj', u, vVsr2)
        gVsr = np.zeros_like(Vsr); gVsr[:, :2, :] = -(p[:, None, None]) * np.einsum('ek,ej->ekj', v, uVd)
        grad = np.zeros_like(Vs); np.add.at(grad, dst, gVd); np.add.at(grad, src, gVsr)
        Vs = cgn._polar(Vs - lr * grad)
    return best, bestV


def verify_covariant(m, k, d, r, Vs):
    """check (TR') <=> ||a_dst - e^{i phi} a_src|| < ||a_dst||, i.e. agree on which edges hold."""
    al = Vs[:, 0, :] + 1j * Vs[:, 1, :]
    S = cgn.std_frame(d, r); E = cgn._edge_list(m, k); mism = 0
    for (s_, t_, phi) in E:
        A_dst = Vs[t_][:2, :]; A_src = Vs[s_][:2, :]
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        tr = np.trace(R.T @ (A_dst @ A_src.T)); trE = 2 - np.trace(A_src @ A_src.T)
        tr_holds = tr > 1 - trE / 2                                  # (TR')
        cov = np.linalg.norm(al[t_] - np.exp(1j * phi) * al[s_]) < np.linalg.norm(al[t_])  # covariant
        mism += (tr_holds != cov)
    return mism


def main():
    d, r, k, m = 6, 3, 1, 6
    print(f"d={d} r={r} m={m} k={k}\n")
    # reformulation check
    Vr = cgn._polar(np.random.default_rng(0).standard_normal((m * m, d, r)))
    print(f"(TR') <=> covariant form: mismatched edges = {verify_covariant(m,k,d,r,Vr)} (expect 0)\n")
    print("Minimizing PB from inits of varied Pi-shadow Chern (does winding reach PB<1?):")
    print(f"{'init':>16} {'init Chern':>10} {'init PB':>8} {'final PB':>9} {'final Chern':>12} {'PB<1?':>6}")
    out = {}; any_below = False
    S = cgn.std_frame(d, r)
    inits = {"in-core S": np.broadcast_to(S, (m * m, d, r)).copy()}
    for s in range(8):
        inits[f"random{s}"] = cgn._polar(np.random.default_rng(100 + s).standard_normal((m * m, d, r)))
    for name, V0 in inits.items():
        c0 = shadow_chern(m, V0)
        S2 = cgn.std_frame(d, r); E = cgn._edge_list(m, k)
        Tf2 = np.stack([(S2 @ cgn.rot_core(phi, r) @ S2.T)[:, :2] for (_, _, phi) in E])
        src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
        M2 = Tf2 - np.einsum('eij,ekj->eik', V0[dst], V0[src][:, :2, :])
        pb0 = float(np.linalg.svd(M2, full_matrices=False)[1][:, 0].max())
        pbf, Vf = pb_min(m, k, d, r, V0)
        cf = shadow_chern(m, Vf)
        below = pbf < 0.999; any_below = any_below or below
        out[name] = dict(init_chern=c0, init_pb=round(pb0, 4), final_pb=round(pbf, 4),
                         final_chern=cf)
        cs = lambda c: f"{c:+.2f}" if c is not None else "undef"
        print(f"{name:>16} {cs(c0):>10} {pb0:>8.4f} {pbf:>9.4f} {cs(cf):>12} {str(below):>6}")
    print("\nverdict:",
          "a WINDING competitor reached PB<1 -> codim>=2 bound FAILS for r>2!" if any_below else
          "no competitor reaches PB<1, in ANY shadow-Chern sector visited -> the obstruction is "
          "sector-independent; min PB = 1 is robust (not a trapped optimizer). r>2 bound stands "
          "numerically; the PROOF (why winding can't beat the magnitude constraint) remains open.")
    with open(os.path.join(HERE, "winding.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote winding.json")


if __name__ == "__main__":
    main()
