"""expI_support_slice.py — Q2 nibble: does the binding distOpFlat competitor live on P?

Lemma A (`02-compression-seam-Q1Q2.md`) rigorously bounds flat competitors whose support is
EXACTLY the core projection P: there distOpFlat >= rho_m, with no leak penalty. The open Q2
seam is whether competitors that LEAVE the support (P' != P) can do better (a "cheap channel"
via support freedom). This script asks the cheap empirical version:

  (a) in-core slice:  minimize distOpFlat restricted to support-P competitors V(s)=S.R(theta_s).
      A support-P flat competitor has per-edge distance  ||R(phi_e) - R(dtheta_e)|| on the core
      = 2|sin((phi_e - dtheta_e)/2)|,  dtheta_e = theta_dst - theta_src.  So this slice is the
      scalar U(1) flat-vs-flux problem; its inf is the unitary radius rho_m.
  (b) free:           minimize over the FULL site-frame Stiefel product (Exp G), and measure the
      SUPPORT DISTANCE of the resulting minimizer, ||P_s - P'_s|| per site, P'_s = V*(s)V*(s)^T.

Read-out:
  * if free-min ~= in-core-min  AND  the free minimizer sits at support distance ~0, the binding
    competitor is INSIDE Lemma A's provable slice -> the near (compression) argument reaches the
    operative competitor and the far regime is empirically slack (Q2 seam narrowed).
  * if free-min < in-core-min (leaving support helps), that is a genuine Q2 alarm.

Honest scope: both are UPPER bounds (descent), so this maps where good competitors are found, it
does not prove a floor. Reuses coarse-geometry-numerics.py primitives. Writes support_slice.json.
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
M_LIST = (4, 6, 8, 12, 16)
D, R, K = 3, 2, 1
SQRT2 = np.sqrt(2.0)


# ---- (a) in-core (support-P) slice: scalar phase problem -------------------------- #
def in_core_distflat(m, k=K, steps=1000, restarts=40):
    """min over site phases theta of  max_e 2|sin((phi_e - dtheta_e)/2)|."""
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    phi = np.array([e[2] for e in E])
    n = m * m

    def run(theta0, seed):
        rng = np.random.default_rng(seed)
        theta = theta0.copy()
        best = np.inf
        for t in range(steps):
            beta = 5.0 + 55.0 * t / (steps - 1)
            lr = 0.5 * (1.0 - 0.9 * t / (steps - 1))
            psi = phi - (theta[dst] - theta[src])
            f = 2.0 * np.abs(np.sin(psi / 2.0))
            best = min(best, f.max())
            mx = f.max(); p = np.exp(beta * (f - mx)); p /= p.sum()
            # d f_e/d psi = sign(sin(psi/2)) cos(psi/2);  d psi/d theta_dst = -1, /theta_src = +1
            dfdpsi = np.sign(np.sin(psi / 2.0)) * np.cos(psi / 2.0)
            ge = p * dfdpsi
            grad = np.zeros(n)
            np.add.at(grad, dst, -ge)
            np.add.at(grad, src, +ge)
            theta = theta - lr * grad
        return best

    best = run(np.zeros(n), 0)                         # theta = 0 (V=S) baseline init
    for rs in range(restarts):
        best = min(best, run(2 * np.pi * np.random.default_rng(rs).random(n), 10 + rs))
    return float(best)


# ---- (b) free minimizer, returning the optimal frames so we can measure its support - #
def free_distflat_with_frames(m, k, d, r, steps=400, seed=0, V_init=None):
    S = cgn.std_frame(d, r)
    E = cgn._edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf = np.stack([S @ cgn.rot_core(phi, r) @ S.T for (_, _, phi) in E])
    rng = np.random.default_rng(seed)
    Vs = (np.broadcast_to(S, (m * m, d, r)).copy() if V_init == "S"
          else cgn._polar(rng.standard_normal((m * m, d, r))))
    best = np.inf; bestV = Vs.copy()
    for t in range(steps):
        beta = 5.0 + 55.0 * t / (steps - 1)
        lr = 0.5 * (1.0 - 0.9 * t / (steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M = Tf - np.einsum('eij,ekj->eik', Vd, Vsr)
        U_, Sg, Vt_ = np.linalg.svd(M)
        sig = Sg[:, 0]; u = U_[:, :, 0]; w = Vt_[:, 0, :]
        if sig.max() < best:
            best = float(sig.max()); bestV = Vs.copy()
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        Vsrtw = np.einsum('edr,ed->er', Vsr, w)
        Vdtu = np.einsum('edr,ed->er', Vd, u)
        gVd = -(p[:, None, None]) * np.einsum('ed,er->edr', u, Vsrtw)
        gVs = -(p[:, None, None]) * np.einsum('ed,er->edr', w, Vdtu)
        grad = np.zeros_like(Vs)
        np.add.at(grad, dst, gVd)
        np.add.at(grad, src, gVs)
        Vs = cgn._polar(Vs - lr * grad)
    return best, bestV


def support_distance(V, d=D, r=R):
    """mean / max over sites of ||P_s - P'_s||_2,  P_s = S S^T (core),  P'_s = V_s V_s^T."""
    S = cgn.std_frame(d, r); Pcore = S @ S.T
    ds = [float(np.linalg.norm(Pcore - V[s] @ V[s].T, 2)) for s in range(V.shape[0])]
    return float(np.mean(ds)), float(np.max(ds))


def main():
    print(f"{'m':>4} {'in-core min':>12} {'free min':>10} {'/sqrt2':>8} "
          f"{'supp dist (mean/max of free min)':>34}")
    out = {}
    for m in M_LIST:
        inc = in_core_distflat(m)
        best, bestV = free_distflat_with_frames(m, K, D, R, V_init="S")
        for rs in range(3):                                  # a few random restarts
            b2, V2 = free_distflat_with_frames(m, K, D, R, seed=20 + rs)
            if b2 < best:
                best, bestV = b2, V2
        sd_mean, sd_max = support_distance(bestV)
        out[str(m)] = dict(in_core=round(inc, 4), free=round(best, 4),
                           supp_mean=round(sd_mean, 4), supp_max=round(sd_max, 4))
        print(f"{m:>4} {inc:>12.4f} {best:>10.4f} {best / SQRT2:>8.3f} "
              f"{sd_mean:>16.4f} /{sd_max:>14.4f}", flush=True)
    with open(os.path.join(HERE, "support_slice.json"), "w") as f:
        json.dump(out, f, indent=2)
    # verdict
    leaves = any(v["free"] < v["in_core"] - 1e-3 for v in out.values())
    onP = all(v["supp_mean"] < 0.05 for v in out.values())
    print("\nverdict:",
          "FREE BEATS IN-CORE -> support freedom helps -> Q2 ALARM" if leaves else
          ("free == in-core AND minimizer sits on P -> binding competitor in Lemma A's slice; "
           "far regime slack" if onP else
           "free == in-core but free minimizer drifts off P (equivalent off-support optimum)"))
    print("wrote support_slice.json")


if __name__ == "__main__":
    main()
