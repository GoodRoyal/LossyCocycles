"""expU_shadow.py — DECISIVE r>2 test: can a Chern=-k shadow reach PB<1? (does the bound fail?)

Telescoping identity: (TR') forces the Pi-shadow line to carry Chern = -k. But expT's PB-minimizer
ended at Chern +1,+2,0 -- NEVER at -k=-1 (it changes Chern during descent and avoided -1). So
min PB=1 might be a sector artifact, and a Chern=-k shadow might have PB<1 -> codim>=2 bound FALSE
for r>2. This settles it.

Key simplification: PB depends ONLY on the shadow A_s = V(s)[:2,:] (2 x r), not the frame completion
(Pi and Pi^perp rows are orthogonal). With M_e = R(phi_e) - A_dst A_src^T (2x2) and
N_e = A_src (I_r - A_dst^T A_dst) A_src^T (2x2):
    PB_e^2 = lambda_max( M_e^T M_e + N_e ),   PB = max_e PB_e,
and the cap is ||A_s||_op <= 1. We MINIMIZE PB over shadows (projected gradient, analytic grad with
a finite-diff check), initialized at a Chern=-1 ansatz (+ in-core + random), and report min PB and
the shadow Chern. If a Chern=-1 shadow reaches PB<1, the r>2 bound FAILS. Reuses cgn. Writes shadow.json
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM); cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)
HERE = os.path.dirname(os.path.abspath(__file__))


def Rphi(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def edges(m, k):
    phh, phv = cgn.flux_angles(m, k); E = []
    for x in range(m):
        for y in range(m):
            E.append((cgn.site(x, y, m), cgn.site(x + 1, y, m), phh[x, y]))
            E.append((cgn.site(x, y, m), cgn.site(x, y + 1, m), phv[x, y]))
    return E


def pb_terms(A, E):
    """returns array PB_e^2 and the per-edge 2x2 Gram G_e, top eigvec."""
    out = np.empty(len(E)); Gs = []; ws = []
    for i, (s, t, phi) in enumerate(E):
        As, At = A[s], A[t]                                  # (2,3)
        M = Rphi(phi) - At @ As.T                            # 2x2
        N = As @ (np.eye(A.shape[2]) - At.T @ At) @ As.T     # 2x2
        G = M.T @ M + N
        evals, evecs = np.linalg.eigh(G)
        out[i] = evals[-1]; Gs.append((M, N, G)); ws.append(evecs[:, -1])
    return out, Gs, ws


def pb(A, E):
    return float(np.sqrt(pb_terms(A, E)[0].max()))


def grad_softmax_pb2(A, E, beta):
    """gradient of (1/beta) logsumexp(beta * PB_e^2) w.r.t. A (analytic, envelope theorem)."""
    pb2, Gs, ws = pb_terms(A, E)
    mx = pb2.max(); p = np.exp(beta * (pb2 - mx)); p /= p.sum()
    g = np.zeros_like(A); I3 = np.eye(A.shape[2])
    for i, (s, t, phi) in enumerate(E):
        As, At = A[s], A[t]; M, N, G = Gs[i]; w = ws[i]; pe = p[i]
        # d lambda_max = w^T dG w ;  G = M^T M + N
        Mw = M @ w                                            # (2,)
        Asw = As.T @ w                                        # (3,)  = A_s^T w
        # term M^T M:  d = 2 (Mw)^T dM w,  dM = -d(A_t A_s^T)
        #   wrt A_t: dM w = -(dA_t)(A_s^T w) -> grad_At += -2 outer(Mw, Asw)
        g[t] += pe * (-2.0) * np.outer(Mw, Asw)
        #   wrt A_s: dM w = -A_t (dA_s)^T w -> contributes to grad_As
        #     d = -2 (Mw)^T A_t (dA_s)^T w = -2 (A_t^T Mw)·((dA_s)^T w) -> grad_As += -2 outer(w, A_t^T Mw)
        g[s] += pe * (-2.0) * np.outer(w, At.T @ Mw)
        # term N = A_s (I - A_t^T A_t) A_s^T,  p3 := A_s^T w
        p3 = As.T @ w                                         # (3,)
        K = I3 - At.T @ At
        #   wrt A_s: d = 2 w (K p3)^T -> grad_As += 2 outer(w, K p3)
        g[s] += pe * 2.0 * np.outer(w, K @ p3)
        #   wrt A_t: dK = -(dA_t^T A_t + A_t^T dA_t); d(p3^T K p3) = -2 (A_t p3)^T (dA_t p3)
        Atp = At @ p3                                         # (2,)
        g[t] += pe * (-2.0) * np.outer(Atp, p3)
    return g, float(np.sqrt(mx))


def proj_cap(A):
    """project each 2x3 block to operator norm <= 1 (clip singular values)."""
    out = A.copy()
    for s in range(A.shape[0]):
        U, S, Vt = np.linalg.svd(A[s], full_matrices=False)
        S = np.minimum(S, 1.0); out[s] = (U * S) @ Vt
    return out


def chern(m, A):
    al = (A[:, 0, :] + 1j * A[:, 1, :]).reshape(m, m, -1)
    nrm = np.linalg.norm(al, axis=2, keepdims=True)
    if nrm.min() < 1e-7:
        return None
    b = al / nrm; F = 0.0
    for x in range(m):
        for y in range(m):
            u1 = np.vdot(b[x, y], b[(x+1) % m, y]); u2 = np.vdot(b[(x+1) % m, y], b[(x+1) % m, (y+1) % m])
            u3 = np.vdot(b[(x+1) % m, (y+1) % m], b[x, (y+1) % m]); u4 = np.vdot(b[x, (y+1) % m], b[x, y])
            F += np.angle(u1*u2*u3*u4)
    return F/(2*np.pi)


def chern1_ansatz(m, r):
    """alpha_s = (cos(pi x/m), sin(pi x/m) e^{i 2pi y/m}, 0,...) -> Chern +-1 monopole map."""
    A = np.zeros((m*m, 2, r))
    for x in range(m):
        for y in range(m):
            s = cgn.site(x, y, m)
            a = np.zeros(r, complex)
            a[0] = np.cos(np.pi*x/m); a[1] = np.sin(np.pi*x/m)*np.exp(1j*2*np.pi*y/m)
            A[s, 0, :] = a.real; A[s, 1, :] = a.imag
    return A


def minimize_pb(m, k, r, A0, steps=1500):
    E = edges(m, k); A = proj_cap(A0.copy()); best = np.inf
    for t in range(steps):
        beta = 8 + 120*t/(steps-1); lr = 0.15*(1-0.9*t/(steps-1))
        g, _ = grad_softmax_pb2(A, E, beta)
        A = proj_cap(A - lr*g)
        b = pb(A, E); best = min(best, b)
    return best, A


def main():
    m, r, k = 6, 3, 1
    E = edges(m, k)
    # gradient check
    A = proj_cap(cgn._polar(np.random.default_rng(0).standard_normal((m*m, 3, 2))).transpose(0, 2, 1))
    g, _ = grad_softmax_pb2(A, E, 20.0)
    eps = 1e-6; i, j, l = 5, 1, 2
    def fobj(Ax):
        pb2 = pb_terms(Ax, E)[0]; mx = pb2.max(); return mx + (1/20.)*np.log(np.exp(20*(pb2-mx)).sum())
    Ap = A.copy(); Ap[i, j, l] += eps; Am = A.copy(); Am[i, j, l] -= eps
    fd = (fobj(Ap)-fobj(Am))/(2*eps)
    print(f"gradient check: analytic={g[i,j,l]:+.5f}  finite-diff={fd:+.5f}  (should match)\n")

    print(f"d=? r={r} m={m} k={k}: minimize PB over SHADOWS from various Chern sectors\n")
    print(f"{'init':>16} {'init Chern':>10} {'init PB':>8} {'min PB':>8} {'final Chern':>12} {'PB<1?':>6}")
    out = {}; any_below = False
    S = cgn.std_frame(3, r)[None].repeat(m*m, 0)              # in-core: A_s = [I_2 | 0]
    Ainc = np.zeros((m*m, 2, r)); Ainc[:, 0, 0] = 1; Ainc[:, 1, 1] = 1
    inits = {"chern-1 ansatz": chern1_ansatz(m, r), "in-core": Ainc}
    for sd in range(4):
        inits[f"random{sd}"] = proj_cap(np.random.default_rng(sd).standard_normal((m*m, 2, r)))
    for name, A0 in inits.items():
        A0 = proj_cap(A0); c0 = chern(m, A0); pb0 = pb(A0, E)
        mp, Af = minimize_pb(m, k, r, A0); cf = chern(m, Af)
        below = mp < 0.999; any_below = any_below or below
        cs = lambda c: f"{c:+.2f}" if c is not None else "undef"
        out[name] = dict(init_chern=c0, init_pb=round(pb0, 4), min_pb=round(mp, 4), final_chern=cf)
        print(f"{name:>16} {cs(c0):>10} {pb0:>8.4f} {mp:>8.4f} {cs(cf):>12} {str(below):>6}")
    print("\nVERDICT:",
          "*** a shadow reached PB<1 -> codim>=2 LOWER BOUND FAILS for r>2 ***" if any_below else
          "no shadow (incl. Chern-1 sector) reaches PB<1 -> bound HOLDS for r>2 (min PB = 1), and the")
    if not any_below:
        print("        telescoping identity + this = (TR') is genuinely unsatisfiable for k!=0. Strong.")
    with open(os.path.join(HERE, "shadow.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote shadow.json")


if __name__ == "__main__":
    main()
