"""expY_gaussbonnet.py — complex reformulation of the SHARP det condition, toward a Gauss-Bonnet proof.

Entry 42 pinned the r>2 obstruction to det(H_e)>0 (the Berry-curvature term), NOT the trace/energy.
To find a GLOBAL (Gauss-Bonnet) identity we first need det(H_e) in clean COMPLEX shadow variables,
since the winding is a phase phenomenon. For shadow alpha_s in C^r (= A_s[0,:]+i A_s[1,:]) define
    P_e = <alpha_t, alpha_s>      (Hermitian overlap; conj-linear in first slot) -- the U(1) connection
    Q_e = (alpha_t, alpha_s)      = sum_j alpha_{t,j} alpha_{s,j}  (bilinear, no conj) -- weight-2
    w_s = (alpha_s, alpha_s)      = sum_j alpha_{s,j}^2            (self-bilinear "anisotropy")
Any 2x2 real M = a I + b J + c sigma_z + d sigma_x  (J=[[0,-1],[1,0]], sigma_z=diag(1,-1), sigma_x=[[0,1],[1,0]]).

CLAIMS to verify to machine precision (then reason about globally):
 (C1) cap ||A_s||_op<=1  <=>  ||alpha_s||^2 + |w_s| <= 2   (since sigma^2_{1,2} = (||a||^2 +- |w_s|)/2).
 (C2) H_e = (2a-s0) I + (2c-sz) sigma_z + (2d-sx) sigma_x with s0=||alpha_s||^2/2, (sz,sx)<->w_s/... ,
      and the conformal part 2(a+ib) and anti-conformal 2(c+id) of B_e=R(phi)^T A_t A_s^T expressed
      via e^{i phi} P_e and e^{i phi} Q_e (signs TBD numerically).
 (C3) tr(H_e) = 2 Re[e^{i phi} P_e] - ||alpha_s||^2 .
 (C4) det(H_e) = (tr(H_e)/2)^2 - | e^{i phi} Q_e - w_s |^2 / (something)   <-- the KEY det formula;
      exact constants found numerically. det>0 is then |conformal trace gap| > |anti-conformal Q-w gap|.
This script fits/verifies the exact coefficients. Reuses expU (Rphi). Writes gaussbonnet.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("expU", os.path.join(HERE, "expU_shadow.py"))
expU = importlib.util.module_from_spec(spec); spec.loader.exec_module(expU)
Rphi = expU.Rphi
J = np.array([[0.0, -1.0], [1.0, 0.0]]); SZ = np.array([[1.0, 0.0], [0.0, -1.0]]); SX = np.array([[0.0, 1.0], [1.0, 0.0]])


def decomp(M):
    """M = a I + b J + c sigma_z + d sigma_x ; return (a,b,c,d)."""
    a = (M[0, 0] + M[1, 1]) / 2; b = (M[1, 0] - M[0, 1]) / 2
    c = (M[0, 0] - M[1, 1]) / 2; d = (M[0, 1] + M[1, 0]) / 2
    return a, b, c, d


def cplx(A):
    """2 x r real -> alpha in C^r."""
    return A[0, :] + 1j * A[1, :]


def rand_capped(r, rng):
    A = rng.standard_normal((2, r))
    U, S, Vt = np.linalg.svd(A, full_matrices=False); S = np.minimum(S, 1.0)
    return (U * S) @ Vt


def main():
    rng = np.random.default_rng(0); r = 4
    maxerr = dict(C1=0.0, C3=0.0, trid=0.0)
    # gather samples to FIT C2/C4 coefficients, then verify
    rows = []
    for _ in range(4000):
        As = rand_capped(r, rng); At = rand_capped(r, rng); phi = rng.uniform(0, 2 * np.pi)
        al_s, al_t = cplx(As), cplx(At)
        P = np.vdot(al_t, al_s)              # <alpha_t,alpha_s> = sum conj(al_t) al_s  (conj-linear 1st)
        Q = np.sum(al_t * al_s)              # (alpha_t,alpha_s) bilinear
        w_s = np.sum(al_s * al_s); w_t = np.sum(al_t * al_t)
        S = As @ As.T; B = Rphi(phi).T @ (At @ As.T)
        a, b, c, d = decomp(B)
        s0 = 0.5 * np.sum(np.abs(al_s) ** 2)
        # C1: cap holds by construction (sigma<=1); check ||a||^2+|w_s| <= 2 and sing val formula
        sv = np.linalg.svd(As, compute_uv=False)
        na2 = np.sum(np.abs(al_s) ** 2)
        maxerr["C1"] = max(maxerr["C1"], abs((sv[0] ** 2) - (na2 + abs(w_s)) / 2))
        # C3: tr(H)=2Re[e^{i phi}P]-||a_s||^2 ;  tr(H)=2(2a-s0)? no: H=2Sym(B)-S, tr=2*tr(SymB)-tr S=2*(2a)-2s0
        trH = 2 * (2 * a) - 2 * s0
        maxerr["C3"] = max(maxerr["C3"], abs(trH - (2 * np.real(np.exp(1j * phi) * P) - na2)))
        # record conformal (a,b) and anti-conformal (c,d) of B vs e^{i phi}*P, e^{i phi}*Q (and conjugates)
        rows.append((phi, a, b, c, d, P, Q, w_s, w_t, s0))

    # FIT: is 2(a+ib) = e^{-i phi} P  or  e^{i phi} conj(P) ... ; 2(c+id) = e^{i phi} Q or e^{-i phi} conj(Q)?
    rows_arr = rows
    def test_conformal(predict):
        e = 0.0
        for (phi, a, b, c, d, P, Q, w_s, w_t, s0) in rows_arr:
            e = max(e, abs((a + 1j * b) - predict(phi, P, Q, w_s, w_t)))
        return e
    def test_anti(predict):
        e = 0.0
        for (phi, a, b, c, d, P, Q, w_s, w_t, s0) in rows_arr:
            e = max(e, abs((c + 1j * d) - predict(phi, P, Q, w_s, w_t)))
        return e

    conf_cands = {
        "a+ib = 1/2 e^{i phi} conj(P)":  lambda phi, P, Q, ws, wt: 0.5 * np.exp(1j * phi) * np.conj(P),
        "a+ib = 1/2 e^{-i phi} P":       lambda phi, P, Q, ws, wt: 0.5 * np.exp(-1j * phi) * P,
        "a+ib = 1/2 e^{i phi} P":        lambda phi, P, Q, ws, wt: 0.5 * np.exp(1j * phi) * P,
    }
    anti_cands = {
        "c+id = 1/2 e^{i phi} Q":        lambda phi, P, Q, ws, wt: 0.5 * np.exp(1j * phi) * Q,
        "c+id = 1/2 e^{-i phi} Q":       lambda phi, P, Q, ws, wt: 0.5 * np.exp(-1j * phi) * Q,
        "c+id = 1/2 e^{i phi} conj(Q)":  lambda phi, P, Q, ws, wt: 0.5 * np.exp(1j * phi) * np.conj(Q),
    }
    print("=== conformal part (a+ib) of B_e candidates (max err over 4000 samples) ===")
    best_conf = min(conf_cands.items(), key=lambda kv: test_conformal(kv[1]))
    for name, f in conf_cands.items():
        print(f"   {name:35s} maxerr={test_conformal(f):.2e}")
    print("=== anti-conformal part (c+id) of B_e candidates ===")
    best_anti = min(anti_cands.items(), key=lambda kv: test_anti(kv[1]))
    for name, f in anti_cands.items():
        print(f"   {name:35s} maxerr={test_anti(f):.2e}")

    print(f"\nC1 (cap: sigma1^2=(||a||^2+|w_s|)/2) maxerr = {maxerr['C1']:.2e}")
    print(f"C3 (tr H = 2Re[e^i^phi P]-||a_s||^2)  maxerr = {maxerr['C3']:.2e}")
    print(f"\nBEST conformal: {best_conf[0]}   (err {test_conformal(best_conf[1]):.1e})")
    print(f"BEST anti-conf: {best_anti[0]}   (err {test_anti(best_anti[1]):.1e})")

    # S traceless: (alpha_s,alpha_s)=w_s = (S00-S11)+2i S01 = 2 sz + 2i sx  =>  sz+i sx = w_s/2  (NO conj)
    err_S = 0.0
    for _ in range(500):
        As = rand_capped(r, rng); al_s = cplx(As); S = As @ As.T
        _, _, sz, sx = decomp(S); w_s = np.sum(al_s * al_s)
        err_S = max(err_S, abs((sz + 1j * sx) - 0.5 * w_s))
    print(f"\nS traceless (sz+i sx) = w_s/2 ? maxerr = {err_S:.2e}")

    # CORRECTED det formula. H=(2a-s0)I+(2c-sz)sigma_z+(2d-sx)sigma_x ; det = (2a-s0)^2 - (2c-sz)^2-(2d-sx)^2.
    #  2a-s0 = Re[e^{i phi}P] - ||a_s||^2/2 =: g_e (real, = tr(H)/2)
    #  (2c-sz)+i(2d-sx) = 2(c+id)-(sz+i sx) = e^{-i phi}Q - w_s/2 =: G_e  (complex)
    # => det(H_e) = g_e^2 - |G_e|^2 .  And H_e > 0  <=>  g_e > |G_e|  (single sharp inequality, g_e>=0 implied).
    det_err = 0.0; sharp_err = 0
    for _ in range(3000):
        As = rand_capped(r, rng); At = rand_capped(r, rng); phi = rng.uniform(0, 2 * np.pi)
        al_s, al_t = cplx(As), cplx(At)
        P = np.vdot(al_t, al_s); Q = np.sum(al_t * al_s); w_s = np.sum(al_s * al_s)
        n_s = np.sum(np.abs(al_s) ** 2)
        B = Rphi(phi).T @ (At @ As.T); H = (B + B.T) - (As @ As.T)
        g_e = np.real(np.exp(1j * phi) * P) - 0.5 * n_s
        G_e = np.exp(-1j * phi) * Q - 0.5 * w_s
        det_err = max(det_err, abs(np.linalg.det(H) - (g_e ** 2 - np.abs(G_e) ** 2)))
        # sharp characterization: H>0 (both eig>0) <=> g_e>|G_e|
        posdef = np.linalg.eigvalsh(H).min() > 1e-9
        if posdef != (g_e > np.abs(G_e) + 1e-9):
            sharp_err += 1
    print(f"\nCORRECTED DET:  det(H_e) = g_e^2 - |G_e|^2,  g_e=Re[e^i^phi P]-||a_s||^2/2,  "
          f"G_e=e^-i^phi Q - w_s/2   maxerr = {det_err:.2e}")
    print(f"SHARP CHARACTERIZATION:  PB_e<1  <=>  H_e>0  <=>  g_e > |G_e|   (mismatches/3000 = {sharp_err})")

    # ---- TOPOLOGICAL PAYLOAD: w_s=(alpha_s,alpha_s) is a section of the SQUARED line bundle (Chern -2k),
    # so it must vanish/wind. Test on a (T)-feasible WINDING config from the flux family. ----
    print("\n--- topological probe: does w_s=(alpha_s,alpha_s) wind with 2*Chern on a winding config? ---")
    m, k = 6, 1
    spec2 = importlib.util.spec_from_file_location("expX", os.path.join(HERE, "expX_energy.py"))
    expX = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(expX)
    E = expU.edges(m, k); rngw = np.random.default_rng(3)
    bestmt, bestA = -np.inf, None
    for sd in range(4):
        A0 = expU.proj_cap(rngw.standard_normal((m * m, 2, r)))
        mt, Af = expX.maximize_trace_margin(m, k, r, A0, steps=1200)
        if mt > bestmt:
            bestmt, bestA = mt, Af
    al = bestA[:, 0, :] + 1j * bestA[:, 1, :]            # (m*m, r)
    ch = expU.chern(m, bestA)
    w = np.array([np.sum(al[s] * al[s]) for s in range(m * m)]).reshape(m, m)   # w_s field
    # lattice DEGREE of the complex field w over the torus = total enclosed zeros = (expected) 2*Chern.
    # principal-branch plaquette phase sum, NO guard; jitter w off any exact zero so phases are defined.
    wj = w + 1e-7 * (np.random.default_rng(7).standard_normal(w.shape) + 1j * np.random.default_rng(8).standard_normal(w.shape))
    Wtot = 0.0
    for x in range(m):
        for y in range(m):
            loop = [wj[x, y], wj[(x+1) % m, y], wj[(x+1) % m, (y+1) % m], wj[x, (y+1) % m], wj[x, y]]
            for i in range(4):
                Wtot += np.angle(loop[i+1] / loop[i])    # principal branch in (-pi,pi]
    w_winding = Wtot / (2 * np.pi)
    n_cores = int(np.sum(np.abs(w) < 0.05))              # near-zeros of w_s = isotropic cores
    print(f"   (T)-feasible config: min_e tr={bestmt:+.3f}, line Chern={ch:+.2f}; "
          f"min|w_s|={np.abs(w).min():.4f}, #cores(|w|<.05)={n_cores}, w-degree={w_winding:+.2f}  "
          f"(expect 2*Chern={2*ch if ch else None:+.2f})")

    out = dict(C1_caperr=float(maxerr["C1"]), C3_trerr=float(maxerr["C3"]),
               best_conformal=best_conf[0], best_anti=best_anti[0],
               S_traceless_err=float(err_S), det_formula_err=float(det_err),
               sharp_mismatches=int(sharp_err), line_chern=ch,
               w_min_abs=float(np.abs(w).min()), w_winding=float(w_winding))
    with open(os.path.join(HERE, "gaussbonnet.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote gaussbonnet.json")


if __name__ == "__main__":
    main()
