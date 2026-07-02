"""expX_energy.py — does the TRACE condition (T) alone obstruct winding, or is the sharp det (D) needed?

The vortex-core lemma is being attacked via the covariant Dirichlet energy. Summing the necessary
trace condition (T)  2 Re[e^{i phi} <a_t,a_s>] > ||a_s||^2  over all directed edges gives (Entry 41)
    (T) on every edge  ==>  E < 2 Sum_s ||a_s||^2 ,
    E := Sum_e ||a_t - e^{i phi_e} a_s||^2 = 4 Sum_s||a_s||^2 - 2 Sum_e Re[e^{i phi_e}<a_t,a_s>]
(covariant Dirichlet energy under the flux connection). The energy/Bogomolny route would prove the
lemma using ONLY (T) -- IF (T) on all edges is already infeasible for k != 0.

THIS SCRIPT decides whether (T)-alone suffices. It maximizes the trace-feasibility margin
    g(alpha) := min_e tr(H_e),   tr(H_e) = 2 tr(R(phi_e)^T A_t A_s^T) - ||A_s||_F^2
over capped shadows ||A_s||_op <= 1 (softmin gradient ascent, analytic grad finite-diff checked).
  * max_alpha min_e tr(H_e) <= 0  ->  (T) can NEVER hold strictly on all edges  ->  PB>=1 by the
        TRACE/ENERGY route alone (no det needed). Cleanest proof; the energy lemma is the whole story.
  * max_alpha min_e tr(H_e) >  0  ->  some config satisfies all (T) but still PB>=1  ->  the
        obstruction lives in det(H_e); energy route needs the sharp (D) too.
Also reports the optimum's Chern and the energy ratio E/(2 Sum||a||^2) (should be < 1 iff (T) all hold).
Controls: r=2 (proven case) and a constant-vector config. Reuses expU (edges, chern). Writes energy.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("expU", os.path.join(HERE, "expU_shadow.py"))
expU = importlib.util.module_from_spec(spec); spec.loader.exec_module(expU)
edges, chern, Rphi, proj_cap = expU.edges, expU.chern, expU.Rphi, expU.proj_cap


def tr_terms(A, E):
    """per-edge tr(H_e) = 2 tr(R^T A_t A_s^T) - ||A_s||_F^2."""
    out = np.empty(len(E))
    for i, (s, t, phi) in enumerate(E):
        As, At = A[s], A[t]
        out[i] = 2.0 * np.trace(Rphi(phi).T @ At @ As.T) - np.sum(As * As)
    return out


def energy_ratio(A, E):
    """E = Sum_e ||a_t - e^{i phi} a_s||^2  and  2 Sum_s ||a_s||^2 ; returns (E, 2*sum, ratio).
    a_s = A[s,0,:] + i A[s,1,:]."""
    al = A[:, 0, :] + 1j * A[:, 1, :]
    Etot = 0.0
    for (s, t, phi) in E:
        Etot += float(np.sum(np.abs(al[t] - np.exp(1j * phi) * al[s]) ** 2))
    two_sum = 2.0 * float(np.sum(np.abs(al) ** 2))
    return Etot, two_sum, Etot / two_sum


def grad_softmin_tr(A, E, beta):
    """gradient of (1/beta) (-logsumexp(-beta tr_e)) ~ min_e tr_e (ascent direction)."""
    tr = tr_terms(A, E); mn = tr.min()
    w = np.exp(-beta * (tr - mn)); w /= w.sum()       # softmin weights
    g = np.zeros_like(A)
    for i, (s, t, phi) in enumerate(E):
        R = Rphi(phi); we = w[i]
        g[t] += we * (2.0 * R @ A[s])                  # d tr_e / dA_t = 2 R A_s
        g[s] += we * (2.0 * R.T @ A[t] - 2.0 * A[s])   # d tr_e / dA_s = 2 R^T A_t - 2 A_s
    return g, float(mn)


def maximize_trace_margin(m, k, r, A0, steps=1500):
    E = edges(m, k); A = proj_cap(A0.copy()); best = -np.inf; bestA = A
    for t in range(steps):
        beta = 8 + 160 * t / (steps - 1); lr = 0.2 * (1 - 0.9 * t / (steps - 1))
        g, _ = grad_softmin_tr(A, E, beta)
        A = proj_cap(A + lr * g)                        # ASCENT
        mn = tr_terms(A, E).min()
        if mn > best:
            best, bestA = mn, A.copy()
    return best, bestA


def gradient_check(m, k, r):
    E = edges(m, k); rng = np.random.default_rng(0)
    A = proj_cap(rng.standard_normal((m * m, 2, r)))
    g, _ = grad_softmin_tr(A, E, 20.0)
    def f(Ax):
        tr = tr_terms(Ax, E); mn = tr.min(); return mn - (1 / 20.) * np.log(np.exp(-20 * (tr - mn)).sum())
    eps = 1e-6; i, j, l = 7, 1, 2
    Ap = A.copy(); Ap[i, j, l] += eps; Am = A.copy(); Am[i, j, l] -= eps
    fd = (f(Ap) - f(Am)) / (2 * eps)
    print(f"gradient check: analytic={g[i,j,l]:+.5f}  finite-diff={fd:+.5f}\n")


def main():
    m, k = 6, 1
    gradient_check(m, k, 3)
    out = {}
    print(f"max_alpha min_e tr(H_e) over capped shadows (m={m}, k={k}).  <=0 => (T) infeasible => trace/energy")
    print(f"route proves PB>=1 alone;  >0 => need the sharp det (D).\n")
    print(f"{'case':>16} {'max min tr':>11} {'Chern':>7} {'E/(2S||a||^2)':>14}  reading")
    for r in (2, 3, 4):
        best = -np.inf; bestA = None
        for sd in range(4):
            A0 = proj_cap(np.random.default_rng(sd).standard_normal((m * m, 2, r)))
            mt, Af = maximize_trace_margin(m, k, r, A0)
            if mt > best:
                best, bestA = mt, Af
        E = edges(m, k); Et, two, ratio = energy_ratio(bestA, E); cf = chern(m, bestA)
        cs = f"{cf:+.2f}" if cf is not None else "undef"
        reading = "(T) infeasible: trace route OK" if best <= 1e-4 else "(T) feasible: need det (D)"
        print(f"{'r='+str(r):>16} {best:>+11.5f} {cs:>7} {ratio:>14.4f}  {reading}")
        out[f"r={r}"] = dict(max_min_tr=round(float(best), 5), chern=cf, energy_ratio=round(ratio, 4))
    # constant-vector control (Chern 0): expect (T) violated badly (cos phi_e < 1/2 on many edges)
    r = 3; v = np.zeros((2, r)); v[0, 0] = 1.0
    Acst = np.broadcast_to(v, (m * m, 2, r)).copy()
    E = edges(m, k); mt_c = tr_terms(Acst, E).min(); Et, two, ratio_c = energy_ratio(Acst, E)
    print(f"{'const-v (Chern0)':>16} {mt_c:>+11.5f} {'+0.00':>7} {ratio_c:>14.4f}  (reference: flat unit vector)")
    out["const_v"] = dict(min_tr=round(float(mt_c), 5), energy_ratio=round(ratio_c, 4))

    print("\nVERDICT:", end=" ")
    worst = max(out[f"r={r}"]["max_min_tr"] for r in (2, 3, 4))
    if worst <= 1e-4:
        print(f"max_alpha min_e tr(H_e) <= 0 for all r (largest = {worst:+.5f}) -> (T) is infeasible on")
        print(f"        the full edge set; the TRACE/ENERGY route proves PB>=1 for ALL r (no det needed).")
    else:
        print(f"some r reaches min_e tr(H_e) = {worst:+.5f} > 0 -> (T) alone is satisfiable; the r>2")
        print(f"        obstruction genuinely needs the sharp det (D). Energy route must add (D).")
    with open(os.path.join(HERE, "energy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote energy.json")


if __name__ == "__main__":
    main()
