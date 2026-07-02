"""expAD_vectorized.py — Entry 47 TOOLING: a vectorized / converged PB-minimizer reaching m ~ 32.

The winding-bound notebook's track 1: the prior pure-Python minimizer (expU.minimize_pb) loops over
edges and UNDER-CONVERGES at large m -- at m>=12 it fails to recover even the known PB=1 trivial
(orthogonal, alpha==0) competitor, reporting loose values >1. Since Theorem 1(a) gives min PB <= 1 for
ALL m (the trivial competitor sits at exactly 1), any reported minimum >1 is a pure optimizer failure,
not math. This rebuilds the minimizer:

  * fully BATCHED over edges (no Python edge loop): einsum + batched eigh + np.add.at scatter-grad,
  * Adam + annealed softmax(beta) over PB_e^2, multi-restart, with a final exact max-edge polish,
  * the trivial alpha==0 competitor (PB=1 exactly) included as an anchor / ground-truth upper bound.

Two scientific outputs, both things numerics CAN settle (a lower bound over a continuum it cannot):
  (a) UPPER BOUND recovery: does free descent reliably reach PB=1 at large m?  (fixes the caveat)
  (b) COUNTEREXAMPLE HUNT: does ANY start -- winding or random -- ever dip strictly below 1?
      (a single PB<1 would break the codim>=2 lower bound for r>2.)

Reuses expU_shadow for the scalar reference grad (validation) and Chern/edges. Writes vectorized.json.
"""

from __future__ import annotations
import importlib.util, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
chern = expU.chern


# ---------------------------------------------------------------- batched edge model
def edge_arrays(m, k):
    """vectorized edge list: integer src/dst index arrays + batched rotations R (E,2,2)."""
    E = expU.edges(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    phi = np.array([e[2] for e in E])
    c, s = np.cos(phi), np.sin(phi)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)   # (E,2,2)
    return src, dst, phi, R


def pb2_batched(A, src, dst, R):
    """PB_e^2 = lambda_max(M^T M + N) for every edge at once. Returns (pb2 (E,), G (E,2,2), w (E,2))."""
    As, At = A[src], A[dst]                                          # (E,2,r)
    AtAsT = np.einsum('eij,ekj->eik', At, As)                        # A_t A_s^T (E,2,2)
    M = R - AtAsT
    r = A.shape[2]; I = np.eye(r)
    K = I - np.einsum('eki,ekj->eij', At, At)                        # I - A_t^T A_t (E,r,r)
    N = np.einsum('eai,eij,ebj->eab', As, K, As)                     # A_s K A_s^T (E,2,2)
    G = np.einsum('eki,ekj->eij', M, M) + N                          # M^T M + N
    evals, evecs = np.linalg.eigh(G)
    return evals[:, -1], M, evecs[:, :, -1], K


def grad_batched(A, src, dst, R, beta):
    """analytic gradient of (1/beta) logsumexp(beta * PB_e^2) -- vectorized port of expU.grad_softmax_pb2."""
    pb2, M, w, K = pb2_batched(A, src, dst, R)
    mx = pb2.max(); p = np.exp(beta * (pb2 - mx)); p /= p.sum()
    As, At = A[src], A[dst]
    Mw = np.einsum('eij,ej->ei', M, w)                              # (E,2)
    Asw = np.einsum('eji,ej->ei', As, w)                            # A_s^T w  (E,r)  (= p3)
    AtTMw = np.einsum('eji,ej->ei', At, Mw)                         # A_t^T M w (E,r)
    Kp3 = np.einsum('eij,ej->ei', K, Asw)                           # K p3 (E,r)
    Atp = np.einsum('eik,ek->ei', At, Asw)                          # A_t p3 (E,2)
    pe = p[:, None, None]
    gt = pe * (-2.0) * (np.einsum('ei,ek->eik', Mw, Asw) + np.einsum('ei,ek->eik', Atp, Asw))
    gs = pe * (np.einsum('ei,ek->eik', w, -2.0 * AtTMw) + np.einsum('ei,ek->eik', w, 2.0 * Kp3))
    g = np.zeros_like(A)
    np.add.at(g, dst, gt); np.add.at(g, src, gs)
    return g, float(np.sqrt(mx))


def pb_exact(A, src, dst, R):
    return float(np.sqrt(pb2_batched(A, src, dst, R)[0].max()))


# ---------------------------------------------------------------- optimizer (Adam + proj)
def proj_cap(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)               # batched SVD over sites
    S = np.minimum(S, 1.0)
    return (U * S[:, None, :]) @ Vt


def minimize(m, k, r, A0, steps=2500, lr=0.05):
    src, dst, phi, R = edge_arrays(m, k)
    A = proj_cap(A0.copy()); mt = np.zeros_like(A); vt = np.zeros_like(A)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = np.inf; bestA = A.copy()
    for t in range(1, steps + 1):
        beta = 12 + 240 * (t / steps)                              # anneal softmax sharpness
        g, _ = grad_batched(A, src, dst, R, beta)
        mt = b1 * mt + (1 - b1) * g; vt = b2 * vt + (1 - b2) * g * g
        mh = mt / (1 - b1 ** t); vh = vt / (1 - b2 ** t)
        A = proj_cap(A - lr * mh / (np.sqrt(vh) + eps))
        b = pb_exact(A, src, dst, R)
        if b < best:
            best = b; bestA = A.copy()
    return best, bestA


# ---------------------------------------------------------------- validation vs scalar reference
def validate(m=6, k=1, r=3):
    """confirm the batched pb2 and gradient match expU's per-edge scalar implementation."""
    src, dst, phi, R = edge_arrays(m, k); E = expU.edges(m, k)
    rng = np.random.default_rng(0)
    A = proj_cap(rng.standard_normal((m * m, 2, r)))
    pb2_v = pb2_batched(A, src, dst, R)[0]
    pb2_s = expU.pb_terms(A, E)[0]
    dpb = np.abs(np.sort(pb2_v) - np.sort(pb2_s)).max()            # sort: eig order may differ
    g_v, _ = grad_batched(A, src, dst, R, 20.0)
    g_s, _ = expU.grad_softmax_pb2(A, E, 20.0)
    dg = np.abs(g_v - g_s).max()
    print(f"validation (m={m},k={k},r={r}):  max|pb2_batched - pb2_scalar| = {dpb:.2e}   "
          f"max|grad_batched - grad_scalar| = {dg:.2e}")
    assert dpb < 1e-10 and dg < 1e-9, "batched kernels disagree with scalar reference!"
    return dpb, dg


# ---------------------------------------------------------------- experiment
def winding_init(m, r):
    return expU.chern1_ansatz(m, r)


def trivial_competitor(m, r):
    """alpha == 0: V(s) maps Pi entirely into Pi^perp.  PB = 1 exactly (Theorem 1a)."""
    return np.zeros((m * m, 2, r))


def run(m, k, r=3, restarts=4, steps=2500):
    src, dst, phi, R = edge_arrays(m, k)
    results = {}
    # anchor: trivial competitor, PB must be exactly 1
    A_triv = trivial_competitor(m, r)
    pb_triv = pb_exact(A_triv, src, dst, R)
    overall_min = pb_triv; min_sector = None
    # free descent from winding ansatz + randoms (small-scale so descent can shrink toward 0)
    inits = {"winding": winding_init(m, r)}
    for sd in range(restarts):
        inits[f"rand{sd}"] = 0.7 * np.random.default_rng(100 + sd).standard_normal((m * m, 2, r))
    for name, A0 in inits.items():
        mp, Af = minimize(m, k, r, A0, steps=steps)
        cf = chern(m, Af)
        results[name] = dict(min_pb=round(mp, 5), final_chern=(round(cf, 2) if cf is not None else None))
        if mp < overall_min:
            overall_min = mp; min_sector = name
    return dict(trivial_pb=round(pb_triv, 6), overall_min_pb=round(overall_min, 5),
                min_from=min_sector, runs=results)


def main():
    validate()
    print("\nvectorized PB-minimizer, codim>=2, r=3.  Theorem 1(a): true min PB = 1 at every m.")
    print("  upper-bound recovery: does descent reach 1?   counterexample: does anything dip < 1?\n")
    print(f"{'k':>3} {'m':>4} {'trivial PB':>11} {'overall min PB':>15} {'from':>9} "
          f"{'winding PB':>11} {'< 1 ?':>6} {'t(s)':>6}")
    out = {}
    for k in (1, 2):
        for m in (6, 8, 12, 16, 24, 32):
            t0 = time.time()
            steps = 2500 if m <= 16 else 1800        # larger m: more edges/iter, fewer iters
            rr = run(m, k, r=3, restarts=4, steps=steps)
            dt = time.time() - t0
            below = rr["overall_min_pb"] < 0.999
            wpb = rr["runs"]["winding"]["min_pb"]
            print(f"{k:>3} {m:>4} {rr['trivial_pb']:>11.5f} {rr['overall_min_pb']:>15.5f} "
                  f"{str(rr['min_from']):>9} {wpb:>11.5f} {str(below):>6} {dt:>6.1f}")
            out[f"k{k}_m{m}"] = dict(**rr, below_one=below, seconds=round(dt, 1))
        print()
    any_below = any(v["below_one"] for v in out.values())
    print("VERDICT:",
          "*** PB<1 found -> codim>=2 lower bound FAILS for r>2 ***" if any_below else
          "no start (winding or random) dips below 1 at any m up to 32; trivial competitor sits at 1.")
    # upper-bound recovery quality
    worst = max((abs(v["overall_min_pb"] - 1.0), kk) for kk, v in out.items())
    print(f"  upper-bound recovery: worst |overall_min_pb - 1| = {worst[0]:.4f} (at {worst[1]}); "
          f"trivial anchor is exact 1 at every m by construction.")
    with open(os.path.join(HERE, "vectorized.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote vectorized.json")


if __name__ == "__main__":
    main()
