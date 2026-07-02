"""e4 — pair-level rigidity probe for FluxPairRigidity (PlateauConjecture.lean).

Pre-registered kill test (solvingplateauideas.md, Opus eval 2026-06-15):
directly measure dist(flux magnetic pair -> nearest commuting contraction pair) vs m.
If that distance decays with m (rather than flattening to a positive constant), the
dimension-free reading is wrong for the contraction setting and the route dies.

Two sectors:
  z = 0   : the unitary core (r = d = 2 complexified: classical magnetic translations
            U, V on m^2 sites, total flux 2*pi*k, commutation scalar e^{2*pi*i*k/m^2}).
  z > 0   : rank-deficient sector (U + 0_z, V + 0_z direct sums) — the lossy directions
            the Stage B floor must survive. Adversary may mix core and padding.

Method: penalty optimization over complex pairs (A, B),
    f = ||Up-A||_F^2 + ||Vp-B||_F^2 + lam * ||[A,B]||_F^2,  lam ramped,
with singular-value clipping to the contraction ball, then CERTIFICATION: joint Schur
purification to an exactly commuting normal contraction pair (A', B'), reporting
    d_cert = max(sigma_max(Up - A'), sigma_max(Vp - B')),
a rigorous upper bound on the distance. The question is whether min-found d_cert
plateaus in m and survives padding.

INTERPRETATION CORRECTION (2026-07-01, same day — do not misread the table):
the TRIVIAL commuting contraction pair (A,B) = (0,0) sits at distance exactly
max(||U||,||V||) = 1 from the flux pair, at every m and every padding. The optimizer
never found it (all d_cert > 1) — the same missed-trivial-basin failure documented for
the edge-level fits in speculativepaper5 §4a. Correct reading of this experiment:
  * true pair-level min over commuting contraction pairs is <= 1 ALWAYS (so any
    rigidity constant delta(k) is <= 1);
  * e4's contribution is only: no competitor BELOW 1 was found at any m or padding,
    i.e. the kill condition (decay toward 0) did not fire — weak evidence, not a floor;
  * d_cert values above 1 measure optimizer weakness, not distance;
  * the bott == k and ||[U,V]|| == 2|sin(pi k/m^2)| columns are exact and stand.
"""

import json
import numpy as np
from scipy.linalg import schur, svd

rng = np.random.default_rng(7)


def flux_pair(m, k):
    """Complex magnetic translations on the m x m torus, Landau gauge + boundary twist.
    U[(x+1)%m, y <- x, y] = exp(i*phi_h(x,y)),  phi_h = -2*pi*k*y/m if x == m-1 else 0
    V[x, (y+1)%m <- x, y] = exp(i*phi_v(x,y)),  phi_v = 2*pi*k*x/m^2
    (fluxAngle in Defs.lean, complexified on the (0,1)-plane core)."""
    n = m * m
    U = np.zeros((n, n), dtype=complex)
    V = np.zeros((n, n), dtype=complex)
    for x in range(m):
        for y in range(m):
            s = x * m + y
            ph = -2 * np.pi * k * y / m if x == m - 1 else 0.0
            U[((x + 1) % m) * m + y, s] = np.exp(1j * ph)
            pv = 2 * np.pi * k * x / m**2
            V[x * m + ((y + 1) % m), s] = np.exp(1j * pv)
    return U, V


def bott_index(U, V):
    """(1/2*pi) * sum of principal args of eig(V U V^* U^*)."""
    W = V @ U @ V.conj().T @ U.conj().T
    ev = np.linalg.eigvals(W)
    return np.sum(np.angle(ev)) / (2 * np.pi)


def clip_contraction(A):
    W, s, Xh = svd(A)
    return W @ (np.minimum(s, 1.0)[:, None] * Xh)


def specnorm(A):
    return svd(A, compute_uv=False)[0]


def grads(A, B, Up, Vp, lam):
    """Wirtinger gradients d f / d conj(A), d f / d conj(B)."""
    C = A @ B - B @ A
    gA = -(Up - A) + lam * (C @ B.conj().T - B.conj().T @ C)
    gB = -(Vp - B) + lam * (A.conj().T @ C - C @ A.conj().T)
    return gA, gB


def gradient_check():
    """Finite-difference check of the hand-coded gradients on a random small case."""
    n = 6
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Up = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Vp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    lam = 3.7

    def f(A, B):
        C = A @ B - B @ A
        return (np.linalg.norm(Up - A, "fro") ** 2 + np.linalg.norm(Vp - B, "fro") ** 2
                + lam * np.linalg.norm(C, "fro") ** 2)

    gA, gB = grads(A, B, Up, Vp, lam)
    eps = 1e-6
    errs = []
    for _ in range(6):
        dA = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        dB = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        num = (f(A + eps * dA, B + eps * dB) - f(A - eps * dA, B - eps * dB)) / (2 * eps)
        ana = 2 * np.real(np.sum(np.conj(gA) * dA) + np.sum(np.conj(gB) * dB))
        errs.append(abs(num - ana) / max(1.0, abs(num)))
    return max(errs)


def purify(A, B, Up, Vp, tries=8):
    """Joint Schur purification -> exactly commuting normal contraction pair.
    Returns the best certified max-spectral distance over random mixing weights."""
    best = np.inf
    n = A.shape[0]
    for _ in range(tries):
        mu = np.exp(2j * np.pi * rng.random())
        T, Q = schur(A + mu * B, output="complex")
        a = np.diag(Q.conj().T @ A @ Q)
        b = np.diag(Q.conj().T @ B @ Q)
        a = a / np.maximum(1.0, np.abs(a))
        b = b / np.maximum(1.0, np.abs(b))
        A2 = Q @ np.diag(a) @ Q.conj().T
        B2 = Q @ np.diag(b) @ Q.conj().T
        d = max(specnorm(Up - A2), specnorm(Vp - B2))
        best = min(best, d)
    return best


def optimize(Up, Vp, stages=(1.0, 10.0, 100.0, 1000.0), iters=1500, eta=2e-3):
    """Adam on the penalty objective, warm-started across the lambda ramp."""
    n = Up.shape[0]
    A = Up + 0.01 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    B = Vp + 0.01 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    mA = np.zeros_like(A); vA = np.zeros_like(A, dtype=float)
    mB = np.zeros_like(B); vB = np.zeros_like(B, dtype=float)
    b1, b2, epsn = 0.9, 0.999, 1e-8
    t = 0
    for lam in stages:
        for _ in range(iters):
            t += 1
            gA, gB = grads(A, B, Up, Vp, lam)
            mA = b1 * mA + (1 - b1) * gA
            vA = b2 * vA + (1 - b2) * np.abs(gA) ** 2
            mB = b1 * mB + (1 - b1) * gB
            vB = b2 * vB + (1 - b2) * np.abs(gB) ** 2
            c1, c2 = 1 - b1**t, 1 - b2**t
            A = A - eta * (mA / c1) / (np.sqrt(vA / c2) + epsn)
            B = B - eta * (mB / c1) / (np.sqrt(vB / c2) + epsn)
            if t % 25 == 0:
                A = clip_contraction(A)
                B = clip_contraction(B)
    A = clip_contraction(A)
    B = clip_contraction(B)
    return A, B


def run_config(m, k, pad_ratio):
    U, V = flux_pair(m, k)
    n = m * m
    z = int(round(pad_ratio * n))
    N = n + z
    Up = np.zeros((N, N), dtype=complex); Up[:n, :n] = U
    Vp = np.zeros((N, N), dtype=complex); Vp[:n, :n] = V

    comm = specnorm(U @ V - V @ U)
    bott = bott_index(U, V)

    # baseline certification straight from the pair (no optimization)
    d_base = purify(Up, Vp, Up, Vp)
    # adversarial optimization, then certification
    A, B = optimize(Up, Vp)
    resid = specnorm(A @ B - B @ A)
    d_opt_raw = max(specnorm(Up - A), specnorm(Vp - B))
    d_cert = purify(A, B, Up, Vp)
    return dict(m=m, k=k, n=n, z=z, comm=comm, bott=float(np.real(bott)),
                d_base=float(d_base), d_opt_raw=float(d_opt_raw),
                resid=float(resid), d_cert=float(min(d_cert, d_base)))


if __name__ == "__main__":
    gc = gradient_check()
    print(f"gradient check (max rel err over random directions): {gc:.2e}")
    assert gc < 1e-5, "hand-coded gradient wrong — abort"

    results = []
    print(f"{'m':>3} {'k':>2} {'z':>4} {'||[U,V]||':>10} {'bott':>7} "
          f"{'d_base':>8} {'d_raw':>8} {'resid':>9} {'d_cert':>8}")
    for k in (1,):
        for m in (3, 4, 5, 6, 8, 10):
            for pad in (0.0, 1.0):
                r = run_config(m, k, pad)
                results.append(r)
                print(f"{r['m']:>3} {r['k']:>2} {r['z']:>4} {r['comm']:>10.4f} "
                      f"{r['bott']:>7.3f} {r['d_base']:>8.4f} {r['d_opt_raw']:>8.4f} "
                      f"{r['resid']:>9.2e} {r['d_cert']:>8.4f}", flush=True)
    with open("e4_pair_rigidity.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved e4_pair_rigidity.json")
