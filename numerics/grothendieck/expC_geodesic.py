"""expC_geodesic.py — the index-jump radius for k=1,2,3 on the TRUE U(n) geodesic.

Resolves the honest caveat in `00-orientation-and-open-frontier.md` §8b. The first
k-sweep (`../../../../projects/sonifications/Grothendieck/expC_ksweep.py`, and the original
Exp C in `coarse-geometry-numerics.py`) homotoped flux -> trivial along the *crude* path
`A = (1-t) U0 + t U1` followed by an SVD polar projection. That convex combination leaves
U(n) and the polar retraction bends the path, so both the recorded distance ||U_t - U_0||
and the branch-gap detector can trip at a radius that is an artifact of the path, not the
geometry. The k=3 dip to 1.244 < sqrt2 was flagged as "plausibly partly that path."

Here we move along the minimal geodesic in U(n):
    U_t = U0 . exp(t . Log(U0^dag U1)),  Log = principal (skew-Hermitian) branch.
Diagonalizing Log gives an exact, cheap U_t at any t, and the distance is analytic:
    ||U_t - U0||_2 = max_j |e^{i t theta_j} - 1| = max_j 2|sin(t theta_j / 2)|,
which is MONOTONE in t up to t*theta_max = pi -- so the first index jump has a well-defined
radius, located here by bisection. We also record WHICH event fired (index left k, vs branch
gap closed first), the diagnostic that tells a genuine geometric jump from a detector artifact.

Reuses the exact operators of `coarse-geometry-numerics.py`. Writes jump_radii_geodesic.json.
  python3 expC_geodesic.py
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
from scipy.linalg import logm

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

HERE = os.path.dirname(os.path.abspath(__file__))
M_LIST = (4, 6, 8, 12, 16, 24, 32)
K_LIST = (1, 2, 3)
SQRT2 = np.sqrt(2.0)


def geodesic_evaluator(U0, U1):
    """Return f(t) -> U_t along the minimal geodesic U0 exp(t Log(U0^dag U1)).
    Diagonalize once (Log is skew-Hermitian => -i Log is Hermitian)."""
    L = logm(U0.conj().T @ U1)            # principal skew-Hermitian generator
    A = -1j * L                            # Hermitian
    lam, Q = np.linalg.eigh(A)             # real angles lam_j, unitary Q
    Qh = Q.conj().T

    def U_at(t):
        M = (Q * np.exp(1j * t * lam)) @ Qh    # exp(t L) = Q diag(e^{i t lam}) Q^dag
        return U0 @ M

    return U_at


def flipped(U_at, V_at, t, kref):
    """(integer index left kref?, branch gap) at parameter t."""
    b, gap = cgn.bott_index(U_at(t), V_at(t))
    return abs(round(b) - kref) >= 1, gap


def jump_radius(m, k, coarse=64, bis=26):
    """Radius ||U_t - U0|| at the first INTEGER index flip along the geodesic. The branch
    gap (its closing is the *mechanism* of the flip) is reported as a diagnostic, not the
    trigger -- so the radius is the genuine topological jump, not a threshold crossing."""
    phh, phv = cgn.flux_angles(m, k)
    U0, V0 = cgn.mag_pair_U1(m, phh, phv)
    U1, V1 = cgn.mag_pair_U1(m, np.zeros((m, m)), np.zeros((m, m)))  # trivial, commuting
    b0, _ = cgn.bott_index(U0, V0)
    kref = round(b0)
    U_at, V_at = geodesic_evaluator(U0, U1), geodesic_evaluator(V0, V1)

    # coarse scan for the first t where the integer index leaves kref
    lo, hi = 0.0, None
    min_gap = np.inf
    for t in np.linspace(0.0, 1.0, coarse)[1:]:
        flip, gap = flipped(U_at, V_at, t, kref)
        min_gap = min(min_gap, gap)
        if flip:
            hi = t
            break
        lo = t
    if hi is None:
        return None, kref, min_gap

    # bisect [lo, hi] to the flip
    for _ in range(bis):
        mid = 0.5 * (lo + hi)
        flip, _ = flipped(U_at, V_at, mid, kref)
        if flip:
            hi = mid
        else:
            lo = mid
    radius = float(np.linalg.norm(U_at(hi) - U0, 2))
    _, gap_at = flipped(U_at, V_at, hi, kref)
    return radius, kref, gap_at


def main():
    print(f"{'k':>3} {'m':>4} {'radius ||U_t-U0||':>18} {'/sqrt2':>8} {'gap@flip':>10}")
    radii = {}
    detail = {}
    for k in K_LIST:
        vals = []
        for m in M_LIST:
            r, kref, gap_at = jump_radius(m, k)
            vals.append(r)
            rs = f"{r:.4f}" if r is not None else "no-flip"
            ratio = f"{r / SQRT2:.3f}" if r is not None else "-"
            gs = f"{gap_at:.4f}" if r is not None else f"min {gap_at:.3f}"
            print(f"{k:>3} {m:>4} {rs:>18} {ratio:>8} {gs:>10}", flush=True)
        finite = [v for v in vals if v is not None]
        radii[k] = float(np.median(finite)) if finite else None
        detail[k] = {str(m): (round(v, 4) if v is not None else None)
                     for m, v in zip(M_LIST, vals)}
        med = f"{radii[k]:.4f} ({radii[k] / SQRT2:.3f} sqrt2)" if radii[k] else "no-flip"
        print(f"    -> k={k} median index-jump radius {med}\n")
    with open(os.path.join(HERE, "jump_radii_geodesic.json"), "w") as f:
        json.dump({"median": radii, "per_m": detail}, f, indent=2)
    print("wrote jump_radii_geodesic.json:", {k: round(v, 4) if v else v for k, v in radii.items()})


if __name__ == "__main__":
    main()
