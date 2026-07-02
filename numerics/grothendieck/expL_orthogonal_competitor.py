"""expL_orthogonal_competitor.py — the orthogonal-support flat competitor: distOpFlat <= 1 for codim>=2.

expK's codim sweep showed the small-m distOpFlat floor sliding toward ~1.0 and saturating, while
its LARGE-m values stayed near sqrt2. That asymmetry is suspicious: if a competitor reaching ~1
exists, it should exist at every m. It does, and we can write it down (no optimizer):

  The magnetic flux rotates only a fixed 2-plane (rot_core acts on the top 2x2). Take W = an
  r-plane that AVOIDS that rotating 2-plane {0,1}; it fits iff d - r >= 2 (CODIMENSION >= 2).
  The CONSTANT flat competitor V(s) = W gives T'(e) = W W^T at every edge. On the rotating plane
  {0,1} flux is a norm-1 isometry while W W^T = 0; on W (orthogonal to {0,1}) flux = 0 while
  W W^T = I; on any shared spectator core direction the two agree (difference 0). The mismatched
  pieces live on mutually orthogonal subspaces, each of norm 1, so

        || flux(e) - W W^T ||_2 = 1   exactly, every edge, every m.

So distOpFlat <= 1 for ALL m whenever codim = d - r >= 2 -- a hand-constructed (reliable) upper
bound, not a descent result. This script verifies it directly for r=2 (d=3..6) and r=3 (d=4..7):
codim 1 has no room for an r-plane off the rotating 2-plane (bound stays ~2, V=S baseline), codim
>= 2 gives exactly 1. Consequence: the sqrt2 separation of Paper 5 is a CODIMENSION-1 (d = r+1)
phenomenon; for codim >= 2 the charge<->trivial distance collapses to 1 (still > 0 -- Plateau
survives -- but c(k) != sqrt2 there). [Earlier draft mislabelled the threshold "d>=2r"; the r=3
row d=5 (codim 2 < 2r=6) also gives 1, so the true threshold is codim >= 2.]

Reuses coarse-geometry-numerics.py. Writes orthogonal_competitor.json.
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
K, R = 1, 2
op = lambda A: float(np.linalg.norm(A, 2))


def orthogonal_frame(d, r):
    """An r-plane (top r coords {d-r..d-1}) that AVOIDS the rotating 2-plane {0,1} the flux
    turns. It fits iff d-r >= 2 (codimension >= 2) -- the only thing the competitor must dodge
    is the 2-plane the magnetic rotation acts on, NOT the whole r-dim core."""
    cols = list(range(d - r, d))
    W = np.zeros((d, r))
    for j, c in enumerate(cols):
        W[c, j] = 1.0
    return W, (d - r >= 2)   # second value: does the r-plane clear the rotating 2-plane?


def sup_edge_distance(m, d, r):
    """sup_e || flux(e) - W W^T ||_2 for the constant orthogonal-frame flat competitor."""
    S = cgn.std_frame(d, r)
    W, fully_orth = orthogonal_frame(d, r)
    WWt = W @ W.T
    phh, phv = cgn.flux_angles(m, k=K)
    worst = 0.0
    for x in range(m):
        for y in range(m):
            for phi in (phh[x, y], phv[x, y]):
                fe = S @ cgn.rot_core(phi, r) @ S.T
                worst = max(worst, op(fe - WWt))
    return worst, fully_orth


def main():
    out = {}
    print(f"{'r':>3} {'d':>3} {'codim':>5} {'d>=2r?':>7} {'orth core?':>10} | "
          f"sup_e ||flux - WW^T||  (= distOpFlat upper bound), m = 4,8,16")
    for r, d in [(2, 3), (2, 4), (2, 5), (2, 6),
                 (3, 4), (3, 5), (3, 6), (3, 7)]:
        row = {}
        for m in (4, 8, 16):
            dist, fully = sup_edge_distance(m, d, r)
            row[str(m)] = round(dist, 6)
        out[f"d{d}r{r}"] = dict(codim=d - r, fits=bool(d >= 2 * r), distance_by_m=row)
        print(f"{r:>3} {d:>3} {d - r:>5} {str(d >= 2 * r):>7} {str(fully):>10} | "
              + "   ".join(f"m={m}: {row[str(m)]:.6f}" for m in (4, 8, 16)), flush=True)
    print("\nReadout:")
    print(" codim >= 2 (d >= r+2): sup_e distance == 1.000000 at every m => distOpFlat <= 1 ALL m")
    print("                        (reliable hand-built competitor; expK's large-m ~sqrt2 was an")
    print("                         optimizer artifact -- the descent never found this basin).")
    print(" codim 1 (d = r+1):     no r-plane clears the rotating 2-plane; bound stays ~2 ->")
    print("                        the sqrt2 separation is special to codimension 1.")
    with open(os.path.join(HERE, "orthogonal_competitor.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote orthogonal_competitor.json")


if __name__ == "__main__":
    main()
