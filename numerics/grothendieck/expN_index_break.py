"""expN_index_break.py — watch the index discharge en route to the orthogonal competitor (codim>=2).

Roadmap (06-...md §4): to attack the lower bound distOpFlat>=1 for codim>=2 we want the MECHANISM
by which the charge dies as a partial-isometry pair is moved toward the (flat, index-0) orthogonal
competitor that sits at operator distance exactly 1. This is the mechanism milestone M2 must bound.

Homotopy (keeps a genuine rank-r partial isometry throughout, endpoint = the orthogonal competitor):
  T_s(e) = G_s . [ S R((1-s) phi_e) S^T ] . G_s^T ,   s in [0,1],
where G_s rotates the rotating 2-plane {0,1} into the orthogonal plane {r,r+1} by angle s*pi/2 and
R((1-s)phi) fades the magnetic rotation. s=0 -> flux (charge k); s=1 -> G_1 S S^T G_1^T = W W^T, the
constant orthogonal projection (flat, charge 0) at distance 1. We track vs the operator distance
t(s) = max_e ||flux(e) - T_s(e)||:

  comm      = ||[U_s, V_s]||                         (G1; vanishes, should NOT drive the break)
  chargeAng = max |arg lambda| over eig(M_s) with |lambda|>0.5   (M_s = U_s V_s U_s^T V_s^T)
  chargeMod = min |lambda| over the "charged" eig (|arg|>0.2)    (-> 0 means charge falls into kernel)
  branchGap = min (pi - |arg lambda|) over |lambda|>0.5          (G3; -> 0 means it crosses -1)

These are representation-honest (just the spectrum of the real M_s); the integer Bott index lives
in the complexified core, so we read the charge off the conjugate-pair structure rather than commit
to a possibly-mislabelled integer. The question: at what t, and by WHICH event (chargeMod->0, i.e.
rank/kernel collapse, vs branchGap->0, i.e. crossing the cut), does the charge die?

Reuses coarse-geometry-numerics.py. Writes index_break.json.
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
op = lambda A: float(np.linalg.norm(A, 2))


def G_rot(d, r, psi):
    """Rotate plane {0,1} into the orthogonal plane {r,r+1} by psi (mixes (0,r) and (1,r+1))."""
    G = np.eye(d)
    c, s = np.cos(psi), np.sin(psi)
    for (a, b) in ((0, r), (1, r + 1)):
        G[a, a] = c; G[a, b] = -s; G[b, a] = s; G[b, b] = c
    return G


def edge_maps_s(m, k, d, r, s):
    """T_s(e) for all edges, plus the flux maps (for the distance)."""
    S = cgn.std_frame(d, r)
    G = G_rot(d, r, s * np.pi / 2.0)
    phh, phv = cgn.flux_angles(m, k)
    Th, Tv, Fh, Fv = {}, {}, {}, {}
    for x in range(m):
        for y in range(m):
            for (ph, Tdic, Fdic) in ((phh[x, y], Th, Fh), (phv[x, y], Tv, Fv)):
                F = S @ cgn.rot_core(ph, r) @ S.T
                Fdic[(x, y)] = F
                Tdic[(x, y)] = G @ (S @ cgn.rot_core((1 - s) * ph, r) @ S.T) @ G.T
    return Th, Tv, Fh, Fv


def diagnostics(m, k, d, r, s):
    Th, Tv, Fh, Fv = edge_maps_s(m, k, d, r, s)
    U = cgn.big_U(m, Th, d); V = cgn.big_V(m, Tv, d)
    # operator distance from flux (sup over edges)
    t = max(max(op(Fh[e] - Th[e]) for e in Th), max(op(Fv[e] - Tv[e]) for e in Tv))
    comm = op(U @ V - V @ U)
    M = U @ V @ U.T @ V.T
    ev = np.linalg.eigvals(M)
    big = ev[np.abs(ev) > 0.5]
    arg = np.abs(np.angle(big))
    charged = ev[(np.abs(ev) > 1e-6) & (np.abs(np.angle(ev)) > 0.2)]
    chargeAng = float(arg.max()) if big.size else 0.0
    branchGap = float((np.pi - arg).min()) if big.size else np.pi
    chargeMod = float(np.abs(charged).min()) if charged.size else 0.0
    return dict(t=t, comm=comm, chargeAng=chargeAng, branchGap=branchGap,
                chargeMod=chargeMod, nCharged=int(charged.size))


def main():
    out = {}
    for (d, r, m) in [(4, 2, 4), (4, 2, 6), (6, 3, 4)]:
        key = f"d{d}r{r}_m{m}"
        print("=" * 84)
        print(f"{key}  (codim {d - r})   homotopy flux -> orthogonal competitor (endpoint t=1)")
        print("=" * 84)
        print(f"{'s':>5} {'t=dist':>8} {'comm':>8} {'chargeAng':>10} {'branchGap':>10} "
              f"{'chargeMod':>10} {'nChg':>5}")
        rows = []
        for s in np.linspace(0, 1, 26):
            dg = diagnostics(m, k=1, d=d, r=r, s=s)
            rows.append((round(float(s), 3), dg))
            print(f"{s:>5.2f} {dg['t']:>8.4f} {dg['comm']:>8.4f} {dg['chargeAng']:>10.4f} "
                  f"{dg['branchGap']:>10.4f} {dg['chargeMod']:>10.4f} {dg['nCharged']:>5}",
                  flush=True)
        out[key] = {str(s): dg for s, dg in rows}
        # locate where the charge dies (chargeAng drops below half its s=0 value)
        a0 = rows[0][1]['chargeAng']
        died = next((r_ for (s_, r_) in rows if r_['chargeAng'] < 0.5 * a0), None)
        if died:
            print(f"  -> charge halves near t={died['t']:.3f}; there chargeMod={died['chargeMod']:.3f}, "
                  f"branchGap={died['branchGap']:.3f}, comm={died['comm']:.3f}")
        print()
    with open(os.path.join(HERE, "index_break.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote index_break.json")


if __name__ == "__main__":
    main()
