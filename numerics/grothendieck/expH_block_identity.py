"""expH_block_identity.py — the exact Schur/compression identity behind Q1 (verification).

Q1 wants distOpFlat >= rho_m - E(leak, gamma). The algebraic core is how the commutator
behaves under compression to the support projection P (coreGap gamma = sigma_min on PH).
For ANY operators, with Q = 1 - P, the standard block expansion is exact:

    P[U,V]P = [U_P, V_P] + (P U Q)(Q V P) - (P V Q)(Q U P),       (*)
    U_P := P U P,  V_P := P V P.

So the compressed commutator [U_P,V_P] equals the kept block P[U,V]P up to a CROSS term
made of the off-diagonal leak blocks QUP, QVP (the quantities Exp E/F bound). Hence

    || [U_P,V_P] - P[U,V]P ||  <=  ||PUQ|| ||QVP|| + ||PVQ|| ||QUP||
                               <=  2 * leak * O(1),     leak := max(||QUP||,||QVP||).   (**)

When the support is exactly preserved (ideal family, leak = 0) the compressed unitary pair
carries the kept commutator EXACTLY: [U_P,V_P] = P[U,V]P, and U_P,V_P are honest unitaries on
PH. This script verifies (*) to machine precision and measures every term in (**) for the
ideal family and for an O(eps) wobbled (support-perturbed) family, at several m.

Reuses coarse-geometry-numerics.py. Prints a table; writes block_identity.json.
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
M_LIST = (4, 6, 8, 12)
D, R, K = 3, 2, 1
EPS = 1e-2
op = lambda A: float(np.linalg.norm(A, 2))   # spectral (operator) norm


def core_indices(m, d=D, r=R):
    idx = []
    for x in range(m):
        for y in range(m):
            base = cgn.site(x, y, m) * d
            idx += [base + a for a in range(r)]
    return np.array(idx)


def analyze(U, V, m):
    n = U.shape[0]
    P = cgn.core_proj(m, D, R)
    Q = np.eye(n) - P
    idx = core_indices(m)

    # compressions on PH
    U_P = U[np.ix_(idx, idx)]
    V_P = V[np.ix_(idx, idx)]
    comm_full = U @ V - V @ U
    P_comm_P = comm_full[np.ix_(idx, idx)]              # P[U,V]P on PH
    comm_P = U_P @ V_P - V_P @ U_P                       # [U_P,V_P] on PH

    # off-diagonal leak blocks  Q U P, Q V P  (rows in Q, cols in P)
    cidx = idx
    qmask = np.ones(n, bool); qmask[cidx] = False
    QUP = U[np.ix_(np.where(qmask)[0], cidx)]
    QVP = V[np.ix_(np.where(qmask)[0], cidx)]
    PUQ = U[np.ix_(cidx, np.where(qmask)[0])]
    PVQ = V[np.ix_(cidx, np.where(qmask)[0])]

    # the exact cross term of (*), assembled on PH.  Identity (*):  P[U,V]P = [U_P,V_P] + cross
    cross = PUQ @ QVP - PVQ @ QUP
    resid = op(P_comm_P - (comm_P + cross))             # should be ~0 (machine)

    leak = max(op(QUP), op(QVP))
    bound = op(PUQ) * op(QVP) + op(PVQ) * op(QUP)       # RHS of (**)
    diff = op(comm_P - P_comm_P)                         # LHS of (**)
    full_leak = op(Q @ comm_full @ P)                    # Exp E quantity ||(1-P)[U,V]P||
    unit_def = op(U_P @ U_P.conj().T - np.eye(len(idx))) # is U_P unitary on PH?
    return dict(resid=resid, kept=op(P_comm_P), comm_P=op(comm_P), diff=diff,
                bound=bound, leak=leak, full_leak=full_leak, unit_def=unit_def,
                comm_target=2 * abs(np.sin(np.pi * K / m**2)))


def main():
    out = {"ideal": {}, "wobble": {}}
    for label in ("ideal", "wobble"):
        print("=" * 92)
        print(f"{label.upper()} family   (identity (*) residual, and the bound (**))")
        print("=" * 92)
        print(f"{'m':>4} {'(*)resid':>10} {'||P[U,V]P||':>12} {'2|sin|':>9} "
              f"{'||[U_P,V_P]-P[U,V]P||':>22} {'leak-bound':>11} {'leak':>9} "
              f"{'fullleak':>9} {'unit_def':>9}")
        for m in M_LIST:
            if label == "ideal":
                Th, Tv = cgn.flux_edge_maps(m, K, D, R)
            else:
                rng = np.random.default_rng(7)
                Th, Tv = cgn.flux_edge_maps_tilted(m, K, EPS, D, R, rng=rng)
            U = cgn.big_U(m, Th, D); V = cgn.big_V(m, Tv, D)
            a = analyze(U, V, m)
            out[label][str(m)] = {k: round(v, 6) for k, v in a.items()}
            print(f"{m:>4} {a['resid']:>10.2e} {a['kept']:>12.6f} {a['comm_target']:>9.4f} "
                  f"{a['diff']:>22.6f} {a['bound']:>11.4f} {a['leak']:>9.4f} "
                  f"{a['full_leak']:>9.4f} {a['unit_def']:>9.2e}")
        print()
    with open(os.path.join(HERE, "block_identity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote block_identity.json")


if __name__ == "__main__":
    main()
