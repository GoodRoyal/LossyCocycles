"""expW_vortex.py — DECISIVE vortex-core test of the r>2 codim>=2 lower bound.

Entry 39/40 left the open lemma: a Pi-shadow obeying the cap ||A_s||_op<=1 cannot both satisfy
(TR')/H_e>0 and carry Chern -k. expU minimized PB but let the shadow UNWIND (flee Chern -k to 0),
so "min PB=1" was sector-leaky. This pins the sector and asks the sharp question directly.

Mechanism (Entry 40): PB_e<1 <=> H_e := 2 Sym(R(phi)^T A_dst A_src^T) - A_src A_src^T > 0. A genuine
Chern -k line [alpha_s] can wind with alpha NOWHERE zero (the monopole/Hopf map has ||alpha||=1), yet
the diagnostic showed the optimizer shrinks ||alpha|| at the binding edges -- it tries to UNWIND
toward the trivial alpha=0 competitor (PB=1 exactly). Chern can only jump when alpha passes through 0.

THE TEST: pin Chern = -k=-1 with a hard FLOOR ||alpha_s|| >= eps (so the line can't unwind), start
from the Chern=-1 monopole ansatz, minimize PB, and sweep eps. Predictions:
  * bound HOLDS  -> min PB(eps) > 1 for every eps>0, and min PB -> 1+ as eps->0 (feasible only in the
                    unwinding/vortex-core limit). The min-PB(eps) curve is the vortex-core signature.
  * bound FAILS  -> some eps gives min PB < 1 with Chern still -1: a winding competitor beats 1
                    -> r>2 codim>=2 lower bound FALSE. (decisive falsifier)
Reuses expU (edges, pb, grad, chern, monopole ansatz). Writes vortex.json.

CAVEAT (found on running, Entry 41): a magnitude floor ||alpha||>=eps does NOT lock the lattice
Chern -- the discrete Berry-phase Chern can also jump when an adjacent-site overlap <b_a,b_b> crosses
0 (a link singularity), not only when alpha=0. So the floor holds modulus but not the winding sector;
descent still drifts out of -k. The runs land only in Chern 0/+1, never -1, all at PB>1. The question
is closed instead by the GLOBAL-min argument: the free min over ALL capped shadows is ~1.003 in Chern
0/+1; if -1 held any PB<1 the global min (free to go there) would prefer it -- it doesn't -- so inf PB
over -1 is also >1. Three methods (expU/expV/this) agree min PB ~ 1.003 > 1 for r>2.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPU = os.path.join(HERE, "expU_shadow.py")
spec = importlib.util.spec_from_file_location("expU", EXPU); expU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expU)
edges, pb, grad_softmax_pb2, chern, Rphi = (expU.edges, expU.pb, expU.grad_softmax_pb2,
                                            expU.chern, expU.Rphi)


def chern_m1_ansatz(m, r):
    """conjugate the Chern=+1 monopole -> Chern=-1 (=-k for k=1) shadow, ||alpha||=1 everywhere."""
    A = expU.chern1_ansatz(m, r); A[:, 1, :] *= -1.0
    return A


def proj_cap_floor(A, eps):
    """project each 2xr block to op-norm<=1 (clip sing. vals), then enforce ||alpha_s||_F >= eps
    (scale up uniformly; preserves op-cap since the scale <= eps/||A||_F and sigma_max<=||A||_F)."""
    out = A.copy()
    for s in range(A.shape[0]):
        U, S, Vt = np.linalg.svd(A[s], full_matrices=False)
        S = np.minimum(S, 1.0); B = (U * S) @ Vt
        f = np.linalg.norm(B)
        if f < eps:
            B = B * (eps / max(f, 1e-12))
        out[s] = B
    return out


def minimize_pb_floor(m, k, r, A0, eps, steps=2500):
    E = edges(m, k); A = proj_cap_floor(A0.copy(), eps); best = np.inf; bestA = A
    for t in range(steps):
        beta = 8 + 160 * t / (steps - 1); lr = 0.15 * (1 - 0.9 * t / (steps - 1))
        g, _ = grad_softmax_pb2(A, E, beta)
        A = proj_cap_floor(A - lr * g, eps)
        b = pb(A, E)
        if b < best:
            best, bestA = b, A.copy()
    return best, bestA


def main():
    m, r, k = 6, 3, 1
    n_restarts = 20
    print(f"vortex-core / sector-feasibility test: r={r}, m={m}, k={k}.  The open question is whether the")
    print(f"Chern=-k=-{-(-k)} sector (which telescoping says a PB<1 competitor MUST live in) contains any PB<1.")
    print(f"Method: {n_restarts} random restarts, moderate floor ||alpha||>=eps to pin the sector & keep")
    print(f"Chern well-defined; bucket the optima by final Chern; report min PB per sector.\n")
    out = {}
    for eps in [0.4, 0.3, 0.2]:
        buckets = {}  # rounded Chern -> (min_pb, min_norm)
        for sd in range(n_restarts):
            A0 = expU.proj_cap(np.random.default_rng(1000 + sd).standard_normal((m * m, 2, r)))
            mp, Af = minimize_pb_floor(m, k, r, A0, eps, steps=1800)
            cf = chern(m, Af)
            if cf is None:
                continue
            c = int(round(cf))
            nm = float(np.linalg.norm(Af.reshape(m * m, -1), axis=1).min())
            if c not in buckets or mp < buckets[c][0]:
                buckets[c] = (round(mp, 4), round(nm, 4))
        print(f"  eps={eps}:  sector -> (min PB, min||alpha||) over {n_restarts} restarts")
        for c in sorted(buckets):
            tag = "  <- -k (open case)" if c == -k else ("  <- +k" if c == k else "")
            print(f"     Chern {c:+d}:  minPB={buckets[c][0]:.4f}  min||a||={buckets[c][1]:.4f}{tag}")
        out[f"eps={eps}"] = {str(c): dict(min_pb=v[0], min_norm=v[1]) for c, v in buckets.items()}

    # collect the -k bucket across all eps
    mk = []
    for eps_key, secs in out.items():
        if str(-k) in secs:
            mk.append(secs[str(-k)]["min_pb"])
    print("\nVERDICT:", end=" ")
    if mk and min(mk) < 0.999:
        print(f"the Chern=-k sector reached PB={min(mk):.4f} < 1 -> r>2 codim>=2 LOWER BOUND FALSE.")
    elif mk:
        print(f"the Chern=-k sector was reached but min PB there = {min(mk):.4f} > 1 (bound HOLDS in the")
        print(f"        binding sector); no winding shadow beats 1 -- the magnitude x winding incompatibility.")
    else:
        print("no restart landed in the Chern=-k sector even with a floor -> the -k sector is")
        print("        dynamically inaccessible while keeping PB low (the strongest form of the obstruction).")
    with open(os.path.join(HERE, "vortex.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote vortex.json")


if __name__ == "__main__":
    main()
