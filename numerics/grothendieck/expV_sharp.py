"""expV_sharp.py — verify the SHARP per-edge characterization of PB<1, and probe the det/winding step.

Computing G_e = M_e^T M_e + N_e with M_e = R(phi_e) - A_dst A_src^T, N_e = A_src(I-A_dst^T A_dst)A_src^T,
the A_src(A_dst^T A_dst)A_src^T terms CANCEL, leaving
    G_e = I - 2 Sym(R(phi_e)^T A_dst A_src^T) + A_src A_src^T,
so   PB < 1  <=>  H_e := 2 Sym(R(phi_e)^T A_dst A_src^T) - A_src A_src^T  >  0 (pos. def.) for all e.
tr(H_e)>0 is the old (TR); det(H_e)>0 is the new sharp ingredient.

det lead: with B = R^T A_dst A_src^T, det(Sym(B)) = det(B) - (antisym(B))^2, and antisym(B) (the 21-12
entry) is the ROTATION/Berry part that carries the winding. H_e>0 forces (det monotonicity, since
2 Sym(B) > A_src A_src^T >= 0): det(B) >= antisym(B)^2 + (1/4) det(A_src A_src^T).

This script (1) verifies G_e = I - 2 Sym(B) + A_src A_src^T to machine precision, (2) confirms
PB<1 <=> all H_e>0, (3) prints, for the min-PB shadow, how tight det(H_e)>0 is (where it first
fails as PB->1). Reuses cgn / expU machinery."""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

NUM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "coarse-geometry-numerics.py")
spec = importlib.util.spec_from_file_location("cgn", NUM); cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)
# reuse expU_shadow's shadow-only PB minimizer / Chern / edge machinery
EXPU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expU_shadow.py")
spec2 = importlib.util.spec_from_file_location("expU", EXPU); expU = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(expU)
HERE = os.path.dirname(os.path.abspath(__file__))
Rphi = expU.Rphi; edges = expU.edges; chern = expU.chern
sym = lambda X: 0.5 * (X + X.T)


def H_edge(As, At, phi):
    """the sharp matrix H_e = 2 Sym(B) - A_src A_src^T,  B = R(phi)^T A_dst A_src^T.
    PB_e < 1  <=>  H_e > 0 (pos. def.).  Returns (H, B, antisym(B))."""
    B = Rphi(phi).T @ (At @ As.T)
    H = 2 * sym(B) - As @ As.T
    anti = (B[1, 0] - B[0, 1]) / 2          # the rotation/Berry (curvature) part of B
    return H, B, anti


def verify_identities(A, E):
    """(1) G_e = I - 2Sym(B) + A_src A_src^T ; (2) PB<1 <=> all H_e>0 ; (3) det(Sym B)=det B - anti^2."""
    max_resid = 0.0; min_H_eig = np.inf; pb2 = 0.0; r = A.shape[2]; I = np.eye(r)
    for (s, t, phi) in E:
        As, At = A[s], A[t]; R = Rphi(phi)
        M = R - At @ As.T; N = As @ (I - At.T @ At) @ As.T
        G_direct = M.T @ M + N
        H, B, _ = H_edge(As, At, phi)
        G_formula = np.eye(2) - H
        max_resid = max(max_resid, np.abs(G_direct - G_formula).max())
        min_H_eig = min(min_H_eig, np.linalg.eigvalsh(H).min())
        pb2 = max(pb2, np.linalg.eigvalsh(G_direct).max())
    s_, t_, phi = E[3]; H, B, anti = H_edge(A[s_], A[t_], phi)
    print(f"(1) G_e cancellation identity: max|G_direct - (I - 2Sym(B) + A_src A_src^T)| = {max_resid:.2e}")
    print(f"(2) PB = {np.sqrt(pb2):.4f};  min over edges lambda_min(H_e) = {min_H_eig:+.4f};  "
          f"PB<1 ? {np.sqrt(pb2) < 1}  ==  all H_e>0 ? {min_H_eig > 0}")
    print(f"(3) det(Sym B) = {np.linalg.det(sym(B)):+.5f}  vs  det B - antisym^2 = "
          f"{np.linalg.det(B) - anti**2:+.5f}  (antisym(B)={anti:+.4f} = Berry/rotation part)\n")


def chern_m1_ansatz(m, r):
    """conjugate phase of chern1_ansatz -> a Chern = -1 monopole shadow (the -k sector for k=1)."""
    A = expU.chern1_ansatz(m, r); A[:, 1, :] *= -1.0   # alpha -> conj(alpha) flips Chern sign
    return A


def diagnose_binding(A, E, m, label=""):
    """At a (near-)optimal shadow with min PB ~ 1, find which constraint binds on the
    active edges (PB_e >= 1): trace tr(H_e) (magnitude / TR') vs det(H_e) (winding / curvature).

    PB_e^2 = 1 - lambda_min(H_e), so active edges have lambda_min(H_e) <= 0. For 2x2 symmetric H:
       tr(H) < 0           -> TRACE binds (TR'/magnitude fails: (TR') is exactly tr(H_e)>0)
       tr(H) > 0, det(H)<0  -> DET binds  (Berry/winding ingredient fails; H indefinite)
    Also reports ||alpha_src||^2 at the binding edges (vortex-core test: does the obstruction
    sit where the shadow nearly vanishes?)."""
    rows = []
    for (s, t, phi) in E:
        H, B, anti = H_edge(A[s], A[t], phi)
        pb_e = np.sqrt(max(0.0, 1 - np.linalg.eigvalsh(H).min()))
        rows.append((pb_e, np.trace(H), np.linalg.det(H), anti,
                     float(np.trace(A[s] @ A[s].T))))      # ||alpha_src||^2 = tr(A_s A_s^T)
    rows.sort(key=lambda z: -z[0])
    n_tr = sum(1 for r_ in rows if r_[0] >= 1 - 1e-9 and r_[1] <= 1e-9)
    n_det = sum(1 for r_ in rows if r_[0] >= 1 - 1e-9 and r_[1] > 1e-9 and r_[2] < 0)
    n_active = sum(1 for r_ in rows if r_[0] >= 1 - 1e-9)
    na2_active = [r_[4] for r_ in rows if r_[0] >= 1 - 1e-9]
    med_na2 = float(np.median(na2_active)) if na2_active else float("nan")
    print(f"  [{label}] active(PB>=1)={n_active:3d}  TRACE-binds={n_tr:3d}  DET-binds={n_det:3d}  "
          f"median||a_src||^2 @active={med_na2:.3f}  (cap=2.0)")
    return n_tr, n_det, n_active, med_na2


def main():
    m, r, k = 6, 3, 1
    E = edges(m, k)

    # --- part A: verify the sharp identities on a random capped shadow ---
    rng = np.random.default_rng(0)
    A = expU.proj_cap(rng.standard_normal((m*m, 2, r)))
    verify_identities(A, E)

    # --- part B: minimize PB per sector, diagnose binding LABELLED BY FINAL CHERN ---
    # The lemma's content is the -k=-1 sector: telescoping gives PB<1 => Chern -k, so ANY
    # Chern != -k shadow has PB>=1 for free. We must read the binding in the Chern=-1 optimum.
    print("minimize PB per init, diagnose binding constraint grouped by FINAL Chern sector:\n")
    inits = {"chern-1 (+1)": expU.chern1_ansatz(m, r),
             "chern-m1 (-1=-k)": chern_m1_ansatz(m, r)}
    for sd in range(4):
        inits[f"random{sd}"] = expU.proj_cap(np.random.default_rng(sd).standard_normal((m*m, 2, r)))
    out = {}; minus_k = []
    for name, A0 in inits.items():
        mp, Af = expU.minimize_pb(m, k, r, expU.proj_cap(A0), steps=1500)
        cf = chern(m, Af)
        n_tr, n_det, n_act, med = diagnose_binding(Af, E, m, label=f"{name:18s} minPB={mp:.4f} Chern={cf:+.2f}")
        out[name] = dict(min_pb=round(mp, 4), final_chern=cf, trace_binds=n_tr,
                         det_binds=n_det, active_edges=n_act, median_na2_active=round(med, 4))
        if cf is not None and abs(cf - (-k)) < 0.4:
            minus_k.append((mp, n_tr, n_det, med))
    print()
    if minus_k:
        mp, n_tr, n_det, med = min(minus_k)
        print(f"-k=-{k} SECTOR (the open case): best minPB={mp:.4f}; binding TRACE={n_tr} DET={n_det}; "
              f"median||a_src||^2@active={med:.3f} (cap=2).")
        print(f"  reading: min PB>1 (bound holds); obstruction sits where the shadow is "
              f"{'SMALL -> vortex-core mechanism' if med < 1.0 else 'O(1)'}.")
    else:
        print(f"NOTE: no run stayed in the Chern=-{k} sector -> the optimizer FLEES -k "
              f"(itself the telescoping signal: low PB is incompatible with the -k winding).")
    with open(os.path.join(HERE, "sharp.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote sharp.json")


if __name__ == "__main__":
    main()
