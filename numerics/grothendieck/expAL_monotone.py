"""expAL_monotone.py — Entry 53: the bilinear correction only HELPS; reduce even-k to a Hermitian loop bound.

Threshold x_e^0 = first positive root of f_e(x)=R_e x - 1/2 - |x q~_e - 1/2 omega_s|, R_e:=Re[e^{i phi}p_e].
CLAIM (proved): on a ray edge (R_e>|q_e|):
  * f_e(1/(2R_e)) = -|q~_e/(2R_e) - 1/2 omega_s| <= 0   (linear part cancels),
  * f_e' (x) >= R_e - |q_e| > 0   (strictly increasing),
  => x_e^0 >= 1/(2R_e),  i.e.  theta_e = log x_e^0 >= theta_e^(0) := -log(2 R_e).
So the bilinear (q,omega) correction RAISES theta on every ray edge; sum_C theta_e >= sum_C theta_e^(0).
By the Entry-51/52 reduction (PB>=1 <=> some directed (=non-contractible) loop has sum theta>=0), the
even-k r>2 bound FOLLOWS from the purely HERMITIAN, bilinear-free statement:
   (H)  for a Chern!=0 line with R_e>0 on all edges, some non-contractible loop C has prod_C 2 R_e <= 1
        (equivalently sum_C theta_e^(0) = -sum_C log(2 R_e) >= 0).
That removes the bilinear obstruction by monotonicity; (H) is the r=2 Loring-type content.

This script:
 (1) VERIFY monotonicity theta_e >= theta_e^(0) on ray edges (0 violations);
 (2) DECIDE whether (H) already holds: compare wrap-loop sums of theta^(0) vs theta on all-ray Chern=-k
     lines. If max-loop sum_C theta^(0) >= 0 -> bilinear-free suffices (reduction COMPLETE mod Loring (H)).
     If sum_C theta^(0) < 0 on all loops while sum_C theta >= 0 -> the bilinear correction is ESSENTIAL.
Reuses expU/expAE/expAJ/expAI/expAK. Writes monotone.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
expAE = _load("expAE", "expAE_pq.py")
expAJ = _load("expAJ", "expAJ_phase_margin.py")
expAI = _load("expAI", "expAI_cycle.py")
expAK = _load("expAK", "expAK_loops.py")


def theta0_and_theta(u, src, dst, phi):
    """per-edge theta0 = -log(2 R_e) (Hermitian) and theta = log x_e^0 (full, with bilinear)."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us); q = np.einsum('ej,ej->e', ut, us)
    om = np.einsum('ej,ej->e', u, u); oms = om[src]
    R = (np.exp(1j*phi)*p).real
    theta0 = np.where(R > 0, -np.log(np.maximum(2*R, 1e-12)), np.nan)
    thetas, kinds, _, _ = expAI.edge_thetas(u, src, dst, phi)
    return theta0, thetas, kinds, R, np.abs(q)


def verify_monotone(trials=30000, seed=1):
    """random ray edges: check log x_0 >= -log(2R) (x_0 >= 1/(2R))."""
    rng = np.random.default_rng(seed); worst = np.inf; n = 0
    for _ in range(trials):
        r = 3
        us = rng.standard_normal(r)+1j*rng.standard_normal(r); us/=np.linalg.norm(us)
        ut = rng.standard_normal(r)+1j*rng.standard_normal(r); ut/=np.linalg.norm(ut)
        phi = rng.uniform(0, 2*np.pi)
        p = np.vdot(ut, us); q = np.sum(ut*us); om = np.sum(us*us)
        R = (np.exp(1j*phi)*p).real
        if R > abs(q) and R > 0:                       # ray edge
            n += 1
            # find x_0 by bisection of f
            qc = np.exp(-1j*phi)*q
            f = lambda x: R*x - 0.5 - abs(x*qc - 0.5*om)
            lo, hi = 1e-9, 1e9
            for _ in range(200):
                mid = np.sqrt(lo*hi)
                if f(mid) > 0: hi = mid
                else: lo = mid
            x0 = hi
            worst = min(worst, x0 - 1.0/(2*R))         # want >= 0
    return n, float(worst)


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def main():
    print("Entry 53: bilinear correction RAISES theta -> reduce even-k to a Hermitian loop bound (H).\n")
    n, worst = verify_monotone()
    print(f"(1) monotonicity x_0 >= 1/(2R) on {n} ray edges: min(x_0 - 1/(2R)) = {worst:+.3e}  "
          f"-> {'HOLDS (theta>=theta0)' if worst >= -1e-7 else 'FAILS'}\n")

    out = {"monotone_min_gap": round(worst, 8), "lines": {}}
    print("(2) does the bilinear-FREE bound (H) already hold on all-ray Chern=-k lines?")
    print("    compare wrap-loop sums: theta^(0) (Hermitian) vs theta (full).\n")
    for m, k in ((6, 2), (6, 1)):
        u, src, dst, phi = expAK.find_all_ray_line(m, k)
        if u is None:
            print(f"  m={m} k={k}: no all-ray Chern=-{k} line"); continue
        theta0, thetas, kinds, R, aq = theta0_and_theta(u, src, dst, phi)
        allray = kinds.count('ray') == len(kinds)
        # wrap-loop sums for theta0 and theta
        H0 = [float(sum(theta0[eidx(x, y, True, m)] for x in range(m))) for y in range(m)]
        V0 = [float(sum(theta0[eidx(x, y, False, m)] for y in range(m))) for x in range(m)]
        Hf = [float(sum(thetas[eidx(x, y, True, m)] for x in range(m))) for y in range(m)]
        Vf = [float(sum(thetas[eidx(x, y, False, m)] for y in range(m))) for x in range(m)]
        ch = round(expAE.chern_of_u(m, u), 2)
        best0 = max(max(H0), max(V0)); bestf = max(max(Hf), max(Vf))
        suffices = best0 >= -1e-9
        out["lines"][f"k{k}_m{m}"] = dict(chern=ch, all_ray=allray,
            max_loop_theta0=round(best0, 4), max_loop_theta=round(bestf, 4),
            bilinear_free_suffices=bool(suffices))
        print(f"  m={m} k={k}: Chern={ch} all-ray={allray}")
        print(f"     max wrap-loop sum theta^(0) (Hermitian-only) = {best0:+.4f}")
        print(f"     max wrap-loop sum theta      (full)          = {bestf:+.4f}   (>= theta^(0) by (1))")
        print(f"     -> bilinear-free bound (H) already gives PB>=1 ? {suffices}   "
              f"{'(reduction COMPLETE mod Loring (H))' if suffices else '(bilinear correction ESSENTIAL here)'}")
    print()
    anysuff = any(v["bilinear_free_suffices"] for v in out["lines"].values())
    print("LEARNING:",
          "the bilinear-free Hermitian bound (H) already certifies PB>=1 on the all-ray line(s) -> the even-k\n"
          "  r>2 bound REDUCES (by the proved monotonicity) to the r=2-type loop statement (H). " if anysuff else
          "on some all-ray line theta^(0) loop-sums are all <0 while theta loop-sums reach >=0 -> the bilinear\n"
          "  correction is ESSENTIAL (monotonicity necessary but not sufficient); (H) alone is too weak.")
    with open(os.path.join(HERE, "monotone.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote monotone.json")


if __name__ == "__main__":
    main()
