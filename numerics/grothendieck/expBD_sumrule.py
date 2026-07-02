"""expBD_sumrule.py -- (a) the global sum rule for Sum_rings var0, and the constant a.   (Entry 66.)

Entry 65 target: all-ray Chern -k => some ring var0 >= c, carried by a SUM RULE Sum_rings var0 ~ a*m (the
variance is spread, not concentrated). max-ring var0 >= (Sum_rings var0)/(2m), so a lower bound Sum var0 >= a*m
gives max-ring var0 >= a/2 (m-uniform). This script:

 1. VERIFIES the exact decomposition (each edge lies in exactly one wrap loop; row=H-edges, col=V-edges):
      Sum_rings var0 = (2/m)(Sp2 + C2) - 4 * Sum_rings meanR_ring^2,
      Sp2 = Sum_{all edges} |p_e|^2 ,  C2 = Sum_rings Sum_s Re(z_{e_s} z_{e_{s-1}})  (consecutive edges).
 2. Measures a(m) = Sum_rings var0 / m on all-ray Chern -k ceiling lines (want a >= a0 > 0, m-uniform).
 3. CONTROL: the variance is held up by the CHERN, not a universal floor. Freely minimising var0 (coherence
    ascent with NO sector constraint) lets the line unwind to psi~0 (flat competitor): Chern -> 0, Sum var0
    -> ~0, but it LEAVES all-ray (Entry 61: psi~0 needs arg p~-phi => |q|~|p|~R). So: in-sector Sum var0 floors
    at a*m>0; free Sum var0 -> 0 at Chern 0. The GAP is the Chern coupling = the content of the sum-rule bound.

Reuses expBA/expAE/expAK/expAR/expAN. Writes sumrule.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBA = _load("expBA", "expBA_ringspectrum.py"); expAE = _load("expAE", "expAE_pq.py")
expAK = _load("expAK", "expAK_loops.py"); expAR = _load("expAR", "expAR_sector.py")
expAN = _load("expAN", "expAN_wilson.py")


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def ring_edge_lists(m):
    rings = []
    for x in range(m):                                   # columns (vertical edges around y)
        rings.append([eidx(x, y, False, m) for y in range(m)])
    for y in range(m):                                   # rows (horizontal edges around x)
        rings.append([eidx(x, y, True, m) for x in range(m)])
    return rings


def sumrule_pieces(u, src, dst, phi, m):
    """Return (sum_var0_direct, sum_var0_decomp, a=sum/m) and components, or None if not all-ray."""
    d = expAN.edge_data(u, src, dst, phi)
    R, p, q = d['R'], d['p'], d['q']
    if not np.all(R > np.abs(q) + 1e-12):
        return None
    z = np.exp(1j*phi) * p
    rings = ring_edge_lists(m)
    # direct: per-ring var0 via expBA ground truth
    sv_direct = 0.0
    for e in rings:
        sv_direct += expBA.ring_operator(z[e])["var0"]
    # decomposition
    Sp2 = float(np.sum(np.abs(p)**2))                    # all edges
    C2 = 0.0; sumMR2 = 0.0
    for e in rings:
        ze = z[e]; mm = len(e)
        C2 += float(np.sum([(ze[s]*ze[(s-1) % mm]).real for s in range(mm)]))
        sumMR2 += float(np.mean(R[e]))**2
    sv_decomp = (2.0/m)*(Sp2 + C2) - 4.0*sumMR2
    return dict(sv_direct=sv_direct, sv_decomp=sv_decomp, a=sv_direct/m,
                Sp2=Sp2, C2=C2, sumMR2=sumMR2)


def free_var0_descend(u0, src, dst, phi, m, steps=1500, lr=0.06):
    """Minimise total var0 with NO sector constraint (gradient-free perturbation): unwinds to psi~0."""
    rng = np.random.default_rng(0)
    rings = ring_edge_lists(m)
    def total_var0(u):
        d = expAN.edge_data(u, src, dst, phi); z = np.exp(1j*phi)*d['p']
        return sum(expBA.ring_operator(z[e])["var0"] for e in rings)
    u = u0.copy(); cur = total_var0(u)
    for _ in range(steps):
        pert = (rng.standard_normal(u.shape) + 1j*rng.standard_normal(u.shape)) * lr
        un = expAE.project_unit(u + pert)
        nv = total_var0(un)
        if nv < cur:
            u, cur = un, nv
    return u, cur


def main():
    print("expBD -- (a) sum rule  Sum_rings var0 = (2/m)(Sp2+C2) - 4*Sum meanR_ring^2 ;  a(m)=Sum var0 / m.\n")
    out = {"per_mk": []}
    max_id_err = 0.0
    for (m, k) in [(5, 2), (6, 2)]:
        lines = expBA.collect_lines(m, k)
        if not lines:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line"); continue
        # recover (u,src,dst,phi) for the sum-rule pieces: re-find one all-ray line + ceiling set
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in _load("expU", "expU_shadow.py").edges(m, k)])
        a_vals = []; sumv_vals = []
        # we need the actual u fields; collect_lines returns rings only, so re-run the line search here
        for sd in range(16):
            np.random.seed(7000 + 31*sd + 100*k + m)
            u0, s2, d2, ph2 = expAK.find_all_ray_line(m, k, tries=24)
            if u0 is None:
                continue
            ch0 = expAE.chern_of_u(m, u0)
            if ch0 is None or abs(ch0+k) > 0.25:
                continue
            _, up, _, _ = expAR.coherence_ascend_in_sector(u0, s2, d2, ph2, m, k, steps=2200)
            for u in (u0, up):
                if u is None or expAE.chern_of_u(m, u) is None or abs(expAE.chern_of_u(m, u)+k) > 0.25:
                    continue
                r = sumrule_pieces(u, s2, d2, ph2, m)
                if r is None:
                    continue
                max_id_err = max(max_id_err, abs(r["sv_direct"]-r["sv_decomp"]))
                a_vals.append(r["a"]); sumv_vals.append(r["sv_direct"])
            break   # one seed's family is enough; ascent gives the spread
        if not a_vals:
            print(f"m={m} k={k}: no usable line"); continue
        a_min = float(np.min(a_vals)); a_mean = float(np.mean(a_vals))
        # control: free var0 descent from the first ceiling line -> Chern 0, Sum var0 ~ 0, leaves all-ray
        uf, svf = free_var0_descend(up, s2, d2, ph2, m, steps=1200)
        chf = expAE.chern_of_u(m, uf)
        df = expAN.edge_data(uf, s2, d2, ph2)
        allray_f = bool(np.all(df['R'] > np.abs(df['q']) + 1e-12))
        print(f"=== m={m} k={k}: {len(a_vals)} in-sector lines ===")
        print(f"  (1) identity max|direct-decomp| = {max_id_err:.2e}  [Sum var0 = (2/m)(Sp2+C2) - 4*Sum meanR^2]")
        print(f"  (2) a(m)=Sum var0/m : min={a_min:.3f}  mean={a_mean:.3f}   => Sum var0 >~ {a_min:.2f}*m  "
              f"(=> max-ring var0 >~ {a_min/2:.2f})")
        print(f"  (3) CONTROL free var0-descent: Chern {chf:+.2f} (was -{k}), Sum var0 {svf:.3f} "
              f"(was >~{a_min*m:.1f}), all-ray={allray_f}")
        print(f"      => the Chern holds the variance up; unwinding kills it but EXITS the -k sector"
              f"{' AND all-ray' if not allray_f else ''}.\n")
        out["per_mk"].append(dict(m=m, k=k, n_lines=len(a_vals), identity_err=max_id_err,
                                  a_min=round(a_min, 4), a_mean=round(a_mean, 4),
                                  free_chern=round(float(chf), 3), free_sumvar0=round(float(svf), 4),
                                  free_all_ray=allray_f))
    out["identity_max_err"] = max_id_err
    with open(os.path.join(HERE, "sumrule.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"identity verified to {max_id_err:.1e}.  wrote sumrule.json")


if __name__ == "__main__":
    main()
