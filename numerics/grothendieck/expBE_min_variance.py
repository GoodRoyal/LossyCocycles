"""expBE_min_variance.py -- (b) dedicated minimiser of max-ring var0: is inf_{all-ray, Chern -k} > 0?  (Entry 66.)

Entry 65 caveat: the constant c=min_line max-ring var0 (~0.7-0.8) was only an UPPER estimate of the infimum,
because we minimised var0 indirectly (coherence ascent), not directly. This script minimises max-ring var0
DIRECTLY, gradient-free (annealed random tangent perturbations; var0 is non-smooth in the max but robust to
search), in two modes from the same ceiling starts:

  IN-SECTOR : accept a perturbation only if it lowers max-ring var0 AND keeps Chern=-k AND keeps all-ray.
              -> the achieved floor estimates inf_{all-ray,Chern -k} max-ring var0.  Target: floor > 0.
  FREE      : accept if it lowers max-ring var0, no constraints.
              -> should reach ~0 (unwinds to psi~0, Chern -> 0). The CONTRAST (in-sector floors > 0 while free
                 -> 0) is the decisive evidence that the CHERN forces the variance (inf>0 is real, not slack).

Reuses expBA/expAE/expAK/expAR/expAN/expU. Writes min_variance.json.
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
expAN = _load("expAN", "expAN_wilson.py"); expU = _load("expU", "expU_shadow.py")


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)
def ring_edge_lists(m):
    return ([[eidx(x, y, False, m) for y in range(m)] for x in range(m)] +
            [[eidx(x, y, True, m) for x in range(m)] for y in range(m)])


def max_ring_var0(u, src, dst, phi, rings):
    d = expAN.edge_data(u, src, dst, phi); z = np.exp(1j*phi)*d['p']
    return max(expBA.ring_operator(z[e])["var0"] for e in rings)


def is_all_ray(u, src, dst, phi):
    d = expAN.edge_data(u, src, dst, phi)
    return bool(np.all(d['R'] > np.abs(d['q']) + 1e-12))


def minimise(u0, src, dst, phi, m, k, rings, in_sector, steps=4000, lr0=0.12, seed=0):
    rng = np.random.default_rng(seed)
    u = u0.copy(); cur = max_ring_var0(u, src, dst, phi, rings)
    for t in range(steps):
        lr = lr0 * (1.0 - t/steps) + 0.005                # anneal
        un = expAE.project_unit(u + (rng.standard_normal(u.shape) + 1j*rng.standard_normal(u.shape))*lr)
        nv = max_ring_var0(un, src, dst, phi, rings)
        if nv < cur:
            if in_sector:
                ch = expAE.chern_of_u(m, un)
                if ch is None or abs(ch+k) > 0.25 or not is_all_ray(un, src, dst, phi):
                    continue
            u, cur = un, nv
    return u, cur


def main():
    print("expBE -- (b) minimise max-ring var0 directly. IN-SECTOR floor vs FREE -> 0.  inf_{Chern -k} > 0?\n")
    out = {"per_mk": []}
    for (m, k) in [(5, 2), (6, 2)]:
        src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)])
        rings = ring_edge_lists(m)
        # gather a few all-ray Chern -k ceiling starts
        starts = []
        for sd in range(10):
            np.random.seed(7000 + 31*sd + 100*k + m)
            u0, _, _, _ = expAK.find_all_ray_line(m, k, tries=24)
            if u0 is None or expAE.chern_of_u(m, u0) is None or abs(expAE.chern_of_u(m, u0)+k) > 0.25:
                continue
            _, up, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2000)
            if up is not None and expAE.chern_of_u(m, up) is not None and abs(expAE.chern_of_u(m, up)+k) <= 0.25 \
               and is_all_ray(up, src, dst, phi):
                starts.append(up)
            if len(starts) >= 3:
                break
        if not starts:
            print(f"m={m} k={k}: no all-ray Chern=-{k} start"); continue
        floor_insec = np.inf; floor_free = np.inf; start_max = np.inf
        for j, u0 in enumerate(starts):
            start_max = min(start_max, max_ring_var0(u0, src, dst, phi, rings))
            _, vi = minimise(u0, src, dst, phi, m, k, rings, in_sector=True, steps=4000, seed=10+j)
            floor_insec = min(floor_insec, vi)
            uf, vf = minimise(u0, src, dst, phi, m, k, rings, in_sector=False, steps=4000, seed=20+j)
            chf = expAE.chern_of_u(m, uf)
            floor_free = min(floor_free, vf)
            free_chern = float(chf) if chf is not None else float('nan')
        print(f"=== m={m} k={k}: {len(starts)} starts (start max-ring var0 ~ {start_max:.3f}) ===")
        print(f"  IN-SECTOR floor (min achievable max-ring var0, Chern -{k} + all-ray held) = {floor_insec:.3f}")
        print(f"  FREE      floor (no constraints)                                         = {floor_free:.3f}  "
              f"(ended Chern ~ {free_chern:+.2f})")
        verdict = ("inf > 0 SUPPORTED: in-sector floors well above 0 while free -> ~0 => the Chern forces var0."
                   if floor_insec > 0.3 and floor_free < 0.5*floor_insec else
                   "INCONCLUSIVE: floors too close; need a stronger minimiser.")
        print(f"  => {verdict}\n")
        out["per_mk"].append(dict(m=m, k=k, n_starts=len(starts), start_max_ring_var0=round(float(start_max), 4),
                                  in_sector_floor=round(float(floor_insec), 4), free_floor=round(float(floor_free), 4),
                                  free_chern=round(free_chern, 3),
                                  inf_positive_supported=bool(floor_insec > 0.3 and floor_free < 0.5*floor_insec)))
    with open(os.path.join(HERE, "min_variance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote min_variance.json")


if __name__ == "__main__":
    main()
