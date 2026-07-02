"""expAT_finite.py — is the all-ray Chern=-k regime FINITE in m?  (a potential closure path for the sorry)

The dichotomy (Entries 51-57): a PB<1 competitor must be Chern -k (telescoping). For each m:
  * if NO all-ray Chern=-k line exists (every Chern=-k line has an edge with R_e<=|q_e|) -> empty-edge
    Lemma 51 gives PB>=1 UNCONDITIONALLY (already PROVED), so that m needs nothing more;
  * if an all-ray Chern=-k line DOES exist -> we need the hard estimate (H)|_-k / mu*>=0 at that m.

So if all-ray Chern=-k exists ONLY for finitely many small m, the whole sorry reduces to: Lemma 51 (large m)
+ a FINITE check of (H)|_-k for the few small m. This script MAPS the boundary: for m=3..M, the max worst-edge
phase margin  g(m) := max over Chern=-k lines of  min_e (R_e - |q_e|).  g(m)>0 <=> all-ray Chern=-k exists.
Find where g(m) crosses 0 (M_0). Reuses expU/expAE/expAJ/expAK. Writes finite.json.
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


def best_margin_in_sector(m, k, seeds=16, steps=1400):
    """max over seeds of worst-edge phase margin min_e(R_e-|q_e|), restricted to Chern≈-k lines.
    Also returns the global best (any Chern) for context, and the Chern of the sector-best line."""
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
    best_sec = -np.inf; best_any = -np.inf; best_sec_ch = None
    for sd in range(seeds):
        rng = np.random.default_rng(1000*k + 13*sd + m)
        u0 = expAE.project_unit(rng.standard_normal((N, 3))+1j*rng.standard_normal((N, 3)))
        wm, uf = expAJ.ascend(u0, src, dst, phi, steps=steps)
        ch = expAE.chern_of_u(m, uf)
        best_any = max(best_any, wm)
        if ch is not None and abs(ch + k) < 0.3 and wm > best_sec:
            best_sec = wm; best_sec_ch = round(float(ch), 2)
    return (float(best_sec) if np.isfinite(best_sec) else None), float(best_any), best_sec_ch


def main():
    print("Map g(m) = max over Chern=-k lines of min_e(R_e-|q_e|).  g>0 <=> all-ray Chern=-k EXISTS.")
    print("If g(m)<=0 for all m>=M_0, the sorry is FINITE: Lemma 51 (large m) + check (H)|_-k for m<M_0.\n")
    out = {"k": {}}
    for k in (2, 1):
        rows = []
        print(f"--- k={k} (binding sector Chern=-{k}) ---")
        print(f"{'m':>3} {'g(m)=max margin in -k':>22} {'all-ray -k exists?':>18} {'(best any-Chern margin)':>24}")
        for m in range(3, 13):
            gsec, gany, ch = best_margin_in_sector(m, k)
            exists = (gsec is not None and gsec > 1e-4)
            rows.append(dict(m=m, g_sector=(round(gsec, 4) if gsec is not None else None),
                             all_ray_exists=bool(exists), g_any=round(gany, 4), sector_chern=ch))
            gs = f"{gsec:+.4f}" if gsec is not None else "   (no -k line)"
            print(f"{m:>3} {gs:>22} {str(exists):>18} {gany:>+24.4f}")
        out["k"][str(k)] = rows
        # find M_0: smallest m beyond which all subsequent measured m have g<=0
        exist_ms = [r["m"] for r in rows if r["all_ray_exists"]]
        last_exist = max(exist_ms) if exist_ms else None
        out["k"][str(k)+"_last_allray_m"] = last_exist
        print(f"  => all-ray Chern=-{k} found up to m={last_exist}; "
              f"{'FINITE-looking (dies at larger m)' if last_exist and last_exist < 12 else 'persists -- not obviously finite'}\n")
    print("READING: if g(m)<=0 for m>=M_0 (all-ray dies), then for m>=M_0 every Chern=-k line has an empty")
    print("edge -> Lemma 51 closes it; only m<M_0 need (H)|_-k. A finite, checkable residue + a clean")
    print("'large-m frustration' lemma (g(m)->negative) would close the sorry. The decay of g(m) is the target.")
    with open(os.path.join(HERE, "finite.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote finite.json")


if __name__ == "__main__":
    main()
