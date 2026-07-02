"""expAY_witness_homology.py — WHERE does the (C-H) witness loop live? (homology class of the max-mean-cycle)

The pure-analysis routes for (C-H) stall because the witness is a diagonal STAIRCASE, not a wrap loop:
edge-disjoint cycle covers only yield the m horizontal + m vertical wrap loops, whose averages S_h,S_v
can be >0 (S2 fails, Entry 55). So a constructive proof must target the witness's actual homology class.

This script extracts the Karp max-mean-cycle witness (of theta^(0)=-log 2R_e) on harvested Chern=-k lines,
computes its HOMOLOGY CLASS (net (wrap_x, wrap_y)/m around the torus), and contrasts the best wrap-loop
(class (1,0)/(0,1)) with the best diagonal-class loop. If the witness is consistently a fixed off-axis class
(e.g. (1,1) or (1,-1)), a constructive proof can be aimed there. Reuses expU/expAE/expAK/expAQ/expAV/expAN.
Writes witness_homology.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expU = _load("expU", "expU_shadow.py"); expAE = _load("expAE", "expAE_pq.py")
expAK = _load("expAK", "expAK_loops.py"); expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
expAV = _load("expAV", "expAV_visualize.py"); expAN = _load("expAN", "expAN_wilson.py")
expAR = _load("expAR", "expAR_sector.py")
cgn = expU.cgn


def witness_class(u, src, dst, phi, m):
    """Karp max-mean-cycle of theta^(0); return (homology class, cycle length, mean theta^(0), sum log2R)."""
    N = m*m
    d = expAN.edge_data(u, src, dst, phi)
    R = d['R']
    if not np.all(R > 0):
        return None
    theta0 = -np.log(np.maximum(2*R, 1e-300))
    mu0, cycle = expAV.karp_cycle(N, src, dst, theta0)
    if not cycle or len(cycle) < 3:
        return None
    site2xy = {cgn.site(x, y, m): (x, y) for x in range(m) for y in range(m)}
    edge_of = {(int(src[e]), int(dst[e])): e for e in range(len(src))}
    # net wrap: sum signed unit steps (forward edges only -> all +1), counting horizontal vs vertical
    nx = ny = 0; sum_log2R = 0.0
    for i in range(len(cycle)-1):
        e = edge_of.get((cycle[i], cycle[i+1]))
        if e is None:
            continue
        if e % 2 == 0: nx += 1            # horizontal step
        else:          ny += 1            # vertical step
        sum_log2R += np.log(2*R[e])
    return dict(hclass=(nx//m, ny//m), C_len=len(cycle)-1, nx=nx, ny=ny,
                mu0=round(float(mu0), 4), sum_log2R=round(float(sum_log2R), 3),
                wrap_x=round(nx/m, 2), wrap_y=round(ny/m, 2))


def best_wrap(u, src, dst, phi, m):
    """min over the m horizontal and m vertical wrap loops of sum log(2R) (the S2 family)."""
    d = expAN.edge_data(u, src, dst, phi); L2R = d['log2R']
    def eidx(x, y, h): return 2*((x % m)*m + (y % m)) + (0 if h else 1)
    H = [sum(L2R[eidx(x, y, True)] for x in range(m)) for y in range(m)]
    V = [sum(L2R[eidx(x, y, False)] for y in range(m)) for x in range(m)]
    return round(float(min(min(H), min(V))), 3)


def main():
    print("Where does the (C-H) witness loop live? Homology class of the Karp max-mean-cycle.\n")
    print("Using COHERENCE-PUSHED lines (where wrap-loop S2 fails) -- the hard regime.\n")
    print(f"{'m':>3} {'k':>2} {'Chern':>6} {'src':>7} | {'witness hclass':>15} {'len':>4} {'sumlog2R':>9} {'mu0':>7} "
          f"| {'best wrap sumlog2R':>18} {'wrap?':>6}")
    out = {"lines": []}
    classes = {}
    for (m, k) in [(6, 2), (5, 2), (6, 1), (5, 1)]:
        np.random.seed(6100 + 100*k + m)
        u0, src, dst, phi = expAK.find_all_ray_line(m, k, tries=16)
        if u0 is None:
            continue
        ch0 = expAE.chern_of_u(m, u0)
        if ch0 is None or abs(ch0+k) > 0.25:
            continue
        # two sources: the margin-harvested line, and its coherence-pushed (hard) version
        cands = [("margin", u0)]
        _, up, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2500)
        if up is not None:
            chp = expAE.chern_of_u(m, up); dd = expAN.edge_data(up, src, dst, phi)
            if chp is not None and abs(chp+k) < 0.25 and np.all(dd['R'] > 0):
                cands.append(("pushed", up))
        for srctag, u in cands:
            ch = expAE.chern_of_u(m, u)
            w = witness_class(u, src, dst, phi, m)
            if w is None:
                continue
            bw = best_wrap(u, src, dst, phi, m)
            rec = dict(m=m, k=k, chern=round(float(ch), 2), source=srctag, **w, best_wrap=bw,
                       wrap_works=bool(bw <= 1e-9), witness_works=bool(w['sum_log2R'] <= 1e-9))
            out["lines"].append(rec)
            hc = f"{w['hclass']}"
            classes[hc] = classes.get(hc, 0) + 1
            print(f"{m:>3} {k:>2} {rec['chern']:>+6.1f} {srctag:>7} | {hc:>15} {w['C_len']:>4} {w['sum_log2R']:>+9.2f} "
                  f"{w['mu0']:>+7.3f} | {bw:>+18.3f} {str(rec['wrap_works']):>6}")
    print(f"\nwitness homology classes seen: {classes}")
    # how often does the best WRAP loop already satisfy (C-H) vs needing the staircase?
    wrap_ok = sum(1 for r in out['lines'] if r['wrap_works'])
    wit_ok = sum(1 for r in out['lines'] if r['witness_works'])
    print(f"(C-H) satisfied by best WRAP loop: {wrap_ok}/{len(out['lines'])};  "
          f"by the witness (any loop): {wit_ok}/{len(out['lines'])}")
    print("\nREAD: if witness hclass is consistently off-axis (e.g. (1,1)/(1,-1)) while best wrap fails,")
    print("the proof of (C-H) must target that diagonal class -- the wrap-loop (S2) family is provably")
    print("insufficient. If wrap already works on some, those m,k are elementary; the hard ones are the rest.")
    out["classes"] = classes
    with open(os.path.join(HERE, "witness_homology.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote witness_homology.json")


if __name__ == "__main__":
    main()
