"""expAX_anatomy.py — WHICH mechanism carries (C-H)? Anatomy of the witness loop.

(C-H): some non-contractible loop C has sum_C log(2R_e) <= 0, R_e=|p_e|cos psi_e, psi_e=phi_e+arg p_e.
Exact split on any loop:
    sum_C log(2R_e) = |C| log2  +  log|W_C|  +  sum_C log cos psi_e,
  W_C = prod_C p_e (Berry Wilson loop, |W_C|<=1, the MAGNITUDE/contraction defect),
  psi_e = phi_e + arg p_e (gauge-invariant, total-Chern-0 connection, the PHASE/Loring part).
Jensen (log cos concave): sum_C log cos psi_e <= |C| log cos(Hol_C/|C|), Hol_C = sum_C psi_e.
=> (H) holds via C if  |C| log(2 cos(Hol_C/|C|)) + log|W_C| <= 0   (a 2-number sufficient condition).

This script harvests Chern=-2 lines (margin-optimized AND coherence-pushed), finds the Karp max-mean-cycle
witness C (theta = full bilinear weight; also the Hermitian theta0 witness), and reports for the witness:
  |C|, sum log(2R), and its split [|C|log2, log|W_C|, sum log cos psi]; the holonomy Hol_C and avg|psi|;
  the Jensen sufficient bound. Tells us whether the witness is carried by PHASE (Hol/|psi|) or MAGNITUDE
  (|W_C|) -- i.e. whether (C-H) is 'really' a Loring holonomy bound or a Fubini-Study deficit bound.
Reuses expU/expAE/expAK/expAQ/expAR/expAV/expAN. Writes anatomy.json.
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
expAK = _load("expAK", "expAK_loops.py")
expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
expAR = _load("expAR", "expAR_sector.py")
expAV = _load("expAV", "expAV_visualize.py")
expAN = _load("expAN", "expAN_wilson.py")


def anatomize(u, src, dst, phi, N, edge_of):
    """find Hermitian (theta0) max-mean-cycle witness; split its sum_C log(2R)."""
    d = expAN.edge_data(u, src, dst, phi)
    R = d['R']; absp = d['absp']; psi = d['psi']
    if not np.all(R > 0):
        return None
    theta0 = -np.log(np.maximum(2*R, 1e-300))            # (H) uses theta0; witness = its max-mean-cycle
    mu0, cycle = expAV.karp_cycle(N, src, dst, theta0)
    if not cycle or len(cycle) < 3:
        return None
    # collect edge indices along the witness cycle
    eidx = [edge_of.get((cycle[i], cycle[i+1])) for i in range(len(cycle)-1)]
    eidx = [e for e in eidx if e is not None]
    C = len(eidx)
    sum_log2R = float(sum(np.log(2*R[e]) for e in eidx))
    log_W = float(sum(np.log(np.maximum(absp[e], 1e-300)) for e in eidx))   # log|W_C| (magnitude)
    sum_logcos = float(sum(np.log(np.maximum(np.cos(psi[e]), 1e-300)) for e in eidx))  # phase
    const = C*np.log(2)
    Hol = float(sum(psi[e] for e in eidx))               # loop holonomy (gauge-invariant)
    avg_abspsi_deg = float(np.degrees(np.mean([abs(psi[e]) for e in eidx])))
    jensen = float(C*np.log(2*np.cos(Hol/C)) + log_W) if abs(Hol/C) < np.pi/2 else None
    return dict(C_len=C, mu0=float(mu0), sum_log2R=round(sum_log2R, 3),
                split_const=round(const, 3), split_logW=round(log_W, 3), split_logcos=round(sum_logcos, 3),
                Hol_over_pi=round(Hol/np.pi, 3), avg_abspsi_deg=round(avg_abspsi_deg, 1),
                jensen_bound=(round(jensen, 3) if jensen is not None else None),
                phase_share=round(sum_logcos/(log_W+sum_logcos), 3) if (log_W+sum_logcos) != 0 else None)


def main():
    m, k = 6, 2
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
    edge_of = {(int(src[e]), int(dst[e])): e for e in range(len(src))}
    print("Anatomy of the (C-H) witness loop: sum_C log(2R) = |C|log2 + log|W_C| + sum log cos psi.")
    print("phase_share = (sum log cos psi)/(log|W_C| + sum log cos psi): ~1 => PHASE/Loring carries it;")
    print("              ~0 => MAGNITUDE/Fubini-Study deficit carries it.\n")
    out = {"lines": []}
    # (a) margin-optimized -2 lines (higher slack)  (b) coherence-pushed -2 lines (lower slack)
    lines = []
    for trial in range(5):
        np.random.seed(5000 + 13*trial)
        u0, _, _, _ = expAK.find_all_ray_line(m, k, tries=12)
        if u0 is None: continue
        ch = expAE.chern_of_u(m, u0)
        if ch is None or abs(ch+k) > 0.25: continue
        lines.append(("margin", u0))
        _, bestu, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2000)
        if bestu is not None:
            chc = expAE.chern_of_u(m, bestu)
            dd = expAN.edge_data(bestu, src, dst, phi)
            if chc is not None and abs(chc+k) < 0.25 and np.all(dd['R'] > 0):
                lines.append(("pushed", bestu))
    print(f"{'kind':>7} {'|C|':>4} {'sumlog2R':>9} {'=const':>7} {'+log|W|':>8} {'+logcos':>8} "
          f"{'Hol/pi':>7} {'avg|psi|':>9} {'phase%':>7} {'Jensen':>7}")
    seen = set()
    for kind, u in lines:
        a = anatomize(u, src, dst, phi, N, edge_of)
        if a is None: continue
        key = (a['sum_log2R'], a['C_len'])
        if key in seen: continue
        seen.add(key)
        a['kind'] = kind; out['lines'].append(a)
        js = f"{a['jensen_bound']:+.2f}" if a['jensen_bound'] is not None else "  n/a"
        print(f"{kind:>7} {a['C_len']:>4} {a['sum_log2R']:>+9.2f} {a['split_const']:>+7.2f} "
              f"{a['split_logW']:>+8.2f} {a['split_logcos']:>+8.2f} {a['Hol_over_pi']:>+7.2f} "
              f"{a['avg_abspsi_deg']:>8.1f} {a['phase_share']:>+7.2f} {js:>7}")
    # aggregate
    if out['lines']:
        ph = np.mean([a['phase_share'] for a in out['lines'] if a['phase_share'] is not None])
        njensen = sum(1 for a in out['lines'] if a['jensen_bound'] is not None and a['jensen_bound'] <= 0)
        out['mean_phase_share'] = round(float(ph), 3)
        out['jensen_certifies'] = f"{njensen}/{len(out['lines'])}"
        print(f"\nmean phase_share = {ph:.3f}  (>0.5 => the Hermitian holonomy/Loring part dominates the witness)")
        print(f"Jensen sufficient bound <=0 on the witness: {njensen}/{len(out['lines'])} lines "
              f"(would certify (H) via that loop from just Hol_C and |W_C|)")
        print("\nREAD: this says which lemma to prove. PHASE-dominated => a Loring holonomy bound on the")
        print("Chern-0 connection psi (with |W|<=1 a bonus). MAGNITUDE-dominated => a Fubini-Study deficit")
        print("bound. Jensen<=0 => the 2-number reduction (Hol_C, |W_C|) already closes it.")
    with open(os.path.join(HERE, "anatomy.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote anatomy.json")


if __name__ == "__main__":
    main()
