"""expAN_wilson.py — fresh attack on (H): the Wilson-loop / averaged reduction.

(H): for an all-ray Chern!=0 line, some non-contractible directed loop C has  sum_C log(2 R_e) <= 0,
     R_e = Re[e^{i phi_e}<u_t,u_s>] = |p_e| cos psi_e,  psi_e = phi_e + arg p_e in (-pi/2,pi/2).

Decompose per edge:   log(2 R_e) = log 2 + log|p_e| + log cos psi_e.

Two structural facts this script EXPLOITS / MEASURES:
 (A) Wilson-loop magnitude help: for a wrap loop C, sum_C log|p_e| = log |W_C|, W_C = prod_C <u_t,u_s>,
     and |W_C| <= 1, so the magnitude part is <= 0 already (a Berry Wilson loop of the line bundle).
 (B) The combined connection psi = phi + arg p has TOTAL Chern 0: flux total +2pi k cancels the Berry
     curvature -2pi k (shadow telescopes to Chern -k). So psi is a Chern-0 U(1) connection -- exactly the
     r=2 Loring situation, now with the extra |p_e|<=1 factors only HELPING.

GOAL of this measurement: identify the cleanest TRUE sufficient condition among
   (S1) min over the m rows  of  S_h(y)=sum_x log(2R^h_{x,y})  <= 0   (some row works), and the col version;
   (S2) the row/col AVERAGES  S_h = sum_{all h} log(2R), S_v = sum_{all v} log(2R):  is min(S_h,S_v) <= 0 ?
   (S3) does the magnitude part alone (sum log|p|) or the cos part alone carry it?
Then report the wrap-loop flux holonomies and Berry Wilson loops to expose the mechanism.
Reuses expU/expAE/expAK/expAI. Writes wilson.json.
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


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def edge_data(u, src, dst, phi):
    """per-edge p, |p|, psi=phi+arg p, R, and the log decomposition pieces."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)
    q = np.einsum('ej,ej->e', ut, us)
    R = (np.exp(1j*phi)*p).real
    absp = np.abs(p)
    psi = phi + np.angle(p)
    psi = (psi + np.pi) % (2*np.pi) - np.pi               # wrap to (-pi,pi]
    return dict(p=p, q=q, absp=absp, psi=psi, R=R,
                log2R=np.log(np.maximum(2*R, 1e-300)),
                logabsp=np.log(np.maximum(absp, 1e-300)),
                logcos=np.log(np.maximum(np.cos(psi), 1e-300)))


def analyze_line(u, src, dst, phi, m, k):
    d = edge_data(u, src, dst, phi)
    allray = bool(np.all(d['R'] > np.abs(d['q']) + 1e-12) and np.all(d['R'] > 0))
    L2R = d['log2R']
    # per-row (horizontal wrap-loop) and per-col (vertical wrap-loop) sums of log(2R)
    Hrows = np.array([sum(L2R[eidx(x, y, True, m)] for x in range(m)) for y in range(m)])
    Vcols = np.array([sum(L2R[eidx(x, y, False, m)] for y in range(m)) for x in range(m)])
    # decomposition pieces per row/col
    def rowsum(arr, horiz):
        if horiz: return np.array([sum(arr[eidx(x, y, True, m)] for x in range(m)) for y in range(m)])
        return np.array([sum(arr[eidx(x, y, False, m)] for y in range(m)) for x in range(m)])
    Hmag = rowsum(d['logabsp'], True);  Hcos = rowsum(d['logcos'], True)
    Vmag = rowsum(d['logabsp'], False); Vcos = rowsum(d['logcos'], False)
    # flux holonomy around each wrap loop (gauge piece of psi)
    Hflux = np.array([sum(phi[eidx(x, y, True, m)] for x in range(m)) for y in range(m)])
    Vflux = np.array([sum(phi[eidx(x, y, False, m)] for y in range(m)) for x in range(m)])
    # totals
    S_h = float(L2R[0::2].sum()); S_v = float(L2R[1::2].sum())  # 0::2 = horiz edges, 1::2 = vert
    # LINEAR (AM-GM) reduction: 2R_e = 2 - ||u_t - e^{i phi}u_s||^2; mean_dir R <= 1/2 => (H) via that dir.
    Rh = d['R'][0::2]; Rv = d['R'][1::2]
    meanRh = float(Rh.mean()); meanRv = float(Rv.mean())
    rec = dict(
        chern=round(float(expAE.chern_of_u(m, u)), 2), all_ray=allray, m=m, k=k,
        # (S1) best single wrap loop
        min_row_log2R=round(float(Hrows.min()), 4), min_col_log2R=round(float(Vcols.min()), 4),
        H_some_loop_works=bool(Hrows.min() <= 1e-9), V_some_loop_works=bool(Vcols.min() <= 1e-9),
        # (S2) averaged totals
        S_h=round(S_h, 4), S_v=round(S_v, 4), min_S=round(min(S_h, S_v), 4),
        S2_holds=bool(min(S_h, S_v) <= 1e-9),
        # (S3) LINEAR target: mean R per direction <= 1/2
        meanRh=round(meanRh, 4), meanRv=round(meanRv, 4), min_meanR=round(min(meanRh, meanRv), 4),
        S3_holds=bool(min(meanRh, meanRv) <= 0.5 + 1e-9),
        # decomposition of the WINNING direction's total
        H_total_mag=round(float(Hmag.sum()), 3), H_total_cos=round(float(Hcos.sum()), 3),
        V_total_mag=round(float(Vmag.sum()), 3), V_total_cos=round(float(Vcos.sum()), 3),
        Hconst=round(m*m*np.log(2), 3),
        # mechanism numbers
        min_absp=round(float(d['absp'].min()), 3), max_abspsi_deg=round(float(np.degrees(np.abs(d['psi']).max())), 1),
        mean_abspsi_deg=round(float(np.degrees(np.abs(d['psi']).mean())), 1),
    )
    return rec, dict(Hrows=Hrows, Vcols=Vcols, Hmag=Hmag, Hcos=Hcos, Vmag=Vmag, Vcos=Vcos,
                     Hflux=Hflux, Vflux=Vflux)


def main():
    print("Fresh attack on (H): Wilson-loop / averaged reduction. log(2R_e)=log2+log|p_e|+log cos psi_e.\n")
    out = {"lines": []}
    for m, k in ((6, 2), (4, 2), (5, 2), (6, 1)):
        u, src, dst, phi = expAK.find_all_ray_line(m, k)
        if u is None:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line (empty-edge regime; Lemma 51).")
            out["lines"].append(dict(m=m, k=k, all_ray=False, note="no all-ray line found"))
            continue
        rec, det = analyze_line(u, src, dst, phi, m, k)
        out["lines"].append(rec)
        print(f"=== m={m} k={k}: Chern={rec['chern']} all_ray={rec['all_ray']} ===")
        print(f"  (S1) best row sum log(2R) = {rec['min_row_log2R']:+.3f} (<=0? {rec['H_some_loop_works']}); "
              f"best col = {rec['min_col_log2R']:+.3f} (<=0? {rec['V_some_loop_works']})")
        print(f"  (S2) S_h(all h)={rec['S_h']:+.3f}  S_v(all v)={rec['S_v']:+.3f}  "
              f"min={rec['min_S']:+.3f}  (<=0? {rec['S2_holds']})")
        print(f"  (S3) LINEAR: mean_h R={rec['meanRh']:+.4f}  mean_v R={rec['meanRv']:+.4f}  "
              f"min={rec['min_meanR']:+.4f}  (<=1/2? {rec['S3_holds']})")
        print(f"       H decomp: const={rec['Hconst']:+.2f} + mag={rec['H_total_mag']:+.2f} + "
              f"cos={rec['H_total_cos']:+.2f} = {rec['Hconst']+rec['H_total_mag']+rec['H_total_cos']:+.2f}")
        print(f"       V decomp: const={rec['Hconst']:+.2f} + mag={rec['V_total_mag']:+.2f} + "
              f"cos={rec['V_total_cos']:+.2f} = {rec['Hconst']+rec['V_total_mag']+rec['V_total_cos']:+.2f}")
        print(f"  mech: min|p_e|={rec['min_absp']}  max|psi|={rec['max_abspsi_deg']}deg  "
              f"mean|psi|={rec['mean_abspsi_deg']}deg")
        print(f"  per-row sum log(2R): {np.round(det['Hrows'],2)}")
        print(f"  per-col sum log(2R): {np.round(det['Vcols'],2)}")
        print(f"  row flux holonomy/pi: {np.round(det['Hflux']/np.pi,3)}")
        print(f"  col flux holonomy/pi: {np.round(det['Vflux']/np.pi,3)}\n")

    # Verdict on the sufficient conditions
    rays = [l for l in out["lines"] if l.get("all_ray")]
    s1 = all(l["H_some_loop_works"] or l["V_some_loop_works"] for l in rays)
    s2 = all(l["S2_holds"] for l in rays)
    s3 = all(l["S3_holds"] for l in rays)
    print("VERDICT over all-ray lines found:")
    print(f"  (S1) some single wrap loop has sum log(2R)<=0 : {s1}  <- this is (H) itself, must hold")
    print(f"  (S2) min(S_h,S_v)<=0 (log-sum averaged target) : {s2}")
    print(f"  (S3) min(mean_h R, mean_v R) <= 1/2 (LINEAR target, => S2 => H) : {s3}")
    out["S1_holds_all"] = bool(s1); out["S2_holds_all"] = bool(s2); out["S3_holds_all"] = bool(s3)
    with open(os.path.join(HERE, "wilson.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote wilson.json")


if __name__ == "__main__":
    main()
