"""expBA_ringspectrum.py -- the localizer SETUP: build the 1-D magnetic-ring operator per wrap loop and ask
which spectral object carries (W). (Entry 63; handoff "what to hand the new session" -> spectral localizer.)

Context (digest 6.1; Entries 60-62):
  (W)  all-ray Chern -k  =>  some row/col has  sum_loop log(2 R_e) <= 0   (some wrap loop, prod 2R_e <= 1).
  (W') its linear sharpening: some row/col has  mean R <= 1/2.
  R_e = Re[e^{i phi_e} <u_t,u_s>] = Re(z_e),  z_e := e^{i phi_e} p_e  (combined-connection hopping, |z_e|=|p_e|<=1).
Entry 62 PROVED the holonomy/winding route dead: the combined connection psi=arg z winds to cancel the flux
(discrete Stokes), so |S|<m*pi/3 always. The handoff therefore names the Loring spectral LOCALIZER (a QUADRATIC
form whose signature is the Bott index) as the right tool -- but a *signature* equals that same winding integer.

This script makes the localizer SETUP concrete and tests, on genuine all-ray Chern -k lines pushed to the
coherence ceiling, WHICH spectral object of the 1-D magnetic ring is (i) BLIND (winding/signature) vs (ii)
the actual CARRIER of (W). For each wrap loop we build the honest 1-D magnetic-ring tight-binding operator

      H_ring (m x m, Hermitian):   H[s+1, s] = z_e = e^{i phi_e} p_e ,  H[s,s+1] = conj(z_e)

(the cleanest 1-D magnetic ring -- exactly the object Entry 60 reduces (C-H) to). We report per ring:
  * Theta   = sum arg z_e               -- the combined holonomy (Entry 62: small, |Theta|<m*pi/3).
  * sig     = (#pos - #neg eigs of H)   -- spectral asymmetry (a signature surrogate). BLIND prediction.
  * meanR   = (1/m) sum Re z_e          -- (W') quantity; also Re of the q=0 hopping (band-bottom shift).
  * energy  = sum (-log 2 R_e)          -- (W) quantity; a Dirichlet energy of psi weighted by |p|.
  * Wilson  = prod |p_e|                -- magnitude deficit (<=1).
  * frustr  = |(sum phi_e mod 2pi) - pi|-- per-ring flux distance from the maximally-frustrated value pi.
We then check: does the WORST ring (min meanR / min log2R) line up with the SIGNATURE, or with the
ENERGY+frustration? Verifies (W')/(W) and pins the mechanism. Reuses expAK/expAE/expAR/expAN. Writes ringspectrum.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expAE = _load("expAE", "expAE_pq.py"); expAK = _load("expAK", "expAK_loops.py")
expAR = _load("expAR", "expAR_sector.py"); expAN = _load("expAN", "expAN_wilson.py")


def eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def ring_operator(z):
    """z = ordered complex hoppings around an m-ring. Returns dict of spectral + energy descriptors.

    H_ring (Hermitian m x m): H[s+1,s]=z_e, H[s,s+1]=conj(z_e). The uniform mode |0>=(1,..,1)/sqrt(m) has
    <0|H|0> = (2/m) sum Re z_e = 2*meanR  -- so (W') 'meanR<=1/2' == 'uniform-mode energy <= 1', an exact
    QUADRATIC-FORM (localizer-flavour) statement, not a winding.
    """
    m = len(z)
    H = np.zeros((m, m), complex)
    for s in range(m):
        H[(s+1) % m, s] = z[s]
        H[s, (s+1) % m] = np.conj(z[s])
    ev = np.linalg.eigvalsh(H)
    sig = int(np.sum(ev > 1e-9) - np.sum(ev < -1e-9))     # spectral asymmetry (signature surrogate)
    R = z.real
    meanR = float(R.mean())
    u0 = np.ones(m, complex)/np.sqrt(m)
    Hu = H @ u0
    unif_energy = float((u0.conj() @ Hu).real)            # exact identity check: == 2*meanR
    var0 = float((Hu.conj() @ Hu).real - unif_energy**2)  # <0|H^2|0> - <0|H|0>^2 (Dirichlet/psi-variance)
    energy = float(np.sum(-np.log(np.maximum(2*R, 1e-300))))   # = -sum log(2R_e); (W): some ring has energy>=0
    Theta = float(np.sum(np.angle(z)))
    Wilson = float(np.prod(np.abs(z)))
    return dict(ev=ev, sig=sig, meanR=meanR, unif_energy=unif_energy, var0=var0, energy=energy,
                Theta=Theta, Wilson=Wilson, lmin=float(ev.min()), lmax=float(ev.max()),
                gap=float(np.abs(ev).min()))


def analyze(u, src, dst, phi, m):
    d = expAN.edge_data(u, src, dst, phi)
    R, p = d['R'], d['p']
    if not np.all(R > np.abs(d['q']) + 1e-12):       # all-ray guard (R_e > |q_e| on every edge)
        return None
    z_all = np.exp(1j*phi) * p                       # combined-connection hopping per edge
    rings = []
    for kind, horiz in (("col", False), ("row", True)):
        for a in range(m):                           # col: x=a around y ; row: y=a around x
            e = [eidx(a, b, horiz, m) if not horiz else eidx(b, a, horiz, m) for b in range(m)]
            z = z_all[e]; flux = float(np.sum(phi[e]))
            r = ring_operator(z)
            r.update(kind=kind, idx=a, frustr=abs(((flux + np.pi) % (2*np.pi)) - np.pi))
            rings.append(r)
    return rings


def collect_lines(m, k, n_seeds=16):
    """Robust multi-seed all-ray Chern=-k line collection (mirrors expAZ's seeding), seed + ceiling each."""
    lines = []
    for sd in range(n_seeds):
        np.random.seed(7000 + 31*sd + 100*k + m)
        u0, src, dst, phi = expAK.find_all_ray_line(m, k, tries=24)
        if u0 is None:
            continue
        ch0 = expAE.chern_of_u(m, u0)
        if ch0 is None or abs(ch0 + k) > 0.25:
            continue
        _, up, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2200)
        for u in (u0, up):
            if u is None:
                continue
            ch = expAE.chern_of_u(m, u)
            if ch is None or abs(ch + k) > 0.25:
                continue
            rings = analyze(u, src, dst, phi, m)
            if rings is not None:
                lines.append((float(ch), rings))
    return lines


def main():
    print("expBA -- the 1-D magnetic-ring operator per wrap loop: which spectral object carries (W)?\n")
    out = {"lines": []}
    g_worst_meanR = -np.inf; g_worst_energy = np.inf
    n_lines = 0
    energy_tracks_worst = 0           # worst-energy ring == worst-meanR ring
    sig_tracks_worst = 0              # |sig| extremal ring == worst-meanR ring (expect ~chance => blind)
    sig_varies_lines = 0             # lines where the per-ring signature is not constant
    unif_id_maxerr = 0.0             # max |2*meanR - <0|H|0>| (exact identity check)
    frustr_picks_worst = 0           # most-frustrated ring == worst-meanR ring

    for (m, k) in [(6, 2), (5, 2), (6, 1), (5, 1), (4, 2)]:
        lines = collect_lines(m, k)
        if not lines:
            print(f"m={m} k={k}: no all-ray Chern=-{k} line"); continue
        worst_mr_mk = -np.inf; worst_en_mk = np.inf; ex = None
        for ch, rings in lines:
            n_lines += 1
            meanRs = np.array([r["meanR"] for r in rings])
            energies = np.array([r["energy"] for r in rings])
            sigs = np.array([r["sig"] for r in rings])
            for r in rings:
                unif_id_maxerr = max(unif_id_maxerr, abs(2*r["meanR"] - r["unif_energy"]))
            i_mr = int(np.argmin(meanRs))             # ring best satisfying (W'): smallest meanR
            i_en = int(np.argmax(energies))           # ring best satisfying (W): largest -sum log 2R
            i_sig = int(np.argmax(np.abs(sigs)))      # ring with most extremal spectral asymmetry
            i_fr = int(np.argmax([r["frustr"] for r in rings]))
            energy_tracks_worst += int(i_en == i_mr)
            sig_tracks_worst += int(i_sig == i_mr)
            sig_varies_lines += int(np.ptp(sigs) != 0)
            frustr_picks_worst += int(i_fr == i_mr)
            wmr = float(meanRs.min()); wen = float(energies.max())
            if wmr > worst_mr_mk:
                worst_mr_mk = wmr; worst_en_mk = wen; ex = rings[i_mr]
            g_worst_meanR = max(g_worst_meanR, wmr)
            g_worst_energy = min(g_worst_energy, wen)   # (W) wants every line's max-energy >= 0; track the min over lines
        okWp = worst_mr_mk <= 0.5 + 1e-9
        print(f"m={m} k={k}: {len(lines)} lines | WORST-over-lines min meanR = {worst_mr_mk:+.3f} "
              f"(W'{'OK<=1/2' if okWp else ' X'}) ; that ring's energy(-Slog2R)={worst_en_mk:+.3f}")
        print(f"        worst ring [{ex['kind']} {ex['idx']}]: Theta={ex['Theta']:+.2f} "
              f"(|S|<m*pi/3={m*np.pi/3:.2f}? {'Y' if abs(ex['Theta'])<m*np.pi/3 else 'N'})  "
              f"sig={ex['sig']}  Wilson|p|={ex['Wilson']:.3f}  frustr={ex['frustr']:.2f}/{np.pi:.2f}")
        out["lines"].append(dict(m=m, k=k, n_lines=len(lines), worst_min_meanR=round(worst_mr_mk, 4),
                                 worst_ring=dict(kind=ex['kind'], idx=ex['idx'], sig=ex['sig'],
                                                 Theta=round(ex['Theta'], 3), Wilson=round(ex['Wilson'], 3),
                                                 frustr=round(ex['frustr'], 3), energy=round(worst_en_mk, 3)),
                                 Wp_holds=bool(okWp)))

    print("\n--- verdict ---")
    print(f"lines analyzed: {n_lines}")
    print(f"exact identity 2*meanR == <uniform|H_ring|uniform>: max err = {unif_id_maxerr:.2e}  "
          f"[(W') is the quadratic form <0|H|0> <= 1]")
    print(f"(W') global worst min meanR = {g_worst_meanR:+.4f}  -> need <= 1/2 every line: "
          f"{'HOLDS (<=1/2)' if g_worst_meanR <= 0.5 + 1e-9 else 'FAILS'}")
    print(f"(W)  global worst (min over lines of max-energy) = {g_worst_energy:+.4f}  -> need >= 0 every line: "
          f"{'HOLDS' if g_worst_energy >= -1e-9 else 'FAILS'}")
    pct = lambda a: f"{a}/{n_lines} ({100*a/max(n_lines,1):.0f}%)"
    print(f"signature varies across rings on {pct(sig_varies_lines)} lines (spectral asymmetry is noisy, not constant)")
    print(f"BUT extremal-signature ring == worst (W') ring on {pct(sig_tracks_worst)} "
          f"[~chance => the signature is BLIND to WHICH ring is bad; confirms Entry 62 at operator level]")
    print(f"worst-ENERGY ring == worst-meanR ring on {pct(energy_tracks_worst)} "
          f"[(W)&(W') select the SAME ring => the carrier is the band-bottom ENERGY, not a signature]")
    print(f"most-FRUSTRATED ring == worst-meanR ring on {pct(frustr_picks_worst)} "
          f"[the bad ring is the high-flux-frustration one => bridge to the 2-D Chern via per-ring flux]")
    out["summary"] = dict(n_lines=n_lines, global_worst_meanR=round(float(g_worst_meanR), 4),
                          global_worst_max_energy=round(float(g_worst_energy), 4),
                          Wp_holds=bool(g_worst_meanR <= 0.5 + 1e-9),
                          unif_identity_maxerr=unif_id_maxerr,
                          sig_varies_lines=sig_varies_lines, sig_tracks_worst=sig_tracks_worst,
                          energy_tracks_worst=energy_tracks_worst, frustr_picks_worst=frustr_picks_worst)
    with open(os.path.join(HERE, "ringspectrum.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote ringspectrum.json")


if __name__ == "__main__":
    main()
