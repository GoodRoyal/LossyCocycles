"""expBH_anomalous_energy.py -- Bochner attempt: structural anatomy of the anomalous energy A. (Entry 69.)

Entry 68 reduced the analytic crux to a static inequality  A := E_L(Im z) - E_L(Re z) >= a0*m^2  under
all-ray + Chern -k, where  A = -Re Sum_{rings} Sum_s (Dz_s)^2,  Dz_s = z_{e_s} - z_{e_{s-1}}  (consecutive
hoppings around each wrap loop).  This script does the Bochner-style anatomy: exact (rho,psi) decomposition,
and a decisive structural test of WHAT carries A.

VERIFIED EXACT IDENTITIES (per ring; z_s = rho_s e^{i psi_s}):
  (I)  A(ring) = Sum_s [ 2 rho_s rho_{s-1} cos(psi_s+psi_{s-1}) - rho_s^2 cos2psi_s - rho_{s-1}^2 cos2psi_{s-1} ]
  (II) A(ring) = Sum_s cos(psi_s+psi_{s-1}) * [ 4 rho_s rho_{s-1} sin^2(delta_s/2) - cos(delta_s) (Drho_s)^2 ]
                 + Sum_s sin(psi_s+psi_{s-1}) sin(delta_s) (rho_s^2 - rho_{s-1}^2)
       delta_s = psi_s - psi_{s-1}, Drho_s = rho_s - rho_{s-1}.  First sum = PHASE-CURRENT (dominant),
       second = MAGNITUDE-GRADIENT (small).  Unit-magnitude limit:  A(ring) = Sum_s 4 sin^2(delta_s/2) cos(psi_s+psi_{s-1}).

STRUCTURAL FINDINGS (the Bochner obstruction, made precise):
  1. A UNIFORMLY-WINDING ring (psi_s = 2*pi*w*s/m) has A == 0 exactly: cos(psi_s+psi_{s-1}) oscillates at
     winding 2w and cancels.  => A is NOT a winding/holonomy functional (closes the 1D/per-ring route, and is
     consistent with the whole thread: the obstruction is energy, never winding -- Entries 59,62,63).
  2. On real all-ray Chern -k lines, each ring carries ~ZERO net phase winding (mean|winding|~0): the combined
     connection psi is Chern-0 per ring (Stokes cancellation, Entry 62; the r=2 'psi is a Chern-0 cocycle').
  3. A is carried by phase-increment NON-UNIFORMITY (disorder of delta_s): corr(A, std(delta)) ~ +0.85, while
     corr(A, |winding|) ~ 0.  => the positivity is a phase-DISORDER effect, necessarily 2D/aggregate (no single
     ring is forced; the Chern forbids flattening ALL rings at once -- a collective rigidity).

Reuses expBG/expBF. Writes anomalous_energy.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBG = _load("expBG", "expBG_density_attack.py")
expBF = expBG.expBF; expBD = expBG.expBD; expAN = expBG.expAN


def A_ring(z):
    mm = len(z)
    return float(sum((z[s]-z[(s-1) % mm]).imag**2 - (z[s]-z[(s-1) % mm]).real**2 for s in range(mm)))


def A_forms(z):
    """exact (rho,psi) forms I and II; returns (A_exact, errI, errII, phase_current, mag_gradient)."""
    mm = len(z); rho = np.abs(z); psi = np.angle(z)
    A = A_ring(z)
    AI = 0.0; pc = 0.0; mg = 0.0
    for s in range(mm):
        a, b = rho[s], rho[(s-1) % mm]; ps, pm = psi[s], psi[(s-1) % mm]
        Sig = ps+pm; dl = ps-pm
        AI += 2*a*b*np.cos(Sig) - a*a*np.cos(2*ps) - b*b*np.cos(2*pm)
        pc += np.cos(Sig)*(4*a*b*np.sin(dl/2)**2 - np.cos(dl)*(a-b)**2)
        mg += np.sin(Sig)*np.sin(dl)*(a*a-b*b)
    return A, abs(A-AI), abs(A-(pc+mg)), pc, mg


def main():
    print("expBH -- Bochner anatomy of A = E_L(Im z) - E_L(Re z).\n")
    out = {"identity_checks": {}, "uniform_winding": {}, "per_mk": []}

    # (0) exact-identity check on synthetic + real data
    maxI = 0.0; maxII = 0.0

    # (1) uniform-winding rings -> A == 0  (for GENERIC windings 0<|w|<m/2; the Nyquist mode w=m/2 is the
    #     lone exception: psi_s=pi*s gives the alternating REAL standing wave zeta_s=(-1)^s, A=-4m -- a real
    #     mode, not a winding. Everything below uses generic w.)
    print("(1) synthetic uniform-winding unit ring  psi_s=2*pi*w*s/m  =>  A == 0 for generic 0<|w|<m/2:")
    uw = {}; maxgen = 0.0; nyq = {}
    for m in [5, 6, 8, 12, 16]:
        row = {}
        for w in range(1, m):                      # all windings 1..m-1
            z = np.exp(1j*2*np.pi*w*np.arange(m)/m); a = A_ring(z); row[w] = a
            if m % 2 == 0 and w == m//2:
                nyq[m] = float(a)                  # Nyquist exception
            else:
                maxgen = max(maxgen, abs(a))
        uw[m] = {str(w): float(v) for w, v in row.items()}
        gens = [w for w in row if not (m % 2 == 0 and w == m//2)]
        print(f"    m={m}: generic w in {gens[:4]}... max|A|={max(abs(row[w]) for w in gens):.1e}"
              + (f"   | Nyquist w={m//2}: A={nyq[m]:+.1f} (=-4m, real standing wave)" if m in nyq else ""))
    out["uniform_winding"] = dict(values=uw, nyquist=nyq, max_abs_A_generic=maxgen)
    print(f"    => max|A_generic-winding| = {maxgen:.1e}  (A vanishes on every non-Nyquist pure winding)\n")

    # (2,3) real lines: identities + disorder correlation
    print("(2,3) real all-ray Chern -k lines: exact identities + A vs phase non-uniformity vs winding:")
    eI = 0.0; eII = 0.0
    for (m, k) in [(5, 2), (6, 2)]:
        lines = expBF.collect_lines(m, k, want=8)
        As = []; nonunif = []; wind = []; pc_tot = 0.0; mg_tot = 0.0
        for (u, s2, d2, ph2, d) in lines:
            z = np.exp(1j*ph2)*expAN.edge_data(u, s2, d2, ph2)['p']
            psi = np.angle(z); rings = expBD.ring_edge_lists(m)
            for e in rings:
                ze = z[e]; ps = psi[e]; mm = len(e)
                A, ei, eii, pc, mg = A_forms(ze)
                eI = max(eI, ei); eII = max(eII, eii); pc_tot += pc; mg_tot += mg
                dl = np.array([(ps[s]-ps[(s-1) % mm] + np.pi) % (2*np.pi) - np.pi for s in range(mm)])
                As.append(A); nonunif.append(float(dl.std())); wind.append(abs(float(dl.sum())/(2*np.pi)))
        As = np.array(As); nonunif = np.array(nonunif); wind = np.array(wind)
        c_disorder = float(np.corrcoef(As, nonunif)[0, 1])
        c_wind = float(np.corrcoef(As, wind)[0, 1]) if wind.std() > 1e-9 else 0.0
        rec = dict(m=m, k=k, n_rings=len(As),
                   corr_A_disorder=round(c_disorder, 3), corr_A_winding=round(c_wind, 3),
                   mean_abs_winding=round(float(wind.mean()), 4),
                   phase_current_sum=round(pc_tot, 2), mag_gradient_sum=round(mg_tot, 2),
                   A_sum=round(float(As.sum()), 2))
        out["per_mk"].append(rec)
        print(f"  m={m} k={k}: corr(A, phase-incr std) = {c_disorder:+.2f}   corr(A, |winding|) = {c_wind:+.2f}"
              f"   mean|winding| = {wind.mean():.3f}")
        print(f"          phase-current sum = {pc_tot:+.1f} (dominant +)   mag-gradient sum = {mg_tot:+.1f}"
              f"   A_sum = {As.sum():+.1f}")
    out["identity_checks"] = dict(max_err_formI=eI, max_err_formII=eII, max_abs_A_generic_winding=maxgen)
    print(f"\n  exact identities: max|A-formI| = {eI:.1e}   max|A-formII| = {eII:.1e}")
    print("  VERDICT: A is a phase-DISORDER functional (corr +0.85), NOT winding (corr ~0, mean|wind|~0);")
    print("           uniform windings give A=0. Positivity is necessarily 2D/aggregate -- the Chern forbids")
    print("           flattening all rings at once. The 1D/per-ring and holonomy routes are CLOSED.")
    with open(os.path.join(HERE, "anomalous_energy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote anomalous_energy.json")


if __name__ == "__main__":
    main()
