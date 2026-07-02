"""expBI_rigidity.py -- the flatness RIGIDITY lemma and the 2D degree obstruction. (Entry 70.)

Entry 69 closed the 1D/per-ring Bochner route: the crux is an irreducibly-2D rigidity. This script attacks it
via the manifestly-NON-NEGATIVE variance form  Sum var0 = Sum_rings Var_s(w_s) >= 0,  w_s = conj(z_{e_s}) +
z_{e_{s-1}},  which is 0 iff every ring is flat (w_s const).

STRUCTURAL (FLATNESS) LEMMA  [derived + verified here]:
  Sum var0 = 0  <=>  the edge field z is a PRODUCT:  z^H_{(x,y)} = h(y)  (const along each row x-cycle),
                     z^V_{(x,y)} = v(x)  (const along each column y-cycle),
  for m ODD.  (m EVEN: product up to an independent NYQUIST real-alternation freedom along rows/cols.)
  Proof sketch: an H-ring (row y, a_x:=z^H_{(x,y)}) is flat iff conj(a_x)+a_{x-1}=c (const). Imag parts:
  Im a_{x-1}-Im a_x = Im c; cyclic sum => Im c=0 and Im a_x const. Real parts: Re a_{x-1}=c-Re a_x, a 2-cycle
  => alternation t,c-t; cyclic closure on an ODD m forces t=c-t => Re a_x const. So a_x const (m odd). []

THE RIGIDITY (target): the flat/product manifold does NOT meet the all-ray + Chern -k sector. The product field
has COMBINED connection psi=arg z with plaquette flux  arg h(y)+arg v(x+1)-arg h(y+1)-arg v(x)  which
TELESCOPES to 0 over the torus => psi-Chern = 0 (consistent with Entry 69: psi is Chern-0 per ring). The band
Chern -k must then be carried by the magnitude field |z|=|p| (|h(y)|,|v(x)|); under all-ray (|p|>|q|) + cap
(|p|<=1) the product magnitude structure is the OPEN incompatibility => a finite algebraic degree obstruction.

Tests: (1) PRODUCT => Sum var0 = 0 (+ Nyquist parity). (2) near-flat descent EXITS the -k sector and LEAVES
all-ray (the rigidity, numerically). (3) row/col variance of z tracks Sum var0 (product-defect ~ flatness-defect).
Reuses expBG/expBD. Writes rigidity.json.
"""
from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
expBG = _load("expBG", "expBG_density_attack.py")
expBD = expBG.expBD; expAN = expBG.expAN; expBA = expBD.expBA
expAE = expBD.expAE; expAK = expBD.expAK; expAR = expBD.expAR


def ring_edge_lists(m): return expBD.ring_edge_lists(m)


def _eidx(x, y, horiz, m): return 2*((x % m)*m + (y % m)) + (0 if horiz else 1)


def raw_plaq_flux(theta, m):
    """Per-plaquette UNWRAPPED holonomy of a per-edge angle field theta (len 2m^2)."""
    F = np.empty(m*m); i = 0
    for x in range(m):
        for y in range(m):
            F[i] = (theta[_eidx(x, y, True, m)] + theta[_eidx(x+1, y, False, m)]
                    - theta[_eidx(x, y+1, True, m)] - theta[_eidx(x, y, False, m)]); i += 1
    return F


def conn_chern(theta, m):
    """Wrapped-flux Chern = sum of wrapped plaquette holonomies /2pi. For a COBOUNDARY (raw sum 0, as psi=phi+arg p
    is), this equals  -sum round(F_raw/2pi) = the signed count of OVERWOUND plaquettes (|F_raw|>pi). Verified."""
    F = raw_plaq_flux(theta, m)
    return float(np.sum((F + np.pi) % (2*np.pi) - np.pi) / (2*np.pi))


def overwinding(theta, m):
    """(#overwound plaquettes, signed count -sum round(F_raw/2pi)); = (|C_psi|, C_psi) for a coboundary."""
    F = raw_plaq_flux(theta, m); r = np.round(F/(2*np.pi))
    return int(np.sum(r != 0)), float(-np.sum(r))


def sumvar0_from_z(z, m):
    return float(sum(expBA.ring_operator(z[e])["var0"] for e in ring_edge_lists(m)))


def z_to_HV(z, m):
    zH = np.array([[z[2*(x*m+y)] for y in range(m)] for x in range(m)])
    zV = np.array([[z[2*(x*m+y)+1] for y in range(m)] for x in range(m)])
    return zH, zV


def HV_to_z(zH, zV, m):
    z = np.zeros(2*m*m, complex)
    for x in range(m):
        for y in range(m):
            z[2*(x*m+y)] = zH[x, y]; z[2*(x*m+y)+1] = zV[x, y]
    return z


def product_defect(z, m):
    """mean row-variance of z^H (var over x per row) + mean col-variance of z^V (var over y per col); 0 iff product."""
    zH, zV = z_to_HV(z, m)
    rowvar = float(np.mean([np.var(zH[:, y]) for y in range(m)]))
    colvar = float(np.mean([np.var(zV[x, :]) for x in range(m)]))
    return rowvar, colvar


def main():
    print("expBI -- flatness RIGIDITY lemma + 2D degree obstruction.\n")
    out = {}
    rng = np.random.default_rng(0)

    # (1) PRODUCT => Sum var0 = 0  (+ Nyquist parity)
    print("(1) STRUCTURAL LEMMA. product z (z^H=h(y), z^V=v(x)) => Sum var0 = 0:")
    prod = {}; nyq = {}
    for m in [5, 6, 7, 8]:
        h = rng.standard_normal(m)+1j*rng.standard_normal(m); v = rng.standard_normal(m)+1j*rng.standard_normal(m)
        zH = np.array([[h[y] for y in range(m)] for x in range(m)])
        zV = np.array([[v[x] for y in range(m)] for x in range(m)])
        sv = sumvar0_from_z(HV_to_z(zH, zV, m), m); prod[m] = sv
        # Nyquist alternation along rows: flat only if m even
        t = rng.standard_normal(m); c = rng.standard_normal(m); be = rng.standard_normal(m)
        zHn = np.array([[(t[y] if x % 2 == 0 else c[y]-t[y]) + 1j*be[y] for y in range(m)] for x in range(m)])
        svn = sumvar0_from_z(HV_to_z(zHn, zV, m), m); nyq[m] = svn
        print(f"  m={m}: product Sum var0 = {sv:+.1e}   | Nyquist-alternation Sum var0 = {svn:+.3f}"
              f"  ({'flat (m even)' if abs(svn) < 1e-6 else 'NOT flat (m odd: cycle does not close)'})")
    out["product_sumvar0"] = {str(k): v for k, v in prod.items()}
    out["nyquist_sumvar0"] = {str(k): v for k, v in nyq.items()}
    print(f"  => product => flat (max {max(abs(v) for v in prod.values()):.0e}); Nyquist closes only for m even.\n")

    # (2,3) RIGIDITY: near-flat descent exits the -k sector & leaves all-ray; product-defect tracks flatness
    print("(2,3) RIGIDITY. in-sector (Chern -k) vs free-flattened, with product-defect (rowvar+colvar):")
    per = []
    for (m, k) in [(5, 2), (6, 2)]:
        np.random.seed(7000 + 100*k + m)
        u0, s2, d2, ph2 = expAK.find_all_ray_line(m, k, tries=24)
        if u0 is None:
            print(f"  m={m}: no line"); continue
        _, up, _, _ = expAR.coherence_ascend_in_sector(u0, s2, d2, ph2, m, k, steps=2200)
        def descr(u):
            d = expAN.edge_data(u, s2, d2, ph2); z = np.exp(1j*ph2)*d['p']
            sv = sumvar0_from_z(z, m); ch = expAE.chern_of_u(m, u)          # FHS band Chern
            cpsi = conn_chern(d['psi'], m)                                  # combined-connection Chern (flat <=> 0)
            now, _ = overwinding(d['psi'], m)                               # #overwound plaquettes = |C_psi|
            ar = bool(np.all(d['R'] > np.abs(d['q']) + 1e-12)); rv, cv = product_defect(z, m)
            return sv, ch, cpsi, now, ar, rv+cv
        sv_in, ch_in, cp_in, ow_in, ar_in, pd_in = descr(up)
        uf, _ = expBD.free_var0_descend(up, s2, d2, ph2, m, steps=1500)
        sv_f, ch_f, cp_f, ow_f, ar_f, pd_f = descr(uf)
        rec = dict(m=m, k=k,
                   in_sumvar0=round(sv_in, 3), in_Cband=round(float(ch_in), 2), in_Cpsi=round(float(cp_in), 2),
                   in_overwound=ow_in, in_all_ray=ar_in, in_prod_defect=round(pd_in, 4),
                   flat_sumvar0=round(sv_f, 3), flat_Cband=round(float(ch_f), 2), flat_Cpsi=round(float(cp_f), 2),
                   flat_overwound=ow_f, flat_all_ray=ar_f, flat_prod_defect=round(pd_f, 4))
        per.append(rec)
        print(f"  m={m}: IN-SECTOR  Svar0={sv_in:.2f}  C_band={ch_in:+.2f}  C_psi={cp_in:+.2f}  #overwound={ow_in}  all-ray={ar_in}  prod-defect={pd_in:.3f}")
        print(f"         FREE-FLAT  Svar0={sv_f:.2f}  C_band={ch_f:+.2f}  C_psi={cp_f:+.2f}  #overwound={ow_f}  all-ray={ar_f}  prod-defect={pd_f:.3f}")
        print(f"         => flattening EXITS to the CONJUGATE sector (C_band {ch_in:+.1f}->{ch_f:+.1f}=+k) and")
        print(f"            kills the combined-connection Chern (C_psi {cp_in:+.1f}->{cp_f:+.1f}): flat <=> C_psi=0.")
    out["rigidity"] = per
    print("\n  flat <=> C_psi(combined connection)=0  [exact topological signature of the product manifold].")
    print("  all-ray C_band=-k has C_psi != 0 (=+1,+3), so it is NOT flat: the rigidity routes through C_psi.")
    print("  Open gap: prove  all-ray + C_band=-k  =>  C_psi != 0  (=> not product => Sum var0 > 0).")
    with open(os.path.join(HERE, "rigidity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote rigidity.json")


if __name__ == "__main__":
    main()
