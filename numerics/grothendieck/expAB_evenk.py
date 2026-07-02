"""expAB_evenk.py — is the r=3 ISOTROPIC sector feasible for EVEN k? (parity allows deg=-k/2; does (star) too?)

Entry 44: r=3 isotropic shadow alpha_s = rho_s e(zeta_s), zeta_s in CP^1, and the sharp PB_e<1 reduces to
    (star)  rho_t (2 mu_e^2 - 1) > (1/2) rho_s ,   mu_e = Re( e^{i phi_e/2} <zeta_t,zeta_s> ),
i.e. a SPINOR problem at HALF the flux. Parity (Chern[alpha]=2 Chern[zeta]) kills ODD k. For EVEN k it
allows deg[zeta] = -k/2, so the question is whether (star) itself obstructs. Heuristic says NO for large m:
a smooth degree-(-1) lowest-Landau-level spinor (k=2) has |<zeta_t,zeta_s>|->1 and arg ~ -phi_e/2, giving
mu_e->1, margin = rho_t - rho_s/2 = sqrt2 - sqrt2/2 > 0. If so, the ISOTROPIC sector is FEASIBLE (PB<1)
for even k -> the PB>=1 lower bound FAILS for even k (never tested directly; all prior PB runs were k=1).

This settles it: maximize the (star)-margin over (zeta unit spinors, rho in (0,sqrt2]) via softmin ascent
(analytic Wirtinger grads), bucket by zeta-degree, report max margin per degree for k=1 (odd, expect <=0
all degrees) and k=2 (even, deg -1 is the test). CROSS-CHECK: build alpha=rho e(zeta) and compute the REAL
PB (expU.pb) for the actual flux-k family; margin>0 must match PB<1. Reuses expU. Writes evenk.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("expU", os.path.join(HERE, "expU_shadow.py"))
expU = importlib.util.module_from_spec(spec); spec.loader.exec_module(expU)


def e_iso(z):
    s, u = z
    v = np.array([s * s - u * u, 1j * (s * s + u * u), 2 * s * u])
    return v / (np.sqrt(2) * (abs(s) ** 2 + abs(u) ** 2))


def edges_half(m, k):
    """edges with HALF flux psi_e = phi_e/2 (the isotropic spinor sees half the flux)."""
    phh, phv = expU.cgn.flux_angles(m, k); E = []
    for x in range(m):
        for y in range(m):
            E.append((expU.cgn.site(x, y, m), expU.cgn.site(x + 1, y, m), phh[x, y] / 2))
            E.append((expU.cgn.site(x, y, m), expU.cgn.site(x, y + 1, m), phv[x, y] / 2))
    return E


def margins(Z, rho, E):
    out = np.empty(len(E)); mus = np.empty(len(E))
    for i, (s, t, psi) in enumerate(E):
        c = np.vdot(Z[t], Z[s])                     # <zeta_t,zeta_s>
        mu = np.real(np.exp(1j * psi) * c)
        mus[i] = mu
        out[i] = rho[t] * (2 * mu * mu - 1) - 0.5 * rho[s]
    return out, mus


def ascend(m, k, Z0, rho0, steps=2000):
    E = edges_half(m, k); Z = Z0.copy(); rho = rho0.copy(); best = -np.inf; bestZ = Z; bestrho = rho
    for it in range(steps):
        beta = 8 + 200 * it / (steps - 1); lr = 0.1 * (1 - 0.9 * it / (steps - 1))
        mg, mus = margins(Z, rho, E)
        w = np.exp(-beta * (mg - mg.min())); w /= w.sum()
        gZ = np.zeros_like(Z); grho = np.zeros_like(rho)
        for i, (s, t, psi) in enumerate(E):
            pe = w[i]; mu = mus[i]
            grho[t] += pe * (2 * mu * mu - 1); grho[s] += pe * (-0.5)
            gZ[s] += pe * 4 * rho[t] * mu * np.exp(-1j * psi) * Z[t]
            gZ[t] += pe * 4 * rho[t] * mu * np.exp(1j * psi) * Z[s]
        Z = Z + lr * gZ
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)          # unit spinors
        rho = np.clip(rho + lr * grho, 1e-3, np.sqrt(2))
        cur = margins(Z, rho, E)[0].min()
        if cur > best:
            best, bestZ, bestrho = cur, Z.copy(), rho.copy()
    return best, bestZ, bestrho


def zeta_chern(m, Z):
    A = np.zeros((m * m, 2, 2)); A[:, 0, :] = Z.real; A[:, 1, :] = Z.imag
    return expU.chern(m, A)


def real_pb(m, k, Z, rho):
    """build alpha = rho * e(zeta) in C^3, return REAL PB for the actual flux-k family (cross-check)."""
    A = np.zeros((m * m, 2, 3))
    for sct in range(m * m):
        al = rho[sct] * e_iso(Z[sct]); A[sct, 0, :] = al.real; A[sct, 1, :] = al.imag
    return expU.pb(A, expU.edges(m, k))


def main():
    m = 8
    out = {}
    for k in (1, 2):
        rng = np.random.default_rng(100 + k)
        buckets = {}
        for sd in range(10):
            Z0 = rng.standard_normal((m * m, 2)) + 1j * rng.standard_normal((m * m, 2))
            Z0 /= np.linalg.norm(Z0, axis=1, keepdims=True)
            rho0 = rng.uniform(0.5, np.sqrt(2), m * m)
            best, Zf, rhof = ascend(m, k, Z0, rho0)
            cz = zeta_chern(m, Zf)
            if cz is None:
                continue
            dz = int(round(cz))
            pb = real_pb(m, k, Zf, rhof)
            if dz not in buckets or best > buckets[dz][0]:
                buckets[dz] = (round(float(best), 4), round(float(pb), 4))
        print(f"k={k}: zeta-degree -> (max (star)-margin, real PB of alpha)   [feasible iff margin>0 <=> PB<1]")
        for dz in sorted(buckets):
            tag = "  <- deg=-k/2 (even-k test)" if (k % 2 == 0 and dz == -k // 2) else ""
            print(f"   deg[zeta]={dz:+d}:  max margin={buckets[dz][0]:+.4f}   real PB={buckets[dz][1]:.4f}{tag}")
        # the binding question
        feas = [(d, v) for d, v in buckets.items() if v[0] > 1e-4]
        target = -k // 2 if k % 2 == 0 else None
        out[f"k={k}"] = {str(d): dict(margin=v[0], real_pb=v[1]) for d, v in buckets.items()}
        if k % 2 == 0:
            tv = buckets.get(target)
            if tv and tv[0] > 1e-4:
                print(f"   *** EVEN-k ISOTROPIC FEASIBLE: deg={target} gives margin {tv[0]:+.4f} (PB {tv[1]:.4f}<1)")
                print(f"       => PB>=1 lower bound FAILS for k={k} in the isotropic sector. RECONCILE w/ Entry 30.")
            else:
                print(f"   even-k deg={target}: margin {tv[0] if tv else 'n/a'} -> isotropic NOT feasible "
                      f"(star obstructs even k too).")
        else:
            print(f"   odd k: any degree feasible? {bool(feas)}  (parity predicts NO winding feasible)")
        print()
    with open(os.path.join(HERE, "evenk.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote evenk.json")


if __name__ == "__main__":
    main()
