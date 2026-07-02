"""expAE_pq.py — Entry 48: is the necessary condition |p_e|>|q_e| (everywhere) compatible with Chern=-k?

Derived lemma (Entry 48, analytic): if a capped shadow has PB<1 (i.e. g_e>|G_e| on every edge), then
writing u_s=alpha_s/||alpha_s|| (unit), p_e=<u_t,u_s> (Hermitian overlap), q_e=(u_t,u_s) (bilinear),
omega_s=(u_s,u_s), rho_e=sqrt(n_t/n_s):
    g_e>|G_e|  =>  rho_e Re[e^{i phi} p_e] - 1/2 > |rho_e e^{-i phi} q_e - 1/2 omega_s|
                =>  (reverse triangle + Re<=|.|)  rho_e(|p_e|-|q_e|) > 1/2 (1-|omega_s|) >= 0
                =>  |p_e| > |q_e|  on EVERY edge.
Since q_e=<conj(u_t),u_s>, this says every line [u_s] is strictly closer to its neighbor [u_t] than to
the conjugate line [conj(u_t)]. The Hermitian overlap dominates the bilinear overlap everywhere.

THE TEST: is |p_e|>|q_e| (on all edges) by itself incompatible with Chern[u]=-k != 0?
 (1) Direct: evaluate min_e(|p|-|q|) on the Chern=+-1 monopole ansatz (no optimization).
 (2) Decisive: MAXIMIZE the worst-edge margin min_e(|p_e|-|q_e|) by projected ascent on unit u_s in C^r,
     from several inits, reporting the FINAL Chern of each optimum. If every optimum that keeps Chern=-k
     has worst-margin <= 0, the necessary condition alone forbids winding -> with the lemma, PB>=1.
     If some Chern=-k config reaches worst-margin > 0, the cap/det margin is essential (lemma not enough).
Analytic Wirtinger gradient (checked vs finite-diff). Reuses expU edges/chern. Writes pq.json.
"""

from __future__ import annotations
import importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
chern_of_alpha = expU.chern


def edge_idx(m, k):
    E = expU.edges(m, k)
    return np.array([e[0] for e in E]), np.array([e[1] for e in E])


def pq(u, src, dst):
    """u: (N,r) complex unit. p_e=<u_t,u_s>, q_e=(u_t,u_s).  returns |p|,|q|,p,q and us,ut."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us)      # <u_t,u_s>
    q = np.einsum('ej,ej->e', ut, us)             # (u_t,u_s)
    return np.abs(p), np.abs(q), p, q, us, ut


def margin_min(u, src, dst):
    ap, aq, *_ = pq(u, src, dst)
    return float((ap - aq).min())


def softmin_obj(u, src, dst, beta):
    """F ~ min_e (|p_e|^2-|q_e|^2); returns (F, weights, p,q,us,ut)."""
    ap, aq, p, q, us, ut = pq(u, src, dst)
    f = ap**2 - aq**2
    mn = f.min(); w = np.exp(-beta * (f - mn)); w /= w.sum()
    F = mn - (1.0/beta) * np.log(np.exp(-beta*(f-mn)).sum())
    return F, w, p, q, us, ut, f


def grad_wirtinger(u, src, dst, beta):
    """ascent gradient of soft-min of f_e=|p_e|^2-|q_e|^2 w.r.t. u (Euclidean, then caller projects).
    df_e/d conj(u_s) = p_e u_t - q_e conj(u_t);  df_e/d conj(u_t) = conj(p_e) u_s - q_e conj(u_s)."""
    F, w, p, q, us, ut, f = softmin_obj(u, src, dst, beta)
    gs = w[:, None] * (p[:, None]*ut - q[:, None]*ut.conj())          # into src
    gt = w[:, None] * (p.conj()[:, None]*us - q[:, None]*us.conj())   # into dst
    g = np.zeros_like(u)
    np.add.at(g, src, gs); np.add.at(g, dst, gt)
    return 2.0 * g, F, f          # Euclidean ascent dir = 2 d/d conj(u)


def project_unit(u):
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def ascend(u0, src, dst, m, steps=2000, lr=0.05):
    u = project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8; best = -np.inf; bestu = u.copy()
    for t in range(1, steps+1):
        beta = 20 + 380*(t/steps)
        g, F, f = grad_wirtinger(u, src, dst, beta)
        # tangential part (remove component along u_s): g_perp = g - <u,g> u
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None] * u
        gt = g - proj
        mt = b1*mt + (1-b1)*gt; vt = b2*vt + (1-b2)*(gt.conj()*gt).real
        mh = mt/(1-b1**t); vh = vt/(1-b2**t)
        u = project_unit(u + lr * mh/(np.sqrt(vh)+eps))
        wm = margin_min(u, src, dst)
        if wm > best:
            best = wm; bestu = u.copy()
    return best, bestu


def alpha_to_u(A):
    al = A[:, 0, :] + 1j*A[:, 1, :]
    nrm = np.linalg.norm(al, axis=1, keepdims=True)
    return al/np.maximum(nrm, 1e-15)


def u_to_alpha(u):
    A = np.zeros((u.shape[0], 2, u.shape[1]))
    A[:, 0, :] = u.real; A[:, 1, :] = u.imag
    return A


def chern_of_u(m, u):
    return chern_of_alpha(m, u_to_alpha(u))


def check_grad(m=6, k=1, r=3):
    src, dst = edge_idx(m, k)
    rng = np.random.default_rng(1)
    u = project_unit(rng.standard_normal((m*m, r)) + 1j*rng.standard_normal((m*m, r)))
    g, F0, _ = grad_wirtinger(u, src, dst, 30.0)
    # finite diff of F wrt real & imag of u[3,1]
    def Fval(uu):
        return softmin_obj(project_unit(uu), src, dst, 30.0)[0]
    eps = 1e-6; i, j = 3, 1
    ur = u.copy(); ur[i, j] += eps; ul = u.copy(); ul[i, j] -= eps
    dRe = (Fval(ur)-Fval(ul))/(2*eps)
    ur = u.copy(); ur[i, j] += 1j*eps; ul = u.copy(); ul[i, j] -= 1j*eps
    dIm = (Fval(ur)-Fval(ul))/(2*eps)
    # Euclidean grad wrt Re = Re(g), wrt Im = Im(g) (since g=2 d/dconj). but Fval renormalizes ->
    # finite diff includes the projection; compare only the tangential agreement loosely.
    print(f"grad check (informal, projection-coupled): analytic g[{i},{j}]={g[i,j]:.4f}  "
          f"fd dRe={dRe:+.4f} (Re g={g[i,j].real:+.4f})  fd dIm={dIm:+.4f} (Im g={g[i,j].imag:+.4f})")


def main():
    out = {}
    print("LEMMA (Entry 48): PB<1 => |p_e|>|q_e| on every edge (Hermitian overlap dominates bilinear).")
    print("TEST: is |p_e|>|q_e| everywhere compatible with Chern=-k?\n")
    check_grad()
    print()
    for k in (1, 2):
        for m in (6, 8, 12):
            src, dst = edge_idx(m, k)
            # (1) direct on the +-1 (here +-k via ansatz) monopole ansatz, no optimization
            u_ans = alpha_to_u(expU.chern1_ansatz(m, 3))
            c_ans = chern_of_u(m, u_ans); wm_ans = margin_min(u_ans, src, dst)
            # (2) maximize worst-edge margin from several inits; record final Chern
            inits = {"monopole": u_ans,
                     "monopole_conj": u_ans.conj()}
            for sd in range(4):
                rng = np.random.default_rng(200 + 17*k + sd)
                inits[f"rand{sd}"] = project_unit(rng.standard_normal((m*m, 3)) + 1j*rng.standard_normal((m*m, 3)))
            best_in_mk = -np.inf; best_overall = -np.inf; sectors = {}
            for name, u0 in inits.items():
                wm, uf = ascend(u0, src, dst, m, steps=1500)
                cf = chern_of_u(m, uf)
                cf_r = round(cf, 1) if cf is not None else None
                sectors.setdefault(str(cf_r), []).append(round(wm, 4))
                best_overall = max(best_overall, wm)
                if cf is not None and abs(cf - (-k)) < 0.4:
                    best_in_mk = max(best_in_mk, wm)
            rec = dict(ansatz_chern=(round(c_ans,2) if c_ans is not None else None),
                       ansatz_worst_margin=round(wm_ans, 4),
                       best_worst_margin_in_minus_k=(round(best_in_mk,4) if best_in_mk>-np.inf else None),
                       best_worst_margin_overall=round(best_overall,4),
                       sectors=sectors)
            out[f"k{k}_m{m}"] = rec
            mk = rec["best_worst_margin_in_minus_k"]
            print(f"k={k} m={m}: ansatz(Chern={rec['ansatz_chern']}) worst |p|-|q| = {wm_ans:+.4f}  | "
                  f"max worst-margin in -k sector = {str(mk):>8}  | overall = {best_overall:+.4f}")
            print(f"          sectors(final Chern -> best worst-margins): {sectors}")
        print()
    # verdict
    incompat = all((v["best_worst_margin_in_minus_k"] is None or v["best_worst_margin_in_minus_k"] <= 1e-3)
                   for v in out.values())
    print("VERDICT:",
          "no Chern=-k config achieves |p|>|q| on all edges (max worst-margin <=0) -> the necessary "
          "condition ALONE forbids winding; with the lemma this gives PB>=1." if incompat else
          "some Chern=-k config reaches |p|>|q| everywhere (worst-margin>0) -> |p|>|q| is NOT the whole "
          "obstruction; the cap/omega margin in g_e>|G_e| is essential.")
    with open(os.path.join(HERE, "pq.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote pq.json")


if __name__ == "__main__":
    main()
