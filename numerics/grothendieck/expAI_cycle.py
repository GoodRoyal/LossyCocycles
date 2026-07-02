"""expAI_cycle.py — Entry 51b: the difference-constraint / positive-cycle reformulation.

KEY (Entry 51): bracket_e is homogeneous degree-1 in nu and depends ONLY on Delta y_e = log(nu_t/nu_s).
Indeed bracket_e/nu_s = f_e(x), x=e^{Delta y_e}>0, with
    f_e(x) = Re[e^{i phi}p_e]*x - 1/2 - | x e^{-i phi}q_e - 1/2 omega_s |,
which is CONCAVE in x (linear minus modulus-of-linear), f_e(0)=-1/2-1/2|omega_s|<0. So {x: f_e(x)>0} is
an interval; when f_e(+inf)=(Re[e^{i phi}p]-|q|)x->+inf (i.e. Re[e^{i phi}p_e]>|q_e|) it is a single ray
x>x_e^0, i.e. a LOWER BOUND  Delta y_e > theta_e := log x_e^0.

If every (relevant) edge gives a single lower bound, the competitor's whole feasibility is the
DIFFERENCE-CONSTRAINT system  y_t - y_s > theta_e  for all directed edges e:s->t.  Standard fact: feasible
iff the directed graph with weights theta_e has NO cycle of total weight >= 0.  Equivalently
(Karp) max-mean-cycle  mu* := max_C (1/|C|) sum_{e in C} theta_e  <  0.  So:
    PB>=1 (bound holds on this line)  <=>  mu* >= 0  (some directed cycle has nonneg theta-sum).
The open claim becomes: a Chern!=0 line forces mu* >= 0.  theta_e is a per-edge holonomy weight; the
positive cycle is the topological obstruction.  THIS entry: verify the single-ray structure, compute
theta_e and mu* (Karp), and test mu* vs Chern (esp. the binding even-k -2 sector).
Reuses expU/expAE/expAF/expAG. Writes cycle.json.
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
expAF = _load("expAF", "expAF_magnitude.py")
expAG = _load("expAG", "expAG_holonomy.py")


def f_edge(x, Rep, qc, omhalf):
    """f_e(x) = Rep*x - 1/2 - |x*qc - omhalf|, qc=e^{-i phi}q_e, omhalf=1/2 omega_s."""
    return Rep*x - 0.5 - np.abs(x*qc - omhalf)


def feasible_set(Rep, qc, omh, xs):
    """sample f over a log-grid xs; return (is_single_ray, theta) where theta=log(smallest x with f>0)."""
    fv = f_edge(xs, Rep, qc, omh)
    pos = fv > 0
    if not pos.any():
        return ("empty", None)              # edge infeasible at any ratio
    idx = np.where(pos)[0]
    # single ray if positive region is a suffix [i0:] (up to grid)
    single = np.all(pos[idx[0]:])
    theta = np.log(xs[idx[0]])
    kind = "ray" if single else "interval"
    return (kind, theta)


def edge_thetas(u, src, dst, phi):
    """per-edge theta_e (lower bound on Delta y_e) + structure tags."""
    us, ut = u[src], u[dst]
    p = np.einsum('ej,ej->e', ut.conj(), us); q = np.einsum('ej,ej->e', ut, us)
    om = np.einsum('ej,ej->e', u, u); oms = om[src]
    Rep = (np.exp(1j*phi)*p).real
    qc = np.exp(-1j*phi)*q; omh = 0.5*oms
    xs = np.exp(np.linspace(-8, 8, 4000))
    thetas = np.full(len(src), np.nan); kinds = []
    for e in range(len(src)):
        kind, th = feasible_set(Rep[e], qc[e], omh[e], xs)
        kinds.append(kind)
        if th is not None: thetas[e] = th
    return thetas, kinds, Rep, np.abs(q)


def karp_max_mean_cycle(N, src, dst, w):
    """Karp's algorithm: max over directed cycles of mean edge weight. O(N*E).
    dp[k][v] = max weight of a walk of exactly k edges ending at v (from a virtual all-source)."""
    E = len(src)
    NEG = -1e18
    dp = np.full((N+1, N), NEG)
    dp[0, :] = 0.0
    for k in range(1, N+1):
        nxt = np.full(N, NEG)
        cand = dp[k-1, src] + w                       # extend walks ending at src by edge ->dst
        np.maximum.at(nxt, dst, cand)
        dp[k] = nxt
    mu = -np.inf
    for v in range(N):
        if dp[N, v] <= NEG/2: continue
        vals = []
        for k in range(N):
            if dp[k, v] <= NEG/2: continue
            vals.append((dp[N, v] - dp[k, v])/(N - k))
        if vals: mu = max(mu, min(vals))
    return mu


def trace_feasible_lines(m, k, n=6):
    """lines from the FULL PB-minimizer (line+magnitude co-optimized to PB~1): trace-feasible Chern=-k,
    the genuinely binding regime (Re[e^{i phi}p_e]>0 on ~all edges)."""
    lines = []
    for sd in range(n):
        A0 = expU.proj_cap(np.random.default_rng(3000+sd).standard_normal((m*m, 2, 3)))
        mp, Af = expU.minimize_pb(m, k, 3, A0, steps=1500)
        u = expAE.alpha_to_u(Af)
        lines.append((round(mp,4), u))
    return lines


def main():
    out = {}; m = 8; k = 2
    print(f"Entry 51b: per-edge feasible-set (ray/interval/empty) + Karp mu*, m={m} k={k}.")
    print("Line source = FULL PB-minimizer (trace-feasible, the binding regime), + margin-ascent for contrast.\n")
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m

    def analyze(u, tag):
        thetas, kinds, Rep, aq = edge_thetas(u, src, dst, phi)
        n_ray = kinds.count("ray"); n_int = kinds.count("interval"); n_emp = kinds.count("empty")
        frac_halfplane = float(np.mean(Rep > 0))      # Re[e^{i phi}p_e]>0 : half-plane achievable
        ch = expAE.chern_of_u(m, u)
        rec = dict(tag=tag, chern=(round(ch,2) if ch is not None else None),
                   n_ray=n_ray, n_interval=n_int, n_empty=n_emp,
                   frac_halfplane_pos=round(frac_halfplane,3))
        if n_emp == 0 and n_int == 0:                 # all-ray: clean difference-constraint system
            mu = karp_max_mean_cycle(N, src, dst, thetas)
            rec["mu_star"] = round(float(mu),5); rec["infeasible_via"] = ("cycle" if mu>=-1e-9 else "FEASIBLE?!")
        elif n_emp > 0:
            rec["mu_star"] = None; rec["infeasible_via"] = "empty-edge (PB_e>=1 unconditionally)"
        else:
            rec["mu_star"] = None; rec["infeasible_via"] = "interval-edges-present"
        return rec

    print("--- trace-feasible lines (full PB-minimizer, PB~1) ---")
    rows = []
    for mp, u in trace_feasible_lines(m, k, n=6):
        r = analyze(u, f"minPB={mp}"); rows.append(r)
        print(f"  [{r['tag']}] Chern={r['chern']}: ray/int/empty={r['n_ray']}/{r['n_interval']}/{r['n_empty']} "
              f"| half-plane Re>0 frac={r['frac_halfplane_pos']} | mu*={r['mu_star']} | via {r['infeasible_via']}")
    print("\n--- margin-ascent lines (contrast; trace-INfeasible) ---")
    for sd in range(3):
        rng = np.random.default_rng(11000+sd)
        u0 = expAE.project_unit(rng.standard_normal((N,3))+1j*rng.standard_normal((N,3)))
        _, uf = expAE.ascend(u0, src, dst, m, steps=900)
        r = analyze(uf, f"margin-asc{sd}"); rows.append(r)
        print(f"  [{r['tag']}] Chern={r['chern']}: ray/int/empty={r['n_ray']}/{r['n_interval']}/{r['n_empty']} "
              f"| half-plane Re>0 frac={r['frac_halfplane_pos']} | via {r['infeasible_via']}")
    out[f"m{m}"] = rows
    print("\nLEARNINGS:")
    tf = [r for r in rows if r['tag'].startswith('minPB')]
    print(f"  trace-feasible lines: half-plane Re>0 frac = {[r['frac_halfplane_pos'] for r in tf]}")
    print(f"  their infeasibility route: {[r['infeasible_via'] for r in tf]}")
    print(f"  all lines infeasible (PB>=1): {all(r.get('infeasible_via','')!='FEASIBLE?!' for r in rows)}")
    with open(os.path.join(HERE,"cycle.json"),"w") as f:
        json.dump(out, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print("wrote cycle.json")


if __name__ == "__main__":
    main()
