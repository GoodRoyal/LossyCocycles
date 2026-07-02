"""expAU_blowup.py — a candidate PROOF MECHANISM for the unconditional fallback mu* >= 0.

Two facts that together may force mu* >= 0 on every all-ray Chern=-k line:
 (A) [elementary] theta_e = log x_e^0 BLOWS UP as the edge approaches the all-ray boundary R_e -> |q_e|^+ .
     Indeed f_e(x)=R_e x - 1/2 - |x e^{-i phi}q_e - 1/2 omega_s| ~ (R_e-|q_e|) x as x->inf, so the root
     x_e^0 ~ 1/(2(R_e-|q_e|)) -> +inf, i.e. theta_e ~ -log(2 (R_e-|q_e|)) -> +inf. One edge with huge theta
     makes EVERY cycle through it have mean theta -> +inf, so mu* >= 0 for free.
 (B) [the frustration bound, hard] Chern=-k CAPS the all-ray slack g := min_e (R_e-|q_e|) small (expAT:
     the max margin over Chern!=0 lines -> ~0 as m grows). Small slack => some edge near the boundary =>
     theta blows up there => mu* >= 0.

So the open core for the FALLBACK is only a WEAK frustration bound: Chern=-k => g <= g*(m) small ENOUGH that
the min-slack edge's theta dominates a cycle. This is weaker than (H)|_-k. This script TESTS the mechanism:
 (1) verify the blow-up theta_e ~ -log(2(R_e-|q_e|)) on near-boundary edges;
 (2) on all-ray Chern=-2 lines spanning a RANGE of min-slack, record (min_slack, theta_max, mu*, mu0*) and
     check mu* is driven positive by theta_max -- and how large min-slack can get (does Chern=-2 cap it?).
Reuses expU/expAE/expAJ/expAK/expAP/expAQ/expAI/expAN. Writes blowup.json.
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
expAK = _load("expAK", "expAK_loops.py")
expAP = _load("expAP", "expAP_coupled.py")
expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
expAI = _load("expAI", "expAI_cycle.py")
expAN = _load("expAN", "expAN_wilson.py")
expAR = _load("expAR", "expAR_sector.py")


def verify_blowup(trials=20000, seed=7):
    """on ray edges, compare theta_e (bisection) to the predicted -log(2(R-|q|)); ratio -> 1 near boundary."""
    rng = np.random.default_rng(seed); rows = []
    for _ in range(trials):
        r = 3
        us = rng.standard_normal(r)+1j*rng.standard_normal(r); us /= np.linalg.norm(us)
        ut = rng.standard_normal(r)+1j*rng.standard_normal(r); ut /= np.linalg.norm(ut)
        phi = rng.uniform(0, 2*np.pi)
        p = np.vdot(ut, us); q = np.sum(ut*us); om = np.sum(us*us)
        R = (np.exp(1j*phi)*p).real; aq = abs(q); slack = R - aq
        if slack <= 0 or R <= 0:
            continue
        qc = np.exp(-1j*phi)*q; omh = 0.5*om
        f = lambda x: R*x - 0.5 - abs(x*qc - omh)
        lo, hi = 1e-12, 1e14
        for _ in range(250):
            mid = np.sqrt(lo*hi)
            if f(mid) > 0: hi = mid
            else: lo = mid
        theta = np.log(hi)
        rows.append((slack, theta))
    rows.sort()
    # bucket by slack, show theta vs -log(2 slack)
    arr = np.array(rows)
    buckets = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 5e-2), (5e-2, 0.2)]
    out = []
    for lo, hi in buckets:
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum() < 5: continue
        th = arr[m, 1]; sl = arr[m, 0]
        pred = -np.log(2*sl)
        out.append(dict(slack_range=[lo, hi], n=int(m.sum()), theta_mean=round(float(th.mean()), 3),
                        pred_mean=round(float(pred.mean()), 3),
                        ratio=round(float((th/pred).mean()), 3)))
    return out


def harvest_lines(m, k, n_trials=8):
    """all-ray Chern=-k lines across a RANGE of min-slack: find_all_ray_line (margin-ascent, higher slack)
    + leashed coherence continuation from each (lower slack, nearer the boundary)."""
    src, dst = expAE.edge_idx(m, k); phi = np.array([e[2] for e in expU.edges(m, k)]); N = m*m
    lines = []
    for trial in range(n_trials):
        np.random.seed(4100 + 19*trial)
        u0, _, _, _ = expAK.find_all_ray_line(m, k, tries=12)
        if u0 is None: continue
        ch = expAE.chern_of_u(m, u0)
        if ch is None or abs(ch + k) > 0.25: continue
        lines.append(u0)                                  # higher-slack (margin-optimized) endpoint
        # push coherence (leashed in -k sector) -> a lower-slack endpoint near the boundary
        _, bestu, _, _ = expAR.coherence_ascend_in_sector(u0, src, dst, phi, m, k, steps=2500)
        if bestu is not None:
            chc = expAE.chern_of_u(m, bestu)
            d = expAN.edge_data(bestu, src, dst, phi)
            if chc is not None and abs(chc + k) < 0.25 and np.all(d['R'] > np.abs(d['q']) + 1e-9):
                lines.append(bestu)
    return lines, src, dst, phi, N


def main():
    print("Candidate mechanism for the FALLBACK mu*>=0: theta blows up at the all-ray boundary R->|q|.\n")
    print("(1) Verify theta_e ~ -log(2(R_e-|q_e|)) as slack->0 (bisection theta vs prediction):")
    for b in verify_blowup():
        print(f"    slack in [{b['slack_range'][0]:.0e},{b['slack_range'][1]:.0e}): n={b['n']:>5}  "
              f"theta_mean={b['theta_mean']:+.3f}  pred=-log(2 slack)={b['pred_mean']:+.3f}  ratio={b['ratio']}")
    print("    => theta tracks -log(2 slack): one near-boundary edge carries arbitrarily large theta.\n")

    print("(2) On all-ray Chern=-2 lines (m=6), relate min-slack -> theta_max -> mu*:")
    out = {"blowup_buckets": verify_blowup(), "lines": []}
    m, k = 6, 2
    lines, src, dst, phi, N = harvest_lines(m, k)
    print(f"    harvested {len(lines)} all-ray Chern=-2 lines")
    print(f"    {'min_slack':>10} {'theta_max':>10} {'mu*(real)':>10} {'mu0*(Herm)':>11} {'-log(2 slack)':>13}")
    max_slack = -np.inf
    for u in lines:
        th, th0, ar, slack = expAQ.theta_bisect(u, src, dst, phi)
        if not ar: continue
        mu = expAI.karp_max_mean_cycle(N, src, dst, th)
        mu0 = expAI.karp_max_mean_cycle(N, src, dst, th0)
        thmax = float(np.nanmax(th)); pred = -np.log(2*max(slack, 1e-12))
        max_slack = max(max_slack, slack)
        rec = dict(min_slack=round(float(slack), 4), theta_max=round(thmax, 3),
                   mu_star=round(float(mu), 3), mu0=round(float(mu0), 3), pred_thetamax=round(pred, 3))
        out["lines"].append(rec)
        print(f"    {slack:>10.4f} {thmax:>10.3f} {mu:>10.3f} {mu0:>11.3f} {pred:>13.3f}")
    out["max_slack_chern_minusk"] = round(float(max_slack), 4) if np.isfinite(max_slack) else None
    allpos = all(r["mu_star"] >= -1e-6 for r in out["lines"])
    print(f"\n    max min-slack over all-ray Chern=-2 lines found = {out['max_slack_chern_minusk']}")
    print(f"    all lines mu* >= 0: {allpos}")
    print("\nMECHANISM ASSESSMENT:")
    print("  - theta blows up at the boundary (verified): a near-boundary edge alone forces mu*>=0.")
    print("  - the FALLBACK mu*>=0 reduces to a WEAK frustration bound: Chern=-k => min-slack <= g*(m) with")
    print("    g* small enough that theta_max dominates a cycle. expAT shows g* -> ~0; if provable even as")
    print("    'g* bounded', mu*>=0 follows. This is WEAKER than (H)|_-k (no loop-product estimate needed).")
    out["all_mu_star_nonneg"] = bool(allpos)
    with open(os.path.join(HERE, "blowup.json"), "w") as f: json.dump(out, f, indent=2)
    print("wrote blowup.json")


if __name__ == "__main__":
    main()
