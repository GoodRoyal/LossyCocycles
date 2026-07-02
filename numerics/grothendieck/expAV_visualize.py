"""expAV_visualize.py — SEE the obstruction: frustration + the Escher-staircase loop on the torus.

Renders, from a real all-ray Chern=-2 competitor (m=6), the three faces of why PB>=1 is hard:
 (1) TOPOLOGY: per-plaquette Berry curvature of the line (sums to Chern=-2) -- the trapped charge, locally
     tiny (~2pi*2/36 each) but globally an integer; cancels the uniform flux (+2 total) to combined Chern 0.
 (2) FRUSTRATION: each directed bond colored by theta_e=log x_e^0 (red=near the all-ray boundary R->|q|, the
     most frustrated; blue=happy, theta<0). No single-valued height y can satisfy y_t-y_s>theta_e everywhere.
 (3) THE OBSTRUCTION LOOP (Karp max-mean-cycle): the non-contractible staircase the topology forces to have
     sum theta >= 0 -- drawn UNROLLED as a monotone lattice path (the 'Escher staircase' that climbs forever
     yet must close on the torus). Its winding (W_x,W_y) and sum theta are annotated.

Reuses expU/expAE/expAK/expAQ. Writes obstruction.png.
"""

from __future__ import annotations
import importlib.util, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
expAE = _load("expAE", "expAE_pq.py")
expAK = _load("expAK", "expAK_loops.py")
expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
cgn = expU.cgn


def berry_curvature(m, u):
    """per-plaquette discrete Berry curvature F_(x,y) in (-pi,pi]; sum/2pi = Chern."""
    b = u / np.linalg.norm(u, axis=1, keepdims=True)
    bg = b.reshape(m, m, -1)
    F = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            u1 = np.vdot(bg[x, y], bg[(x+1) % m, y]); u2 = np.vdot(bg[(x+1) % m, y], bg[(x+1) % m, (y+1) % m])
            u3 = np.vdot(bg[(x+1) % m, (y+1) % m], bg[x, (y+1) % m]); u4 = np.vdot(bg[x, (y+1) % m], bg[x, y])
            F[x, y] = np.angle(u1*u2*u3*u4)
    return F


def karp_cycle(N, src, dst, w):
    """Karp max-mean-cycle WITH cycle recovery. Returns (mu_star, cycle_nodes list)."""
    NEG = -1e18
    dp = np.full((N+1, N), NEG); par = np.full((N+1, N), -1, dtype=int)
    dp[0, :] = 0.0
    for k in range(1, N+1):
        for e in range(len(src)):
            s, t = src[e], dst[e]
            if dp[k-1, s] > NEG/2 and dp[k-1, s] + w[e] > dp[k, t]:
                dp[k, t] = dp[k-1, s] + w[e]; par[k, t] = s
    # best node v* maximizing min_k (dp[N,v]-dp[k,v])/(N-k)
    mu = -np.inf; vstar = -1
    for v in range(N):
        if dp[N, v] <= NEG/2: continue
        worst = np.inf
        for k in range(N):
            if dp[k, v] <= NEG/2: continue
            worst = min(worst, (dp[N, v]-dp[k, v])/(N-k))
        if worst > mu: mu = worst; vstar = v
    # recover a cycle: walk back N steps from (N, v*), find a repeated node
    seq = [vstar]; v = vstar
    for k in range(N, 0, -1):
        v = par[k, v]
        if v < 0: break
        seq.append(v)
    seq = seq[::-1]
    seen = {}; cycle = None
    for i, node in enumerate(seq):
        if node in seen:
            cycle = seq[seen[node]:i+1]; break
        seen[node] = i
    return float(mu), cycle


def main():
    m, k = 6, 2
    np.random.seed(20)
    u, src, dst, phi = expAK.find_all_ray_line(m, k, tries=20)
    assert u is not None, "no all-ray Chern=-2 line"
    N = m*m
    ch = expAE.chern_of_u(m, u)
    theta, theta0, allray, slack = expAQ.theta_bisect(u, src, dst, phi)
    F = berry_curvature(m, u)
    mu, cycle = karp_cycle(N, src, dst, theta)
    print(f"m={m} Chern={ch:+.2f} all_ray={allray} min-slack={slack:+.4f}  Karp mu*={mu:+.3f}")

    # site<->(x,y) maps (cgn.site convention)
    site2xy = {}
    for x in range(m):
        for y in range(m):
            site2xy[cgn.site(x, y, m)] = (x, y)
    # edge lookup: (src,dst)->index, and H/V
    edge_of = {(int(src[e]), int(dst[e])): e for e in range(len(src))}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # ---- Panel 1: Berry curvature (topology) ----
    ax = axes[0]
    im = ax.imshow(F.T/(2*np.pi), origin="lower", cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                   extent=[0, m, 0, m])
    ax.set_title(f"(1) Topology: trapped charge\nBerry curvature, $\\sum=$Chern$={ch:+.0f}$ "
                 f"(flux $=+{k}$ cancels $\\to$ combined 0)", fontsize=10.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("$F_p/2\\pi$", fontsize=9)

    # ---- Panel 2: frustration field theta_e on the bonds ----
    ax = axes[1]
    segs = []; cols = []
    tmin, tmax = np.nanmin(theta), np.nanmax(theta)
    norm = plt.Normalize(vmin=-max(abs(tmin), abs(tmax)), vmax=max(abs(tmin), abs(tmax)))
    cmap = plt.cm.RdBu_r
    for e in range(len(src)):
        x0, y0 = site2xy[int(src[e])]; x1, y1 = site2xy[int(dst[e])]
        dx, dy = (x1-x0), (y1-y0)
        # unwrap to short forward step for drawing
        if dx > 1: dx = -1
        if dx < -1: dx = 1
        if dy > 1: dy = -1
        if dy < -1: dy = 1
        segs.append([(x0, y0), (x0+0.82*dx, y0+0.82*dy)]); cols.append(cmap(norm(theta[e])))
    lc = LineCollection(segs, colors=cols, linewidths=2.4)
    ax.add_collection(lc)
    ax.scatter([site2xy[s][0] for s in range(N)], [site2xy[s][1] for s in range(N)],
               s=10, color="0.25", zorder=3)
    ax.set_xlim(-0.6, m-0.4); ax.set_ylim(-0.6, m-0.4); ax.set_aspect("equal")
    ax.set_title("(2) Frustration on the bonds\n"
                 r"red $=\theta_e$ high (near boundary $R\!\to\!|q|$), blue $=$ happy", fontsize=10.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"$\theta_e=\log x_e^0$", fontsize=9)

    # ---- Panel 3: the obstruction loop, unrolled (Escher staircase) ----
    ax = axes[2]
    if cycle and len(cycle) > 2:
        # unroll: cumulative monotone lattice path
        X = [site2xy[cycle[0]][0]]; Y = [site2xy[cycle[0]][1]]
        ths = []
        for i in range(len(cycle)-1):
            a, b = cycle[i], cycle[i+1]
            e = edge_of.get((a, b))
            ax0, ay0 = site2xy[a]
            horiz = (e % 2 == 0)
            X.append(X[-1] + (1 if horiz else 0)); Y.append(Y[-1] + (0 if horiz else 1))
            ths.append(theta[e] if e is not None else 0.0)
        X = np.array(X); Y = np.array(Y); ths = np.array(ths)
        pts = np.column_stack([X, Y]).reshape(-1, 1, 2)
        segs2 = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc2 = LineCollection(segs2, cmap="RdBu_r",
                             norm=plt.Normalize(-abs(ths).max(), abs(ths).max()), linewidths=4)
        lc2.set_array(ths); ax.add_collection(lc2)
        ax.scatter(X, Y, s=14, color="0.2", zorder=3)
        Wx = (X[-1]-X[0]) / m; Wy = (Y[-1]-Y[0]) / m
        sumt = float(ths.sum())
        ax.set_xlim(X.min()-0.5, X.max()+0.5); ax.set_ylim(Y.min()-0.5, Y.max()+0.5)
        ax.set_aspect("equal")
        ax.set_title(f"(3) The obstruction loop (unrolled)\n"
                     f"winding $({Wx:.0f},{Wy:.0f})$, $\\sum_C\\theta={sumt:+.2f}\\geq0\\Rightarrow$ PB$\\geq1$",
                     fontsize=10.5)
        ax.set_xlabel("cumulative x  (climbs $\\to$ never closes on the torus)")
        ax.set_ylabel("cumulative y")
    else:
        ax.text(0.5, 0.5, "cycle recovery failed", ha="center")

    fig.suptitle("Why $PB\\geq1$ for $r>2$: a topological charge (Chern $-2$) frustrates the line so that some "
                 "non-contractible loop cannot relax ($\\mu^*\\geq0$)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    outp = os.path.join(HERE, "obstruction.png")
    fig.savefig(outp, dpi=140, bbox_inches="tight")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
