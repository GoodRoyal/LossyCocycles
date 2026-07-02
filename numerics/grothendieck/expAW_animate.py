"""expAW_animate.py — ANIMATE the obstruction: push to the boundary, watch it refuse to relax.

At fixed m=6 in the BINDING sector Chern=-2, push the competitor toward maximal coherence (toward the all-ray
boundary R_e -> |q_e|, i.e. slack -> 0 -- the regime where the bound is TIGHT). Animate, per frame:
  (left)  the frustration field theta_e on the torus bonds (intensifies as we approach the boundary);
  (mid)   the obstruction loop (Karp max-mean-cycle) UNROLLED as a climbing staircase, colored by theta,
          annotated with sum_C theta and winding;
  (right) running traces of the all-ray slack (-> 0) and mu* (PINNED ~1.8): tight but unbroken.

The story: as the line is squeezed to the boundary the frustration grows, but the non-contractible staircase
keeps sum_C theta >= 0 (mu* stays positive) -- the topological charge never lets the loop close. That is why
PB>=1 is hard: the bound is exactly tight (slack->0) yet never fails.
Reuses expU/expAE/expAK/expAQ/expAP/expAV. Writes obstruction.gif.
"""

from __future__ import annotations
import importlib.util, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
expU = _load("expU", "expU_shadow.py")
expAE = _load("expAE", "expAE_pq.py")
expAK = _load("expAK", "expAK_loops.py")
expAQ = _load("expAQ", "expAQ_verify_Hfalse.py")
expAP = _load("expAP", "expAP_coupled.py")
expAV = _load("expAV", "expAV_visualize.py")
cgn = expU.cgn


def collect_snapshots(u0, src, dst, phi, m, k, steps=3000, lr=0.02, n_snaps=120):
    """leashed coherence ascent in the -k sector; snapshot u at increasing coherence (slack shrinking)."""
    u = expAE.project_unit(u0.copy()); mt = np.zeros_like(u); vt = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    snaps = []; last_good = u.copy(); seen_coh = []
    def coherence(uu):
        us, ut = uu[src], uu[dst]
        R = (np.exp(1j*phi)*np.einsum('ej,ej->e', ut.conj(), us)).real
        return min(R[0::2].mean(), R[1::2].mean())
    snaps.append(u.copy())
    every = max(1, steps // n_snaps)
    for t in range(1, steps+1):
        beta = 50 + 250*(t/steps); lam_t = 3.0*(0.5+0.5*t/steps)
        J, mn, mh, mv, g, R, absq = expAP.obj_and_grad(u, src, dst, phi, m, beta, lam_t, 0.012)
        proj = np.einsum('ej,ej->e', u.conj(), g)[:, None]*u; gt = g - proj
        mt = b1*mt+(1-b1)*gt; vt = b2*vt+(1-b2)*(gt.conj()*gt).real
        step = lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+eps)
        ucand = expAE.project_unit(u + step)
        ch = expAE.chern_of_u(m, ucand)
        if ch is not None and abs(ch + k) < 0.25:
            u = ucand; last_good = ucand.copy()
        else:
            u = last_good.copy(); mt *= 0.3; vt *= 0.3
        if t % every == 0:
            # keep only all-ray snapshots (so theta is defined on every edge)
            us, ut = u[src], u[dst]
            Rr = (np.exp(1j*phi)*np.einsum('ej,ej->e', ut.conj(), us)).real
            qq = np.abs(np.einsum('ej,ej->e', ut, us))
            if np.all(Rr > qq + 1e-9):
                snaps.append(u.copy())
    return snaps


def frame_data(u, src, dst, phi, m, N, site2xy, edge_of):
    theta, theta0, allray, slack = expAQ.theta_bisect(u, src, dst, phi)
    mu, cycle = expAV.karp_cycle(N, src, dst, theta)
    us, ut = u[src], u[dst]
    R = (np.exp(1j*phi)*np.einsum('ej,ej->e', ut.conj(), us)).real
    coh = min(R[0::2].mean(), R[1::2].mean())
    # unroll cycle
    X = [site2xy[cycle[0]][0]]; Y = [site2xy[cycle[0]][1]]; ths = []
    for i in range(len(cycle)-1):
        e = edge_of.get((cycle[i], cycle[i+1]))
        horiz = (e % 2 == 0) if e is not None else True
        X.append(X[-1] + (1 if horiz else 0)); Y.append(Y[-1] + (0 if horiz else 1))
        ths.append(theta[e] if e is not None else 0.0)
    return dict(theta=theta, slack=float(slack), mu=float(mu), coh=float(coh),
               X=np.array(X), Y=np.array(Y), ths=np.array(ths), sumt=float(np.sum(ths)))


def main():
    m, k = 6, 2; np.random.seed(20)
    u0, src, dst, phi = expAK.find_all_ray_line(m, k, tries=20)
    assert u0 is not None
    N = m*m
    site2xy = {cgn.site(x, y, m): (x, y) for x in range(m) for y in range(m)}
    edge_of = {(int(src[e]), int(dst[e])): e for e in range(len(src))}

    print("collecting snapshots along the coherence push (slack -> 0)...")
    snaps = collect_snapshots(u0, src, dst, phi, m, k)
    if len(snaps) > 30:                       # subsample to ~30 frames for a snappy GIF
        idx = np.linspace(0, len(snaps)-1, 30).round().astype(int)
        snaps = [snaps[i] for i in idx]
    frames = [frame_data(s, src, dst, phi, m, N, site2xy, edge_of) for s in snaps]
    print(f"{len(frames)} frames: slack {frames[0]['slack']:.3f} -> {frames[-1]['slack']:.3f}, "
          f"mu* {frames[0]['mu']:.2f} -> {frames[-1]['mu']:.2f}")

    tabs = max(np.nanmax(np.abs(f['theta'])) for f in frames)
    norm = plt.Normalize(-tabs, tabs); cmap = plt.cm.RdBu_r
    sl = [f['slack'] for f in frames]; mus = [f['mu'] for f in frames]; cohs = [f['coh'] for f in frames]

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Squeezing a Chern$-2$ line to the all-ray boundary: frustration grows, "
                 "but the obstruction loop never relaxes ($\\mu^*\\geq0$ pinned)", fontsize=12.5, y=1.0)

    def draw(i):
        f = frames[i]
        for ax in (axL, axM, axR): ax.clear()
        # LEFT: frustration bonds
        segs = []; cols = []
        for e in range(len(src)):
            x0, y0 = site2xy[int(src[e])]; x1, y1 = site2xy[int(dst[e])]
            dx, dy = x1-x0, y1-y0
            dx = -1 if dx > 1 else (1 if dx < -1 else dx)
            dy = -1 if dy > 1 else (1 if dy < -1 else dy)
            segs.append([(x0, y0), (x0+0.82*dx, y0+0.82*dy)]); cols.append(cmap(norm(f['theta'][e])))
        axL.add_collection(LineCollection(segs, colors=cols, linewidths=2.6))
        axL.scatter([site2xy[s][0] for s in range(N)], [site2xy[s][1] for s in range(N)], s=8, color="0.3")
        axL.set_xlim(-0.6, m-0.4); axL.set_ylim(-0.6, m-0.4); axL.set_aspect("equal")
        axL.set_title(f"frustration  $\\theta_e$   (slack $R\\!-\\!|q| = {f['slack']:.3f}$)", fontsize=10.5)
        axL.set_xticks([]); axL.set_yticks([])
        # MID: unrolled staircase
        X, Y, ths = f['X'], f['Y'], f['ths']
        pts = np.column_stack([X, Y]).reshape(-1, 1, 2)
        s2 = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(s2, cmap=cmap, norm=norm, linewidths=4.5); lc.set_array(ths)
        axM.add_collection(lc); axM.scatter(X, Y, s=12, color="0.2")
        axM.set_xlim(X.min()-0.5, X.max()+0.5); axM.set_ylim(Y.min()-0.5, Y.max()+0.5); axM.set_aspect("equal")
        Wx = (X[-1]-X[0])/m; Wy = (Y[-1]-Y[0])/m
        axM.set_title(f"obstruction staircase  $\\sum_C\\theta={f['sumt']:+.1f}$  wind$({Wx:.0f},{Wy:.0f})$",
                      fontsize=10.5)
        axM.set_xlabel("climbs $\\to$ never closes"); axM.set_xticks([]); axM.set_yticks([])
        # RIGHT: traces
        fr = np.arange(i+1)
        axR.plot(fr, sl[:i+1], "-o", ms=3, color="tab:red", label="all-ray slack $\\to 0$")
        axR.plot(fr, mus[:i+1], "-o", ms=3, color="tab:blue", label="$\\mu^*$ (pinned $\\geq0$)")
        axR.axhline(0, color="0.6", lw=0.8, ls="--")
        axR.set_xlim(0, len(frames)-1); axR.set_ylim(-0.2, 2.2)
        axR.set_title("tight but unbroken", fontsize=10.5); axR.set_xlabel("push step")
        axR.legend(loc="center right", fontsize=9)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=200, blit=False)
    fig.tight_layout()
    outp = os.path.join(HERE, "obstruction.gif")
    anim.save(outp, writer=PillowWriter(fps=5))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
