#!/usr/bin/env python3
"""
Plateau attack — pre-registered numerical falsification battery.
Tests the coarse-geometry reduction's claims for the EXPLICIT magnetic-translation
flux family (matching LossyCocycles/Defs.lean), to confirm-or-kill the Bott route.

Experiments:
  A. Bott index identification:  bott(flux) == k,  bott(flat) == 0,  across m.
     (validates "totalFlux/2pi == Bott index", the cited input A of the reduction)
  B. Polar (unitary-part) Lipschitz constant under the gap, across m.
     (the NEW estimate: bounded geometry => m-free Lipschitz => contraction transport)
  C. Dimension-free stability: homotopy flux -> trivial-flat; the operator distance
     at which the Bott index jumps from k to 0.  Is that distance bounded below,
     independent of m?  (the decisive dimension-free / delta_0 test)
  E. Compression / "cheap channel" test: compress to the coreGap support
     projection P and measure the off-diagonal leak ||(1-P)[U,V]P||.  Ideal family
     leak == 0 (clean); wobbled frames give leak/eps == O(1), flat in m (the
     coreGap-controlled Schur estimate that the section-9 reframe rests on).

Scalar U(1) core used for A,C (the SO(2) core complexifies to U(1); operators are
genuinely unitary there).  Real d>r partial-isometry used for B (to exercise the
actual non-unitary / rank-loss structure).
"""
import os
import numpy as np
from scipy.linalg import logm, expm, svd

np.random.seed(0)
TWO_PI = 2 * np.pi

# ----------------------------------------------------------------------------- #
#  Scalar U(1) magnetic translations (Defs.lean fluxAngle, complexified core)    #
# ----------------------------------------------------------------------------- #
def site(x, y, m):           # flatten (x,y) -> index
    return (x % m) * m + (y % m)

def flux_angles(m, k):
    """vertical edge (x,y): 2*pi*k*x/m^2 ; horizontal: 0 except x=m-1: -2*pi*k*y/m"""
    phv = np.zeros((m, m)); phh = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            phv[x, y] = TWO_PI * k * x / (m * m)
            phh[x, y] = (-TWO_PI * k * y / m) if x == m - 1 else 0.0
    return phh, phv

def mag_pair_U1(m, phh, phv):
    """U[(x+1,y),(x,y)] = e^{i phh}, V[(x,y+1),(x,y)] = e^{i phv}; unitary shifts."""
    n = m * m
    U = np.zeros((n, n), complex); V = np.zeros((n, n), complex)
    for x in range(m):
        for y in range(m):
            U[site(x + 1, y, m), site(x, y, m)] = np.exp(1j * phh[x, y])
            V[site(x, y + 1, m), site(x, y, m)] = np.exp(1j * phv[x, y])
    return U, V

def bott_index(U, V):
    """(1/2pi i) Tr log(U V U^dag V^dag), principal branch.
    Returns (real value, min distance of spec(M) from the -1 branch cut)."""
    M = U @ V @ U.conj().T @ V.conj().T
    ev = np.linalg.eigvals(M)
    ang = np.angle(ev)
    # distance of eigenvalue arguments from +/-pi (the branch cut at -1)
    gap = np.min(np.pi - np.abs(ang))
    # (1/2pi i) Tr log M = (1/2pi) sum arg(eigvals)  for unitary M, principal branch
    val = np.sum(ang) / (2 * np.pi)
    return val, gap

def flat_pair_U1(m, theta):
    """flat: edge phase = theta(dst) - theta(src); pure gauge."""
    phh = np.zeros((m, m)); phv = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            phh[x, y] = theta[(x + 1) % m, y] - theta[x, y]
            phv[x, y] = theta[x, (y + 1) % m] - theta[x, y]
    return mag_pair_U1(m, phh, phv)

# ----------------------------------------------------------------------------- #
#  Experiment A : index identification                                           #
# ----------------------------------------------------------------------------- #
def expA():
    print("=" * 72)
    print("A. Bott index identification (expect bott(flux)=k, bott(flat)=0)")
    print("=" * 72)
    print(f"{'m':>4} {'k':>3} {'bott(flux)':>12} {'branchgap':>10} "
          f"{'bott(flat)':>12} {'2|sin(pik/m^2)|':>16}")
    for k in (1, 2):
        for m in (4, 6, 8, 12, 16, 24, 32):
            phh, phv = flux_angles(m, k)
            U, V = mag_pair_U1(m, phh, phv)
            bf, gap = bott_index(U, V)
            theta = TWO_PI * np.random.rand(m, m)
            Uf, Vf = flat_pair_U1(m, theta)
            bflat, _ = bott_index(Uf, Vf)
            comm = 2 * abs(np.sin(np.pi * k / m**2))
            print(f"{m:>4} {k:>3} {bf:>12.6f} {gap:>10.4f} "
                  f"{bflat:>12.2e} {comm:>16.6f}")
        print()

# ----------------------------------------------------------------------------- #
#  Experiment B : polar Lipschitz constant under the gap (d>r, partial isometry) #
# ----------------------------------------------------------------------------- #
def rot_core(theta, d=3):
    R = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R

def std_frame(d=3, r=2):
    S = np.zeros((d, r))
    for j in range(r):
        S[j, j] = 1.0
    return S

def flux_edge_maps(m, k, d=3, r=2):
    """T(e) = S R(phi_e) S^T : d x d rank-r partial isometry (kills coord d-1)."""
    S = std_frame(d, r)
    phh, phv = flux_angles(m, k)
    Th = {}; Tv = {}
    for x in range(m):
        for y in range(m):
            Th[(x, y)] = S @ rot_core(phh[x, y], r) @ S.T
            Tv[(x, y)] = S @ rot_core(phv[x, y], r) @ S.T
    return Th, Tv

def big_U(m, Th, d=3):
    n = m * m * d
    U = np.zeros((n, n))
    def idx(x, y, a): return (site(x, y, m)) * d + a
    for x in range(m):
        for y in range(m):
            T = Th[(x, y)]
            for a in range(d):
                for b in range(d):
                    U[idx(x + 1, y, a), idx(x, y, b)] = T[a, b]
    return U

def big_V(m, Tv, d=3):
    n = m * m * d
    V = np.zeros((n, n))
    def idx(x, y, a): return (site(x, y, m)) * d + a
    for x in range(m):
        for y in range(m):
            T = Tv[(x, y)]
            for a in range(d):
                for b in range(d):
                    V[idx(x, y + 1, a), idx(x, y, b)] = T[a, b]
    return V

def polar_unitary_core(A, r_total):
    """Unitary part on the top-r_total singular subspace: W Z^T from SVD."""
    W, sig, Zt = svd(A)
    return W[:, :r_total] @ Zt[:r_total, :], sig

def expB():
    print("=" * 72)
    print("B. Polar (unitary-part) Lipschitz constant under the gap, d=3 r=2")
    print("   ||U_polar(A+dA) - U_polar(A)|| / ||dA||  -- should stay O(1) in m")
    print("=" * 72)
    d, r = 3, 2
    print(f"{'m':>4} {'sigma_min(core)':>16} {'polarLip(mean)':>16} {'polarLip(max)':>14}")
    for m in (4, 6, 8, 12, 16):
        Th, _ = flux_edge_maps(m, 1, d, r)
        U = big_U(m, Th, d)
        rtot = m * m * r            # core rank of the whole operator
        Up, sig = polar_unitary_core(U, rtot)
        sigmin = sig[rtot - 1]      # smallest *core* singular value (the gap)
        lips = []
        for _ in range(12):
            dA = np.random.randn(*U.shape)
            dA *= (1e-4 / np.linalg.norm(dA, 2))   # ||dA||_2 = 1e-4
            Up2, _ = polar_unitary_core(U + dA, rtot)
            lips.append(np.linalg.norm(Up2 - Up, 2) / 1e-4)
        print(f"{m:>4} {sigmin:>16.4f} {np.mean(lips):>16.4f} {np.max(lips):>14.4f}")
    print()

# ----------------------------------------------------------------------------- #
#  Experiment C : dimension-free stability (homotopy flux -> trivial flat)       #
# ----------------------------------------------------------------------------- #
def expC():
    print("=" * 72)
    print("C. Dimension-free stability: homotopy flux -> trivial-flat (pure shifts).")
    print("   Operator distance from flux at which Bott index leaves k.")
    print("   DECISIVE: does this jump-distance stay bounded BELOW as m grows?")
    print("=" * 72)
    k = 1
    print(f"{'m':>4} {'jump_dist(||U_t-U_0||_2)':>26} {'bott just before':>18}")
    for m in (4, 6, 8, 12, 16):
        phh, phv = flux_angles(m, k)
        U0, V0 = mag_pair_U1(m, phh, phv)                  # flux
        Ut_target, Vt_target = mag_pair_U1(m, np.zeros((m, m)), np.zeros((m, m)))  # trivial shifts (flat, commute)
        b0, gap0 = bott_index(U0, V0)
        kref = round(b0)                                   # = -k in this orientation
        jump = None; bprev = b0
        ts = np.linspace(0, 1, 161)
        for t in ts[1:]:
            # geodesic-ish: polar of convex combination, keeps unitarity
            A = (1 - t) * U0 + t * Ut_target
            B = (1 - t) * V0 + t * Vt_target
            Wa, _, Za = svd(A); Ut = Wa @ Za
            Wb, _, Zb = svd(B); Vt = Wb @ Zb
            b, gap = bott_index(Ut, Vt)
            if abs(round(b) - kref) >= 1 or gap < 0.05:    # index left k0 or ill-defined
                jump = np.linalg.norm(Ut - U0, 2)
                break
            bprev = b
        jd = f"{jump:.4f}" if jump is not None else "no jump on [0,1]"
        print(f"{m:>4} {jd:>26} {bprev:>18.4f}")
    print()

# ----------------------------------------------------------------------------- #
#  Experiment E : the compression / "cheap channel" test (the §9 reframe)         #
#  ---------------------------------------------------------------------------   #
#  Reframe: don't transport the unitary distance bound across polar decomposition #
#  -- compress to the coreGap support projection P (where the partial isometries  #
#  are honest unitaries) and apply the unitary theorem there.  The ONLY residue   #
#  is the off-diagonal block ||(1-P)[U,V]P||: the "cheap channel" the kernel might #
#  open.  For the ideal family the support is exactly preserved (leak == 0); the  #
#  real test is whether, once the support frames WOBBLE site-to-site (so P is no  #
#  longer exactly preserved), the leak stays controlled by coreGap and m-uniform. #
#  Schur prediction: leak <= (1/coreGap)*||wobble|| ~ eps, ratio O(1), flat in m. #
# ----------------------------------------------------------------------------- #
def core_proj(m, d=3, r=2):
    """Block-diagonal projection onto the surviving SO(2) core at every site."""
    n = m * m * d
    P = np.zeros((n, n))
    for x in range(m):
        for y in range(m):
            for a in range(r):
                i = site(x, y, m) * d + a
                P[i, i] = 1.0
    return P

def ortho_near_I(d, eps, rng):
    """Orthogonal matrix exp(eps*skew) -- a small site-local frame rotation."""
    X = rng.standard_normal((d, d))
    return expm(eps * (X - X.T) / 2.0)

def flux_edge_maps_tilted(m, k, eps, d=3, r=2, rng=None):
    """As flux_edge_maps, but each edge's rank-r frame is tilted by a random
    O(eps) rotation, so the support 2-plane no longer equals coords {0,1}."""
    phh, phv = flux_angles(m, k)
    Th = {}; Tv = {}
    for x in range(m):
        for y in range(m):
            Sh = ortho_near_I(d, eps, rng)[:, :r]
            Sv = ortho_near_I(d, eps, rng)[:, :r]
            Th[(x, y)] = Sh @ rot_core(phh[x, y], r) @ Sh.T
            Tv[(x, y)] = Sv @ rot_core(phv[x, y], r) @ Sv.T
    return Th, Tv

def expE():
    print("=" * 72)
    print("E. Compression / 'cheap channel' test (the section-9 reframe), d=3 r=2")
    print("   P = core support projection. leak = ||(1-P)[U,V]P||_2.")
    print("   ideal family: leak should be ~0 (support exactly preserved).")
    print("   wobbled frames (eps=1e-2): leak/eps should be O(1) and FLAT in m.")
    print("=" * 72)
    d, r, k, eps = 3, 2, 1, 1e-2
    rng = np.random.default_rng(0)
    print(f"{'m':>4} {'sig_min(core)':>14} {'leak_ideal':>12} {'||P[U,V]P||':>12} "
          f"{'leak_wobble':>12} {'leak/eps':>10}")
    for m in (4, 6, 8, 12, 16, 24, 32):
        P = core_proj(m, d, r)
        Q = np.eye(P.shape[0]) - P
        # ideal family
        Th, Tv = flux_edge_maps(m, k, d, r)
        U, V = big_U(m, Th, d), big_V(m, Tv, d)
        C = U @ V - V @ U
        sig = svd(U, compute_uv=False)
        sigmin = sig[m * m * r - 1]
        leak_ideal = np.linalg.norm(Q @ C @ P, 2)
        kept = np.linalg.norm(P @ C @ P, 2)
        # wobbled frames
        Thw, Tvw = flux_edge_maps_tilted(m, k, eps, d, r, rng)
        Uw, Vw = big_U(m, Thw, d), big_V(m, Tvw, d)
        Cw = Uw @ Vw - Vw @ Uw
        leak_w = np.linalg.norm(Q @ Cw @ P, 2)
        print(f"{m:>4} {sigmin:>14.4f} {leak_ideal:>12.2e} {kept:>12.4f} "
              f"{leak_w:>12.2e} {leak_w / eps:>10.4f}")
    print()

# ----------------------------------------------------------------------------- #
#  Experiment F : ADVERSARIAL leak (worst-case competitor frame)                 #
#  ---------------------------------------------------------------------------   #
#  Exp E used RANDOM frame wobble, which partially cancels across edges.  The     #
#  distOpFlat metric is sup_e ||T(e)-T'(e)||, so an adversary may spend an eps    #
#  budget on EVERY one of the m^2 edges at once; coherent choices could ACCUMULATE #
#  into one large singular direction that grows with m -- the only way the cheap  #
#  channel reopens.  We maximise leak = ||(1-P)[U,V]P||_2 by projected gradient   #
#  ascent over per-edge so(d) frame generators A_e with per-edge budget ||A_e||<=eps. #
#  Analytic top-singular-value gradient (no m^3 commutator formed): matvec only.  #
#  FALSIFIER: if max-leak/eps grows like a power of m, the channel is real.       #
# ----------------------------------------------------------------------------- #
def skew_basis(d=3):
    B = []
    for i in range(d):
        for j in range(i + 1, d):
            A = np.zeros((d, d)); A[i, j] = 1.0; A[j, i] = -1.0; B.append(A)
    return B

def core_mask(m, d=3, r=2):
    n = m * m * d
    mask = np.zeros(n)
    for x in range(m):
        for y in range(m):
            for a in range(r):
                mask[site(x, y, m) * d + a] = 1.0
    return mask

def edge_maps_from_gens(m, k, gh, gv, d=3, r=2):
    """frames S_e = expm(A_e)[:, :r]; T_e = S_e R(phi_e) S_e^T (rank r)."""
    phh, phv = flux_angles(m, k)
    Th = {}; Tv = {}
    for x in range(m):
        for y in range(m):
            Sh = expm(gh[(x, y)])[:, :r]; Sv = expm(gv[(x, y)])[:, :r]
            Th[(x, y)] = Sh @ rot_core(phh[x, y], r) @ Sh.T
            Tv[(x, y)] = Sv @ rot_core(phv[x, y], r) @ Sv.T
    return Th, Tv

def leak_top_sv(U, V, pmask, iters=50, seed=1):
    """sigma_max(M), top vecs u,v for M = Q[U,V]P (Q=1-P), via matvec power iter."""
    q = 1.0 - pmask
    Mv  = lambda x: q * (U @ (V @ (pmask * x)) - V @ (U @ (pmask * x)))
    MTv = lambda x: pmask * (V.T @ (U.T @ (q * x)) - U.T @ (V.T @ (q * x)))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(U.shape[0]); v /= np.linalg.norm(v)
    for _ in range(iters):
        u = Mv(v); s = np.linalg.norm(u)
        if s < 1e-14: return 0.0, None, None
        u /= s
        v = MTv(u); sv = np.linalg.norm(v)
        if sv < 1e-14: return 0.0, None, None
        v /= sv
    Mvv = Mv(v); sigma = np.linalg.norm(Mvv)
    return sigma, (Mvv / sigma if sigma > 0 else Mvv), v

def _rand_gens(m, d, eps, scale, seed):
    rng = np.random.default_rng(seed)
    mk = lambda: (lambda A: (A - A.T) / 2 * (eps * scale))(rng.standard_normal((d, d)))
    return ({(x, y): mk() for x in range(m) for y in range(m)},
            {(x, y): mk() for x in range(m) for y in range(m)})

def expF(ms=(4, 6, 8, 12, 16), eps=1e-2, iters=80, restarts=2, d=3, r=2, k=1):
    print("=" * 72)
    print("F. ADVERSARIAL leak: max ||(1-P)[U,V]P|| over per-edge frames, ||A_e||<=eps")
    print(f"   projected gradient ascent, eps={eps}, iters={iters}, restarts={restarts}+warm")
    print("   FALSIFIER: max-leak/eps growing as a power of m => cheap channel real.")
    print("=" * 72)
    SK = skew_basis(d); hh = 1e-6
    print(f"{'m':>4} {'rand leak/eps':>14} {'ADV leak/eps':>14} {'adv/rand':>10}")
    for m in ms:
        pmask = core_mask(m, d, r)
        phh, phv = flux_angles(m, k)
        # random baseline (same construction as Exp E), projected into the eps-ball
        gh0, gv0 = _rand_gens(m, d, eps, 1.0, seed=0)
        for g in (gh0, gv0):
            for key, A in g.items():
                an = np.linalg.norm(A, 2)
                if an > eps: g[key] = A * (eps / an)
        Th, Tv = edge_maps_from_gens(m, k, gh0, gv0, d, r)
        Ur, Vr = big_U(m, Th, d), big_V(m, Tv, d)
        rand_leak, _, _ = leak_top_sv(Ur, Vr, pmask)

        best_adv = 0.0
        # restart 0 is WARM-STARTED from the random baseline (=> adv >= rand always);
        # the rest are random restarts to escape local maxima.
        inits = [({k_: v.copy() for k_, v in gh0.items()},
                  {k_: v.copy() for k_, v in gv0.items()})]
        for rs in range(restarts):
            inits.append(_rand_gens(m, d, eps, 0.3, seed=100 + rs))
        for gh, gv in inits:
            cur = 0.0
            for it in range(iters):
                Th, Tv = edge_maps_from_gens(m, k, gh, gv, d, r)
                U, V = big_U(m, Th, d), big_V(m, Tv, d)
                sigma, u, v = leak_top_sv(U, V, pmask)
                cur = max(cur, sigma)
                if u is None: break
                q = 1.0 - pmask
                # gradient of sigma w.r.t. blocks of U and V (rank-2 outer products)
                a1 = q * u; b1 = V @ (pmask * v)
                a2 = V.T @ (q * u); b2 = pmask * v
                c1 = U.T @ (q * u); d2v = U @ (pmask * v)
                # ascent on each edge generator via local chain rule
                for x in range(m):
                    for y in range(m):
                        R = site(x + 1, y, m) * d; C = site(x, y, m) * d
                        GUblk = (np.outer(a1[R:R + d], b1[C:C + d])
                                 - np.outer(a2[R:R + d], b2[C:C + d]))
                        Rv = site(x, y + 1, m) * d
                        GVblk = (np.outer(c1[Rv:Rv + d], b2[C:C + d])  # Pv=b2
                                 - np.outer(a1[Rv:Rv + d], d2v[C:C + d]))
                        for (g, GB, phi) in ((gh, GUblk, phh[x, y]), (gv, GVblk, phv[x, y])):
                            A = g[(x, y)]; S0 = expm(A)[:, :r]; T0 = S0 @ rot_core(phi, r) @ S0.T
                            grad = np.zeros((d, d))
                            for B in SK:
                                S1 = expm(A + hh * B)[:, :r]
                                dT = (S1 @ rot_core(phi, r) @ S1.T - T0) / hh
                                grad += np.sum(GB * dT) * B
                            gn = np.linalg.norm(grad, 2)
                            if gn > 1e-18:
                                A = A + (0.4 * eps) * grad / gn          # ascent step
                            an = np.linalg.norm(A, 2)                     # project to ball
                            g[(x, y)] = A * (eps / an) if an > eps else A
            best_adv = max(best_adv, cur)
        ratio = best_adv / rand_leak if rand_leak > 0 else float('nan')
        print(f"{m:>4} {rand_leak / eps:>14.4f} {best_adv / eps:>14.4f} {ratio:>10.3f}")
    print()

# ----------------------------------------------------------------------------- #
#  Experiment G : direct distOpFlat (the conjectured quantity itself)            #
#  ---------------------------------------------------------------------------   #
#  distOpFlat(r, flux) = inf over FLAT competitors T'(e)=V(dst)V(src)^T (site     #
#  frames V, V^T V = I_r) of  sup_e ||flux(e) - T'(e)||_2.  We minimise this max  #
#  directly by Riemannian gradient descent over the product of Stiefel manifolds  #
#  (analytic top-singular-value gradient, batched 3x3 SVD, polar retraction,      #
#  softmax-smoothed max, multi-restart incl. the V=S baseline).                   #
#  LOGIC: this is an UPPER bound on distOpFlat.  If it -> 0 with m, Plateau is     #
#  FALSE (strongest possible kill).  If it stays >= c(k) > 0, the best competitor  #
#  we can find cannot beat the floor -- consistent with (not proof of) Plateau.   #
# ----------------------------------------------------------------------------- #
def _edge_list(m, k):
    """edges as (src_site, dst_site, flux_angle): horizontal then vertical."""
    phh, phv = flux_angles(m, k)
    E = []
    for x in range(m):
        for y in range(m):
            E.append((site(x, y, m), site(x + 1, y, m), phh[x, y]))
            E.append((site(x, y, m), site(x, y + 1, m), phv[x, y]))
    return E

def _polar(V):  # batched retraction onto Stiefel (orthonormal columns)
    U_, _, Vt_ = np.linalg.svd(V, full_matrices=False)
    return U_ @ Vt_

def _distflat_minimize(m, k, d, r, steps, beta0, beta1, lr0, seed, V_init=None):
    S = std_frame(d, r)
    E = _edge_list(m, k)
    src = np.array([e[0] for e in E]); dst = np.array([e[1] for e in E])
    Tf = np.stack([S @ rot_core(phi, r) @ S.T for (_, _, phi) in E])   # (E,d,d)
    rng = np.random.default_rng(seed)
    if V_init == "S":
        Vs = np.broadcast_to(S, (m * m, d, r)).copy()
    else:
        Vs = _polar(rng.standard_normal((m * m, d, r)))
    best = np.inf
    for t in range(steps):
        beta = beta0 + (beta1 - beta0) * t / max(1, steps - 1)
        lr = lr0 * (1.0 - 0.9 * t / max(1, steps - 1))
        Vd = Vs[dst]; Vsr = Vs[src]
        M = Tf - np.einsum('eij,ekj->eik', Vd, Vsr)        # flux - V_dst V_src^T
        U_, Sg, Vt_ = np.linalg.svd(M)
        sig = Sg[:, 0]; u = U_[:, :, 0]; w = Vt_[:, 0, :]   # top singular triplet
        best = min(best, sig.max())
        mx = sig.max(); p = np.exp(beta * (sig - mx)); p /= p.sum()
        Vsrtw = np.einsum('edr,ed->er', Vsr, w)            # V_src^T w  (E,r)
        Vdtu  = np.einsum('edr,ed->er', Vd, u)             # V_dst^T u  (E,r)
        gVd = -(p[:, None, None]) * np.einsum('ed,er->edr', u, Vsrtw)
        gVs = -(p[:, None, None]) * np.einsum('ed,er->edr', w, Vdtu)
        grad = np.zeros_like(Vs)
        np.add.at(grad, dst, gVd)
        np.add.at(grad, src, gVs)
        Vs = _polar(Vs - lr * grad)                        # descend + retract
    return best

def expG(ms=(4, 6, 8, 12, 16, 24, 32), d=3, r=2, k=1, steps=300, restarts=3):
    print("=" * 72)
    print("G. Direct distOpFlat: min over flat competitors of  sup_e ||flux-T'||_2")
    print(f"   Riemannian GD on Stiefel^(m^2), steps={steps}, restarts={restarts}+S-init")
    print("   FALSIFIER: best-competitor distance -> 0 with m  =>  Plateau is FALSE.")
    print("=" * 72)
    print(f"{'m':>4} {'V=S baseline':>14} {'best distOpFlat (upper bnd)':>28}")
    for m in ms:
        base = _distflat_minimize(m, k, d, r, steps=1, beta0=1, beta1=1, lr0=0,
                                  seed=0, V_init="S")  # objective at V=S, no step
        best = _distflat_minimize(m, k, d, r, steps, 5.0, 60.0, 0.5, 0, V_init="S")
        for rs in range(restarts):
            best = min(best, _distflat_minimize(m, k, d, r, steps, 5.0, 60.0, 0.5,
                                                seed=10 + rs))
        print(f"{m:>4} {base:>14.4f} {best:>28.4f}")
    print()

if __name__ == "__main__":
    if os.environ.get("EXPG_ONLY"):
        ms = tuple(int(s) for s in os.environ.get("EXPG_MS", "4,6,8,12,16,24,32").split(","))
        expG(ms=ms,
             steps=int(os.environ.get("EXPG_STEPS", "300")),
             restarts=int(os.environ.get("EXPG_RESTARTS", "3")))
    elif os.environ.get("EXPF_ONLY"):
        ms = tuple(int(s) for s in os.environ.get("EXPF_MS", "4,6,8,12,16").split(","))
        expF(ms=ms,
             iters=int(os.environ.get("EXPF_ITERS", "80")),
             restarts=int(os.environ.get("EXPF_RESTARTS", "2")))
    else:
        expA()
        expB()
        expC()
        expE()
        expF()
        expG()
    print("Interpretation:")
    print("  A ok  => Bott==totalFlux/2pi confirmed (reduction input A).")
    print("  B flat in m => bounded geometry / gap gives m-free polar transport (NEW).")
    print("  C jump-dist bounded below in m => dimension-free delta_0 holds for OUR")
    print("     explicit pair => the Plateau route is real. C decaying => route dies.")
