"""expAH_spectral.py — Entry 51: spectral/duality route to the even-k holonomy bound.

From Entry 50: feasibility (PB<1) <=> EXISTS node magnitudes nu>0 with, on every edge e:s->t,
    E_e(nu) := a_e nu_t^2 - b_e nu_s nu_t + c_s nu_s^2 > 0   AND   L_e(nu):=nu_t Re[e^{i phi}p_e]-1/2 nu_s>0.
E_e is a homogeneous degree-2 form supported on the 2 sites {s,t}. Build the GLOBAL NxN symmetric matrix
    M(w) = sum_e w_e E_e ,   w_e>=0,
with M[t,t]+=w_e a_e, M[s,s]+=w_e c_s, M[s,t]=M[t,s]+= -w_e b_e/2.

DUALITY: if there is w>=0 (not all 0) with nu^T M(w) nu <= 0 for ALL nu>=0 (M(w) 'copositive-negative'),
then no nu>0 can make every E_e>0 -> PB>=1.  The open claim: Chern!=0 forces such a certificate to exist.
SPECTRAL question probed here: does the INERTIA (# negative eigenvalues) of M(w) -- at the minimax-optimal
dual weights -- track the Chern sector?  And can we drive M(w) copositive-negative only when Chern!=0?

Computes, per harvested line in several Chern sectors:
 (1) at minimax-optimal nu*, softmin dual weights w*, form M(w*): inertia (n_neg,n_zero,n_pos),
     nu*^T M nu* (should be <=0), and Rayleigh of nu* (is it the bottom mode?);
 (2) a copositivity probe: max over nu>=0 of nu^T M(w*) nu (projected power iteration) -- <=0 means cert;
 (3) correlation of inertia / copositivity with Chern across sectors {0,-2,-4} (k=2).
Reuses expU/expAE/expAF/expAG. Writes spectral.json.
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


def build_M(w, a, b, c, src, dst, N):
    cs = c[src]
    M = np.zeros((N, N))
    np.add.at(M, (dst, dst), w*a)
    np.add.at(M, (src, src), w*cs)
    np.add.at(M, (src, dst), -0.5*w*b)
    np.add.at(M, (dst, src), -0.5*w*b)
    return M


def minimax_nu(u, src, dst, phi, m, steps=4000, restarts=4):
    rep, req, om = expAF.edge_data(u, src, dst, phi)
    cap = np.sqrt(2.0/(1.0+np.abs(om))); best=-np.inf; besty=None
    for sd in range(restarts):
        rng=np.random.default_rng(321+sd); y=np.log(np.clip(rng.uniform(.3,1,u.shape[0])*cap,1e-3,cap))
        mt=np.zeros_like(y); vt=np.zeros_like(y)
        for t in range(1,steps+1):
            nu=np.exp(y); g_nu,F,br=expAF.grad_mag(nu,src,dst,rep,req,om,30+600*t/steps); g=g_nu*nu
            mt=.9*mt+.1*g; vt=.999*vt+.001*g*g; y=y+0.03*mt/(1-.9**t)/(np.sqrt(vt/(1-.999**t))+1e-8)
            y=np.minimum(y,np.log(cap))
        brn=expAF.brackets(np.exp(y),src,dst,rep,req,om)[0]
        if brn.min()>best: best=brn.min(); besty=y.copy()
    return np.exp(besty), best, (rep,req,om)


def copositive_max(M, restarts=20, iters=400):
    """approx max over nu>=0, ||nu||=1 of nu^T M nu (projected gradient power iteration). <=0 => certificate."""
    N=M.shape[0]; best=-np.inf
    for sd in range(restarts):
        rng=np.random.default_rng(sd); v=np.abs(rng.standard_normal(N)); v/=np.linalg.norm(v)
        for _ in range(iters):
            g=M@v
            v=v+0.1*g; v=np.maximum(v,0); n=np.linalg.norm(v)
            if n<1e-12: break
            v/=n
        best=max(best, float(v@M@v))
    return best


def analyze(u, src, dst, phi, m, label):
    N=u.shape[0]
    a,b,c,Rep,p,q,om=expAG.line_coeffs(u,src,dst,phi)
    nu,minbr,_=minimax_nu(u,src,dst,phi,m)
    rep,req,omf=expAF.edge_data(u,src,dst,phi)
    br=expAF.brackets(nu,src,dst,rep,req,omf)[0]
    # softmin dual weights concentrate on binding (smallest-bracket) edges
    beta=400.0; wv=np.exp(-beta*(br-br.min())); wv/=wv.sum()
    M=build_M(wv,a,b,c,src,dst,N)
    ev=np.linalg.eigvalsh(M)
    n_neg=int(np.sum(ev<-1e-9)); n_zero=int(np.sum(np.abs(ev)<=1e-9)); n_pos=int(np.sum(ev>1e-9))
    quad=float(nu@M@nu)
    rayl=quad/float(nu@nu)
    copo=copositive_max(M)
    chern=expAE.chern_of_u(m,u)
    return dict(label=label, chern=(round(chern,2) if chern is not None else None),
                minimax_bracket=round(float(minbr),5),
                inertia=[n_neg,n_zero,n_pos], nuMnu=round(quad,6), rayleigh=round(rayl,6),
                bottom_eig=round(float(ev[0]),6), copositive_max=round(float(copo),6),
                cert=(copo<=1e-6))


def main():
    out={}; m=8
    print(f"Entry 51: spectral/duality probe, m={m}.  Inertia of M(w*) and copositivity vs Chern sector.\n")
    # harvest lines across sectors for k=2: target sectors 0, -2, -4 (and +2)
    src,dst=expAE.edge_idx(m,2); phi=np.array([e[2] for e in expU.edges(m,2)])
    buckets={}
    for sd in range(80):
        rng=np.random.default_rng(9000+sd)
        u0=expAE.project_unit(rng.standard_normal((m*m,3))+1j*rng.standard_normal((m*m,3)))
        _,uf=expAE.ascend(u0,src,dst,m,steps=900)
        ch=expAE.chern_of_u(m,uf)
        if ch is None: continue
        key=round(ch)
        buckets.setdefault(key,[])
        if len(buckets[key])<2: buckets[key].append(uf)
    print(f"sectors harvested: { {k:len(v) for k,v in sorted(buckets.items())} }\n")
    rows=[]
    for key in sorted(buckets):
        for j,u in enumerate(buckets[key]):
            r=analyze(u,src,dst,phi,m,label=f"chern{key}_{j}")
            rows.append(r)
            print(f"  Chern={r['chern']:+.1f}: minimax_br={r['minimax_bracket']:+.5f} inertia(n-,0,+)={r['inertia']} "
                  f"nu^T M nu={r['nuMnu']:+.5f} bottom_eig={r['bottom_eig']:+.5f} copos_max={r['copositive_max']:+.5f} "
                  f"cert={r['cert']}")
    out[f"m{m}"]=rows
    # does anything separate Chern=0 from Chern!=0?
    z=[r for r in rows if r['chern'] is not None and abs(r['chern'])<0.4]
    nz=[r for r in rows if r['chern'] is not None and abs(r['chern'])>=1.6]
    print("\nLEARNINGS:")
    if z and nz:
        print(f"  Chern=0  : minimax_br median={np.median([r['minimax_bracket'] for r in z]):+.5f}  "
              f"inertia n_neg={[r['inertia'][0] for r in z]}  copos_max={[r['copositive_max'] for r in z]}")
        print(f"  Chern!=0 : minimax_br median={np.median([r['minimax_bracket'] for r in nz]):+.5f}  "
              f"inertia n_neg={[r['inertia'][0] for r in nz]}  copos_max={[r['copositive_max'] for r in nz]}")
    print("  (looking for: a spectral quantity of M(w*) that is <=0 / jumps exactly when Chern!=0.)")
    with open(os.path.join(HERE,"spectral.json"),"w") as f: json.dump(out,f,indent=2)
    print("wrote spectral.json")


if __name__ == "__main__":
    main()
