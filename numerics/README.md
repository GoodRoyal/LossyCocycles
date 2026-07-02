# Numerics — ancillary scripts for the companion papers

Numerical evidence base for the papers accompanying this formalization, in
particular *Three gaps, one index* (the experimental-mathematics note) and the
empirical-status passages of the companion papers. Everything is plain
`numpy`/`scipy`, single-threaded, and mirrors the Lean definitions in
`../LossyCocycles/`; the core script cross-checks itself against the
machine-checked theorems (`‖[U,V]‖ = 2|sin(πk/m²)|`, `totalFlux = 2πk / 0`).

## Contents

- **`coarse-geometry-numerics.py`** — the seven core experiments **A–G** of
  *Three gaps, one index* (§3–§8 and Appendix A there): Bott index and branch gap
  (A), core gap / polar Lipschitz (B), distance-to-commuting (C), magnetic
  Hamiltonian spectral gap (D), compression leak (E), adversarial leak (F,
  `EXPF_ONLY=1` to run alone), direct `distOpFlat` descent (G, `EXPG_ONLY=1`).
- **`e4_pair_rigidity.py`** (+ recorded output `e4_pair_rigidity.json`) — the
  pre-registered pair-level adversarial probe of `FluxPairRigidity`
  (the library's one open estimate): penalty optimization toward the nearest
  commuting contraction pair with exact-commuting Schur certification. The
  recorded outcome: no competitor below the trivial zero-pair distance 1;
  the kill condition did not fire.
- **`grothendieck/`** — the follow-up experiments (each a standalone script
  loading the core module by relative path), with their recorded `*.json`
  outputs. Highlights, keyed to *Three gaps, one index*:
  - codimension analysis (§4.1): `expJ_codim.py`, `expK_codim_sweep.py`,
    `expL_orthogonal_competitor.py`, `expM_floor.py`; geodesic index-jump radius
    `expC_geodesic.py`; `distOpFlat` k-sweep `expG_ksweep.py`; compression Schur
    identity `expH_block_identity.py`; support slice `expI_support_slice.py`.
  - the (R1)–(R7) reduction chain (§4.2): `expU_shadow.py` … `expAU_blowup.py`
    (per-edge characterization, telescoping/Chern, factorization, empty-edge
    lemma, max-mean-cycle reformulation, Hermitian bound, sector dichotomy,
    shortcut closures).
  - the variance / branch-cut sharpening (§4.3): `expBA_ringspectrum.py` …
    `expBI_rigidity.py` (ring operator, sum rule, static bound, product
    characterization, the (R) statement's ingredients).

## Running

```
pip install numpy scipy        # matplotlib only for the two visualization scripts
python3 coarse-geometry-numerics.py
python3 grothendieck/expL_orthogonal_competitor.py   # etc.
```

Each experiment runs in seconds to minutes on one core and writes/updates its
`.json` next to itself. The committed `.json` files are the recorded runs the
papers quote; re-running reproduces them up to optimizer/RNG jitter in the
optimization-based experiments (the certified/identity checks reproduce exactly).
