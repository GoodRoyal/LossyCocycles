# LossyCocycles

A Lean 4 formalization of the **Plateau Conjecture** for lossy (rank-deficient)
lattice gauge fields on a discrete torus, and of its machine-checked reduction to a
single dimension-free rigidity estimate.

## The problem

A charge-`k` magnetic flux family on an `m × m` discrete torus assigns
parallel-transport contractions to edges (Landau gauge, rank `r ≤ d`). The Plateau
Conjecture asserts a size-uniform separation from flatness:

> For every `k ≠ 0` there is `c(k) > 0` such that for all `m ≥ 3` and all ranks
> `2 ≤ r ≤ d`, the charge-`k` flux family is at operator-norm distance `≥ c(k)`
> from every rank-`r` flat family.

## What is machine-checked

- **Collapse algebra** for flat families (`Basic`, `Defs`).
- **Falsification of a 1D invariant**: the column-winding sum is carried equally by
  the *shear family*, which is `O(1/m)`-close to flat — a machine-checked
  counterexample that killed the 1D route (`Winding`, `Shear`).
- **The corrected 2D invariant**: total flux distinguishes flux (`2πk`) from flat
  (`0`) and shear (`0`) (`PlaqIndex`).
- **Magnetic-translation pair**: commutator locality (`‖[𝒰,𝒱]‖` bounded by the max
  plaquette discrepancy, never amplified by `m`) and the almost-commuting estimate
  `‖[𝒰,𝒱]‖ ≤ 2|sin(πk/m²)|` (`MagPair`).
- **Elementary lower bounds**: Stage A/B norm kit, gap stability (`StageA`, `StageB`).
- **The reduction** `plateau_of_rigidity` (`PlateauConjecture`): the full conjecture,
  uniformly in `m` and with the constant preserved, follows from one scoped estimate.

## The one open estimate

The library contains exactly **one `sorry`**: `flux_pair_rigidity`, asserting the
Prop `FluxPairRigidity` — for each `k ≠ 0` a constant `δ(k) > 0`, independent of
`m`, `d`, `r`, separating the magnetic-translation flux pair from every commuting
contraction pair on the same space. This is a transported (non-unitary,
rank-deficient) form of Exel–Loring / Voiculescu rigidity: at full rank `r = d` the
pair's complexified core is the classical Voiculescu pair, where rigidity is known.
The trivial zero pair caps the constant at `δ(k) ≤ 1`; the open content is strict
positivity, uniform in rank deficiency.

Everything else compiles without axioms beyond mathlib.

## Building

Requires [elan](https://github.com/leanprover/elan). Then:

```
lake exe cache get   # fetch mathlib build cache
lake build
```

The build reports the single `sorry` warning in
`LossyCocycles/PlateauConjecture.lean` (the open estimate) and nothing else.

## Companion paper

*Formalized Mathematics as a Debugging Tool: Correcting a False Invariant in
Lattice Gauge Theory* (draft; link to be added on posting). The paper narrates the
two debugging events the formalization produced: the shear-family falsification of
the winding invariant, and the discovery during the reduction proof that a
distance-translation step previously believed to be half the open problem is not
load-bearing.

## License

Apache 2.0 (see `LICENSE`).
