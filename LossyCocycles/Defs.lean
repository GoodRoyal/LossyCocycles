import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import LossyCocycles.Basic

/-! # LossyCocycles.Defs — L4: real norms, the torus flux family, distance-to-flat

DEBT-2 payoff. All definitions are elementary (no fragile norm-instance names):
`opNorm` is the supremum of image norms over the unit sphere, `frobNorm` the
root-sum-of-squares. The torus, the Landau-gauge flux family (boundary twist included),
flatness (Lemma-1 family), and the operator distance-to-flat are all explicit. After
this file, the Plateau Conjecture is a sorried *theorem about defined objects*, not a
Prop over placeholders.
-/

open Matrix

namespace LossyCocycles

/-- Operator norm, elementarily: sup of image ℓ²-norms over the unit sphere. -/
noncomputable def opNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  sSup {c | ∃ v : Fin n → ℝ, (∑ i, v i ^ 2) = 1 ∧ c = Real.sqrt (∑ i, (A.mulVec v i) ^ 2)}

/-- Frobenius norm, elementarily. -/
noncomputable def frobNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  Real.sqrt (∑ i, ∑ j, A i j ^ 2)

/-- Torus edges: a site (x,y) and a direction (`true` = step in x, `false` = step in y). -/
abbrev TorusEdge (m : ℕ) := (Fin m × Fin m) × Bool

variable {m : ℕ} [NeZero m]

def src (e : TorusEdge m) : Fin m × Fin m := e.1

def dst (e : TorusEdge m) : Fin m × Fin m :=
  if e.2 then (e.1.1 + 1, e.1.2) else (e.1.1, e.1.2 + 1)

/-- A family of edge maps over the torus. -/
abbrev EdgeFamily (m d : ℕ) := TorusEdge m → Matrix (Fin d) (Fin d) ℝ

/-- The Lemma-1 flat family: edge maps `V_dst · V_srcᵀ` for orthonormal frames. -/
def IsFlatFamily {d : ℕ} (r : ℕ) (T : EdgeFamily m d) : Prop :=
  ∃ V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ,
    (∀ v, IsFrame (V v)) ∧ ∀ e, T e = V (dst e) * (V (src e))ᵀ

/-- Operator-norm distance from an edge family to the rank-r flat family. -/
noncomputable def distOpFlat {d : ℕ} (r : ℕ) (T : EdgeFamily m d) : ℝ :=
  sInf {c | ∃ T' : EdgeFamily m d, IsFlatFamily r T' ∧ ∀ e, opNorm (T e - T' e) ≤ c}

/-- Landau-gauge angle: vertical edges at column x carry `2πk·x/m²`; horizontal wrap
edges at `x = m−1` carry the boundary twist `−2πk·y/m`. Total flux = 2πk. -/
noncomputable def fluxAngle (k : ℤ) (e : TorusEdge m) : ℝ :=
  if e.2 then
    (if (e.1.1 : ℕ) = m - 1 then -(2 * Real.pi * k * (e.1.2 : ℕ)) / m else 0)
  else
    2 * Real.pi * k * (e.1.1 : ℕ) / (m ^ 2 : ℕ)

/-- Rotation by θ in the (0,1)-plane of ℝʳ, identity elsewhere. -/
noncomputable def rotCore (r : ℕ) (θ : ℝ) : Matrix (Fin r) (Fin r) ℝ :=
  fun i j =>
    if (i : ℕ) = 0 ∧ (j : ℕ) = 0 then Real.cos θ
    else if (i : ℕ) = 0 ∧ (j : ℕ) = 1 then -Real.sin θ
    else if (i : ℕ) = 1 ∧ (j : ℕ) = 0 then Real.sin θ
    else if (i : ℕ) = 1 ∧ (j : ℕ) = 1 then Real.cos θ
    else if i = j then 1 else 0

/-- The standard d×r frame (first r coordinates). -/
def stdFrame (d r : ℕ) : Matrix (Fin d) (Fin r) ℝ :=
  fun i j => if (i : ℕ) = (j : ℕ) then 1 else 0

/-- The torus flux family: charge-k Landau connection in the surviving r-subspace,
carried by the standard frame at every site. -/
noncomputable def fluxFamily (k : ℤ) (d r : ℕ) : EdgeFamily m d :=
  fun e => stdFrame d r * rotCore r (fluxAngle k e) * (stdFrame d r)ᵀ

end LossyCocycles
