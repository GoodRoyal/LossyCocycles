import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import LossyCocycles.Defs

/-! # LossyCocycles.Winding — L5a/L5b: the core winding of an edge family

The CT-4 lemma target (`ct4-skeleton.md`): winding is locally constant under a spectral
gap, so flattening a charged family costs O(1) in operator norm. This file defines the
winding (L5a) and proves the flat anchor (DEBT-3b): flat families with a core gap have
winding zero.

Refactor note (2026-06-11): `vertCycle` is now a recursion (`vertProd`), not a
`List.prod` — same composite, one-step inductions. The Fin-m wraparound (`(m : Fin m)
= 0`) closes the cycle.

Remaining debts:
* DEBT-3a `windingSum_flux`: the charge-k flux family has winding k (for m > 2|k|).
* L5b: local constancy of `windingSum` under the gap + gap-collapse cost ⟹ Plateau.
-/

open Matrix

namespace LossyCocycles

variable {m d : ℕ} [NeZero m]

/-- The 2×2 core block of a d×d matrix (first two coordinates), for d ≥ 2. -/
def coreBlock (hd : 1 < d) (H : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i j => H ⟨i, lt_of_le_of_lt (Nat.le_of_lt_succ i.isLt) hd⟩
               ⟨j, lt_of_le_of_lt (Nat.le_of_lt_succ j.isLt) hd⟩

/-- The core vector of a 2×2 matrix: (trace, skew). Nonzero ⟺ the rotation angle of the
polar factor is well-defined. -/
def coreVec (A : Matrix (Fin 2) (Fin 2) ℝ) : ℝ × ℝ :=
  (A 0 0 + A 1 1, A 1 0 - A 0 1)

/-- The rotation-part angle of a 2×2 matrix: `arg(trace + i·skew)`. -/
noncomputable def rotAngle (A : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  Complex.arg ⟨(coreVec A).1, (coreVec A).2⟩

/-- ℕ → Fin m by explicit reduction mod m (independent of any NatCast instance). -/
def natFin (n : ℕ) : Fin m :=
  ⟨n % m, Nat.mod_lt _ (Nat.pos_of_ne_zero (NeZero.ne m))⟩

theorem natFin_zero : (natFin 0 : Fin m) = 0 := by
  apply Fin.ext; simp [natFin]

theorem natFin_succ (n : ℕ) : (natFin n : Fin m) + 1 = natFin (n + 1) := by
  apply Fin.ext
  rw [Fin.val_add]
  have h1 : ((1 : Fin m) : ℕ) = 1 % m := rfl
  have h2 : ((natFin n : Fin m) : ℕ) = n % m := rfl
  rw [h1, h2]
  exact (Nat.add_mod n 1 m).symm

theorem natFin_self : (natFin m : Fin m) = 0 := by
  apply Fin.ext; simp [natFin]

/-- Partial vertical product at column x: the first n vertical steps (later steps on
the left). -/
def vertProd (T : EdgeFamily m d) (x : Fin m) : ℕ → Matrix (Fin d) (Fin d) ℝ
  | 0 => 1
  | n + 1 => T ((x, (natFin n : Fin m)), false) * vertProd T x n

/-- The full vertical cycle composite at column x. -/
def vertCycle (T : EdgeFamily m d) (x : Fin m) : Matrix (Fin d) (Fin d) ℝ :=
  vertProd T x m

/-- The core angle of the vertical cycle at column x. -/
noncomputable def cycleAngle (hd : 1 < d) (T : EdgeFamily m d) (x : Fin m) : ℝ :=
  rotAngle (coreBlock hd (vertCycle T x))

/-- Principal-branch difference: the representative of b − a in (−π, π]. -/
noncomputable def angleDiff (a b : ℝ) : ℝ :=
  b - a - 2 * Real.pi * round ((b - a) / (2 * Real.pi))

/-- The gap hypothesis: every vertical-cycle core vector has length ≥ γ. -/
def CoreGap (γ : ℝ) (hd : 1 < d) (T : EdgeFamily m d) : Prop :=
  ∀ x : Fin m, γ ≤ Real.sqrt ((coreVec (coreBlock hd (vertCycle T x))).1 ^ 2 +
                              (coreVec (coreBlock hd (vertCycle T x))).2 ^ 2)

/-- L5a — the winding of an edge family: principal-branch-summed angle change of the
vertical-cycle core angle around the torus, in units of 2π. -/
noncomputable def windingSum (hd : 1 < d) (T : EdgeFamily m d) : ℝ :=
  (∑ x : Fin m, angleDiff (cycleAngle hd T x) (cycleAngle hd T (x + 1))) / (2 * Real.pi)

/-! ## DEBT-3b: flat families with a core gap have winding zero -/

/-- Telescoping: for a flat family, the partial vertical product reaches
`V(x, n) · V(x, 0)ᵀ`. -/
theorem flat_vertProd {r : ℕ} (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hT : ∀ e, T e = V (dst e) * (V (src e))ᵀ)
    (x : Fin m) : ∀ n : ℕ, 1 ≤ n →
    vertProd T x n = V (x, (natFin n : Fin m)) * (V (x, (0 : Fin m)))ᵀ := by
  intro n hn
  induction n with
  | zero => omega
  | succ k ih =>
      rcases Nat.eq_or_lt_of_le hn with h1 | h2
      · -- k + 1 = 1 : base case
        have hk : k = 0 := by omega
        subst hk
        show T ((x, (natFin 0 : Fin m)), false) * 1 = _
        rw [Matrix.mul_one, hT]
        simp only [src, dst, Bool.false_eq_true, if_false]
        rw [natFin_succ, natFin_zero]
      · -- k ≥ 1 : telescope one step
        have hk : 1 ≤ k := by omega
        show T ((x, (natFin k : Fin m)), false) * vertProd T x k = _
        rw [ih hk, hT]
        simp only [src, dst, Bool.false_eq_true, if_false]
        rw [collapse _ _ _ (hfr (x, (natFin k : Fin m))), natFin_succ]

/-- A flat family's vertical cycle is the projection `V(x,0) · V(x,0)ᵀ`. -/
theorem flat_vertCycle {r : ℕ} (hm : 1 ≤ m) (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hT : ∀ e, T e = V (dst e) * (V (src e))ᵀ)
    (x : Fin m) : vertCycle T x = V (x, 0) * (V (x, 0))ᵀ := by
  have h := flat_vertProd T V hfr hT x m hm
  rw [natFin_self] at h
  exact h

/-- The core vector of a projection `V·Vᵀ` is (nonnegative trace, zero skew). -/
theorem coreVec_proj {r : ℕ} (hd : 1 < d) (V : Matrix (Fin d) (Fin r) ℝ) :
    (coreVec (coreBlock hd (V * Vᵀ))).2 = 0 ∧
    0 ≤ (coreVec (coreBlock hd (V * Vᵀ))).1 := by
  constructor
  · -- skew = 0: the (1,0) and (0,1) entries agree
    simp only [coreVec, coreBlock, Matrix.mul_apply, Matrix.transpose_apply]
    rw [sub_eq_zero]
    exact Finset.sum_congr rfl fun k _ => mul_comm _ _
  · -- trace ≥ 0: diagonal entries are sums of squares
    simp only [coreVec, coreBlock, Matrix.mul_apply, Matrix.transpose_apply]
    have h0 : (0:ℝ) ≤ ∑ k, V ⟨0, by omega⟩ k * V ⟨0, by omega⟩ k :=
      Finset.sum_nonneg fun k _ => mul_self_nonneg _
    have h1 : (0:ℝ) ≤ ∑ k, V ⟨1, hd⟩ k * V ⟨1, hd⟩ k :=
      Finset.sum_nonneg fun k _ => mul_self_nonneg _
    exact add_nonneg h0 h1

/-- DEBT-3b — a flat family with a core gap has winding 0. -/
theorem windingSum_flat {γ : ℝ} (hγ : 0 < γ) (hd : 1 < d) (r : ℕ)
    (T : EdgeFamily m d) (hT : IsFlatFamily r T) (hgap : CoreGap γ hd T) :
    windingSum hd T = 0 := by
  obtain ⟨V, hfr, hTe⟩ := hT
  have hm : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr (NeZero.ne m)
  -- every cycle angle is zero
  have hang : ∀ x : Fin m, cycleAngle hd T x = 0 := by
    intro x
    have hproj := flat_vertCycle hm T V hfr hTe x
    obtain ⟨hskew, htr⟩ := coreVec_proj hd (V (x, 0))
    -- the gap forces the trace strictly positive
    have hlen := hgap x
    rw [hproj, hskew] at hlen
    norm_num [Real.sqrt_sq_eq_abs] at hlen
    have hpos : 0 < (coreVec (coreBlock hd (V (x, 0) * (V (x, 0))ᵀ))).1 := by
      have habs : |(coreVec (coreBlock hd (V (x, 0) * (V (x, 0))ᵀ))).1|
          = (coreVec (coreBlock hd (V (x, 0) * (V (x, 0))ᵀ))).1 := abs_of_nonneg htr
      rw [habs] at hlen
      linarith
    -- arg of (positive, 0) is zero
    unfold cycleAngle rotAngle
    rw [hproj, hskew, Complex.arg_eq_zero_iff]
    exact ⟨le_of_lt hpos, rfl⟩
  -- all increments vanish, hence the sum
  unfold windingSum
  have : ∀ x : Fin m, angleDiff (cycleAngle hd T x) (cycleAngle hd T (x + 1)) = 0 := by
    intro x
    rw [hang x, hang (x + 1)]
    unfold angleDiff
    norm_num
  rw [Finset.sum_congr rfl fun x _ => this x]
  simp

/-! ## Flux structure (toward DEBT-3a and Stage A): rotation-core algebra,
standard-frame collapse, and the flux column composite -/

theorem rotCore_row_two {r : ℕ} (θ : ℝ) {i : Fin r} (hi : 2 ≤ (i : ℕ)) (x : Fin r) :
    rotCore r θ i x = if i = x then 1 else 0 := by
  have h0 : ¬((i : ℕ) = 0) := by omega
  have h1 : ¬((i : ℕ) = 1) := by omega
  simp [rotCore, h0, h1]

theorem rotCore_row_zero {r : ℕ} (θ : ℝ) {i : Fin r} (hi : (i : ℕ) = 0) (x : Fin r) :
    rotCore r θ i x =
      if (x : ℕ) = 0 then Real.cos θ else if (x : ℕ) = 1 then -Real.sin θ else 0 := by
  by_cases hx0 : (x : ℕ) = 0
  · simp [rotCore, hi, hx0]
  · by_cases hx1 : (x : ℕ) = 1
    · simp [rotCore, hi, hx0, hx1]
    · have hne : ¬(i = x) := by rw [Fin.ext_iff]; omega
      simp [rotCore, hi, hx0, hx1, hne]

theorem rotCore_row_one {r : ℕ} (θ : ℝ) {i : Fin r} (hi : (i : ℕ) = 1) (x : Fin r) :
    rotCore r θ i x =
      if (x : ℕ) = 0 then Real.sin θ else if (x : ℕ) = 1 then Real.cos θ else 0 := by
  by_cases hx0 : (x : ℕ) = 0
  · simp [rotCore, hi, hx0]
  · by_cases hx1 : (x : ℕ) = 1
    · simp [rotCore, hi, hx0, hx1]
    · have hne : ¬(i = x) := by rw [Fin.ext_iff]; omega
      simp [rotCore, hi, hx0, hx1, hne]

theorem rotCore_zero {r : ℕ} : rotCore r (0 : ℝ) = 1 := by
  ext i j
  rw [Matrix.one_apply]
  by_cases h0 : (i : ℕ) = 0
  · rw [rotCore_row_zero 0 h0 j]
    by_cases hj0 : (j : ℕ) = 0
    · have hij : i = j := Fin.ext (by omega)
      simp [hj0, hij]
    · by_cases hj1 : (j : ℕ) = 1
      · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
        simp [hj0, hj1, hij]
      · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
        simp [hj0, hj1, hij]
  · by_cases h1 : (i : ℕ) = 1
    · rw [rotCore_row_one 0 h1 j]
      by_cases hj0 : (j : ℕ) = 0
      · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
        simp [hj0, hij]
      · by_cases hj1 : (j : ℕ) = 1
        · have hij : i = j := Fin.ext (by omega)
          simp [hj0, hj1, hij]
        · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
          simp [hj0, hj1, hij]
    · have h2 : 2 ≤ (i : ℕ) := by omega
      rw [rotCore_row_two 0 h2 j]

/-- Two-slot sum collapse: a function supported on indices 0 and 1 sums to the two
slot values. -/
theorem sum_two_slot {r : ℕ} (h0r : 0 < r) (hr : 1 < r) (C₀ C₁ : ℝ) :
    (∑ x : Fin r, ((if (x : ℕ) = 0 then C₀ else 0) + (if (x : ℕ) = 1 then C₁ else 0)))
      = C₀ + C₁ := by
  rw [Finset.sum_add_distrib]
  congr 1
  · rw [Finset.sum_eq_single (⟨0, h0r⟩ : Fin r)]
    · simp
    · intro b _ hb
      have hb0 : ¬((b : ℕ) = 0) := fun h => hb (Fin.ext h)
      simp [hb0]
    · intro h; exact absurd (Finset.mem_univ _) h
  · rw [Finset.sum_eq_single (⟨1, hr⟩ : Fin r)]
    · simp
    · intro b _ hb
      have hb1 : ¬((b : ℕ) = 1) := fun h => hb (Fin.ext h)
      simp [hb1]
    · intro h; exact absurd (Finset.mem_univ _) h

/-- Rotation cores compose additively (needs r ≥ 2: for r = 1 the core is truncated
and the identity is false). -/
theorem rotCore_mul {r : ℕ} (hr : 1 < r) (α β : ℝ) :
    rotCore r α * rotCore r β = rotCore r (α + β) := by
  have h0r : 0 < r := by omega
  ext i j
  rw [Matrix.mul_apply]
  by_cases hi0 : (i : ℕ) = 0
  · have hterm : ∀ x : Fin r, rotCore r α i x * rotCore r β x j
        = (if (x : ℕ) = 0 then Real.cos α * rotCore r β ⟨0, h0r⟩ j else 0)
          + (if (x : ℕ) = 1 then -Real.sin α * rotCore r β ⟨1, hr⟩ j else 0) := by
      intro x
      rw [rotCore_row_zero α hi0 x]
      by_cases hx0 : (x : ℕ) = 0
      · have hx : x = (⟨0, h0r⟩ : Fin r) := Fin.ext hx0
        subst hx
        simp
      · by_cases hx1 : (x : ℕ) = 1
        · have hx : x = (⟨1, hr⟩ : Fin r) := Fin.ext hx1
          subst hx
          simp
        · simp [hx0, hx1]
    rw [Finset.sum_congr rfl fun x _ => hterm x, sum_two_slot h0r hr]
    by_cases hj0 : (j : ℕ) = 0
    · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
        rotCore_row_zero (α + β) hi0 j]
      simp [hj0, Real.cos_add] <;> ring
    · by_cases hj1 : (j : ℕ) = 1
      · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
          rotCore_row_zero (α + β) hi0 j]
        simp [hj0, hj1, Real.sin_add] <;> ring
      · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
          rotCore_row_zero (α + β) hi0 j]
        simp [hj0, hj1]
  · by_cases hi1 : (i : ℕ) = 1
    · have hterm : ∀ x : Fin r, rotCore r α i x * rotCore r β x j
          = (if (x : ℕ) = 0 then Real.sin α * rotCore r β ⟨0, h0r⟩ j else 0)
            + (if (x : ℕ) = 1 then Real.cos α * rotCore r β ⟨1, hr⟩ j else 0) := by
        intro x
        rw [rotCore_row_one α hi1 x]
        by_cases hx0 : (x : ℕ) = 0
        · have hx : x = (⟨0, h0r⟩ : Fin r) := Fin.ext hx0
          subst hx
          simp
        · by_cases hx1 : (x : ℕ) = 1
          · have hx : x = (⟨1, hr⟩ : Fin r) := Fin.ext hx1
            subst hx
            simp
          · simp [hx0, hx1]
      rw [Finset.sum_congr rfl fun x _ => hterm x, sum_two_slot h0r hr]
      by_cases hj0 : (j : ℕ) = 0
      · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
          rotCore_row_one (α + β) hi1 j]
        simp [hj0, Real.sin_add] <;> ring
      · by_cases hj1 : (j : ℕ) = 1
        · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
            rotCore_row_one (α + β) hi1 j]
          simp [hj0, hj1, Real.cos_add] <;> ring
        · rw [rotCore_row_zero β rfl j, rotCore_row_one β rfl j,
            rotCore_row_one (α + β) hi1 j]
          simp [hj0, hj1]
    · have hi2 : 2 ≤ (i : ℕ) := by omega
      have hterm : ∀ x : Fin r, rotCore r α i x * rotCore r β x j
          = if i = x then rotCore r β x j else 0 := by
        intro x
        rw [rotCore_row_two α hi2 x]
        by_cases hx : i = x
        · simp [hx]
        · simp [hx]
      rw [Finset.sum_congr rfl fun x _ => hterm x, Finset.sum_ite_eq]
      simp only [Finset.mem_univ, if_pos]
      rw [rotCore_row_two β hi2 j, rotCore_row_two (α + β) hi2 j]

/-- The standard frame has orthonormal columns (needs r ≤ d). -/
theorem stdFrame_isFrame {d r : ℕ} (hrd : r ≤ d) : IsFrame (stdFrame d r) := by
  unfold IsFrame
  ext a b
  rw [Matrix.mul_apply, Matrix.one_apply]
  have hterm : ∀ i : Fin d, (stdFrame d r)ᵀ a i * stdFrame d r i b
      = if i = (⟨(a : ℕ), lt_of_lt_of_le a.isLt hrd⟩ : Fin d) then
          (if (a : ℕ) = (b : ℕ) then (1 : ℝ) else 0) else 0 := by
    intro i
    simp only [Matrix.transpose_apply, stdFrame]
    by_cases hia : (i : ℕ) = (a : ℕ)
    · have hi : i = (⟨(a : ℕ), lt_of_lt_of_le a.isLt hrd⟩ : Fin d) := Fin.ext hia
      subst hi
      simp
    · have hi : ¬(i = (⟨(a : ℕ), lt_of_lt_of_le a.isLt hrd⟩ : Fin d)) :=
        fun h => hia (by rw [h])
      simp [hia, hi]
  rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_ite_eq']
  simp only [Finset.mem_univ, if_pos]
  by_cases hab : a = b
  · subst hab; simp
  · have : ¬((a : ℕ) = (b : ℕ)) := fun h => hab (Fin.ext h)
    simp [this, hab]

/-- Collapse of a standard-frame conjugation onto its core entries. -/
theorem stdFrame_conj_apply {d r : ℕ} (hrd : r ≤ d) (A : Matrix (Fin r) (Fin r) ℝ)
    (i j : Fin d) (hi : (i : ℕ) < r) (hj : (j : ℕ) < r) :
    (stdFrame d r * A * (stdFrame d r)ᵀ) i j = A ⟨(i : ℕ), hi⟩ ⟨(j : ℕ), hj⟩ := by
  rw [Matrix.mul_apply]
  have houter : ∀ b : Fin r, (stdFrame d r * A) i b * (stdFrame d r)ᵀ b j
      = if b = (⟨(j : ℕ), hj⟩ : Fin r) then (stdFrame d r * A) i b else 0 := by
    intro b
    simp only [Matrix.transpose_apply, stdFrame]
    by_cases hjb : (j : ℕ) = (b : ℕ)
    · have hb : b = (⟨(j : ℕ), hj⟩ : Fin r) := Fin.ext hjb.symm
      subst hb
      simp
    · have hb : ¬(b = (⟨(j : ℕ), hj⟩ : Fin r)) := fun h => hjb (by rw [h])
      simp [hjb, hb]
  rw [Finset.sum_congr rfl fun b _ => houter b, Finset.sum_ite_eq']
  simp only [Finset.mem_univ, if_pos]
  rw [Matrix.mul_apply]
  have hinner : ∀ a : Fin r, stdFrame d r i a * A a ⟨(j : ℕ), hj⟩
      = if a = (⟨(i : ℕ), hi⟩ : Fin r) then A a ⟨(j : ℕ), hj⟩ else 0 := by
    intro a
    simp only [stdFrame]
    by_cases hia : (i : ℕ) = (a : ℕ)
    · have ha : a = (⟨(i : ℕ), hi⟩ : Fin r) := Fin.ext hia.symm
      subst ha
      simp
    · have ha : ¬(a = (⟨(i : ℕ), hi⟩ : Fin r)) := fun h => hia (by rw [h])
      simp [hia, ha]
  rw [Finset.sum_congr rfl fun a _ => hinner a, Finset.sum_ite_eq']
  simp

/-- Conjugated products collapse: `(W A Wᵀ)(W B Wᵀ) = W (AB) Wᵀ`. -/
theorem stdFrame_conj_mul {d r : ℕ} (hrd : r ≤ d) (A B : Matrix (Fin r) (Fin r) ℝ) :
    (stdFrame d r * A * (stdFrame d r)ᵀ) * (stdFrame d r * B * (stdFrame d r)ᵀ)
      = stdFrame d r * (A * B) * (stdFrame d r)ᵀ := by
  have h : (stdFrame d r)ᵀ * stdFrame d r = 1 := stdFrame_isFrame hrd
  calc (stdFrame d r * A * (stdFrame d r)ᵀ) * (stdFrame d r * B * (stdFrame d r)ᵀ)
      = stdFrame d r * A * (((stdFrame d r)ᵀ * stdFrame d r) * (B * (stdFrame d r)ᵀ)) := by
        rw [Matrix.mul_assoc (stdFrame d r * A), Matrix.mul_assoc (stdFrame d r)ᵀ,
          Matrix.mul_assoc (stdFrame d r) B]
    _ = stdFrame d r * (A * B) * (stdFrame d r)ᵀ := by
        rw [h, Matrix.one_mul, ← Matrix.mul_assoc, ← Matrix.mul_assoc]

/-- The vertical flux edge at column x carries the Landau angle `2πk·x/m²`,
independent of the row. -/
theorem fluxFamily_vert (k : ℤ) (d r : ℕ) (x y : Fin m) :
    fluxFamily (m := m) k d r ((x, y), false)
      = stdFrame d r
        * rotCore r (2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ))
        * (stdFrame d r)ᵀ := by
  simp [fluxFamily, fluxAngle]

/-- The partial vertical product of the flux family is the conjugated rotation by
n steps of the column angle. -/
theorem flux_vertProd (k : ℤ) (d r : ℕ) (hr : 1 < r) (hrd : r ≤ d) (x : Fin m) :
    ∀ n : ℕ, 1 ≤ n →
      vertProd (fluxFamily (m := m) k d r) x n
        = stdFrame d r
          * rotCore r (n * (2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ)))
          * (stdFrame d r)ᵀ := by
  intro n hn
  induction n with
  | zero => omega
  | succ p ih =>
      rcases Nat.eq_or_lt_of_le hn with h1 | h2
      · -- p + 1 = 1
        have hp : p = 0 := by omega
        subst hp
        show fluxFamily (m := m) k d r ((x, (natFin 0 : Fin m)), false) * 1 = _
        rw [Matrix.mul_one, fluxFamily_vert]
        norm_num
      · -- p ≥ 1
        have hp : 1 ≤ p := by omega
        show fluxFamily (m := m) k d r ((x, (natFin p : Fin m)), false)
            * vertProd (fluxFamily (m := m) k d r) x p = _
        rw [ih hp, fluxFamily_vert, stdFrame_conj_mul hrd, rotCore_mul hr]
        have hang : 2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ)
            + (p : ℝ) * (2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ))
            = ((p + 1 : ℕ) : ℝ) * (2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ)) := by
          push_cast
          ring
        rw [hang]

/-- The full vertical cycle of the flux family at column x is the conjugated
rotation by `2πk·x/m`. -/
theorem flux_vertCycle (k : ℤ) (d r : ℕ) (hr : 1 < r) (hrd : r ≤ d) (x : Fin m) :
    vertCycle (fluxFamily (m := m) k d r) x
      = stdFrame d r * rotCore r (2 * Real.pi * k * (x : ℕ) / m) * (stdFrame d r)ᵀ := by
  have hm : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr (NeZero.ne m)
  have h := flux_vertProd k d r hr hrd x m hm
  have hmR : ((m : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne m)
  unfold vertCycle
  have hang : ((m : ℕ) : ℝ) * (2 * Real.pi * k * (x : ℕ) / ((m : ℕ) ^ 2 : ℕ))
      = 2 * Real.pi * k * (x : ℕ) / m := by
    push_cast
    field_simp
  rw [h, hang]

/-- The core block of a conjugated r×r matrix is the matrix's own top block
(needs r ≥ 2 so the two core indices land inside Fin r). -/
theorem coreBlock_conj {d r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (A : Matrix (Fin r) (Fin r) ℝ) (i j : Fin 2) :
    coreBlock hd (stdFrame d r * A * (stdFrame d r)ᵀ) i j
      = A ⟨(i : ℕ), lt_of_lt_of_le i.isLt hr⟩ ⟨(j : ℕ), lt_of_lt_of_le j.isLt hr⟩ := by
  unfold coreBlock
  exact stdFrame_conj_apply hrd A _ _
    (lt_of_lt_of_le i.isLt hr) (lt_of_lt_of_le j.isLt hr)

/-- The core vector of a conjugated rotation reads off the angle. -/
theorem coreVec_conj_rot {d r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (θ : ℝ) :
    coreVec (coreBlock hd (stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ))
      = (2 * Real.cos θ, 2 * Real.sin θ) := by
  unfold coreVec
  rw [coreBlock_conj hd hr hrd _ 0 0, coreBlock_conj hd hr hrd _ 1 1,
    coreBlock_conj hd hr hrd _ 1 0, coreBlock_conj hd hr hrd _ 0 1]
  rw [rotCore_row_zero θ rfl, rotCore_row_one θ rfl, rotCore_row_one θ rfl,
    rotCore_row_zero θ rfl]
  norm_num
  constructor <;> ring

/-- The cycle angle of the flux family at column x is `2πk·x/m`, up to the
principal-branch shift — stated with the explicit integer. -/
theorem cycleAngle_flux (k : ℤ) (d r : ℕ) (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (x : Fin m) :
    ∃ n : ℤ, cycleAngle hd (fluxFamily (m := m) k d r) x
      = 2 * Real.pi * k * (x : ℕ) / m + 2 * Real.pi * n := by
  set θ : ℝ := 2 * Real.pi * k * (x : ℕ) / m with hθ
  refine ⟨⌊(Real.pi - θ) / (2 * Real.pi)⌋, ?_⟩
  unfold cycleAngle rotAngle
  rw [flux_vertCycle k d r hr hrd x, ← hθ, coreVec_conj_rot hd hr hrd θ]
  have hz : (⟨(2 * Real.cos θ, 2 * Real.sin θ).1, (2 * Real.cos θ, 2 * Real.sin θ).2⟩ : ℂ)
      = (2 : ℝ) * (Complex.cos θ + Complex.sin θ * Complex.I) := by
    rw [← Complex.ofReal_cos, ← Complex.ofReal_sin]
    apply Complex.ext <;>
      simp [Complex.cos_ofReal_re, Complex.sin_ofReal_re]
  rw [hz]
  have key := Complex.arg_mul_cos_add_sin_mul_I_sub (r := 2) two_pos θ
  linarith [key]

/-- Principal-branch evaluation: if `b − a` is `c` up to a full-turn integer and `c`
lies in `[−π, π)`, then `angleDiff a b = c`. -/
theorem angleDiff_eq_of_int (a b c : ℝ) (n : ℤ) (h : b - a = c + 2 * Real.pi * n)
    (hc1 : -Real.pi ≤ c) (hc2 : c < Real.pi) : angleDiff a b = c := by
  have h2π : (0 : ℝ) < 2 * Real.pi := Real.two_pi_pos
  unfold angleDiff
  rw [h]
  have hdiv : (c + 2 * Real.pi * n) / (2 * Real.pi) = c / (2 * Real.pi) + n := by
    field_simp
  rw [hdiv, round_add_intCast]
  have hround : round (c / (2 * Real.pi)) = 0 := by
    rw [round_eq_zero_iff, Set.mem_Ico]
    constructor
    · rw [show -(1 / 2 : ℝ) = -Real.pi / (2 * Real.pi) from by
        rw [eq_div_iff (ne_of_gt h2π)]; ring]
      exact div_le_div_of_nonneg_right hc1 h2π.le
    · rw [show (1 / 2 : ℝ) = Real.pi / (2 * Real.pi) from by
        rw [eq_div_iff (ne_of_gt h2π)]; ring]
      exact div_lt_div_of_pos_right hc2 h2π
  rw [hround]
  push_cast
  ring

/-- DEBT-3a — the charge-k flux family has winding k (valid for m > 2|k|, where
consecutive column angles differ by 2πk/m < π and the principal branch is exact). -/
theorem windingSum_flux (k : ℤ) (d r : ℕ) (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (hm : 2 * k.natAbs < m) :
    windingSum hd (fluxFamily (m := m) k d r) = k := by
  have hm0 : 0 < m := by omega
  have hmR : (0 : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hπ := Real.pi_pos
  set c : ℝ := 2 * Real.pi * k / m with hc
  -- |c| < π from 2|k| < m
  have hkm : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm
  have habs : |(k : ℝ)| = ((k.natAbs : ℕ) : ℝ) := by
    rw [Nat.cast_natAbs, Int.cast_abs]
  have hcabs : |c| < Real.pi := by
    rw [hc, abs_div, abs_mul, abs_mul, abs_two, abs_of_pos hπ, habs, abs_of_pos hmR]
    rw [div_lt_iff₀ hmR]
    nlinarith [hkm, hπ]
  have hc1 : -Real.pi ≤ c := le_of_lt (abs_lt.mp hcabs).1
  have hc2 : c < Real.pi := (abs_lt.mp hcabs).2
  -- every principal-branch increment is exactly c
  have hstep : ∀ x : Fin m,
      angleDiff (cycleAngle hd (fluxFamily (m := m) k d r) x)
        (cycleAngle hd (fluxFamily (m := m) k d r) (x + 1)) = c := by
    intro x
    obtain ⟨n₁, h₁⟩ := cycleAngle_flux k d r hd hr hrd x
    obtain ⟨n₂, h₂⟩ := cycleAngle_flux k d r hd hr hrd (x + 1)
    have hv : ((x + 1 : Fin m) : ℕ) = ((x : ℕ) + 1) % m := by
      rw [Fin.val_add, Fin.val_one']
      conv_lhs => rw [Nat.add_mod]
      conv_rhs => rw [Nat.add_mod]
      rw [Nat.mod_mod]
    by_cases hxm : (x : ℕ) + 1 < m
    · have hv' : ((x + 1 : Fin m) : ℕ) = (x : ℕ) + 1 := by
        rw [hv]; exact Nat.mod_eq_of_lt hxm
      apply angleDiff_eq_of_int _ _ c (n₂ - n₁) _ hc1 hc2
      rw [h₁, h₂, hv', hc]
      push_cast
      field_simp
      ring
    · have hx1 : (x : ℕ) + 1 = m := by have := x.isLt; omega
      have hv' : ((x + 1 : Fin m) : ℕ) = 0 := by
        rw [hv, hx1]; exact Nat.mod_self m
      apply angleDiff_eq_of_int _ _ c (n₂ - n₁ - k) _ hc1 hc2
      rw [h₁, h₂, hv', hc]
      have hxval : (((x : ℕ) : ℕ) : ℝ) = ((m : ℕ) : ℝ) - 1 := by
        have hxe : (x : ℕ) = m - 1 := by omega
        rw [hxe]
        have h1m : 1 ≤ m := hm0
        push_cast [Nat.cast_sub h1m]
        ring
      rw [hxval]
      push_cast
      field_simp
      ring
  unfold windingSum
  rw [Finset.sum_congr rfl fun x _ => hstep x, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, hc]
  have hπ0 : Real.pi ≠ 0 := ne_of_gt hπ
  have hm0' : ((m : ℕ) : ℝ) ≠ 0 := ne_of_gt hmR
  field_simp

end LossyCocycles
