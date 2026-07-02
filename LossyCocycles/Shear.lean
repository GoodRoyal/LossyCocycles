import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import LossyCocycles.StageB

/-! # LossyCocycles.Shear — UnifGapLiftCont is FALSE (machine-checked falsification)

The **shear family**: the flux family's vertical edges with TRIVIAL horizontal edges.
Its vertical column composites are exactly the flux family's, so `windingSum = k`
(by `windingSum_flux`) and `CoreGap 2` holds — yet it lies within `2π|k|/m` of the
**constant flat family** on every edge (verticals carry angle ≤ 2πk/m² each;
horizontals agree exactly). At large m this violates any m-uniform margin:
`unifGapLiftCont_false`.

What this means (Response 38): the formalized `windingSum` reads only vertical
composites; the shear family has ZERO total flux (its wrap-column plaquettes carry −k,
which the diagnostic never sees). So the column-path winding is a 1D shadow, not a
topological obstruction — CT-4's named kill criterion ("a winding-changing,
gap-preserving path") FIRES via `s ↦ shear(s·angles)`. The Plateau Conjecture itself
(about `fluxFamily` and `distOpFlat`) is untouched; what dies is the reduction route
through `UnifGapLiftCont`. The corrected invariant must read horizontal data
(Bott index of the magnetic-translation pair — the pre-registered next target).
-/

open Matrix

namespace LossyCocycles

variable {m d : ℕ} [NeZero m]

/-- The shear family: flux verticals, trivial horizontals. -/
noncomputable def shearFamily (k : ℤ) (d r : ℕ) : EdgeFamily m d :=
  fun e => if e.2 then stdFrame d r * (stdFrame d r)ᵀ else fluxFamily (m := m) k d r e

/-- The constant standard-frame flat family. -/
def constFlat (d r : ℕ) : EdgeFamily m d :=
  fun _ => stdFrame d r * (stdFrame d r)ᵀ

theorem constFlat_isFlat {r : ℕ} (hrd : r ≤ d) :
    IsFlatFamily (m := m) r (constFlat d r) :=
  ⟨fun _ => stdFrame d r, fun _ => stdFrame_isFrame hrd, fun _ => rfl⟩

theorem shear_vert (k : ℤ) (d r : ℕ) (x y : Fin m) :
    shearFamily (m := m) k d r ((x, y), false)
      = fluxFamily (m := m) k d r ((x, y), false) := by
  simp [shearFamily]

/-- Families agreeing on vertical edges have equal vertical partial products. -/
theorem vertProd_congr (T T' : EdgeFamily m d)
    (h : ∀ x y, T ((x, y), false) = T' ((x, y), false)) (x : Fin m) :
    ∀ n, vertProd T x n = vertProd T' x n := by
  intro n
  induction n with
  | zero => rfl
  | succ p ih =>
      show T ((x, (natFin p : Fin m)), false) * vertProd T x p
          = T' ((x, (natFin p : Fin m)), false) * vertProd T' x p
      rw [h x (natFin p), ih]

theorem shear_vertCycle (k : ℤ) (d r : ℕ) (x : Fin m) :
    vertCycle (shearFamily (m := m) k d r) x
      = vertCycle (fluxFamily (m := m) k d r) x :=
  vertProd_congr _ _ (shear_vert k d r) x m

theorem shear_windingSum (k : ℤ) (d r : ℕ) (hd : 1 < d) :
    windingSum hd (shearFamily (m := m) k d r)
      = windingSum hd (fluxFamily (m := m) k d r) := by
  unfold windingSum
  congr 1
  apply Finset.sum_congr rfl
  intro x _
  unfold cycleAngle
  rw [shear_vertCycle, shear_vertCycle]

theorem shear_coreGap {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ) :
    CoreGap 2 hd (shearFamily (m := m) k d r) := by
  intro x
  rw [shear_vertCycle k d r x]
  exact fluxFamily_coreGap hr hrd hd k m x

theorem constFlat_coreGap {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) :
    CoreGap 2 hd (constFlat (m := m) d r) := by
  intro x
  have hm : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr (NeZero.ne m)
  rw [flat_vertCycle hm (constFlat d r) (fun _ => stdFrame d r)
    (fun _ => stdFrame_isFrame hrd) (fun _ => rfl) x]
  obtain ⟨hskew, htr⟩ := coreVec_proj hd (stdFrame d r)
  rw [hskew]
  have key : ∀ (i : Fin 2),
      coreBlock hd (stdFrame d r * (stdFrame d r)ᵀ) i i = 1 := by
    intro i
    have hcb : coreBlock hd (stdFrame d r * (stdFrame d r)ᵀ) i i
        = (stdFrame d r * (stdFrame d r)ᵀ)
            ⟨(i : ℕ), lt_of_le_of_lt (Nat.le_of_lt_succ i.isLt) hd⟩
            ⟨(i : ℕ), lt_of_le_of_lt (Nat.le_of_lt_succ i.isLt) hd⟩ := rfl
    rw [hcb]
    exact stdFrame_proj_diag hrd _ (lt_of_lt_of_le i.isLt hr)
  have htrace : (coreVec (coreBlock hd (stdFrame d r * (stdFrame d r)ᵀ))).1 = 2 := by
    show coreBlock hd (stdFrame d r * (stdFrame d r)ᵀ) 0 0
        + coreBlock hd (stdFrame d r * (stdFrame d r)ᵀ) 1 1 = 2
    rw [key 0, key 1]
    norm_num
  rw [htrace, show ((2:ℝ)) ^ 2 + (0:ℝ) ^ 2 = 2 ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]

/-! ## The distance bound: a conjugated rotation is |θ|-close to the projection -/

/-- `‖(R(θ) − 1)w‖ ≤ |θ|·‖w‖` — exact identity `A²+B² = (2−2cosθ)(w₀²+w₁²)` plus the
quadratic cosine bound. -/
theorem vnorm_rot_sub_one {r : ℕ} (hr : 1 < r) (θ : ℝ) (w : Fin r → ℝ) :
    vnorm ((rotCore r θ - 1) *ᵥ w) ≤ |θ| * vnorm w := by
  have h0r : 0 < r := by omega
  set A : ℝ := (Real.cos θ - 1) * w ⟨0, h0r⟩ - Real.sin θ * w ⟨1, hr⟩ with hA
  set B : ℝ := Real.sin θ * w ⟨0, h0r⟩ + (Real.cos θ - 1) * w ⟨1, hr⟩ with hB
  have hrow : ∀ i : Fin r, ((rotCore r θ - 1) *ᵥ w) i
      = if (i : ℕ) = 0 then A else if (i : ℕ) = 1 then B else 0 := by
    intro i
    show (∑ j, (rotCore r θ - 1) i j * w j) = _
    by_cases hi0 : (i : ℕ) = 0
    · have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j
          = (if (j : ℕ) = 0 then (Real.cos θ - 1) * w ⟨0, h0r⟩ else 0)
            + (if (j : ℕ) = 1 then -Real.sin θ * w ⟨1, hr⟩ else 0) := by
        intro j
        rw [Matrix.sub_apply, rotCore_row_zero θ hi0 j, Matrix.one_apply]
        by_cases hj0 : (j : ℕ) = 0
        · have hj : j = (⟨0, h0r⟩ : Fin r) := Fin.ext hj0
          subst hj
          have hij : i = (⟨0, h0r⟩ : Fin r) := Fin.ext hi0
          simp [hij]
        · by_cases hj1 : (j : ℕ) = 1
          · have hj : j = (⟨1, hr⟩ : Fin r) := Fin.ext hj1
            subst hj
            have hij : ¬(i = (⟨1, hr⟩ : Fin r)) := by
              rw [Fin.ext_iff]; omega
            simp [hj0, hij]
          · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
            simp [hj0, hj1, hij]
      rw [Finset.sum_congr rfl fun j _ => hterm j, sum_two_slot h0r hr, hi0]
      simp [hA]
      ring
    · by_cases hi1 : (i : ℕ) = 1
      · have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j
            = (if (j : ℕ) = 0 then Real.sin θ * w ⟨0, h0r⟩ else 0)
              + (if (j : ℕ) = 1 then (Real.cos θ - 1) * w ⟨1, hr⟩ else 0) := by
          intro j
          rw [Matrix.sub_apply, rotCore_row_one θ hi1 j, Matrix.one_apply]
          by_cases hj0 : (j : ℕ) = 0
          · have hj : j = (⟨0, h0r⟩ : Fin r) := Fin.ext hj0
            subst hj
            have hij : ¬(i = (⟨0, h0r⟩ : Fin r)) := by
              rw [Fin.ext_iff]; omega
            simp [hij]
          · by_cases hj1 : (j : ℕ) = 1
            · have hj : j = (⟨1, hr⟩ : Fin r) := Fin.ext hj1
              subst hj
              have hij : i = (⟨1, hr⟩ : Fin r) := Fin.ext hi1
              simp [hij]
            · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
              simp [hj0, hj1, hij]
        rw [Finset.sum_congr rfl fun j _ => hterm j, sum_two_slot h0r hr, hi1]
        simp [hi0, hB]
      · have hi2 : 2 ≤ (i : ℕ) := by omega
        have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j = 0 := by
          intro j
          rw [Matrix.sub_apply, rotCore_row_two θ hi2 j, Matrix.one_apply, sub_self,
            zero_mul]
        rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_const_zero]
        simp [hi0, hi1]
  have hsum : (∑ i, (((rotCore r θ - 1) *ᵥ w) i) ^ 2) = A ^ 2 + B ^ 2 := by
    have hpt : ∀ i : Fin r, (((rotCore r θ - 1) *ᵥ w) i) ^ 2
        = (if (i : ℕ) = 0 then A ^ 2 else 0) + (if (i : ℕ) = 1 then B ^ 2 else 0) := by
      intro i
      rw [hrow i]
      by_cases hi0 : (i : ℕ) = 0
      · simp [hi0]
      · by_cases hi1 : (i : ℕ) = 1
        · simp [hi0, hi1]
        · simp [hi0, hi1]
    rw [Finset.sum_congr rfl fun i _ => hpt i, sum_two_slot h0r hr]
  have hid : A ^ 2 + B ^ 2
      = (2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2) := by
    rw [hA, hB]
    linear_combination (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2) * (Real.sin_sq_add_cos_sq θ)
  have hcos : 2 - 2 * Real.cos θ ≤ θ ^ 2 := by
    have := Real.one_sub_sq_div_two_le_cos (x := θ)
    linarith
  have hw2 : w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2 ≤ ∑ j, w j ^ 2 := by
    have hne : (⟨1, hr⟩ : Fin r) ≠ (⟨0, h0r⟩ : Fin r) :=
      fun hcon => absurd (congrArg Fin.val hcon) Nat.one_ne_zero
    have e1 : w ⟨1, hr⟩ ^ 2 ≤ ∑ j ∈ Finset.univ.erase ⟨0, h0r⟩, w j ^ 2 :=
      Finset.single_le_sum (fun j _ => sq_nonneg _)
        (Finset.mem_erase.mpr ⟨hne, Finset.mem_univ _⟩)
    have e0 : w ⟨0, h0r⟩ ^ 2 + ∑ j ∈ Finset.univ.erase ⟨0, h0r⟩, w j ^ 2
        = ∑ j, w j ^ 2 :=
      Finset.add_sum_erase Finset.univ (fun j => w j ^ 2)
        (Finset.mem_univ (⟨0, h0r⟩ : Fin r))
    linarith
  have hWnn : (0:ℝ) ≤ ∑ j, w j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  have hw01 : (0:ℝ) ≤ w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2 := by positivity
  unfold vnorm
  rw [hsum, hid]
  have hbound : (2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2)
      ≤ θ ^ 2 * (∑ j, w j ^ 2) := by
    have h2c : (0:ℝ) ≤ 2 - 2 * Real.cos θ := by
      have := Real.cos_le_one θ
      linarith
    calc (2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2)
        ≤ θ ^ 2 * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2) :=
          mul_le_mul_of_nonneg_right hcos hw01
      _ ≤ θ ^ 2 * (∑ j, w j ^ 2) := mul_le_mul_of_nonneg_left hw2 (sq_nonneg θ)
  calc Real.sqrt ((2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2))
      ≤ Real.sqrt (θ ^ 2 * (∑ j, w j ^ 2)) := Real.sqrt_le_sqrt hbound
    _ = |θ| * Real.sqrt (∑ j, w j ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg θ), Real.sqrt_sq_eq_abs]

/-- **Sharp form:** `‖(R(θ) − 1)w‖ ≤ 2|sin(θ/2)|·‖w‖` — the exact chord length
`A² + B² = (2 − 2cosθ)(w₀² + w₁²)` with the half-angle identity `2 − 2cosθ =
(2sin(θ/2))²`. This is the quantitatively correct distance (it knows `θ ↦ 0` and
`θ ↦ 2π` both collapse), where `vnorm_rot_sub_one`'s `|θ|` only sees the first order. -/
theorem vnorm_rot_sub_one_sharp {r : ℕ} (hr : 1 < r) (θ : ℝ) (w : Fin r → ℝ) :
    vnorm ((rotCore r θ - 1) *ᵥ w) ≤ 2 * |Real.sin (θ / 2)| * vnorm w := by
  have h0r : 0 < r := by omega
  set A : ℝ := (Real.cos θ - 1) * w ⟨0, h0r⟩ - Real.sin θ * w ⟨1, hr⟩ with hA
  set B : ℝ := Real.sin θ * w ⟨0, h0r⟩ + (Real.cos θ - 1) * w ⟨1, hr⟩ with hB
  have hrow : ∀ i : Fin r, ((rotCore r θ - 1) *ᵥ w) i
      = if (i : ℕ) = 0 then A else if (i : ℕ) = 1 then B else 0 := by
    intro i
    show (∑ j, (rotCore r θ - 1) i j * w j) = _
    by_cases hi0 : (i : ℕ) = 0
    · have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j
          = (if (j : ℕ) = 0 then (Real.cos θ - 1) * w ⟨0, h0r⟩ else 0)
            + (if (j : ℕ) = 1 then -Real.sin θ * w ⟨1, hr⟩ else 0) := by
        intro j
        rw [Matrix.sub_apply, rotCore_row_zero θ hi0 j, Matrix.one_apply]
        by_cases hj0 : (j : ℕ) = 0
        · have hj : j = (⟨0, h0r⟩ : Fin r) := Fin.ext hj0
          subst hj
          have hij : i = (⟨0, h0r⟩ : Fin r) := Fin.ext hi0
          simp [hij]
        · by_cases hj1 : (j : ℕ) = 1
          · have hj : j = (⟨1, hr⟩ : Fin r) := Fin.ext hj1
            subst hj
            have hij : ¬(i = (⟨1, hr⟩ : Fin r)) := by
              rw [Fin.ext_iff]; omega
            simp [hj0, hij]
          · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
            simp [hj0, hj1, hij]
      rw [Finset.sum_congr rfl fun j _ => hterm j, sum_two_slot h0r hr, hi0]
      simp [hA]
      ring
    · by_cases hi1 : (i : ℕ) = 1
      · have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j
            = (if (j : ℕ) = 0 then Real.sin θ * w ⟨0, h0r⟩ else 0)
              + (if (j : ℕ) = 1 then (Real.cos θ - 1) * w ⟨1, hr⟩ else 0) := by
          intro j
          rw [Matrix.sub_apply, rotCore_row_one θ hi1 j, Matrix.one_apply]
          by_cases hj0 : (j : ℕ) = 0
          · have hj : j = (⟨0, h0r⟩ : Fin r) := Fin.ext hj0
            subst hj
            have hij : ¬(i = (⟨0, h0r⟩ : Fin r)) := by
              rw [Fin.ext_iff]; omega
            simp [hij]
          · by_cases hj1 : (j : ℕ) = 1
            · have hj : j = (⟨1, hr⟩ : Fin r) := Fin.ext hj1
              subst hj
              have hij : i = (⟨1, hr⟩ : Fin r) := Fin.ext hi1
              simp [hij]
            · have hij : ¬(i = j) := by rw [Fin.ext_iff]; omega
              simp [hj0, hj1, hij]
        rw [Finset.sum_congr rfl fun j _ => hterm j, sum_two_slot h0r hr, hi1]
        simp [hi0, hB]
      · have hi2 : 2 ≤ (i : ℕ) := by omega
        have hterm : ∀ j : Fin r, (rotCore r θ - 1) i j * w j = 0 := by
          intro j
          rw [Matrix.sub_apply, rotCore_row_two θ hi2 j, Matrix.one_apply, sub_self,
            zero_mul]
        rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_const_zero]
        simp [hi0, hi1]
  have hsum : (∑ i, (((rotCore r θ - 1) *ᵥ w) i) ^ 2) = A ^ 2 + B ^ 2 := by
    have hpt : ∀ i : Fin r, (((rotCore r θ - 1) *ᵥ w) i) ^ 2
        = (if (i : ℕ) = 0 then A ^ 2 else 0) + (if (i : ℕ) = 1 then B ^ 2 else 0) := by
      intro i
      rw [hrow i]
      by_cases hi0 : (i : ℕ) = 0
      · simp [hi0]
      · by_cases hi1 : (i : ℕ) = 1
        · simp [hi0, hi1]
        · simp [hi0, hi1]
    rw [Finset.sum_congr rfl fun i _ => hpt i, sum_two_slot h0r hr]
  have hid : A ^ 2 + B ^ 2
      = (2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2) := by
    rw [hA, hB]
    linear_combination (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2) * (Real.sin_sq_add_cos_sq θ)
  have hhalf : 2 - 2 * Real.cos θ = (2 * Real.sin (θ / 2)) ^ 2 := by
    have hcos2 : Real.cos θ = 2 * Real.cos (θ / 2) ^ 2 - 1 := by
      have h := Real.cos_two_mul (θ / 2)
      rwa [show 2 * (θ / 2) = θ by ring] at h
    have hpyth := Real.sin_sq_add_cos_sq (θ / 2)
    linear_combination (-2 : ℝ) * hcos2 + (-4 : ℝ) * hpyth
  have hge : (0:ℝ) ≤ 2 - 2 * Real.cos θ := by
    have := Real.cos_le_one θ; linarith
  have hw2 : w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2 ≤ ∑ j, w j ^ 2 := by
    have hne : (⟨1, hr⟩ : Fin r) ≠ (⟨0, h0r⟩ : Fin r) :=
      fun hcon => absurd (congrArg Fin.val hcon) Nat.one_ne_zero
    have e1 : w ⟨1, hr⟩ ^ 2 ≤ ∑ j ∈ Finset.univ.erase ⟨0, h0r⟩, w j ^ 2 :=
      Finset.single_le_sum (fun j _ => sq_nonneg _)
        (Finset.mem_erase.mpr ⟨hne, Finset.mem_univ _⟩)
    have e0 : w ⟨0, h0r⟩ ^ 2 + ∑ j ∈ Finset.univ.erase ⟨0, h0r⟩, w j ^ 2
        = ∑ j, w j ^ 2 :=
      Finset.add_sum_erase Finset.univ (fun j => w j ^ 2)
        (Finset.mem_univ (⟨0, h0r⟩ : Fin r))
    linarith
  unfold vnorm
  rw [hsum, hid]
  have hbound : (2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2)
      ≤ (2 - 2 * Real.cos θ) * (∑ j, w j ^ 2) :=
    mul_le_mul_of_nonneg_left hw2 hge
  calc Real.sqrt ((2 - 2 * Real.cos θ) * (w ⟨0, h0r⟩ ^ 2 + w ⟨1, hr⟩ ^ 2))
      ≤ Real.sqrt ((2 - 2 * Real.cos θ) * (∑ j, w j ^ 2)) := Real.sqrt_le_sqrt hbound
    _ = Real.sqrt ((2 * Real.sin (θ / 2)) ^ 2 * (∑ j, w j ^ 2)) := by rw [hhalf]
    _ = |2 * Real.sin (θ / 2)| * Real.sqrt (∑ j, w j ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs]
    _ = 2 * |Real.sin (θ / 2)| * Real.sqrt (∑ j, w j ^ 2) := by
        rw [abs_mul, abs_two]

/-- The conjugated rotation is |θ|-close to the standard projection in operator norm. -/
theorem opNorm_conj_rot_sub {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (θ : ℝ) :
    opNorm (stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ
        - stdFrame d r * (stdFrame d r)ᵀ)
      ≤ |θ| := by
  have hW : (stdFrame d r)ᵀ * stdFrame d r = 1 := stdFrame_isFrame hrd
  have hfact : stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ
      - stdFrame d r * (stdFrame d r)ᵀ
      = stdFrame d r * ((rotCore r θ - 1) * (stdFrame d r)ᵀ) := by
    rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_assoc]
  apply opNorm_le _ _ (abs_nonneg θ)
  intro v hv
  have hv1 : vnorm v = 1 := by
    unfold vnorm
    rw [hv, Real.sqrt_one]
  rw [hfact, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    frame_isometry (stdFrame d r) hW]
  calc vnorm ((rotCore r θ - 1) *ᵥ ((stdFrame d r)ᵀ *ᵥ v))
      ≤ |θ| * vnorm ((stdFrame d r)ᵀ *ᵥ v) := vnorm_rot_sub_one hr θ _
    _ ≤ |θ| * vnorm v :=
        mul_le_mul_of_nonneg_left (transpose_contract _ hW v) (abs_nonneg θ)
    _ = |θ| := by rw [hv1, mul_one]

/-- Every shear edge is within `2π|k|/m` of the constant flat family. -/
theorem shear_close {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (k : ℤ) (e : TorusEdge m) :
    opNorm (shearFamily (m := m) k d r e - constFlat d r e)
      ≤ 2 * Real.pi * |(k : ℝ)| / m := by
  have hm0 : 0 < m := Nat.pos_of_ne_zero (NeZero.ne m)
  have hmR : (0:ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hbnonneg : (0:ℝ) ≤ 2 * Real.pi * |(k : ℝ)| / m := by positivity
  by_cases he : e.2
  · have hzero : shearFamily (m := m) k d r e = constFlat d r e := by
      simp [shearFamily, he, constFlat]
    rw [hzero, sub_self]
    apply opNorm_le _ _ hbnonneg
    intro v _
    rw [Matrix.zero_mulVec, vnorm_zero]
    exact hbnonneg
  · have heq : shearFamily (m := m) k d r e
        = stdFrame d r * rotCore r (fluxAngle k e) * (stdFrame d r)ᵀ := by
      simp [shearFamily, he, fluxFamily]
    rw [heq, show constFlat (m := m) d r e = stdFrame d r * (stdFrame d r)ᵀ from rfl]
    refine le_trans (opNorm_conj_rot_sub hr hrd _) ?_
    -- |fluxAngle| ≤ 2π|k|·x/m² ≤ 2π|k|/m, since x < m
    have hangle : fluxAngle (m := m) k e = 2 * Real.pi * k * (e.1.1 : ℕ) / ((m : ℕ) ^ 2 : ℕ) := by
      simp [fluxAngle, he]
    rw [hangle]
    have hx : ((e.1.1 : ℕ) : ℝ) ≤ ((m : ℕ) : ℝ) := by
      exact_mod_cast le_of_lt e.1.1.isLt
    have hxnn : (0:ℝ) ≤ ((e.1.1 : ℕ) : ℝ) := by positivity
    rw [abs_div, abs_mul, abs_mul, abs_mul, abs_two, abs_of_pos Real.pi_pos,
      abs_of_nonneg hxnn, abs_of_pos (by positivity : (0:ℝ) < (((m : ℕ) ^ 2 : ℕ) : ℝ))]
    push_cast
    rw [div_le_div_iff₀ (by positivity) hmR]
    have hπ := Real.pi_pos
    have hknn : (0:ℝ) ≤ |(k : ℝ)| := abs_nonneg _
    nlinarith [mul_le_mul_of_nonneg_left hx
      (by positivity : (0:ℝ) ≤ 2 * Real.pi * |(k : ℝ)|), hmR]

/-- **UnifGapLiftCont is FALSE.** The shear family winds k = 1 while sitting within
`2π/m` of the constant flat family, with both families gapped at 2. -/
theorem unifGapLiftCont_false (d r : ℕ) (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) :
    ¬ UnifGapLiftCont d r := by
  intro h
  obtain ⟨margin, hmpos, hP⟩ := h 2 two_pos
  -- choose m with 3 ≤ m and 2π/m < margin
  obtain ⟨m, hm3, hmlt⟩ : ∃ m : ℕ, 3 ≤ m ∧ 2 * Real.pi / m < margin := by
    refine ⟨max 3 (⌈2 * Real.pi / margin⌉₊ + 1), le_max_left _ _, ?_⟩
    have hMnat : 3 ≤ max 3 (⌈2 * Real.pi / margin⌉₊ + 1) := le_max_left _ _
    have hM0 : (0:ℝ) < ((max 3 (⌈2 * Real.pi / margin⌉₊ + 1) : ℕ) : ℝ) := by
      have : 0 < max 3 (⌈2 * Real.pi / margin⌉₊ + 1) := by omega
      exact_mod_cast this
    rw [div_lt_iff₀ hM0]
    have h1 : 2 * Real.pi / margin
        < ((max 3 (⌈2 * Real.pi / margin⌉₊ + 1) : ℕ) : ℝ) := by
      have h2 : 2 * Real.pi / margin ≤ (⌈2 * Real.pi / margin⌉₊ : ℝ) := Nat.le_ceil _
      have h3 : (⌈2 * Real.pi / margin⌉₊ : ℝ)
          < ((max 3 (⌈2 * Real.pi / margin⌉₊ + 1) : ℕ) : ℝ) := by
        have hle : ⌈2 * Real.pi / margin⌉₊ + 1 ≤ max 3 (⌈2 * Real.pi / margin⌉₊ + 1) :=
          le_max_right _ _
        exact_mod_cast lt_of_lt_of_le (Nat.lt_succ_self _) hle
      linarith
    calc 2 * Real.pi = (2 * Real.pi / margin) * margin := by field_simp
      _ < ((max 3 (⌈2 * Real.pi / margin⌉₊ + 1) : ℕ) : ℝ) * margin :=
          mul_lt_mul_of_pos_right h1 hmpos
      _ = margin * ((max 3 (⌈2 * Real.pi / margin⌉₊ + 1) : ℕ) : ℝ) := mul_comm _ _
  haveI : NeZero m := ⟨by omega⟩
  have hwS : windingSum (m := m) hd (shearFamily 1 d r) = 1 := by
    rw [shear_windingSum, windingSum_flux 1 d r hd hr hrd
      (by simp only [Int.natAbs_one]; omega)]
    norm_num
  have hwF : windingSum (m := m) hd (constFlat d r) = 0 :=
    windingSum_flat two_pos hd r _ (constFlat_isFlat hrd) (constFlat_coreGap hd hr hrd)
  have hclose : ∀ e : TorusEdge m,
      opNorm (shearFamily (m := m) 1 d r e - constFlat d r e) < margin := by
    intro e
    refine lt_of_le_of_lt (shear_close hr hrd 1 e) ?_
    have : (2 : ℝ) * Real.pi * |((1 : ℤ) : ℝ)| / m = 2 * Real.pi / m := by
      norm_num
    rw [this]
    exact hmlt
  have heq := hP m hd (shearFamily 1 d r) (constFlat d r) (constFlat_isFlat hrd)
    (shear_coreGap hd hr hrd 1) (constFlat_coreGap hd hr hrd) hclose
  rw [hwS, hwF] at heq
  norm_num at heq

end LossyCocycles
