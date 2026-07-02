import LossyCocycles.Shear

/-! # LossyCocycles.PlaqIndex — L5a′: the plaquette index (the corrected invariant)

Response 38 proved the vertical-only `windingSum` is a 1D shadow (the shear family
winds k near flat). The corrected charge must read ALL edges. This file defines the
**plaquette holonomy** (the 4-edge loop composite), its core angle, and the **total
flux** Σ_p plaqAngle — and proves the three anchor theorems, with the shear test FIRST
in spirit (falsification-first: the corrected invariant must separate flux from shear):

* `totalFlux_flat`   : flat families (with plaquette gap) have total flux 0;
* `totalFlux_flux`   : the charge-k flux family has total flux 2πk;
* `totalFlux_shear`  : the shear family has total flux 0  ← the test the old
                       invariant failed (it read k); the new one passes.

Honesty note: totalFlux is the correct CHARGE (it distinguishes the families), but it
is a continuous functional — the m-uniform RIGIDITY must come from the quantized index
of the magnetic-translation pair (L5a′-b, next rung), for which these anchors are the
evaluation layer. Nothing here claims stability.
-/

open Matrix

namespace LossyCocycles

variable {m d : ℕ} [NeZero m]

/-- Fin wraparound value (extracted pattern). -/
theorem fin_add_one_val (x : Fin m) : ((x + 1 : Fin m) : ℕ) = ((x : ℕ) + 1) % m := by
  rw [Fin.val_add, Fin.val_one']
  conv_lhs => rw [Nat.add_mod]
  conv_rhs => rw [Nat.add_mod]
  rw [Nat.mod_mod]

/-- The plaquette holonomy at v = (x,y): right, up, then the two reverse edges —
`(T(v,↑))ᵀ · (T((x,y+1),→))ᵀ · T((x+1,y),↑) · T(v,→)`. Orientation chosen so the
flux family reads +2πk/m² per plaquette. -/
def plaqHol (T : EdgeFamily m d) (v : Fin m × Fin m) : Matrix (Fin d) (Fin d) ℝ :=
  (T (v, false))ᵀ * (T ((v.1, v.2 + 1), true))ᵀ
    * T ((v.1 + 1, v.2), false) * T (v, true)

/-- The plaquette core angle. -/
noncomputable def plaqAngle (hd : 1 < d) (T : EdgeFamily m d) (v : Fin m × Fin m) : ℝ :=
  rotAngle (coreBlock hd (plaqHol T v))

/-- Plaquette-level gap. -/
def PlaqGap (γ : ℝ) (hd : 1 < d) (T : EdgeFamily m d) : Prop :=
  ∀ v : Fin m × Fin m,
    γ ≤ Real.sqrt ((coreVec (coreBlock hd (plaqHol T v))).1 ^ 2 +
                   (coreVec (coreBlock hd (plaqHol T v))).2 ^ 2)

/-- The total flux: the corrected charge, reading every edge. -/
noncomputable def totalFlux (hd : 1 < d) (T : EdgeFamily m d) : ℝ :=
  ∑ v : Fin m × Fin m, plaqAngle hd T v

/-! ## Conjugated-rotation machinery -/

theorem conjRot_transpose {r : ℕ} (θ : ℝ) :
    (stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ)ᵀ
      = stdFrame d r * rotCore r (-θ) * (stdFrame d r)ᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    rotCore_transpose, ← Matrix.mul_assoc]

/-- The plaquette holonomy of an angle family is the conjugated rotation by the
oriented angle sum. -/
theorem plaqHol_ofAngles {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (T : EdgeFamily m d)
    (φ : TorusEdge m → ℝ)
    (hT : ∀ e, T e = stdFrame d r * rotCore r (φ e) * (stdFrame d r)ᵀ)
    (v : Fin m × Fin m) :
    plaqHol T v = stdFrame d r * rotCore r
        (-(φ (v, false)) + -(φ ((v.1, v.2 + 1), true))
          + φ ((v.1 + 1, v.2), false) + φ (v, true)) * (stdFrame d r)ᵀ := by
  unfold plaqHol
  simp only [hT]
  rw [conjRot_transpose, conjRot_transpose, stdFrame_conj_mul hrd,
    stdFrame_conj_mul hrd, stdFrame_conj_mul hrd, rotCore_mul hr, rotCore_mul hr,
    rotCore_mul hr]

/-- The conjugated rotation's core angle, with the explicit branch integer. -/
theorem rotAngle_conj {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (θ : ℝ) :
    rotAngle (coreBlock hd (stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ))
      = θ + 2 * Real.pi * ⌊(Real.pi - θ) / (2 * Real.pi)⌋ := by
  unfold rotAngle
  rw [coreVec_conj_rot hd hr hrd θ]
  have hz : (⟨(2 * Real.cos θ, 2 * Real.sin θ).1, (2 * Real.cos θ, 2 * Real.sin θ).2⟩ : ℂ)
      = (2 : ℝ) * (Complex.cos θ + Complex.sin θ * Complex.I) := by
    rw [← Complex.ofReal_cos, ← Complex.ofReal_sin]
    apply Complex.ext <;> simp [Complex.cos_ofReal_re, Complex.sin_ofReal_re]
  rw [hz]
  have key := Complex.arg_mul_cos_add_sin_mul_I_sub (r := 2) two_pos θ
  linarith [key]

/-- Principal case: the angle is read off exactly. -/
theorem rotAngle_conj_principal {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    {θ : ℝ} (h1 : -Real.pi < θ) (h2 : θ ≤ Real.pi) :
    rotAngle (coreBlock hd (stdFrame d r * rotCore r θ * (stdFrame d r)ᵀ)) = θ := by
  rw [rotAngle_conj hd hr hrd θ]
  have h2π := Real.two_pi_pos
  have hfl : ⌊(Real.pi - θ) / (2 * Real.pi)⌋ = 0 := by
    rw [Int.floor_eq_zero_iff]
    constructor
    · exact div_nonneg (by linarith) (le_of_lt h2π)
    · rw [div_lt_one h2π]
      linarith
  rw [hfl]
  norm_num

/-! ## Anchor 1: flat families -/

/-- The plaquette holonomy of a flat family is the base-vertex projection. -/
theorem plaqHol_flat {r : ℕ} (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hTe : ∀ e, T e = V (dst e) * (V (src e))ᵀ)
    (v : Fin m × Fin m) :
    plaqHol T v = V v * (V v)ᵀ := by
  unfold plaqHol
  simp only [hTe, src, dst, Bool.false_eq_true, if_false, if_true]
  simp only [Matrix.transpose_mul, Matrix.transpose_transpose]
  rw [collapse (V (v.1 + 1, v.2 + 1)) (V (v.1, v.2 + 1)) (V v) (hfr _),
    collapse (V (v.1 + 1, v.2)) (V (v.1 + 1, v.2 + 1)) (V v) (hfr _),
    collapse (V v) (V (v.1 + 1, v.2)) (V v) (hfr _)]

/-- **Anchor 1.** Flat families with a plaquette gap have total flux 0. -/
theorem totalFlux_flat {r : ℕ} {γ : ℝ} (hγ : 0 < γ) (hd : 1 < d)
    (T : EdgeFamily m d) (hT : IsFlatFamily r T) (hgap : PlaqGap γ hd T) :
    totalFlux hd T = 0 := by
  obtain ⟨V, hfr, hTe⟩ := hT
  have hang : ∀ v : Fin m × Fin m, plaqAngle hd T v = 0 := by
    intro v
    have hproj := plaqHol_flat T V hfr hTe v
    obtain ⟨hskew, htr⟩ := coreVec_proj hd (V v)
    have hlen := hgap v
    rw [hproj, hskew] at hlen
    norm_num [Real.sqrt_sq_eq_abs] at hlen
    have hpos : 0 < (coreVec (coreBlock hd (V v * (V v)ᵀ))).1 := by
      have habs : |(coreVec (coreBlock hd (V v * (V v)ᵀ))).1|
          = (coreVec (coreBlock hd (V v * (V v)ᵀ))).1 := abs_of_nonneg htr
      rw [habs] at hlen
      linarith
    unfold plaqAngle rotAngle
    rw [hproj, hskew, Complex.arg_eq_zero_iff]
    exact ⟨le_of_lt hpos, rfl⟩
  unfold totalFlux
  rw [Finset.sum_congr rfl fun v _ => hang v]
  simp

/-! ## Anchor 2: the flux family reads 2πk -/

theorem fluxAngle_vert (k : ℤ) (w : Fin m × Fin m) :
    fluxAngle (m := m) k (w, false)
      = 2 * Real.pi * k * ((w.1 : ℕ) : ℝ) / (((m : ℕ) ^ 2 : ℕ) : ℝ) := by
  simp [fluxAngle]

theorem fluxAngle_hor (k : ℤ) (w : Fin m × Fin m) :
    fluxAngle (m := m) k (w, true)
      = if (w.1 : ℕ) = m - 1 then -(2 * Real.pi * k * ((w.2 : ℕ) : ℝ)) / (m : ℝ)
        else 0 := by
  simp [fluxAngle]

/-- Every plaquette of the charge-k flux family carries angle exactly 2πk/m². -/
theorem plaqAngle_flux {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (hm : 2 * k.natAbs < m) (v : Fin m × Fin m) :
    plaqAngle hd (fluxFamily (m := m) k d r) v
      = 2 * Real.pi * k / ((m : ℝ) ^ 2) := by
  have hm0 : 0 < m := by omega
  have hmR : (0 : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hπ := Real.pi_pos
  have hkm : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm
  have hkabs : |(k : ℝ)| = ((k.natAbs : ℕ) : ℝ) := by
    rw [Nat.cast_natAbs, Int.cast_abs]
  -- the target angle is principal: |2πk/m²| < π
  have hm1R : (1:ℝ) ≤ ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have habs : |2 * Real.pi * k / ((m : ℝ) ^ 2)| < Real.pi := by
    rw [abs_div, abs_mul, abs_mul, abs_two, abs_of_pos hπ, hkabs,
      abs_of_pos (by positivity : (0:ℝ) < ((m : ℕ) : ℝ) ^ 2)]
    rw [div_lt_iff₀ (by positivity)]
    have hsq : ((m : ℕ) : ℝ) ≤ ((m : ℕ) : ℝ) ^ 2 := by nlinarith [hm1R]
    have hstep : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) ^ 2 := by linarith
    nlinarith [hstep, hπ]
  unfold plaqAngle
  rw [plaqHol_ofAngles hr hrd (fluxFamily (m := m) k d r) (fluxAngle (m := m) k)
    (fun _ => rfl) v]
  by_cases hx : (v.1 : ℕ) + 1 < m
  · -- interior column: horizontal angles vanish
    have hx1 : ((v.1 + 1 : Fin m) : ℕ) = (v.1 : ℕ) + 1 := by
      rw [fin_add_one_val]; exact Nat.mod_eq_of_lt hx
    have hxne : ¬((v.1 : ℕ) = m - 1) := by omega
    have hval : -(fluxAngle (m := m) k (v, false))
          + -(fluxAngle (m := m) k ((v.1, v.2 + 1), true))
          + fluxAngle (m := m) k ((v.1 + 1, v.2), false)
          + fluxAngle (m := m) k (v, true)
        = 2 * Real.pi * k / ((m : ℝ) ^ 2) := by
      rw [fluxAngle_vert, fluxAngle_vert, fluxAngle_hor, fluxAngle_hor]
      simp only [hxne, if_false]
      rw [hx1]
      push_cast
      field_simp
      ring
    rw [hval]
    exact rotAngle_conj_principal hd hr hrd (abs_lt.mp habs).1 (le_of_lt (abs_lt.mp habs).2)
  · -- wrap column x = m−1
    have hxe : (v.1 : ℕ) = m - 1 := by have := v.1.isLt; omega
    have hx1 : ((v.1 + 1 : Fin m) : ℕ) = 0 := by
      rw [fin_add_one_val]
      have : (v.1 : ℕ) + 1 = m := by omega
      rw [this, Nat.mod_self]
    have hcast : ((v.1 : ℕ) : ℝ) = ((m : ℕ) : ℝ) - 1 := by
      rw [hxe]
      have h1m : 1 ≤ m := hm0
      push_cast [Nat.cast_sub h1m]
      ring
    by_cases hy : (v.2 : ℕ) + 1 < m
    · -- y interior: the two horizontal twists differ by one step
      have hy1 : ((v.2 + 1 : Fin m) : ℕ) = (v.2 : ℕ) + 1 := by
        rw [fin_add_one_val]; exact Nat.mod_eq_of_lt hy
      have hval : -(fluxAngle (m := m) k (v, false))
            + -(fluxAngle (m := m) k ((v.1, v.2 + 1), true))
            + fluxAngle (m := m) k ((v.1 + 1, v.2), false)
            + fluxAngle (m := m) k (v, true)
          = 2 * Real.pi * k / ((m : ℝ) ^ 2) := by
        rw [fluxAngle_vert, fluxAngle_vert, fluxAngle_hor, fluxAngle_hor]
        simp only [hxe]
        rw [if_pos trivial, if_pos trivial, hx1, hy1,
          Nat.cast_sub (by omega : 1 ≤ m)]
        push_cast
        field_simp
        ring
      rw [hval]
      exact rotAngle_conj_principal hd hr hrd (abs_lt.mp habs).1
        (le_of_lt (abs_lt.mp habs).2)
    · -- the corner (m−1, m−1): raw angle 2πk/m² − 2πk; the branch integer is k
      have hye : (v.2 : ℕ) = m - 1 := by have := v.2.isLt; omega
      have hy1 : ((v.2 + 1 : Fin m) : ℕ) = 0 := by
        rw [fin_add_one_val]
        have : (v.2 : ℕ) + 1 = m := by omega
        rw [this, Nat.mod_self]
      have hycast : ((v.2 : ℕ) : ℝ) = ((m : ℕ) : ℝ) - 1 := by
        rw [hye]
        have h1m : 1 ≤ m := hm0
        push_cast [Nat.cast_sub h1m]
        ring
      have hval : -(fluxAngle (m := m) k (v, false))
            + -(fluxAngle (m := m) k ((v.1, v.2 + 1), true))
            + fluxAngle (m := m) k ((v.1 + 1, v.2), false)
            + fluxAngle (m := m) k (v, true)
          = 2 * Real.pi * k / ((m : ℝ) ^ 2) - 2 * Real.pi * k := by
        rw [fluxAngle_vert, fluxAngle_vert, fluxAngle_hor, fluxAngle_hor]
        simp only [hxe]
        rw [if_pos trivial, if_pos trivial, hx1, hy1, hycast,
          Nat.cast_sub (by omega : 1 ≤ m)]
        push_cast
        field_simp
        ring
      rw [hval, rotAngle_conj hd hr hrd]
      have h2π := Real.two_pi_pos
      have hsplit : (Real.pi - (2 * Real.pi * k / ((m : ℝ) ^ 2) - 2 * Real.pi * k))
            / (2 * Real.pi)
          = (Real.pi - 2 * Real.pi * k / ((m : ℝ) ^ 2)) / (2 * Real.pi) + (k : ℝ) := by
        field_simp
        ring
      have hsplit' : (Real.pi - (2 * Real.pi * k / ((m : ℝ) ^ 2) - 2 * Real.pi * k))
            / (2 * Real.pi)
          = (Real.pi - 2 * Real.pi * k / ((m : ℝ) ^ 2)) / (2 * Real.pi) + (k : ℤ) := by
        push_cast
        exact hsplit
      rw [hsplit', Int.floor_add_intCast]
      have hfl : ⌊(Real.pi - 2 * Real.pi * k / ((m : ℝ) ^ 2)) / (2 * Real.pi)⌋ = 0 := by
        rw [Int.floor_eq_zero_iff]
        constructor
        · apply div_nonneg _ (le_of_lt h2π)
          linarith [(abs_lt.mp habs).2]
        · rw [div_lt_one h2π]
          linarith [(abs_lt.mp habs).1]
      rw [hfl]
      push_cast
      ring

/-- **Anchor 2.** The charge-k flux family has total flux 2πk. -/
theorem totalFlux_flux {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (hm : 2 * k.natAbs < m) :
    totalFlux hd (fluxFamily (m := m) k d r) = 2 * Real.pi * k := by
  have hm0 : 0 < m := by omega
  have hmR : ((m : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne m)
  unfold totalFlux
  rw [Finset.sum_congr rfl fun v _ => plaqAngle_flux hd hr hrd k hm v,
    Finset.sum_const, Finset.card_univ]
  have hcard : Fintype.card (Fin m × Fin m) = m * m := by
    rw [Fintype.card_prod, Fintype.card_fin]
  rw [hcard, nsmul_eq_mul]
  push_cast
  field_simp

/-! ## The decisive test: the shear family reads 0 -/

theorem shearFamily_eq (k : ℤ) (d r : ℕ) (e : TorusEdge m) :
    shearFamily (m := m) k d r e
      = stdFrame d r * rotCore r (if e.2 then 0 else fluxAngle (m := m) k e)
        * (stdFrame d r)ᵀ := by
  by_cases he : e.2
  · simp [shearFamily, he, rotCore_zero, Matrix.mul_one]
  · simp [shearFamily, he, fluxFamily]

/-- The shear plaquette angles: 2πk/m² inside, −2πk(m−1)/m² at the wrap column —
all principal, no branch shift anywhere. -/
theorem plaqAngle_shear {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (hm : 2 * k.natAbs < m) (v : Fin m × Fin m) :
    plaqAngle hd (shearFamily (m := m) k d r) v
      = if (v.1 : ℕ) = m - 1
        then -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)
        else 2 * Real.pi * k / ((m : ℝ) ^ 2) := by
  have hm0 : 0 < m := by omega
  have hmR : (0 : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hπ := Real.pi_pos
  have hkm : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm
  have hkabs : |(k : ℝ)| = ((k.natAbs : ℕ) : ℝ) := by
    rw [Nat.cast_natAbs, Int.cast_abs]
  have hm1R : (1:ℝ) ≤ ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  unfold plaqAngle
  rw [plaqHol_ofAngles hr hrd (shearFamily (m := m) k d r)
    (fun e => if e.2 then 0 else fluxAngle (m := m) k e) (shearFamily_eq k d r) v]
  by_cases hx : (v.1 : ℕ) + 1 < m
  · have hx1 : ((v.1 + 1 : Fin m) : ℕ) = (v.1 : ℕ) + 1 := by
      rw [fin_add_one_val]; exact Nat.mod_eq_of_lt hx
    have hxne : ¬((v.1 : ℕ) = m - 1) := by omega
    have habs : |2 * Real.pi * k / ((m : ℝ) ^ 2)| < Real.pi := by
      rw [abs_div, abs_mul, abs_mul, abs_two, abs_of_pos hπ, hkabs,
        abs_of_pos (by positivity : (0:ℝ) < ((m : ℕ) : ℝ) ^ 2)]
      rw [div_lt_iff₀ (by positivity)]
      have hsq : ((m : ℕ) : ℝ) ≤ ((m : ℕ) : ℝ) ^ 2 := by nlinarith [hm1R]
      have hstep : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) ^ 2 := by linarith
      nlinarith [hstep, hπ]
    have hval : -(if (false : Bool) then (0:ℝ) else fluxAngle (m := m) k (v, false))
          + -(if (true : Bool) then (0:ℝ)
              else fluxAngle (m := m) k ((v.1, v.2 + 1), true))
          + (if (false : Bool) then (0:ℝ)
              else fluxAngle (m := m) k ((v.1 + 1, v.2), false))
          + (if (true : Bool) then (0:ℝ) else fluxAngle (m := m) k (v, true))
        = 2 * Real.pi * k / ((m : ℝ) ^ 2) := by
      simp only [if_true, if_false, Bool.false_eq_true]
      rw [fluxAngle_vert, fluxAngle_vert, hx1]
      push_cast
      field_simp
      ring
    rw [hval, if_neg hxne]
    exact rotAngle_conj_principal hd hr hrd (abs_lt.mp habs).1
      (le_of_lt (abs_lt.mp habs).2)
  · have hxe : (v.1 : ℕ) = m - 1 := by have := v.1.isLt; omega
    have hx1 : ((v.1 + 1 : Fin m) : ℕ) = 0 := by
      rw [fin_add_one_val]
      have : (v.1 : ℕ) + 1 = m := by omega
      rw [this, Nat.mod_self]
    have hcast : ((v.1 : ℕ) : ℝ) = ((m : ℕ) : ℝ) - 1 := by
      rw [hxe]
      have h1m : 1 ≤ m := hm0
      push_cast [Nat.cast_sub h1m]
      ring
    have habs : |-(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)| < Real.pi := by
      have hm1 : (0:ℝ) ≤ ((m : ℕ) : ℝ) - 1 := by
        have : (1:ℝ) ≤ ((m : ℕ) : ℝ) := by exact_mod_cast hm0
        linarith
      rw [abs_div, abs_neg, abs_mul, abs_mul, abs_mul, abs_two, abs_of_pos hπ, hkabs,
        abs_of_nonneg hm1,
        abs_of_pos (by positivity : (0:ℝ) < ((m : ℕ) : ℝ) ^ 2)]
      rw [div_lt_iff₀ (by positivity)]
      have hnA : (0:ℝ) ≤ ((k.natAbs : ℕ) : ℝ) := Nat.cast_nonneg _
      have hmm : 2 * ((k.natAbs : ℕ) : ℝ) * ((m : ℕ) : ℝ)
          < ((m : ℕ) : ℝ) * ((m : ℕ) : ℝ) := by nlinarith [hkm, hm1R]
      nlinarith [hmm, hnA, hπ, hm1R]
    have hval : -(if (false : Bool) then (0:ℝ) else fluxAngle (m := m) k (v, false))
          + -(if (true : Bool) then (0:ℝ)
              else fluxAngle (m := m) k ((v.1, v.2 + 1), true))
          + (if (false : Bool) then (0:ℝ)
              else fluxAngle (m := m) k ((v.1 + 1, v.2), false))
          + (if (true : Bool) then (0:ℝ) else fluxAngle (m := m) k (v, true))
        = -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2) := by
      simp only [if_true, if_false, Bool.false_eq_true]
      rw [fluxAngle_vert, fluxAngle_vert, hx1, hcast]
      push_cast
      field_simp
      ring
    rw [hval, if_pos hxe]
    exact rotAngle_conj_principal hd hr hrd (abs_lt.mp habs).1
      (le_of_lt (abs_lt.mp habs).2)

/-- Splitting a Fin-m sum at the last index. -/
theorem sum_split_last (A B : ℝ) :
    (∑ x : Fin m, if (x : ℕ) = m - 1 then B else A)
      = (((m : ℕ) : ℝ) - 1) * A + B := by
  have hm0 : 0 < m := Nat.pos_of_ne_zero (NeZero.ne m)
  have hlast : ((⟨m - 1, by omega⟩ : Fin m) : ℕ) = m - 1 := rfl
  have hpt : ∀ x : Fin m, (if (x : ℕ) = m - 1 then B else A)
      = A + (if x = (⟨m - 1, by omega⟩ : Fin m) then B - A else 0) := by
    intro x
    by_cases hx : (x : ℕ) = m - 1
    · have : x = (⟨m - 1, by omega⟩ : Fin m) := Fin.ext hx
      simp [hx, this]
    · have : ¬(x = (⟨m - 1, by omega⟩ : Fin m)) := fun h => hx (by rw [h])
      simp [hx, this]
  rw [Finset.sum_congr rfl fun x _ => hpt x, Finset.sum_add_distrib,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, Finset.sum_ite_eq']
  simp only [Finset.mem_univ, if_pos, nsmul_eq_mul]
  have h1m : (1:ℝ) ≤ ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  ring

/-- **The decisive test.** The shear family — which the old invariant read as winding
k — has total flux 0. The corrected invariant separates flux from shear. -/
theorem totalFlux_shear {r : ℕ} (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (hm : 2 * k.natAbs < m) :
    totalFlux hd (shearFamily (m := m) k d r) = 0 := by
  have hm0 : 0 < m := by omega
  have hmR : ((m : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne m)
  unfold totalFlux
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl
    fun y _ => plaqAngle_shear hd hr hrd k hm (x, y)]
  have hinner : ∀ x : Fin m,
      (∑ _y : Fin m, if ((x, _y).1 : ℕ) = m - 1
          then -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)
          else 2 * Real.pi * k / ((m : ℝ) ^ 2))
      = ((m : ℕ) : ℝ) * (if (x : ℕ) = m - 1
          then -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)
          else 2 * Real.pi * k / ((m : ℝ) ^ 2)) := by
    intro x
    have hb : (∑ _y : Fin m, if (((x, _y) : Fin m × Fin m).1 : ℕ) = m - 1
          then -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)
          else 2 * Real.pi * k / ((m : ℝ) ^ 2))
        = (∑ _y : Fin m, if (x : ℕ) = m - 1
          then -(2 * Real.pi * k * (((m : ℕ) : ℝ) - 1)) / ((m : ℝ) ^ 2)
          else 2 * Real.pi * k / ((m : ℝ) ^ 2)) := rfl
    rw [hb, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [Finset.sum_congr rfl fun x _ => hinner x, ← Finset.mul_sum, sum_split_last]
  field_simp
  ring
