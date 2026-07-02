import LossyCocycles.PlaqIndex

/-! # LossyCocycles.MagPair — L5a′-b: the magnetic-translation pair and commutator locality

The corrected route's mechanism, machine-checked: package an edge family T into the two
magnetic translations 𝒰 (shift right composed with horizontal edges) and 𝒱 (shift up
composed with vertical edges) acting on site-indexed vectors. Then:

* `magComm_flat` — for FLAT families the pair commutes **exactly** ([𝒰,𝒱] = 0);
* `opNormG_magComm_le` — for ANY family, ‖[𝒰,𝒱]‖ is bounded by the **sup** of the
  per-plaquette discrepancies — edge-local, hence **m-free**: the commutator never sees
  the system size. This is the mechanism that the column-winding route lacked (sums
  amplified by m) and that `flux_gapStable` exhibited at the vertex level.

The Exel–Loring program (L5a′-c, open): a quantized index of the almost-commuting pair
whose nonvanishing obstructs flat approximation m-uniformly. These are its operands.
-/

open Matrix

namespace LossyCocycles

variable {m d : ℕ} [NeZero m]

/-- Index type for site-indexed vectors. -/
abbrev MagIx (m d : ℕ) := (Fin m × Fin m) × Fin d

/-- The horizontal magnetic translation: (𝒰ψ)(x+1,y) = T((x,y),→)·ψ(x,y). -/
def magU (T : EdgeFamily m d) : Matrix (MagIx m d) (MagIx m d) ℝ :=
  fun p q => if p.1 = (q.1.1 + 1, q.1.2) then T (q.1, true) p.2 q.2 else 0

/-- The vertical magnetic translation: (𝒱ψ)(x,y+1) = T((x,y),↑)·ψ(x,y). -/
def magV (T : EdgeFamily m d) : Matrix (MagIx m d) (MagIx m d) ℝ :=
  fun p q => if p.1 = (q.1.1, q.1.2 + 1) then T (q.1, false) p.2 q.2 else 0

/-- The plaquette discrepancy at s: up-then-right minus right-then-up. -/
def plaqDisc (T : EdgeFamily m d) (s : Fin m × Fin m) : Matrix (Fin d) (Fin d) ℝ :=
  T ((s.1, s.2 + 1), true) * T (s, false) - T ((s.1 + 1, s.2), false) * T (s, true)

theorem magU_mul_magV (T : EdgeFamily m d) (p q : MagIx m d) :
    (magU T * magV T) p q
      = if p.1 = (q.1.1 + 1, q.1.2 + 1)
        then (T ((q.1.1, q.1.2 + 1), true) * T (q.1, false)) p.2 q.2 else 0 := by
  rw [Matrix.mul_apply, Fintype.sum_prod_type]
  rw [Finset.sum_eq_single ((q.1.1, q.1.2 + 1) : Fin m × Fin m)]
  · by_cases hp : p.1 = (q.1.1 + 1, q.1.2 + 1)
    · rw [if_pos hp, Matrix.mul_apply]
      apply Finset.sum_congr rfl
      intro j _
      simp [magU, magV, hp]
    · rw [if_neg hp]
      apply Finset.sum_eq_zero
      intro j _
      simp [magU, magV, hp]
  · intro s _ hs
    apply Finset.sum_eq_zero
    intro j _
    simp [magU, magV, hs]
  · intro h; exact absurd (Finset.mem_univ _) h

theorem magV_mul_magU (T : EdgeFamily m d) (p q : MagIx m d) :
    (magV T * magU T) p q
      = if p.1 = (q.1.1 + 1, q.1.2 + 1)
        then (T ((q.1.1 + 1, q.1.2), false) * T (q.1, true)) p.2 q.2 else 0 := by
  rw [Matrix.mul_apply, Fintype.sum_prod_type]
  rw [Finset.sum_eq_single ((q.1.1 + 1, q.1.2) : Fin m × Fin m)]
  · by_cases hp : p.1 = (q.1.1 + 1, q.1.2 + 1)
    · rw [if_pos hp, Matrix.mul_apply]
      apply Finset.sum_congr rfl
      intro j _
      simp [magU, magV, hp]
    · rw [if_neg hp]
      apply Finset.sum_eq_zero
      intro j _
      simp [magU, magV, hp]
  · intro s _ hs
    apply Finset.sum_eq_zero
    intro j _
    simp [magU, magV, hs]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- The commutator is block-supported on the diagonal shift, with plaquette
discrepancies as blocks. -/
theorem magComm_apply (T : EdgeFamily m d) (p q : MagIx m d) :
    (magU T * magV T - magV T * magU T) p q
      = if p.1 = (q.1.1 + 1, q.1.2 + 1) then plaqDisc T q.1 p.2 q.2 else 0 := by
  rw [Matrix.sub_apply, magU_mul_magV, magV_mul_magU]
  by_cases hp : p.1 = (q.1.1 + 1, q.1.2 + 1)
  · rw [if_pos hp, if_pos hp, if_pos hp]
    rw [show plaqDisc T q.1 p.2 q.2
        = (T ((q.1.1, q.1.2 + 1), true) * T (q.1, false)) p.2 q.2
          - (T ((q.1.1 + 1, q.1.2), false) * T (q.1, true)) p.2 q.2 from by
      unfold plaqDisc; rw [Matrix.sub_apply]]
  · rw [if_neg hp, if_neg hp, if_neg hp, sub_zero]

/-- Flat families have vanishing plaquette discrepancy: both paths collapse to the
same corner-to-corner map. -/
theorem plaqDisc_flat {r : ℕ} (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hTe : ∀ e, T e = V (dst e) * (V (src e))ᵀ)
    (s : Fin m × Fin m) : plaqDisc T s = 0 := by
  unfold plaqDisc
  simp only [hTe, src, dst, Bool.false_eq_true, if_false, if_true]
  rw [collapse (V s) (V (s.1, s.2 + 1)) (V (s.1 + 1, s.2 + 1)) (hfr _),
    collapse (V s) (V (s.1 + 1, s.2)) (V (s.1 + 1, s.2 + 1)) (hfr _), sub_self]

/-- **Exact commutativity for flat families:** [𝒰,𝒱] = 0. -/
theorem magComm_flat {r : ℕ} (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hTe : ∀ e, T e = V (dst e) * (V (src e))ᵀ) :
    magU T * magV T - magV T * magU T = 0 := by
  ext p q
  rw [magComm_apply, plaqDisc_flat T V hfr hTe q.1]
  simp

/-! ## The generic norm kit (arbitrary finite index) and commutator locality -/

/-- ℓ² norm over an arbitrary finite index. -/
noncomputable def vnormG {ι : Type*} [Fintype ι] (v : ι → ℝ) : ℝ :=
  Real.sqrt (∑ i, v i ^ 2)

/-- Operator norm over an arbitrary finite index, elementarily. -/
noncomputable def opNormG {ι : Type*} [Fintype ι] (M : Matrix ι ι ℝ) : ℝ :=
  sSup {c | ∃ v : ι → ℝ, (∑ i, v i ^ 2) = 1 ∧ c = Real.sqrt (∑ i, (M.mulVec v i) ^ 2)}

theorem opNormG_le {ι : Type*} [Fintype ι] (M : Matrix ι ι ℝ) (c : ℝ) (hc : 0 ≤ c)
    (h : ∀ v : ι → ℝ, (∑ i, v i ^ 2) = 1 → vnormG (M *ᵥ v) ≤ c) : opNormG M ≤ c := by
  apply Real.sSup_le _ hc
  rintro x ⟨v, hv, rfl⟩
  exact h v hv

/-- The shift-by-(1,1) self-equivalence of the torus sites. -/
def shiftHat : (Fin m × Fin m) ≃ (Fin m × Fin m) where
  toFun s := (s.1 - 1, s.2 - 1)
  invFun s := (s.1 + 1, s.2 + 1)
  left_inv s := by simp
  right_inv s := by simp

/-- The commutator's action computes block-wise through the shifted site. -/
theorem magComm_mulVec (T : EdgeFamily m d) (ψ : MagIx m d → ℝ) (p : MagIx m d) :
    ((magU T * magV T - magV T * magU T) *ᵥ ψ) p
      = ((plaqDisc T (p.1.1 - 1, p.1.2 - 1)) *ᵥ
          fun j => ψ ((p.1.1 - 1, p.1.2 - 1), j)) p.2 := by
  show (∑ q, (magU T * magV T - magV T * magU T) p q * ψ q) = _
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_eq_single ((p.1.1 - 1, p.1.2 - 1) : Fin m × Fin m)]
  · have hcond : p.1 = ((p.1.1 - 1 : Fin m) + 1, (p.1.2 - 1 : Fin m) + 1) := by
      apply Prod.ext <;> simp
    have hterm : ∀ j : Fin d,
        (magU T * magV T - magV T * magU T) p ((p.1.1 - 1, p.1.2 - 1), j)
          * ψ ((p.1.1 - 1, p.1.2 - 1), j)
        = plaqDisc T (p.1.1 - 1, p.1.2 - 1) p.2 j * ψ ((p.1.1 - 1, p.1.2 - 1), j) := by
      intro j
      rw [magComm_apply, if_pos hcond]
    rw [Finset.sum_congr rfl fun j _ => hterm j]
    rfl
  · intro s _ hs
    apply Finset.sum_eq_zero
    intro j _
    rw [magComm_apply]
    have hns : ¬(p.1 = (s.1 + 1, s.2 + 1)) := by
      intro h
      apply hs
      have h1 : p.1.1 = s.1 + 1 := by rw [h]
      have h2 : p.1.2 = s.2 + 1 := by rw [h]
      apply Prod.ext
      · simp [h1]
      · simp [h2]
    rw [if_neg hns, zero_mul]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- **Commutator locality (m-free).** The magnetic pair's commutator norm is bounded
by the supremum of the plaquette discrepancies: edge-local data, never amplified by
system size. -/
theorem opNormG_magComm_le (hd0 : 0 < d) (T : EdgeFamily m d) (C : ℝ) (hC0 : 0 ≤ C)
    (hC : ∀ s, opNorm (plaqDisc T s) ≤ C) :
    opNormG (magU T * magV T - magV T * magU T) ≤ C := by
  apply opNormG_le _ _ hC0
  intro ψ hψ
  unfold vnormG
  have hrow := magComm_mulVec T ψ
  calc Real.sqrt (∑ p, (((magU T * magV T - magV T * magU T) *ᵥ ψ) p) ^ 2)
      = Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (((plaqDisc T (s'.1 - 1, s'.2 - 1)) *ᵥ
            fun j => ψ ((s'.1 - 1, s'.2 - 1), j)) i) ^ 2) := by
        rw [Fintype.sum_prod_type]
        congr 1
        apply Finset.sum_congr rfl
        intro s' _
        apply Finset.sum_congr rfl
        intro i _
        rw [hrow (s', i)]
    _ ≤ Real.sqrt (∑ s' : Fin m × Fin m, C ^ 2 * ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2 - 1), i)) ^ 2) := by
        apply Real.sqrt_le_sqrt
        apply Finset.sum_le_sum
        intro s' _
        -- per-block: ‖D·w‖² ≤ C²‖w‖²
        have hb := mulVec_vnorm_le (plaqDisc T (s'.1 - 1, s'.2 - 1))
          (fun j => ψ ((s'.1 - 1, s'.2 - 1), j)) hd0
        have hb2 : vnorm ((plaqDisc T (s'.1 - 1, s'.2 - 1)) *ᵥ
            fun j => ψ ((s'.1 - 1, s'.2 - 1), j))
            ≤ C * vnorm (fun j => ψ ((s'.1 - 1, s'.2 - 1), j)) := by
          refine le_trans hb ?_
          exact mul_le_mul_of_nonneg_right (hC _) (vnorm_nonneg _)
        have hsq := pow_le_pow_left₀ (vnorm_nonneg _) hb2 2
        unfold vnorm at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        rw [mul_pow] at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        exact hsq
    _ = Real.sqrt (C ^ 2 * ∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2 - 1), i)) ^ 2) := by
        rw [← Finset.mul_sum]
    _ = C * Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2 - 1), i)) ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg C), Real.sqrt_sq hC0]
    _ = C * Real.sqrt (∑ s : Fin m × Fin m, ∑ i : Fin d, (ψ (s, i)) ^ 2) := by
        congr 1
        conv_rhs => rw [← Equiv.sum_comp (shiftHat (m := m))
          (fun s => ∑ i : Fin d, (ψ (s, i)) ^ 2)]
        rfl
    _ = C := by
        have hsum : (∑ p : MagIx m d, ψ p ^ 2)
            = ∑ s : Fin m × Fin m, ∑ i : Fin d, ψ (s, i) ^ 2 :=
          Fintype.sum_prod_type (fun p : MagIx m d => ψ p ^ 2)
        rw [← hsum, hψ, Real.sqrt_one, mul_one]

/-! ## L5a′-c (quantitative): the flux pair's commutator is `2|sin(πk/m²)|`-small

The locality theorem bounds `‖[𝒰,𝒱]‖` by the worst plaquette discrepancy. For the
charge-k flux family every plaquette is a single conjugated rotation by `2πk/m²` (the
wrap corner differs by the integer `2πk`, invisible to the matrix), so the discrepancy
is the chord `2|sin(πk/m²)|` at every site. Hence the commutator is `2|sin(πk/m²)|`-small
**uniformly in m** — and `→ 0` like `2π|k|/m²`. This is the Exel–Loring "almost
commuting" estimate: the pair commutes in the large-m limit while carrying index k. -/

/-- A rotation core is an ℓ²-isometry (it is orthogonal: `R(θ)ᵀR(θ) = R(0) = 1`). -/
theorem vnorm_rotCore {r : ℕ} (hr : 1 < r) (θ : ℝ) (z : Fin r → ℝ) :
    vnorm (rotCore r θ *ᵥ z) = vnorm z := by
  apply frame_isometry
  rw [rotCore_transpose, rotCore_mul hr, show -θ + θ = 0 by ring, rotCore_zero]

/-- Difference of two conjugated rotations, sharp operator-norm bound: the chord
`2|sin((a−b)/2)|`. Factor `R(a) − R(b) = R(b)(R(a−b) − 1)` and use that `R(b)` and the
frame are isometries, then the sharp chord bound on `R(a−b) − 1`. -/
theorem opNorm_conj_rotDiff {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (a b : ℝ) :
    opNorm (stdFrame d r * rotCore r a * (stdFrame d r)ᵀ
        - stdFrame d r * rotCore r b * (stdFrame d r)ᵀ)
      ≤ 2 * |Real.sin ((a - b) / 2)| := by
  have hW : (stdFrame d r)ᵀ * stdFrame d r = 1 := stdFrame_isFrame hrd
  have hinner : rotCore r b * (rotCore r (a - b) - 1) = rotCore r a - rotCore r b := by
    rw [Matrix.mul_sub, Matrix.mul_one, rotCore_mul hr, show b + (a - b) = a by ring]
  have hfact : stdFrame d r * rotCore r a * (stdFrame d r)ᵀ
      - stdFrame d r * rotCore r b * (stdFrame d r)ᵀ
      = stdFrame d r * (rotCore r b * (rotCore r (a - b) - 1)) * (stdFrame d r)ᵀ := by
    rw [hinner, Matrix.mul_sub, Matrix.sub_mul]
  apply opNorm_le _ _ (by positivity)
  intro v hv
  have hv1 : vnorm v = 1 := by unfold vnorm; rw [hv, Real.sqrt_one]
  rw [hfact, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    frame_isometry (stdFrame d r) hW, ← Matrix.mulVec_mulVec, vnorm_rotCore hr]
  calc vnorm ((rotCore r (a - b) - 1) *ᵥ ((stdFrame d r)ᵀ *ᵥ v))
      ≤ 2 * |Real.sin ((a - b) / 2)| * vnorm ((stdFrame d r)ᵀ *ᵥ v) :=
        vnorm_rot_sub_one_sharp hr (a - b) _
    _ ≤ 2 * |Real.sin ((a - b) / 2)| * vnorm v :=
        mul_le_mul_of_nonneg_left (transpose_contract _ hW v) (by positivity)
    _ = 2 * |Real.sin ((a - b) / 2)| := by rw [hv1, mul_one]

/-- `|sin(−x + nπ)| = |sin x|` for integer `n`: the `π`-antiperiodicity of `|sin|`.
This is what makes the wrap corner's extra `2πk` invisible to the chord. -/
theorem abs_sin_neg_add_int (x : ℝ) (n : ℤ) :
    |Real.sin (-x + (n : ℝ) * Real.pi)| = |Real.sin x| := by
  rw [Real.sin_add_int_mul_pi, abs_mul, Real.sin_neg, abs_neg, abs_zpow, abs_neg,
    abs_one, one_zpow, one_mul]

/-- The plaquette discrepancy of an angle family is a difference of conjugated rotations
by the two corner-path angle sums. -/
theorem plaqDisc_ofAngles {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (T : EdgeFamily m d)
    (φ : TorusEdge m → ℝ)
    (hT : ∀ e, T e = stdFrame d r * rotCore r (φ e) * (stdFrame d r)ᵀ)
    (s : Fin m × Fin m) :
    plaqDisc T s
      = stdFrame d r * rotCore r (φ ((s.1, s.2 + 1), true) + φ (s, false))
          * (stdFrame d r)ᵀ
        - stdFrame d r * rotCore r (φ ((s.1 + 1, s.2), false) + φ (s, true))
          * (stdFrame d r)ᵀ := by
  unfold plaqDisc
  simp only [hT]
  rw [stdFrame_conj_mul hrd, stdFrame_conj_mul hrd, rotCore_mul hr, rotCore_mul hr]

/-- **Per-plaquette flux bound.** Every plaquette discrepancy of the charge-k flux family
has operator norm at most `2|sin(πk/m²)|`. The wrap corner's raw angle differs by `2πk`
from the interior `2πk/m²`, but that is an integer multiple of `2π` — the same matrix,
hence the same chord. No `2|k| < m` hypothesis is needed: this works with the raw angle. -/
theorem opNorm_plaqDisc_flux {r : ℕ} (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (s : Fin m × Fin m) :
    opNorm (plaqDisc (fluxFamily (m := m) k d r) s)
      ≤ 2 * |Real.sin (Real.pi * k / (m : ℝ) ^ 2)| := by
  have hm0 : 0 < m := Nat.pos_of_ne_zero (NeZero.ne m)
  have hmR : (0 : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hmne : ((m : ℕ) : ℝ) ≠ 0 := ne_of_gt hmR
  rw [plaqDisc_ofAngles hr hrd (fluxFamily (m := m) k d r) (fluxAngle (m := m) k)
    (fun _ => rfl) s]
  refine (opNorm_conj_rotDiff hr hrd _ _).trans_eq ?_
  congr 1
  by_cases hx : (s.1 : ℕ) + 1 < m
  · -- interior column: both horizontal twists vanish; raw diff is −2πk/m², n = 0
    have hx1 : ((s.1 + 1 : Fin m) : ℕ) = (s.1 : ℕ) + 1 := by
      rw [fin_add_one_val]; exact Nat.mod_eq_of_lt hx
    have hxne : ¬((s.1 : ℕ) = m - 1) := by omega
    have hAB : ((fluxAngle (m := m) k ((s.1, s.2 + 1), true) + fluxAngle (m := m) k (s, false))
          - (fluxAngle (m := m) k ((s.1 + 1, s.2), false) + fluxAngle (m := m) k (s, true))) / 2
        = -(Real.pi * k / (m : ℝ) ^ 2) + ((0 : ℤ) : ℝ) * Real.pi := by
      rw [fluxAngle_hor, fluxAngle_vert, fluxAngle_vert, fluxAngle_hor]
      simp only [hxne, if_false]
      rw [hx1]
      push_cast
      field_simp
      ring
    rw [hAB, abs_sin_neg_add_int]
  · -- wrap column x = m−1
    have hxe : (s.1 : ℕ) = m - 1 := by have := s.1.isLt; omega
    have hx1c : ((s.1 + 1 : Fin m) : ℕ) = 0 := by
      rw [fin_add_one_val]
      have : (s.1 : ℕ) + 1 = m := by omega
      rw [this, Nat.mod_self]
    by_cases hy : (s.2 : ℕ) + 1 < m
    · -- y interior: the two horizontal twists differ by one step; raw diff is −2πk/m²
      have hy1 : ((s.2 + 1 : Fin m) : ℕ) = (s.2 : ℕ) + 1 := by
        rw [fin_add_one_val]; exact Nat.mod_eq_of_lt hy
      have hAB : ((fluxAngle (m := m) k ((s.1, s.2 + 1), true) + fluxAngle (m := m) k (s, false))
            - (fluxAngle (m := m) k ((s.1 + 1, s.2), false) + fluxAngle (m := m) k (s, true))) / 2
          = -(Real.pi * k / (m : ℝ) ^ 2) + ((0 : ℤ) : ℝ) * Real.pi := by
        rw [fluxAngle_hor, fluxAngle_vert, fluxAngle_vert, fluxAngle_hor]
        simp only [hxe]
        rw [if_pos trivial, if_pos trivial, hx1c, hy1, Nat.cast_sub (by omega : 1 ≤ m)]
        push_cast
        field_simp
        ring
      rw [hAB, abs_sin_neg_add_int]
    · -- the corner (m−1, m−1): raw diff is −2πk/m² + 2πk; the branch integer n = k
      have hye : (s.2 : ℕ) = m - 1 := by have := s.2.isLt; omega
      have hy1c : ((s.2 + 1 : Fin m) : ℕ) = 0 := by
        rw [fin_add_one_val]
        have : (s.2 : ℕ) + 1 = m := by omega
        rw [this, Nat.mod_self]
      have hycast : ((s.2 : ℕ) : ℝ) = ((m : ℕ) : ℝ) - 1 := by
        rw [hye]; push_cast [Nat.cast_sub (by omega : 1 ≤ m)]; ring
      have hAB : ((fluxAngle (m := m) k ((s.1, s.2 + 1), true) + fluxAngle (m := m) k (s, false))
            - (fluxAngle (m := m) k ((s.1 + 1, s.2), false) + fluxAngle (m := m) k (s, true))) / 2
          = -(Real.pi * k / (m : ℝ) ^ 2) + ((k : ℤ) : ℝ) * Real.pi := by
        rw [fluxAngle_hor, fluxAngle_vert, fluxAngle_vert, fluxAngle_hor]
        simp only [hxe]
        rw [if_pos trivial, if_pos trivial, hx1c, hy1c, hycast,
          Nat.cast_sub (by omega : 1 ≤ m)]
        push_cast
        field_simp
        ring
      rw [hAB, abs_sin_neg_add_int]

/-- **L5a′-c (quantitative).** The charge-k flux family's magnetic-translation pair has
commutator norm at most `2|sin(πk/m²)|`, **uniformly in m** — vanishing like `2π|k|/m²`
as the lattice grows, while the index stays k. The almost-commuting estimate. -/
theorem opNormG_magComm_flux {r : ℕ} (hd0 : 0 < d) (hr : 1 < r) (hrd : r ≤ d) (k : ℤ) :
    opNormG (magU (fluxFamily (m := m) k d r) * magV (fluxFamily (m := m) k d r)
        - magV (fluxFamily (m := m) k d r) * magU (fluxFamily (m := m) k d r))
      ≤ 2 * |Real.sin (Real.pi * k / (m : ℝ) ^ 2)| :=
  opNormG_magComm_le hd0 (fluxFamily (m := m) k d r)
    (2 * |Real.sin (Real.pi * k / (m : ℝ) ^ 2)|) (by positivity)
    (fun s => opNorm_plaqDisc_flux hr hrd k s)

end LossyCocycles
