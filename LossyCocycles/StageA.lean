import LossyCocycles.Winding

/-! # LossyCocycles.StageA — L5b-A: the existence stage of the Plateau argument

Stage A of the Response-27 split: for every fixed m (with m > 2|k|), the charge-k flux
family is at operator distance ≥ |sin(2πk/m)|/m from every rank-r flat family.

The mechanism needs no winding apparatus: flat column composites are projections, whose
core skew vanishes (`coreVec_proj`); the flux column composite at column 1 has core skew
2·sin(2πk/m) (`flux_vertCycle` + `coreVec_conj_rot`); and skew is Lipschitz along the
column under edge-wise operator perturbation, with constant 2m (contraction
telescoping). The 1/m² decay of the resulting constant is the honest exhibit that this
mechanism cannot prove Stage B (Entry 14: the plateau is m-uniform and irreducibly 2D).

Norm infrastructure is elementary, matching `Defs.opNorm` (sSup over the unit sphere):
no normed-space instances are invoked.
-/

open Matrix

namespace LossyCocycles

/-! ## Elementary vector-norm kit -/

/-- The ℓ² norm of a coordinate vector, elementarily. -/
noncomputable def vnorm {n : ℕ} (v : Fin n → ℝ) : ℝ :=
  Real.sqrt (∑ i, v i ^ 2)

theorem vnorm_nonneg {n : ℕ} (v : Fin n → ℝ) : 0 ≤ vnorm v :=
  Real.sqrt_nonneg _

theorem vnorm_zero {n : ℕ} : vnorm (0 : Fin n → ℝ) = 0 := by
  unfold vnorm
  simp

theorem vnorm_neg {n : ℕ} (v : Fin n → ℝ) : vnorm (-v) = vnorm v := by
  unfold vnorm
  simp

theorem vnorm_smul {n : ℕ} (c : ℝ) (v : Fin n → ℝ) : vnorm (c • v) = |c| * vnorm v := by
  unfold vnorm
  simp only [Pi.smul_apply, smul_eq_mul, mul_pow]
  rw [← Finset.mul_sum, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

theorem single_le_vnorm {n : ℕ} (v : Fin n → ℝ) (i : Fin n) : |v i| ≤ vnorm v := by
  unfold vnorm
  rw [← Real.sqrt_sq_eq_abs]
  apply Real.sqrt_le_sqrt
  exact Finset.single_le_sum (f := fun j => v j ^ 2) (fun j _ => sq_nonneg _)
    (Finset.mem_univ i)

/-- Minkowski for the elementary ℓ² norm (via discrete Cauchy–Schwarz). -/
theorem vnorm_add_le {n : ℕ} (u v : Fin n → ℝ) : vnorm (u + v) ≤ vnorm u + vnorm v := by
  have hU : (0:ℝ) ≤ ∑ i, u i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hV : (0:ℝ) ≤ ∑ i, v i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hCS : (∑ i, u i * v i)
      ≤ Real.sqrt (∑ i, u i ^ 2) * Real.sqrt (∑ i, v i ^ 2) := by
    have h2 := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u v
    have hle : (∑ i, u i * v i) ≤ |∑ i, u i * v i| := le_abs_self _
    rw [← Real.sqrt_sq_eq_abs] at hle
    refine hle.trans ?_
    rw [← Real.sqrt_mul hU]
    exact Real.sqrt_le_sqrt h2
  have ha : Real.sqrt (∑ i, u i ^ 2) ^ 2 = ∑ i, u i ^ 2 := Real.sq_sqrt hU
  have hb : Real.sqrt (∑ i, v i ^ 2) ^ 2 = ∑ i, v i ^ 2 := Real.sq_sqrt hV
  have hexp : (∑ i, (u + v) i ^ 2)
      = (∑ i, u i ^ 2) + 2 * (∑ i, u i * v i) + (∑ i, v i ^ 2) := by
    have hpt : ∀ i, (u + v) i ^ 2 = u i ^ 2 + 2 * (u i * v i) + v i ^ 2 := by
      intro i
      simp only [Pi.add_apply]
      ring
    rw [Finset.sum_congr rfl fun i _ => hpt i, Finset.sum_add_distrib,
      Finset.sum_add_distrib, ← Finset.mul_sum]
  have hsq : (∑ i, (u + v) i ^ 2)
      ≤ (Real.sqrt (∑ i, u i ^ 2) + Real.sqrt (∑ i, v i ^ 2)) ^ 2 := by
    rw [hexp]
    nlinarith [hCS, ha, hb]
  unfold vnorm
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (by positivity)] at this

theorem vnorm_sub_le {n : ℕ} (u v : Fin n → ℝ) : vnorm (u - v) ≤ vnorm u + vnorm v := by
  have h := vnorm_add_le u (-v)
  rw [vnorm_neg] at h
  simpa [sub_eq_add_neg] using h

/-! ## Quadratic-form identities and contraction facts -/

theorem dot_self_nonneg {n : ℕ} (v : Fin n → ℝ) : 0 ≤ v ⬝ᵥ v :=
  Finset.sum_nonneg fun i _ => mul_self_nonneg _

theorem vnorm_sq_dot {n : ℕ} (v : Fin n → ℝ) : (∑ i, v i ^ 2) = v ⬝ᵥ v := by
  simp [dotProduct, sq]

/-- `(Av)·(Av) = v·(AᵀA v)` — the workhorse identity. -/
theorem mulVec_dot_self {a b : ℕ} (M : Matrix (Fin a) (Fin b) ℝ) (w : Fin b → ℝ) :
    (M *ᵥ w) ⬝ᵥ (M *ᵥ w) = w ⬝ᵥ ((Mᵀ * M) *ᵥ w) := by
  rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec]
  exact dotProduct_comm _ _

/-- Columns-orthonormal maps are isometries. -/
theorem frame_isometry {a b : ℕ} (M : Matrix (Fin a) (Fin b) ℝ) (h : Mᵀ * M = 1)
    (w : Fin b → ℝ) : vnorm (M *ᵥ w) = vnorm w := by
  unfold vnorm
  rw [vnorm_sq_dot, vnorm_sq_dot, mulVec_dot_self, h, Matrix.one_mulVec]

/-- The transpose of a columns-orthonormal map is a contraction. -/
theorem transpose_contract {a b : ℕ} (M : Matrix (Fin a) (Fin b) ℝ) (h : Mᵀ * M = 1)
    (u : Fin a → ℝ) : vnorm (Mᵀ *ᵥ u) ≤ vnorm u := by
  have hMq : (M *ᵥ (Mᵀ *ᵥ u)) ⬝ᵥ (M *ᵥ (Mᵀ *ᵥ u)) = (Mᵀ *ᵥ u) ⬝ᵥ (Mᵀ *ᵥ u) := by
    rw [mulVec_dot_self, h, Matrix.one_mulVec]
  have hu : u ⬝ᵥ (M *ᵥ (Mᵀ *ᵥ u)) = (Mᵀ *ᵥ u) ⬝ᵥ (Mᵀ *ᵥ u) := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose]
  have hu' : (M *ᵥ (Mᵀ *ᵥ u)) ⬝ᵥ u = (Mᵀ *ᵥ u) ⬝ᵥ (Mᵀ *ᵥ u) := by
    rw [dotProduct_comm, hu]
  have hpos : (0:ℝ) ≤ (u - M *ᵥ (Mᵀ *ᵥ u)) ⬝ᵥ (u - M *ᵥ (Mᵀ *ᵥ u)) :=
    dot_self_nonneg _
  have hexp : (u - M *ᵥ (Mᵀ *ᵥ u)) ⬝ᵥ (u - M *ᵥ (Mᵀ *ᵥ u))
      = u ⬝ᵥ u - (Mᵀ *ᵥ u) ⬝ᵥ (Mᵀ *ᵥ u) := by
    rw [sub_dotProduct, dotProduct_sub, dotProduct_sub, hMq, hu, hu']
    ring
  unfold vnorm
  apply Real.sqrt_le_sqrt
  rw [vnorm_sq_dot, vnorm_sq_dot]
  linarith [hpos, hexp ▸ hpos]

/-! ## The elementary operator norm: basic facts -/

theorem basis_unit {n : ℕ} (j : Fin n) :
    (∑ i, (fun i => if i = j then (1:ℝ) else 0) i ^ 2) = 1 := by
  have hpt : ∀ i : Fin n, ((fun i => if i = j then (1:ℝ) else 0) i) ^ 2
      = if i = j then (1:ℝ) else 0 := by
    intro i
    by_cases h : i = j <;> simp [h]
  rw [Finset.sum_congr rfl fun i _ => hpt i, Finset.sum_ite_eq']
  simp

theorem mulVec_basis {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (a b : Fin n) :
    (A *ᵥ (fun j => if j = b then (1:ℝ) else 0)) a = A a b := by
  simp only [Matrix.mulVec, dotProduct, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_ite_eq']
  simp

theorem opNorm_bddAbove {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
    BddAbove {c | ∃ v : Fin n → ℝ, (∑ i, v i ^ 2) = 1
      ∧ c = Real.sqrt (∑ i, (A.mulVec v i) ^ 2)} := by
  refine ⟨Real.sqrt (∑ i, ∑ j, A i j ^ 2), ?_⟩
  rintro c ⟨v, hv, rfl⟩
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro i _
  have hCS := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => A i j) v
  rw [hv, mul_one] at hCS
  calc (A.mulVec v i) ^ 2 = (∑ j, A i j * v j) ^ 2 := by
        simp [Matrix.mulVec, dotProduct]
    _ ≤ ∑ j, A i j ^ 2 := hCS

theorem vnorm_mulVec_le_opNorm {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (v : Fin n → ℝ) (hv : (∑ i, v i ^ 2) = 1) : vnorm (A *ᵥ v) ≤ opNorm A :=
  le_csSup (opNorm_bddAbove A) ⟨v, hv, rfl⟩

theorem opNorm_nonneg {n : ℕ} (hn : 0 < n) (A : Matrix (Fin n) (Fin n) ℝ) :
    0 ≤ opNorm A := by
  have h := vnorm_mulVec_le_opNorm A _ (basis_unit ⟨0, hn⟩)
  exact le_trans (vnorm_nonneg _) h

/-- The defining bound, scaled to arbitrary vectors. -/
theorem mulVec_vnorm_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (w : Fin n → ℝ)
    (hn : 0 < n) : vnorm (A *ᵥ w) ≤ opNorm A * vnorm w := by
  by_cases hw : w = 0
  · subst hw
    rw [Matrix.mulVec_zero, vnorm_zero, mul_zero]
  · have hw0 : 0 < vnorm w := by
      unfold vnorm
      apply Real.sqrt_pos.mpr
      obtain ⟨i, hi⟩ := Function.ne_iff.mp hw
      apply Finset.sum_pos' (fun j _ => sq_nonneg _)
      refine ⟨i, Finset.mem_univ i, ?_⟩
      have h := pow_pos (abs_pos.mpr hi) 2
      rwa [sq_abs] at h
    have hWnn : (0:ℝ) ≤ ∑ i, w i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
    set u : Fin n → ℝ := (vnorm w)⁻¹ • w with hu
    have hu1 : (∑ i, u i ^ 2) = 1 := by
      rw [hu]
      simp only [Pi.smul_apply, smul_eq_mul, mul_pow]
      rw [← Finset.mul_sum]
      have hsq : (∑ i, w i ^ 2) = (vnorm w) ^ 2 := by
        unfold vnorm
        rw [Real.sq_sqrt hWnn]
      rw [hsq]
      field_simp
    have hAw : vnorm w • (A *ᵥ u) = A *ᵥ w := by
      rw [hu, Matrix.mulVec_smul, smul_smul, mul_inv_cancel₀ (ne_of_gt hw0), one_smul]
    have hsplit : vnorm (A *ᵥ w) = vnorm w * vnorm (A *ᵥ u) := by
      rw [← hAw, vnorm_smul, abs_of_pos hw0]
    rw [hsplit, mul_comm (opNorm A) (vnorm w)]
    exact mul_le_mul_of_nonneg_left (vnorm_mulVec_le_opNorm A u hu1) (le_of_lt hw0)

theorem opNorm_le {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) (hc : 0 ≤ c)
    (h : ∀ v : Fin n → ℝ, (∑ i, v i ^ 2) = 1 → vnorm (A *ᵥ v) ≤ c) : opNorm A ≤ c := by
  apply Real.sSup_le _ hc
  rintro x ⟨v, hv, rfl⟩
  exact h v hv

/-! ## Edge maps of both families are vector contractions -/

variable {m d : ℕ} [NeZero m]

/-- Rotation cores transpose to the opposite rotation. -/
theorem rotCore_transpose {r : ℕ} (θ : ℝ) : (rotCore r θ)ᵀ = rotCore r (-θ) := by
  ext i j
  rw [Matrix.transpose_apply]
  by_cases hi0 : (i : ℕ) = 0
  · rw [rotCore_row_zero (-θ) hi0 j]
    by_cases hj0 : (j : ℕ) = 0
    · rw [rotCore_row_zero θ hj0 i]
      simp [hi0, hj0, Real.cos_neg]
    · by_cases hj1 : (j : ℕ) = 1
      · rw [rotCore_row_one θ hj1 i]
        simp [hi0, hj0, hj1, Real.sin_neg]
      · have hj2 : 2 ≤ (j : ℕ) := by omega
        rw [rotCore_row_two θ hj2 i]
        have : ¬(j = i) := by rw [Fin.ext_iff]; omega
        simp [hi0, hj0, hj1, this]
  · by_cases hi1 : (i : ℕ) = 1
    · rw [rotCore_row_one (-θ) hi1 j]
      by_cases hj0 : (j : ℕ) = 0
      · rw [rotCore_row_zero θ hj0 i]
        simp [hi1, hj0, Real.sin_neg]
      · by_cases hj1 : (j : ℕ) = 1
        · rw [rotCore_row_one θ hj1 i]
          simp [hi1, hj0, hj1, Real.cos_neg]
        · have hj2 : 2 ≤ (j : ℕ) := by omega
          rw [rotCore_row_two θ hj2 i]
          have : ¬(j = i) := by rw [Fin.ext_iff]; omega
          simp [hi1, hj0, hj1, this]
    · have hi2 : 2 ≤ (i : ℕ) := by omega
      rw [rotCore_row_two (-θ) hi2 j]
      by_cases hj0 : (j : ℕ) = 0
      · rw [rotCore_row_zero θ hj0 i]
        have : ¬(i = j) := by rw [Fin.ext_iff]; omega
        simp [hi0, hi1, this]
      · by_cases hj1 : (j : ℕ) = 1
        · rw [rotCore_row_one θ hj1 i]
          have : ¬(i = j) := by rw [Fin.ext_iff]; omega
          simp [hi0, hi1, this]
        · have hj2 : 2 ≤ (j : ℕ) := by omega
          rw [rotCore_row_two θ hj2 i]
          by_cases hij : i = j
          · simp [hij]
          · have hji : ¬(j = i) := fun h => hij h.symm
            simp [hij, hji]

/-- Rotation cores are orthogonal (r ≥ 2). -/
theorem rotCore_orth {r : ℕ} (hr : 1 < r) (θ : ℝ) :
    (rotCore r θ)ᵀ * rotCore r θ = 1 := by
  rw [rotCore_transpose, rotCore_mul hr]
  simp [rotCore_zero]

/-- Every flux edge map is a vector contraction. -/
theorem flux_contr (k : ℤ) (d r : ℕ) (hr : 1 < r) (hrd : r ≤ d)
    (e : TorusEdge m) (u : Fin d → ℝ) :
    vnorm ((fluxFamily (m := m) k d r e) *ᵥ u) ≤ vnorm u := by
  show vnorm ((stdFrame d r * rotCore r (fluxAngle k e) * (stdFrame d r)ᵀ) *ᵥ u)
      ≤ vnorm u
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
  rw [frame_isometry (stdFrame d r) (stdFrame_isFrame hrd),
    frame_isometry (rotCore r (fluxAngle k e)) (rotCore_orth hr _)]
  exact transpose_contract (stdFrame d r) (stdFrame_isFrame hrd) u

/-- Every flat edge map is a vector contraction. -/
theorem flat_contr {r : ℕ} (T : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hTe : ∀ e, T e = V (dst e) * (V (src e))ᵀ)
    (e : TorusEdge m) (u : Fin d → ℝ) :
    vnorm ((T e) *ᵥ u) ≤ vnorm u := by
  rw [hTe e, ← Matrix.mulVec_mulVec,
    frame_isometry (V (dst e)) (hfr (dst e))]
  exact transpose_contract (V (src e)) (hfr (src e)) u

/-! ## Telescoping along a column -/

/-- Products of contractive edge maps are contractions. -/
theorem vertProd_contr (T : EdgeFamily m d)
    (hT : ∀ e u, vnorm ((T e) *ᵥ u) ≤ vnorm u) (x : Fin m) :
    ∀ n (u : Fin d → ℝ), vnorm ((vertProd T x n) *ᵥ u) ≤ vnorm u := by
  intro n
  induction n with
  | zero =>
      intro u
      rw [show vertProd T x 0 = 1 from rfl, Matrix.one_mulVec]
  | succ p ih =>
      intro u
      rw [show vertProd T x (p + 1)
          = T ((x, (natFin p : Fin m)), false) * vertProd T x p from rfl]
      rw [← Matrix.mulVec_mulVec]
      exact le_trans (hT _ _) (ih u)

/-- The telescoping bound: edge-wise ε spreads to at most n·ε along a partial column
product (the 1D mechanism, and the whole of it — Response 25). -/
theorem vertProd_diff (hd0 : 0 < d) (T T' : EdgeFamily m d)
    (hT : ∀ e u, vnorm ((T e) *ᵥ u) ≤ vnorm u)
    (hT' : ∀ e u, vnorm ((T' e) *ᵥ u) ≤ vnorm u)
    (ε : ℝ) (hε0 : 0 ≤ ε) (hε : ∀ e, opNorm (T e - T' e) ≤ ε) (x : Fin m) :
    ∀ n (u : Fin d → ℝ),
      vnorm ((vertProd T x n - vertProd T' x n) *ᵥ u) ≤ n * ε * vnorm u := by
  intro n
  induction n with
  | zero =>
      intro u
      have h0 : vertProd T x 0 - vertProd T' x 0 = 0 := sub_self _
      rw [h0, Matrix.zero_mulVec, vnorm_zero]
      simp
  | succ p ih =>
      intro u
      set e : TorusEdge m := ((x, (natFin p : Fin m)), false) with he
      have halg : vertProd T x (p + 1) - vertProd T' x (p + 1)
          = T e * (vertProd T x p - vertProd T' x p)
            + (T e - T' e) * vertProd T' x p := by
        rw [show vertProd T x (p + 1) = T e * vertProd T x p from rfl,
          show vertProd T' x (p + 1) = T' e * vertProd T' x p from rfl]
        noncomm_ring
      rw [halg, Matrix.add_mulVec, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
      have h1 : vnorm (T e *ᵥ ((vertProd T x p - vertProd T' x p) *ᵥ u))
          ≤ p * ε * vnorm u :=
        le_trans (hT _ _) (ih u)
      have h2 : vnorm ((T e - T' e) *ᵥ ((vertProd T' x p) *ᵥ u))
          ≤ ε * vnorm u := by
        calc vnorm ((T e - T' e) *ᵥ ((vertProd T' x p) *ᵥ u))
            ≤ opNorm (T e - T' e) * vnorm ((vertProd T' x p) *ᵥ u) :=
              mulVec_vnorm_le _ _ hd0
          _ ≤ ε * vnorm ((vertProd T' x p) *ᵥ u) :=
              mul_le_mul_of_nonneg_right (hε e) (vnorm_nonneg _)
          _ ≤ ε * vnorm u :=
              mul_le_mul_of_nonneg_left (vertProd_contr T' hT' x p u) hε0
      calc vnorm (T e *ᵥ ((vertProd T x p - vertProd T' x p) *ᵥ u)
              + (T e - T' e) *ᵥ ((vertProd T' x p) *ᵥ u))
          ≤ vnorm (T e *ᵥ ((vertProd T x p - vertProd T' x p) *ᵥ u))
            + vnorm ((T e - T' e) *ᵥ ((vertProd T' x p) *ᵥ u)) := vnorm_add_le _ _
        _ ≤ p * ε * vnorm u + ε * vnorm u := add_le_add h1 h2
        _ = (p + 1 : ℕ) * ε * vnorm u := by push_cast; ring

/-! ## Stage A — the existence bound -/

/-- The cycle-difference entry bound: edge-wise ε bounds every entry of the column-cycle
difference by m·ε. -/
theorem vertCycle_diff_entry (hd0 : 0 < d) (T T' : EdgeFamily m d)
    (hT : ∀ e u, vnorm ((T e) *ᵥ u) ≤ vnorm u)
    (hT' : ∀ e u, vnorm ((T' e) *ᵥ u) ≤ vnorm u)
    (ε : ℝ) (hε0 : 0 ≤ ε) (hε : ∀ e, opNorm (T e - T' e) ≤ ε) (x : Fin m)
    (a b : Fin d) :
    |(vertCycle T x - vertCycle T' x) a b| ≤ m * ε := by
  have hbasis := basis_unit (n := d) b
  have h := vertProd_diff hd0 T T' hT hT' ε hε0 hε x m
    (fun j => if j = b then (1:ℝ) else 0)
  have hb1 : vnorm (fun j => if j = b then (1:ℝ) else 0) = 1 := by
    unfold vnorm
    rw [hbasis, Real.sqrt_one]
  rw [hb1, mul_one] at h
  have hentry : (vertCycle T x - vertCycle T' x) a b
      = ((vertCycle T x - vertCycle T' x) *ᵥ (fun j => if j = b then (1:ℝ) else 0)) a :=
    (mulVec_basis _ a b).symm
  rw [hentry]
  exact le_trans (single_le_vnorm _ a) h

/-- **Stage A (existence, quantitative).** For m > 2|k| and k ≠ 0, every rank-r flat
family is at worst-edge operator distance ≥ |sin(2πk/m)|/m from the charge-k flux
family. The constant decays like 1/m² — Stage A's mechanism is the 1D telescoping
bound and cannot reach the m-uniform Stage B (Entry 14). -/
theorem stageA_skew_bound (k : ℤ) (d r : ℕ) (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (hk : k ≠ 0) (hm : 2 * k.natAbs < m) :
    |Real.sin (2 * Real.pi * k / m)| / m
      ≤ distOpFlat (m := m) r (fluxFamily k d r) := by
  have hd0 : 0 < d := by omega
  have hm3 : 3 ≤ m := by
    have : 1 ≤ k.natAbs := Int.natAbs_pos.mpr hk
    omega
  have hm0 : 0 < m := by omega
  have hmR : (0:ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  unfold distOpFlat
  apply le_csInf
  · -- the constraint set is nonempty: the constant standard-frame family is flat
    -- and within operator distance 2 of every edge
    refine ⟨2, fun _ => stdFrame d r * (stdFrame d r)ᵀ,
      ⟨fun _ => stdFrame d r, fun _ => stdFrame_isFrame hrd, fun _ => rfl⟩, ?_⟩
    intro e
    apply opNorm_le _ _ (by norm_num)
    intro v hv
    have hv1 : vnorm v = 1 := by
      unfold vnorm
      rw [hv, Real.sqrt_one]
    rw [Matrix.sub_mulVec]
    calc vnorm (fluxFamily (m := m) k d r e *ᵥ v
            - (stdFrame d r * (stdFrame d r)ᵀ) *ᵥ v)
        ≤ vnorm (fluxFamily (m := m) k d r e *ᵥ v)
          + vnorm ((stdFrame d r * (stdFrame d r)ᵀ) *ᵥ v) := vnorm_sub_le _ _
      _ ≤ vnorm v + vnorm v := by
          apply add_le_add (flux_contr k d r hr hrd e v)
          rw [← Matrix.mulVec_mulVec,
            frame_isometry (stdFrame d r) (stdFrame_isFrame hrd)]
          exact transpose_contract (stdFrame d r) (stdFrame_isFrame hrd) v
      _ = 2 := by rw [hv1]; norm_num
  · -- every feasible c dominates the skew obstruction
    rintro c ⟨T', ⟨V, hfr, hTe⟩, hc⟩
    have hε0 : 0 ≤ c :=
      le_trans (opNorm_nonneg hd0 _) (hc ((0, 0), false))
    set x₁ : Fin m := ⟨1, by omega⟩ with hx₁
    -- skew of the flux column cycle at column 1 is 2·sin(2πk/m)
    have hsflux : (coreVec (coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁))).2
        = 2 * Real.sin (2 * Real.pi * k / m) := by
      rw [flux_vertCycle k d r hr hrd x₁, coreVec_conj_rot hd hr hrd]
      norm_num
    -- skew of the flat column cycle is 0
    have hsflat : (coreVec (coreBlock hd (vertCycle T' x₁))).2 = 0 := by
      rw [flat_vertCycle (Nat.one_le_iff_ne_zero.mpr (NeZero.ne m)) T' V hfr hTe x₁]
      exact (coreVec_proj hd (V (x₁, 0))).1
    -- the two core entries of the difference are bounded by m·c
    have hΔ := vertCycle_diff_entry hd0 (fluxFamily (m := m) k d r) T'
      (flux_contr k d r hr hrd) (flat_contr T' V hfr hTe) c hε0 hc x₁
    -- assemble: skew is two entries of the cycle difference; |2 sin| ≤ 2 m c
    have hentry : ∀ (i j : Fin 2),
        coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁) i j
          - coreBlock hd (vertCycle T' x₁) i j
        = (vertCycle (fluxFamily (m := m) k d r) x₁ - vertCycle T' x₁)
            ⟨(i : ℕ), lt_of_le_of_lt (Nat.le_of_lt_succ i.isLt) hd⟩
            ⟨(j : ℕ), lt_of_le_of_lt (Nat.le_of_lt_succ j.isLt) hd⟩ := by
      intro i j
      rw [Matrix.sub_apply]
      rfl
    have hS : 2 * Real.sin (2 * Real.pi * k / m)
        = (coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁) 1 0
            - coreBlock hd (vertCycle T' x₁) 1 0)
          - (coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁) 0 1
            - coreBlock hd (vertCycle T' x₁) 0 1) := by
      have h1 : (coreVec (coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁))).2
          = coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁) 1 0
            - coreBlock hd (vertCycle (fluxFamily (m := m) k d r) x₁) 0 1 := rfl
      have h2 : (coreVec (coreBlock hd (vertCycle T' x₁))).2
          = coreBlock hd (vertCycle T' x₁) 1 0
            - coreBlock hd (vertCycle T' x₁) 0 1 := rfl
      linarith [hsflux, hsflat, h1, h2]
    have hbound : |2 * Real.sin (2 * Real.pi * k / m)| ≤ m * c + m * c := by
      rw [hS, hentry 1 0, hentry 0 1]
      exact le_trans (abs_sub _ _) (add_le_add (hΔ _ _) (hΔ _ _))
    -- divide by 2m
    rw [abs_mul, abs_two] at hbound
    have hsin : |Real.sin (2 * Real.pi * k / m)| ≤ m * c := by linarith
    rw [div_le_iff₀ hmR]
    calc |Real.sin (2 * Real.pi * k / m)| ≤ m * c := hsin
      _ = c * m := mul_comm _ _

/-- **Stage A (existence, positivity).** The flux family is at positive operator
distance from every flat family, at every fixed size. -/
theorem stageA_pos (k : ℤ) (d r : ℕ) (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (hk : k ≠ 0) (hm : 2 * k.natAbs < m) :
    0 < distOpFlat (m := m) r (fluxFamily k d r) := by
  have hm0 : 0 < m := by omega
  have hmR : (0:ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm0
  have hπ := Real.pi_pos
  refine lt_of_lt_of_le ?_ (stageA_skew_bound k d r hd hr hrd hk hm)
  apply div_pos _ hmR
  apply abs_pos.mpr
  -- sin(2πk/m) ≠ 0 because 0 < 2π|k|/m < π
  have hkm : 2 * ((k.natAbs : ℕ) : ℝ) < ((m : ℕ) : ℝ) := by exact_mod_cast hm
  have hka : (1:ℝ) ≤ ((k.natAbs : ℕ) : ℝ) := by
    have : 1 ≤ k.natAbs := Int.natAbs_pos.mpr hk
    exact_mod_cast this
  rcases lt_or_gt_of_ne (fun h => hk (by exact_mod_cast h) : (k:ℝ) ≠ 0) with hneg | hpos
  · -- k < 0: sin(2πk/m) = −sin(2π|k|/m) < 0
    have habs : ((k.natAbs : ℕ) : ℝ) = -(k:ℝ) := by
      rw [Nat.cast_natAbs, Int.cast_abs]
      exact abs_of_neg hneg
    have harg : 2 * Real.pi * k / m = -(2 * Real.pi * (k.natAbs : ℕ) / m) := by
      rw [habs]; ring
    rw [harg, Real.sin_neg]
    apply neg_ne_zero.mpr
    apply ne_of_gt
    apply Real.sin_pos_of_pos_of_lt_pi
    · positivity
    · rw [div_lt_iff₀ hmR]
      nlinarith
  · -- k > 0: sin(2πk/m) > 0
    apply ne_of_gt
    apply Real.sin_pos_of_pos_of_lt_pi
    · positivity
    · rw [div_lt_iff₀ hmR]
      have habs : ((k.natAbs : ℕ) : ℝ) = (k:ℝ) := by
        rw [Nat.cast_natAbs, Int.cast_abs]
        exact abs_of_pos hpos
      nlinarith

end LossyCocycles

