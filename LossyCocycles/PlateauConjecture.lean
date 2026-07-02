import LossyCocycles.MagPair
import LossyCocycles.StageB

/-! # LossyCocycles.PlateauConjecture — the conjecture, reduced to its open core

DEBT-2 PAID (2026-06-11): the conjecture is stated over the real L4 definitions —
the explicit Landau-gauge flux family and the elementary operator-norm distance to the
Lemma-1 flat family.

REDUCTION LANDED (2026-07-01): the conjecture is no longer a monolithic `sorry`. The
proved theorem `plateau_of_rigidity` shows `PlateauStatement` follows from a single
clearly-scoped estimate, `FluxPairRigidity` — transported Exel–Loring rigidity for the
magnetic-translation pair. The only `sorry` in the library is that estimate.

What the reduction settles about §6.2.1 of the paper draft:
* **Gap 2 (distance translation) is NOT load-bearing for the lower bound.** Only the
  easy direction is needed: `‖magU T − magU T′‖ ≤ supₑ ‖T e − T′ e‖`, proved here
  m-freely as `opNormG_magU_le` (single-translation locality, the same mechanism as
  `opNormG_magComm_le`). The hard converse direction never enters: rigidity bounds the
  distance to *every* commuting contraction pair from below, flat families give exact
  commuting contraction pairs (`magComm_flat` + `opNormG_magU_flat_le_one`), and the
  locality bound turns any edge-wise flat approximation into a pair-level one.
* **Gap 1 (non-unitary transport) is the whole remaining problem**, isolated as
  `FluxPairRigidity` below.

Empirical status (e2v2 + kill test): plateau 1.29–1.49 at k = 1, r ∈ {2,3,4}, m ≤ 8,
surviving oracle-initialized adversarial overfitting. Codimension caveat (paper5 §4a,
2026-07-01): those fit values are UPPER bounds from optimizers that miss trivial
basins; at codim = d−r ≥ 2 the true distOpFlat is exactly 1 (hand-built orthogonal
competitor ≤ 1 for all r; ≥ 1 proved for r = 2), and at the pair level the zero pair
caps the FluxPairRigidity constant at δ(k) ≤ 1. The conjecture's content is strict
positivity, uniformly in m, r — not any particular height.
-/

open Matrix

namespace LossyCocycles

variable {m d : ℕ} [NeZero m]

/-- **The Plateau Conjecture.** For every nonzero charge k there is c(k) > 0 such that,
for every torus size m ≥ 3 and every rank 2 ≤ r ≤ d, the charge-k flux family is at
operator distance ≥ c(k) from every rank-r flat family — uniformly in m and r:
the index obstruction survives rank loss. -/
def PlateauStatement : Prop :=
  ∀ k : ℤ, k ≠ 0 → ∃ c : ℝ, 0 < c ∧
    ∀ m : ℕ, ∀ _ : NeZero m, 3 ≤ m →
    ∀ d r : ℕ, 2 ≤ r → r ≤ d →
      c ≤ distOpFlat (m := m) r (fluxFamily k d r)

/-! ## Single-translation locality (m-free)

`magU`/`magV` are entrywise linear in the edge family, and each acts block-wise through
one shifted site — so a sup bound on the edge blocks is a bound on the assembled
translation, never amplified by system size. Same mechanism as `opNormG_magComm_le`. -/

theorem magU_apply (T : EdgeFamily m d) (p : MagIx m d) (s : Fin m × Fin m) (j : Fin d) :
    magU T p (s, j) = if p.1 = (s.1 + 1, s.2) then T (s, true) p.2 j else 0 := rfl

theorem magV_apply (T : EdgeFamily m d) (p : MagIx m d) (s : Fin m × Fin m) (j : Fin d) :
    magV T p (s, j) = if p.1 = (s.1, s.2 + 1) then T (s, false) p.2 j else 0 := rfl

/-- `magU` is entrywise linear in the family: differences pass inside. -/
theorem magU_sub (T T' : EdgeFamily m d) :
    magU T - magU T' = magU fun e => T e - T' e := by
  ext p q
  obtain ⟨s, j⟩ := q
  rw [Matrix.sub_apply, magU_apply, magU_apply, magU_apply]
  by_cases h : p.1 = (s.1 + 1, s.2)
  · rw [if_pos h, if_pos h, if_pos h, Matrix.sub_apply]
  · rw [if_neg h, if_neg h, if_neg h, sub_zero]

/-- `magV` is entrywise linear in the family: differences pass inside. -/
theorem magV_sub (T T' : EdgeFamily m d) :
    magV T - magV T' = magV fun e => T e - T' e := by
  ext p q
  obtain ⟨s, j⟩ := q
  rw [Matrix.sub_apply, magV_apply, magV_apply, magV_apply]
  by_cases h : p.1 = (s.1, s.2 + 1)
  · rw [if_pos h, if_pos h, if_pos h, Matrix.sub_apply]
  · rw [if_neg h, if_neg h, if_neg h, sub_zero]

/-- `magU`'s action computes block-wise through the left-shifted site. -/
theorem magU_mulVec (T : EdgeFamily m d) (ψ : MagIx m d → ℝ) (p : MagIx m d) :
    (magU T *ᵥ ψ) p
      = (T ((p.1.1 - 1, p.1.2), true) *ᵥ fun j => ψ ((p.1.1 - 1, p.1.2), j)) p.2 := by
  show (∑ q, magU T p q * ψ q) = _
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_eq_single ((p.1.1 - 1, p.1.2) : Fin m × Fin m)]
  · have hcond : p.1 = ((p.1.1 - 1 : Fin m) + 1, p.1.2) := by
      apply Prod.ext <;> simp
    have hterm : ∀ j : Fin d,
        magU T p ((p.1.1 - 1, p.1.2), j) * ψ ((p.1.1 - 1, p.1.2), j)
          = T ((p.1.1 - 1, p.1.2), true) p.2 j * ψ ((p.1.1 - 1, p.1.2), j) := by
      intro j
      rw [magU_apply, if_pos hcond]
    rw [Finset.sum_congr rfl fun j _ => hterm j]
    rfl
  · intro s _ hs
    apply Finset.sum_eq_zero
    intro j _
    have hns : ¬(p.1 = (s.1 + 1, s.2)) := by
      intro h
      apply hs
      have h1 : p.1.1 = s.1 + 1 := by rw [h]
      have h2 : p.1.2 = s.2 := by rw [h]
      apply Prod.ext
      · simp [h1]
      · simp [h2]
    rw [magU_apply, if_neg hns, zero_mul]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- `magV`'s action computes block-wise through the down-shifted site. -/
theorem magV_mulVec (T : EdgeFamily m d) (ψ : MagIx m d → ℝ) (p : MagIx m d) :
    (magV T *ᵥ ψ) p
      = (T ((p.1.1, p.1.2 - 1), false) *ᵥ fun j => ψ ((p.1.1, p.1.2 - 1), j)) p.2 := by
  show (∑ q, magV T p q * ψ q) = _
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_eq_single ((p.1.1, p.1.2 - 1) : Fin m × Fin m)]
  · have hcond : p.1 = (p.1.1, (p.1.2 - 1 : Fin m) + 1) := by
      apply Prod.ext <;> simp
    have hterm : ∀ j : Fin d,
        magV T p ((p.1.1, p.1.2 - 1), j) * ψ ((p.1.1, p.1.2 - 1), j)
          = T ((p.1.1, p.1.2 - 1), false) p.2 j * ψ ((p.1.1, p.1.2 - 1), j) := by
      intro j
      rw [magV_apply, if_pos hcond]
    rw [Finset.sum_congr rfl fun j _ => hterm j]
    rfl
  · intro s _ hs
    apply Finset.sum_eq_zero
    intro j _
    have hns : ¬(p.1 = (s.1, s.2 + 1)) := by
      intro h
      apply hs
      have h1 : p.1.1 = s.1 := by rw [h]
      have h2 : p.1.2 = s.2 + 1 := by rw [h]
      apply Prod.ext
      · simp [h1]
      · simp [h2]
    rw [magV_apply, if_neg hns, zero_mul]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- The shift-by-(1,0) self-equivalence of the torus sites. -/
def shiftX : (Fin m × Fin m) ≃ (Fin m × Fin m) where
  toFun s := (s.1 - 1, s.2)
  invFun s := (s.1 + 1, s.2)
  left_inv s := by simp
  right_inv s := by simp

/-- The shift-by-(0,1) self-equivalence of the torus sites. -/
def shiftY : (Fin m × Fin m) ≃ (Fin m × Fin m) where
  toFun s := (s.1, s.2 - 1)
  invFun s := (s.1, s.2 + 1)
  left_inv s := by simp
  right_inv s := by simp

/-- **Horizontal locality (m-free).** A sup bound on the edge blocks bounds `magU`. -/
theorem opNormG_magU_le (hd0 : 0 < d) (T : EdgeFamily m d) (C : ℝ) (hC0 : 0 ≤ C)
    (hC : ∀ e, opNorm (T e) ≤ C) : opNormG (magU T) ≤ C := by
  apply opNormG_le _ _ hC0
  intro ψ hψ
  unfold vnormG
  have hrow := magU_mulVec T ψ
  calc Real.sqrt (∑ p, ((magU T *ᵥ ψ) p) ^ 2)
      = Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          ((T ((s'.1 - 1, s'.2), true) *ᵥ fun j => ψ ((s'.1 - 1, s'.2), j)) i) ^ 2) := by
        rw [Fintype.sum_prod_type]
        congr 1
        apply Finset.sum_congr rfl
        intro s' _
        apply Finset.sum_congr rfl
        intro i _
        rw [hrow (s', i)]
    _ ≤ Real.sqrt (∑ s' : Fin m × Fin m, C ^ 2 * ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2), i)) ^ 2) := by
        apply Real.sqrt_le_sqrt
        apply Finset.sum_le_sum
        intro s' _
        have hb := mulVec_vnorm_le (T ((s'.1 - 1, s'.2), true))
          (fun j => ψ ((s'.1 - 1, s'.2), j)) hd0
        have hb2 : vnorm (T ((s'.1 - 1, s'.2), true) *ᵥ fun j => ψ ((s'.1 - 1, s'.2), j))
            ≤ C * vnorm (fun j => ψ ((s'.1 - 1, s'.2), j)) := by
          refine le_trans hb ?_
          exact mul_le_mul_of_nonneg_right (hC _) (vnorm_nonneg _)
        have hsq := pow_le_pow_left₀ (vnorm_nonneg _) hb2 2
        unfold vnorm at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        rw [mul_pow] at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        exact hsq
    _ = Real.sqrt (C ^ 2 * ∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2), i)) ^ 2) := by
        rw [← Finset.mul_sum]
    _ = C * Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1 - 1, s'.2), i)) ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg C), Real.sqrt_sq hC0]
    _ = C * Real.sqrt (∑ s : Fin m × Fin m, ∑ i : Fin d, (ψ (s, i)) ^ 2) := by
        congr 1
        conv_rhs => rw [← Equiv.sum_comp (shiftX (m := m))
          (fun s => ∑ i : Fin d, (ψ (s, i)) ^ 2)]
        rfl
    _ = C := by
        have hsum : (∑ p : MagIx m d, ψ p ^ 2)
            = ∑ s : Fin m × Fin m, ∑ i : Fin d, ψ (s, i) ^ 2 :=
          Fintype.sum_prod_type (fun p : MagIx m d => ψ p ^ 2)
        rw [← hsum, hψ, Real.sqrt_one, mul_one]

/-- **Vertical locality (m-free).** A sup bound on the edge blocks bounds `magV`. -/
theorem opNormG_magV_le (hd0 : 0 < d) (T : EdgeFamily m d) (C : ℝ) (hC0 : 0 ≤ C)
    (hC : ∀ e, opNorm (T e) ≤ C) : opNormG (magV T) ≤ C := by
  apply opNormG_le _ _ hC0
  intro ψ hψ
  unfold vnormG
  have hrow := magV_mulVec T ψ
  calc Real.sqrt (∑ p, ((magV T *ᵥ ψ) p) ^ 2)
      = Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          ((T ((s'.1, s'.2 - 1), false) *ᵥ fun j => ψ ((s'.1, s'.2 - 1), j)) i) ^ 2) := by
        rw [Fintype.sum_prod_type]
        congr 1
        apply Finset.sum_congr rfl
        intro s' _
        apply Finset.sum_congr rfl
        intro i _
        rw [hrow (s', i)]
    _ ≤ Real.sqrt (∑ s' : Fin m × Fin m, C ^ 2 * ∑ i : Fin d,
          (ψ ((s'.1, s'.2 - 1), i)) ^ 2) := by
        apply Real.sqrt_le_sqrt
        apply Finset.sum_le_sum
        intro s' _
        have hb := mulVec_vnorm_le (T ((s'.1, s'.2 - 1), false))
          (fun j => ψ ((s'.1, s'.2 - 1), j)) hd0
        have hb2 : vnorm (T ((s'.1, s'.2 - 1), false) *ᵥ fun j => ψ ((s'.1, s'.2 - 1), j))
            ≤ C * vnorm (fun j => ψ ((s'.1, s'.2 - 1), j)) := by
          refine le_trans hb ?_
          exact mul_le_mul_of_nonneg_right (hC _) (vnorm_nonneg _)
        have hsq := pow_le_pow_left₀ (vnorm_nonneg _) hb2 2
        unfold vnorm at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        rw [mul_pow] at hsq
        rw [Real.sq_sqrt (Finset.sum_nonneg fun j _ => sq_nonneg _)] at hsq
        exact hsq
    _ = Real.sqrt (C ^ 2 * ∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1, s'.2 - 1), i)) ^ 2) := by
        rw [← Finset.mul_sum]
    _ = C * Real.sqrt (∑ s' : Fin m × Fin m, ∑ i : Fin d,
          (ψ ((s'.1, s'.2 - 1), i)) ^ 2) := by
        rw [Real.sqrt_mul (sq_nonneg C), Real.sqrt_sq hC0]
    _ = C * Real.sqrt (∑ s : Fin m × Fin m, ∑ i : Fin d, (ψ (s, i)) ^ 2) := by
        congr 1
        conv_rhs => rw [← Equiv.sum_comp (shiftY (m := m))
          (fun s => ∑ i : Fin d, (ψ (s, i)) ^ 2)]
        rfl
    _ = C := by
        have hsum : (∑ p : MagIx m d, ψ p ^ 2)
            = ∑ s : Fin m × Fin m, ∑ i : Fin d, ψ (s, i) ^ 2 :=
          Fintype.sum_prod_type (fun p : MagIx m d => ψ p ^ 2)
        rw [← hsum, hψ, Real.sqrt_one, mul_one]

/-! ## Flat families give exactly-commuting contraction pairs -/

/-- Flat-family edge maps are operator contractions (frame-pair maps contract). -/
theorem opNorm_flatEdge_le_one {r : ℕ} (T' : EdgeFamily m d)
    (hT' : IsFlatFamily r T') (e : TorusEdge m) : opNorm (T' e) ≤ 1 := by
  obtain ⟨V, hfr, hTe⟩ := hT'
  apply opNorm_le _ _ zero_le_one
  intro v hv
  rw [hTe]
  have hv1 : vnorm v = 1 := by unfold vnorm; rw [hv, Real.sqrt_one]
  have h := frames_pair_contr (V (dst e)) (V (src e)) (hfr _) (hfr _) v
  rw [hv1] at h
  exact h

/-- The horizontal translation of a flat family is a contraction. -/
theorem opNormG_magU_flat_le_one {r : ℕ} (hd0 : 0 < d) (T' : EdgeFamily m d)
    (hT' : IsFlatFamily r T') : opNormG (magU T') ≤ 1 :=
  opNormG_magU_le hd0 T' 1 zero_le_one fun e => opNorm_flatEdge_le_one T' hT' e

/-- The vertical translation of a flat family is a contraction. -/
theorem opNormG_magV_flat_le_one {r : ℕ} (hd0 : 0 < d) (T' : EdgeFamily m d)
    (hT' : IsFlatFamily r T') : opNormG (magV T') ≤ 1 :=
  opNormG_magV_le hd0 T' 1 zero_le_one fun e => opNorm_flatEdge_le_one T' hT' e

/-! ## The open core, scoped -/

/-- **The open estimate: transported Exel–Loring rigidity for the flux pair.**
For each nonzero charge k there is a δ(k) > 0 — independent of m, d, r — such that the
charge-k magnetic-translation pair is at operator distance ≥ δ(k) from *every* commuting
pair of contractions.

This is the contraction-pair transport of the classical dimension-free Bott rigidity
[Voiculescu; Exel–Loring; quantitative form in Hastings–Loring]: the pair almost
commutes (`opNormG_magComm_flux`: commutator ≤ 2|sin(πk/m²)|, m-uniform) while carrying
index k (`totalFlux_flux`: total flux 2πk, machine-checked for all 2 ≤ r ≤ d). The
classical statement is for unitary pairs; here 𝒰, 𝒱 are rank-r partial isometries —
the non-unitary transport is §6.2.1 gap 1 of the paper draft, the load-bearing open
step (= Stage B of the dilation gate, Round 5). Note the transport is only *expected*
to give this for m ≥ m₀(k) (commutator inside the ε₀-window); quantifying over all
m ≥ 3 additionally asserts the finitely many small-m cases, which are a finite check
once m₀(k) is explicit. -/
def FluxPairRigidity : Prop :=
  ∀ k : ℤ, k ≠ 0 → ∃ δ : ℝ, 0 < δ ∧
    ∀ m : ℕ, ∀ _ : NeZero m, 3 ≤ m →
    ∀ d r : ℕ, 2 ≤ r → r ≤ d →
    ∀ A B : Matrix (MagIx m d) (MagIx m d) ℝ,
      opNormG A ≤ 1 → opNormG B ≤ 1 → A * B = B * A →
      δ ≤ max (opNormG (magU (fluxFamily k d r) - A))
              (opNormG (magV (fluxFamily k d r) - B))

/-- **The reduction, machine-checked: pair rigidity implies the Plateau Conjecture,
with the same constant.** Contrapositive mechanism: an edge-wise c-close flat family
assembles (single-translation locality, m-free) into a commuting contraction pair
c-close to the flux pair, so c ≥ δ. Only the *easy* direction of the §6.2.1 distance
translation is used — the hard converse is not load-bearing for the lower bound. -/
theorem plateau_of_rigidity (h : FluxPairRigidity) : PlateauStatement := by
  intro k hk
  obtain ⟨δ, hδ0, hδ⟩ := h k hk
  refine ⟨δ, hδ0, ?_⟩
  intro m hm hm3 d r hr2 hrd
  have hd0 : 0 < d := by omega
  have hr1 : 1 < r := by omega
  unfold distOpFlat
  apply le_csInf
  · -- the distance set is nonempty: the constant standard-frame flat family is within 2
    refine ⟨2, fun _ => stdFrame d r * (stdFrame d r)ᵀ,
      ⟨fun _ => stdFrame d r, fun _ => stdFrame_isFrame hrd, fun _ => rfl⟩, ?_⟩
    intro e
    have hrw : stdFrame d r * (stdFrame d r)ᵀ
        = stdFrame d r * rotCore r 0 * (stdFrame d r)ᵀ := by
      rw [rotCore_zero, Matrix.mul_one]
    show opNorm (stdFrame d r * rotCore r (fluxAngle (m := m) k e) * (stdFrame d r)ᵀ
        - stdFrame d r * (stdFrame d r)ᵀ) ≤ 2
    rw [hrw]
    refine le_trans (opNorm_conj_rotDiff hr1 hrd _ _) ?_
    have hs : |Real.sin ((fluxAngle (m := m) k e - 0) / 2)| ≤ 1 :=
      abs_le.mpr ⟨Real.neg_one_le_sin _, Real.sin_le_one _⟩
    linarith
  · rintro c ⟨T', hT'flat, hT'close⟩
    have hc0 : 0 ≤ c :=
      le_trans (opNorm_nonneg hd0 _) (hT'close ((0, 0), true))
    obtain ⟨V, hfr, hTe⟩ := hT'flat
    have hcomm : magU T' * magV T' = magV T' * magU T' :=
      sub_eq_zero.mp (magComm_flat T' V hfr hTe)
    have hUc : opNormG (magU T') ≤ 1 :=
      opNormG_magU_flat_le_one hd0 T' ⟨V, hfr, hTe⟩
    have hVc : opNormG (magV T') ≤ 1 :=
      opNormG_magV_flat_le_one hd0 T' ⟨V, hfr, hTe⟩
    have hrig := hδ m hm hm3 d r hr2 hrd (magU T') (magV T') hUc hVc hcomm
    have hU : opNormG (magU (fluxFamily k d r) - magU T') ≤ c := by
      rw [magU_sub]
      exact opNormG_magU_le hd0 _ c hc0 fun e => hT'close e
    have hV : opNormG (magV (fluxFamily k d r) - magV T') ≤ c := by
      rw [magV_sub]
      exact opNormG_magV_le hd0 _ c hc0 fun e => hT'close e
    exact le_trans hrig (max_le hU hV)

/-- The lone open estimate — where the library's only `sorry` lives. -/
theorem flux_pair_rigidity : FluxPairRigidity := by
  sorry

/-- The conjecture, now a corollary of the scoped estimate. -/
theorem plateau_conjecture : PlateauStatement :=
  plateau_of_rigidity flux_pair_rigidity

end LossyCocycles
