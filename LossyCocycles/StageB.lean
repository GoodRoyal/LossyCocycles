import LossyCocycles.StageA

/-! # LossyCocycles.StageB — the m-uniform plateau (the genuine open frontier)

Stage A (in `StageA.lean`, fused from the redo branch) gives existence *unconditionally*
per fixed m, with the explicit constant |sin(2πk/m)|/m. Stage B is the QUANTITATIVE,
m-UNIFORM bound c(k) — the part the m-scaling experiment localized as irreducibly
2-dimensional (a 1-D argument gives only γ/m; Stage A's constant decays like 1/m²,
which is the same boundary seen formally).

This file keeps the Opus-arc architecture (Response 31, decision D5) with its two named
analytic Props, and — post-fusion — closes the two mechanical debts that were sorried:
`fluxFamily_coreGap` (DEBT-B3, one step from `coreVec_conj_rot`) and the `stageB`
combination logic (DEBT-B1). With the redo branch's `windingSum_flux` discharging the
flux-winding hypothesis, `plateau_of_props` states the net result — sharpened in Response 36: **the m-uniform
plateau holds modulo exactly ONE Prop, `UnifGapLiftCont`** (gap stability for the flux
family is now the theorem `flux_gapStable`).
-/

open Matrix
namespace LossyCocycles

variable {d r : ℕ}

/-- Core gap is monotone in the threshold. -/
theorem coreGap_mono {γ γ' : ℝ} (hle : γ' ≤ γ) {m : ℕ} [NeZero m] {hd : 1 < d}
    {T : EdgeFamily m d} (h : CoreGap γ hd T) : CoreGap γ' hd T :=
  fun x => le_trans hle (h x)

/-! ## Gap stability for the flux family — PROVED (Response 36)

The conjectured `GapStable` Prop is not needed: for the flux family, gap stability is a
THEOREM, m-uniformly, via source projections. A flux edge satisfies `EᵀE = W·Wᵀ`
*exactly*; a flat edge satisfies `E'ᵀE' = V_src·V_srcᵀ` *exactly*; and `EᵀE` is
2ε-close to `E'ᵀE'` whenever the edges are ε-close — a PER-EDGE transfer with no
m-amplification (the telescoping channel never opens because the projections are pinned
edge-locally, not assembled along the column). Hence every vertex projection of a flat
family near flux keeps core mass ≥ 2 − 4ε, and the flat cycle (a projection by
`flat_vertCycle`) keeps CoreGap 1 for ε ≤ 1/4. -/

/-- Conjugation by a frame of an orthogonal core is a vector contraction. -/
theorem conj_orth_contr {a b : ℕ} (M : Matrix (Fin a) (Fin b) ℝ) (hM : Mᵀ * M = 1)
    (A : Matrix (Fin b) (Fin b) ℝ) (hA : Aᵀ * A = 1) (u : Fin a → ℝ) :
    vnorm ((M * A * Mᵀ) *ᵥ u) ≤ vnorm u := by
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, frame_isometry M hM,
    frame_isometry A hA]
  exact transpose_contract M hM u

/-- A pair-of-frames map `V₁·V₂ᵀ` is a vector contraction. -/
theorem frames_pair_contr {a b : ℕ} (V₁ V₂ : Matrix (Fin a) (Fin b) ℝ)
    (h₁ : V₁ᵀ * V₁ = 1) (h₂ : V₂ᵀ * V₂ = 1) (u : Fin a → ℝ) :
    vnorm ((V₁ * V₂ᵀ) *ᵥ u) ≤ vnorm u := by
  rw [← Matrix.mulVec_mulVec, frame_isometry V₁ h₁]
  exact transpose_contract V₂ h₂ u

/-- Entry-wise closeness of source projections from operator closeness of the maps —
the m-uniform transfer: `|EᵀE − E'ᵀE'| ≤ 2ε` entry-wise when `‖E − E'‖ ≤ ε` and both
transposes are contractions. -/
theorem srcProj_entry_close {d : ℕ} (hd0 : 0 < d) (E E' : Matrix (Fin d) (Fin d) ℝ)
    (ε : ℝ) (hop : opNorm (E - E') ≤ ε)
    (hEt : ∀ u, vnorm (Eᵀ *ᵥ u) ≤ vnorm u) (hE't : ∀ u, vnorm (E'ᵀ *ᵥ u) ≤ vnorm u)
    (i j : Fin d) : |(Eᵀ * E - E'ᵀ * E') i j| ≤ 2 * ε := by
  have halg : Eᵀ * E - E'ᵀ * E' = Eᵀ * (E - E') + (E'ᵀ * (E - E'))ᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_sub, Matrix.transpose_transpose]
    noncomm_ring
  have hbasis : vnorm (fun l => if l = j then (1:ℝ) else 0) = 1 := by
    unfold vnorm
    rw [basis_unit, Real.sqrt_one]
  have hbasis' : ∀ (jj : Fin d), vnorm (fun l => if l = jj then (1:ℝ) else 0) = 1 := by
    intro jj
    unfold vnorm
    rw [basis_unit, Real.sqrt_one]
  have hterm : ∀ (X : Matrix (Fin d) (Fin d) ℝ)
      (hXt : ∀ u, vnorm (Xᵀ *ᵥ u) ≤ vnorm u) (a bb : Fin d),
      |(Xᵀ * (E - E')) a bb| ≤ ε := by
    intro X hXt a bb
    have h1 : (Xᵀ * (E - E')) a bb
        = ((Xᵀ * (E - E')) *ᵥ (fun l => if l = bb then (1:ℝ) else 0)) a :=
      (mulVec_basis _ a bb).symm
    rw [h1, ← Matrix.mulVec_mulVec]
    calc |(Xᵀ *ᵥ ((E - E') *ᵥ fun l => if l = bb then (1:ℝ) else 0)) a|
        ≤ vnorm (Xᵀ *ᵥ ((E - E') *ᵥ fun l => if l = bb then (1:ℝ) else 0)) :=
          single_le_vnorm _ a
      _ ≤ vnorm ((E - E') *ᵥ fun l => if l = bb then (1:ℝ) else 0) := hXt _
      _ ≤ opNorm (E - E') * vnorm (fun l => if l = bb then (1:ℝ) else 0) :=
          mulVec_vnorm_le _ _ hd0
      _ ≤ ε := by rw [hbasis' bb, mul_one]; exact hop
  rw [halg, Matrix.add_apply]
  calc |(Eᵀ * (E - E')) i j + (E'ᵀ * (E - E'))ᵀ i j|
      ≤ |(Eᵀ * (E - E')) i j| + |(E'ᵀ * (E - E'))ᵀ i j| := abs_add_le _ _
    _ ≤ ε + ε := by
        apply add_le_add (hterm E hEt i j)
        rw [Matrix.transpose_apply]
        exact hterm E' hE't j i
    _ = 2 * ε := by ring

/-- A flux edge's source projection is exactly the standard projection `W·Wᵀ`. -/
theorem fluxEdge_srcProj {m d r : ℕ} [NeZero m] (hr : 1 < r) (hrd : r ≤ d) (k : ℤ)
    (e : TorusEdge m) :
    (fluxFamily (m := m) k d r e)ᵀ * fluxFamily (m := m) k d r e
      = stdFrame d r * (stdFrame d r)ᵀ := by
  show (stdFrame d r * rotCore r (fluxAngle k e) * (stdFrame d r)ᵀ)ᵀ
      * (stdFrame d r * rotCore r (fluxAngle k e) * (stdFrame d r)ᵀ)
      = stdFrame d r * (stdFrame d r)ᵀ
  have hF : (stdFrame d r)ᵀ * stdFrame d r = 1 := stdFrame_isFrame hrd
  rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    rotCore_transpose]
  simp only [Matrix.mul_assoc]
  rw [← Matrix.mul_assoc (stdFrame d r)ᵀ (stdFrame d r), hF, Matrix.one_mul,
    ← Matrix.mul_assoc (rotCore r (-fluxAngle k e)), rotCore_mul hr,
    neg_add_cancel, rotCore_zero, Matrix.one_mul]

/-- A flat edge's source projection is exactly the source-vertex projection. -/
theorem flatEdge_srcProj {m d r : ℕ} [NeZero m] (T' : EdgeFamily m d)
    (V : Fin m × Fin m → Matrix (Fin d) (Fin r) ℝ)
    (hfr : ∀ v, IsFrame (V v)) (hTe : ∀ e, T' e = V (dst e) * (V (src e))ᵀ)
    (e : TorusEdge m) :
    (T' e)ᵀ * T' e = V (src e) * (V (src e))ᵀ := by
  rw [hTe e, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc,
    ← Matrix.mul_assoc (V (dst e))ᵀ, ← Matrix.mul_assoc (V (src e)), hfr (dst e),
    Matrix.mul_one]

/-- The standard projection has unit diagonal on the surviving coordinates. -/
theorem stdFrame_proj_diag {d r : ℕ} (hrd : r ≤ d) (i : Fin d) (hi : (i : ℕ) < r) :
    (stdFrame d r * (stdFrame d r)ᵀ) i i = 1 := by
  have h := stdFrame_conj_apply hrd 1 i i hi hi
  rw [Matrix.mul_one] at h
  rw [h, Matrix.one_apply_eq]

/-- **Gap stability for the flux family — PROVED, m-uniformly.** Every flat family
edge-wise within ε ≤ 1/4 of the charge-k flux family has CoreGap 1. -/
theorem flux_gapStable {m d r : ℕ} [NeZero m] (hd : 1 < d) (hr : 1 < r) (hrd : r ≤ d)
    (k : ℤ) (T' : EdgeFamily m d) (hflat : IsFlatFamily r T')
    (ε : ℝ) (hε14 : ε ≤ 1/4)
    (hc : ∀ e, opNorm (fluxFamily (m := m) k d r e - T' e) ≤ ε) :
    CoreGap 1 hd T' := by
  have hd0 : 0 < d := by omega
  have hm : 1 ≤ m := Nat.one_le_iff_ne_zero.mpr (NeZero.ne m)
  obtain ⟨V, hfr, hTe⟩ := hflat
  intro x
  rw [flat_vertCycle hm T' V hfr hTe x]
  obtain ⟨hskew, htr⟩ := coreVec_proj hd (V (x, 0))
  rw [hskew]
  -- the vertical edge sourced at (x,0)
  set e₀ : TorusEdge m := ((x, (0 : Fin m)), false) with he₀
  have hsrc : src e₀ = (x, (0 : Fin m)) := rfl
  -- transpose contractions for both edges
  have hEt : ∀ u, vnorm ((fluxFamily (m := m) k d r e₀)ᵀ *ᵥ u) ≤ vnorm u := by
    intro u
    show vnorm ((stdFrame d r * rotCore r (fluxAngle k e₀) * (stdFrame d r)ᵀ)ᵀ *ᵥ u)
        ≤ vnorm u
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
      rotCore_transpose, ← Matrix.mul_assoc]
    exact conj_orth_contr (stdFrame d r) (stdFrame_isFrame hrd) _
      (rotCore_orth hr _) u
  have hE't : ∀ u, vnorm ((T' e₀)ᵀ *ᵥ u) ≤ vnorm u := by
    intro u
    rw [hTe e₀, Matrix.transpose_mul, Matrix.transpose_transpose]
    exact frames_pair_contr _ _ (hfr (src e₀)) (hfr (dst e₀)) u
  -- the two diagonal core entries of the vertex projection are ≥ 1 − 2ε
  have hdiag : ∀ (i : Fin d), (i : ℕ) < r →
      1 - 2 * ε ≤ (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) i i := by
    intro i hi
    have hclose := srcProj_entry_close hd0 (fluxFamily (m := m) k d r e₀) (T' e₀)
      ε (hc e₀) hEt hE't i i
    rw [fluxEdge_srcProj hr hrd k e₀, flatEdge_srcProj T' V hfr hTe e₀, hsrc] at hclose
    have h1 := stdFrame_proj_diag hrd i hi
    rw [Matrix.sub_apply, h1] at hclose
    have := abs_le.mp hclose
    linarith [this.2]
  have h0r : ((0 : ℕ)) < r := by omega
  have hq : 2 - 4 * ε ≤ (coreVec (coreBlock hd
      (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ))).1 := by
    have e00 := hdiag ⟨0, by omega⟩ h0r
    have e11 := hdiag ⟨1, hd⟩ hr
    show 2 - 4 * ε ≤ coreBlock hd (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) 0 0
        + coreBlock hd (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) 1 1
    have hcb0 : coreBlock hd (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) 0 0
        = (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) ⟨0, by omega⟩ ⟨0, by omega⟩ := rfl
    have hcb1 : coreBlock hd (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) 1 1
        = (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ) ⟨1, hd⟩ ⟨1, hd⟩ := rfl
    rw [hcb0, hcb1]
    linarith [e00, e11]
  -- conclude: the core vector is (q, 0) with q ≥ 2 − 4ε ≥ 1
  have hq1 : (1 : ℝ) ≤ (coreVec (coreBlock hd
      (V (x, (0 : Fin m)) * (V (x, (0 : Fin m)))ᵀ))).1 := by linarith
  rw [show ((0:ℝ)) ^ 2 = 0 by norm_num, add_zero, Real.sqrt_sq (by linarith : (0:ℝ) ≤ _)]
  exact hq1

/-- **m-UNIFORM gap-protected lift continuity, for FLAT targets.**
**STATUS (Response 38): FALSE — machine-checked.** See `Shear.lean`
(`unifGapLiftCont_false`): the shear family (flux verticals, trivial horizontals) winds
k = 1 within 2π/m of the constant flat family, both gapped. The definition is KEPT so
its negation is a statable theorem; nothing may assume it. The vertical-only
`windingSum` is a 1D shadow, not an obstruction — the corrected invariant must read
horizontal data (Bott index of the magnetic-translation pair; pre-registered target). -/
def UnifGapLiftCont (d r : ℕ) : Prop :=
  ∀ γ : ℝ, 0 < γ → ∃ margin : ℝ, 0 < margin ∧
    ∀ (m : ℕ) [NeZero m] (hd : 1 < d) (T T' : EdgeFamily m d),
      IsFlatFamily r T' →
      CoreGap γ hd T → CoreGap γ hd T' →
      (∀ e, opNorm (T e - T' e) < margin) → windingSum hd T = windingSum hd T'

-- `GapStable` (Response 31/35) is GONE: its content for the flux family is the THEOREM
-- `flux_gapStable` above. (Its general-T form remains unproven and possibly false even
-- flat-restricted — the cycle-level gap of an arbitrary T does not pin edge-level
-- structure; tagged OPEN, not asserted. The flux case is all `stageB` ever needed.)

/-- The flux family has a definite, m-independent core gap of 2: the core of a planar
rotation R(φ) is (2cos φ, 2sin φ), of norm 2. (DEBT-B3 — PAID post-fusion.) -/
theorem fluxFamily_coreGap (hr : 1 < r) (hrd : r ≤ d) (hd : 1 < d) (k : ℤ)
    (m : ℕ) [NeZero m] : CoreGap 2 hd (fluxFamily (m := m) k d r) := by
  intro x
  rw [flux_vertCycle k d r hr hrd x, coreVec_conj_rot hd hr hrd]
  have hpy : (2 * Real.cos (2 * Real.pi * k * (x : ℕ) / m)) ^ 2
      + (2 * Real.sin (2 * Real.pi * k * (x : ℕ) / m)) ^ 2 = 4 := by
    nlinarith [Real.sin_sq_add_cos_sq (2 * Real.pi * k * (x : ℕ) / m)]
  rw [hpy, show (4:ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]

/-! ## The retired reduction (Response 38)

`stageB` and `plateau_of_props` (Responses 31/36) reduced the Plateau to
`UnifGapLiftCont` — which `Shear.lean` proves FALSE, so both theorems were vacuously
true and are removed (history preserves them; the combination logic remains valid for
whatever corrected invariant replaces `windingSum`). What SURVIVES of this file's
content, unconditionally: `coreGap_mono`, `fluxFamily_coreGap`, `flux_gapStable`, and
the source-projection toolkit — all true theorems about the objects, independent of the
dead reduction. The Plateau Conjecture (`plateau_conjecture`) stands exactly as before:
open, with its empirical support (Entries 3/10/14) untouched, and now with a sharper
understanding of what any proof must use. -/

end LossyCocycles
