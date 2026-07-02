import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-! # LossyCocycles.Basic — L0/L1 of the formalization ladder

L0 (collapse): for frames with orthonormal columns, `(V₃V₂ᵀ)(V₂V₁ᵀ) = V₃V₁ᵀ`.
L1 (chain/cycle): a flat chain composes to `(last frame)·(first frame)ᵀ`; a cycle
composite is therefore `V·Vᵀ` — idempotent and symmetric: an orthogonal projection.

These are the formal spine of Paper 2's Lemma 1 (⇐). The (⇒) direction is L3 — the
gauge-absorption argument the prose hand-waves — and is the ladder's first real test.

Audit note (2026-06-11): the first version of `chain_collapse` quantified over
`Vs.head?/getLast?` and was FALSE for singleton lists (`chainComposite [V] = 1 ≠ V·Vᵀ`);
caught in the double-check, restated over `V₀ :: mid ++ [Vlast]`, and proved (DEBT-1 paid).
-/

open Matrix

variable {d r : ℕ}

/-- A frame: a d×r matrix with orthonormal columns. -/
def IsFrame (V : Matrix (Fin d) (Fin r) ℝ) : Prop :=
  Vᵀ * V = 1

/-- L0 — the collapse lemma: interior frames cancel. -/
theorem collapse (V₁ V₂ V₃ : Matrix (Fin d) (Fin r) ℝ) (h : IsFrame V₂) :
    (V₃ * V₂ᵀ) * (V₂ * V₁ᵀ) = V₃ * V₁ᵀ := by
  calc (V₃ * V₂ᵀ) * (V₂ * V₁ᵀ)
      = V₃ * ((V₂ᵀ * V₂) * V₁ᵀ) := by
        rw [Matrix.mul_assoc, Matrix.mul_assoc]
    _ = V₃ * V₁ᵀ := by rw [h, Matrix.one_mul]

/-- A flat chain: edge maps `Vᵢ₊₁·Vᵢᵀ` composed along a list of frames
(later maps multiply on the left). -/
def chainComposite : List (Matrix (Fin d) (Fin r) ℝ) → Matrix (Fin d) (Fin d) ℝ
  | [] => 1
  | [_] => 1
  | V₁ :: V₂ :: rest => chainComposite (V₂ :: rest) * (V₂ * V₁ᵀ)

/-- L1a — a flat chain with orthonormal interior frames composes to `(last)·(first)ᵀ`.
Endpoints need no hypothesis; only the `mid` frames cancel. -/
theorem chain_collapse (Vlast : Matrix (Fin d) (Fin r) ℝ)
    (mid : List (Matrix (Fin d) (Fin r) ℝ)) (hmid : ∀ V ∈ mid, IsFrame V)
    (V₀ : Matrix (Fin d) (Fin r) ℝ) :
    chainComposite (V₀ :: mid ++ [Vlast]) = Vlast * V₀ᵀ := by
  induction mid generalizing V₀ with
  | nil => simp [chainComposite]
  | cons W rest ih =>
      have hW : IsFrame W := hmid W List.mem_cons_self
      have hrest : ∀ V ∈ rest, IsFrame V := fun V hV => hmid V (List.mem_cons_of_mem _ hV)
      have step : chainComposite (V₀ :: (W :: rest) ++ [Vlast])
          = chainComposite (W :: rest ++ [Vlast]) * (W * V₀ᵀ) := by
        simp only [List.cons_append, chainComposite]
      rw [step, ih hrest W]
      exact collapse V₀ W Vlast hW

/-- L1a′ — a closed flat cycle composes to the projection `V₀·V₀ᵀ`. -/
theorem cycle_collapse (V₀ : Matrix (Fin d) (Fin r) ℝ)
    (mid : List (Matrix (Fin d) (Fin r) ℝ)) (hmid : ∀ V ∈ mid, IsFrame V) :
    chainComposite (V₀ :: mid ++ [V₀]) = V₀ * V₀ᵀ :=
  chain_collapse V₀ mid hmid V₀

/-- L1b — the cycle composite is idempotent: `(V·Vᵀ)·(V·Vᵀ) = V·Vᵀ`. -/
theorem cycle_idempotent (V : Matrix (Fin d) (Fin r) ℝ) (h : IsFrame V) :
    (V * Vᵀ) * (V * Vᵀ) = V * Vᵀ :=
  collapse V V V h

/-- L1c — the cycle composite is symmetric. -/
theorem cycle_symm (V : Matrix (Fin d) (Fin r) ℝ) :
    (V * Vᵀ)ᵀ = V * Vᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose]
