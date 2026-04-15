"""
Real-world Lean 4 scaffolds taken from
``logs/conjecture_formalization_logs_20mins/formalizations.jsonl``.

Each ``*_SCAFFOLD`` constant is the value of ``lean_statement_without_comment``
(the GT-answer comment has already been stripped by the pipeline).

The corresponding ``*_ABBREV_DECL`` and ``*_GT_ANSWER`` constants reflect the
formalized answer used in the run.
"""

# ---------------------------------------------------------------------------
# putnam_2008_b2  (simple ℝ result, multi-line theorem arguments)
# ---------------------------------------------------------------------------

SIMPLE_SCAFFOLD = """\
import Mathlib

open Filter Topology Set Nat

abbrev putnam_2008_b2_solution : ℝ := sorry
/--
Let $F_0(x)=\\ln x$. For $n \\geq 0$ and $x>0$, let $F_{n+1}(x)=\\int_0^x F_n(t)\\,dt$.
Evaluate $\\lim_{n \\to \\infty} \\frac{n!F_n(1)}{\\ln n}$.
-/
theorem putnam_2008_b2
(F : ℕ → ℝ → ℝ)
(hF0 : ∀ x : ℝ, F 0 x = Real.log x)
(hFn : ∀ n : ℕ, ∀ x > 0, F (n + 1) x = ∫ t in Set.Ioo 0 x, F n t)
: Tendsto (fun n : ℕ => ((n)! * F n 1) / Real.log n) atTop (𝓝 putnam_2008_b2_solution) :=
sorry"""

SIMPLE_ABBREV_NAME = "putnam_2008_b2_solution"
SIMPLE_ABBREV_DECL = "abbrev putnam_2008_b2_solution : ℝ := -1"
SIMPLE_GT_ANSWER = "-1"

# Scaffold with the GT-answer comment still in place (input to
# extract_ground_truth_comment_and_strip_line).
SIMPLE_SCAFFOLD_WITH_GT_COMMENT = """\
import Mathlib

open Filter Topology Set Nat

abbrev putnam_2008_b2_solution : ℝ := sorry
-- -1

/--
Let $F_0(x)=\\ln x$. For $n \\geq 0$ and $x>0$, let $F_{n+1}(x)=\\int_0^x F_n(t)\\,dt$.
Evaluate $\\lim_{n \\to \\infty} \\frac{n!F_n(1)}{\\ln n}$.
-/
theorem putnam_2008_b2
(F : ℕ → ℝ → ℝ)
(hF0 : ∀ x : ℝ, F 0 x = Real.log x)
(hFn : ∀ n : ℕ, ∀ x > 0, F (n + 1) x = ∫ t in Set.Ioo 0 x, F n t)
: Tendsto (fun n : ℕ => ((n)! * F n 1) / Real.log n) atTop (𝓝 putnam_2008_b2_solution) :=
sorry"""

# ---------------------------------------------------------------------------
# putnam_1986_a3  (noncomputable abbrev, ℝ result involving π)
# ---------------------------------------------------------------------------

NONCOMPUTABLE_SCAFFOLD = """\
import Mathlib

open  Real

noncomputable abbrev putnam_1986_a3_solution : ℝ := sorry
/--
Evaluate $\\sum_{n=0}^\\infty \\mathrm{Arccot}(n^2+n+1)$.
-/
theorem putnam_1986_a3
(cot : ℝ → ℝ)
(fcot : cot = fun θ ↦ cos θ / sin θ)
(arccot : ℝ → ℝ)
(harccot : ∀ t : ℝ, t ≥ 0 → arccot t ∈ Set.Ioc 0 (Real.pi / 2) ∧ cot (arccot t) = t)
: (∑' n : ℕ, arccot (n ^ 2 + n + 1) = putnam_1986_a3_solution) :=
sorry"""

NONCOMPUTABLE_ABBREV_NAME = "putnam_1986_a3_solution"
NONCOMPUTABLE_ABBREV_DECL = "noncomputable abbrev putnam_1986_a3_solution : ℝ := π / 2"
NONCOMPUTABLE_GT_ANSWER = "π / 2"

# ---------------------------------------------------------------------------
# putnam_1985_a5  (Set ℕ result, indented theorem)
# ---------------------------------------------------------------------------

SET_SCAFFOLD = """\
import Mathlib

open Set Filter Topology Real

abbrev putnam_1985_a5_solution : Set ℕ := sorry
/--
For which integers $m$, $1 \\leq m \\leq 10$ is $I_m \\neq 0$?
-/
theorem putnam_1985_a5
    (I : ℕ → ℝ)
    (hI : I = fun (m : ℕ) ↦ ∫ x in (0)..(2 * Real.pi), ∏ k ∈ Finset.Icc 1 m, cos (k * x)) :
    {m ∈ Finset.Icc 1 10 | I m ≠ 0} = putnam_1985_a5_solution :=
  sorry"""

SET_ABBREV_NAME = "putnam_1985_a5_solution"
SET_ABBREV_DECL = "abbrev putnam_1985_a5_solution : Set ℕ := ({3, 4, 7, 8} : Set ℕ)"
SET_GT_ANSWER = "({3, 4, 7, 8} : Set ℕ)"

# ---------------------------------------------------------------------------
# putnam_2007_a1  (Set ℝ, multi-line abbrev RHS)
# ---------------------------------------------------------------------------

MULTILINE_DECL_SCAFFOLD = """\
import Mathlib

abbrev putnam_2007_a1_solution : Set ℝ := sorry

/--
Find all values of $\\alpha$ for which two parabolas are tangent to each other.
-/
theorem putnam_2007_a1
    (P : (ℝ → ℝ) → Prop)
    (P_def : ∀ f, P f ↔ ∃ x y, f x = y ∧ f y = x ∧ deriv f x * deriv f y = 1)
    (α : ℝ) :
    P (fun t ↦ α * t ^ 2 + α * t + 1 / 24) ↔ α ∈ putnam_2007_a1_solution :=
  sorry"""

MULTILINE_DECL_ABBREV_NAME = "putnam_2007_a1_solution"
MULTILINE_DECL_ABBREV_DECL = (
    "abbrev putnam_2007_a1_solution : Set ℝ :=\n"
    "  ({(2 : ℝ) / 3, (3 : ℝ) / 2,"
    " (13 - Real.sqrt 601) / 12, (13 + Real.sqrt 601) / 12} : Set ℝ)"
)
MULTILINE_DECL_GT_ANSWER = (
    "({(2 : ℝ) / 3, (3 : ℝ) / 2,"
    " (13 - Real.sqrt 601) / 12, (13 + Real.sqrt 601) / 12} : Set ℝ)"
)

# ---------------------------------------------------------------------------
# putnam_1995_a5  (Prop result, theorem with no args)
# ---------------------------------------------------------------------------

PROP_SCAFFOLD = """\
import Mathlib

open Filter Topology Real

abbrev putnam_1995_a5_solution : Prop := sorry
/--
Are the functions $x_1, \\dots, x_n$ necessarily linearly dependent?
-/
theorem putnam_1995_a5 :
  putnam_1995_a5_solution ↔
  (∀ (n : ℕ) (x : Fin n → (ℝ → ℝ)), True) :=
  sorry"""

PROP_ABBREV_NAME = "putnam_1995_a5_solution"
PROP_ABBREV_DECL = "abbrev putnam_1995_a5_solution : Prop := True"
PROP_GT_ANSWER = "True"
