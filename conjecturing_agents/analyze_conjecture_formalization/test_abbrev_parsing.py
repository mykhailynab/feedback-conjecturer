#!/usr/bin/env python3
from __future__ import annotations

import re
from textwrap import dedent

from conjecturing_agents.agents.conjecture_formalizer import (
    extract_abbrev_name_from_statement,
    extract_ground_truth_comment_and_strip_line,
    extract_last_abbrev_declaration,
    extract_rhs_from_abbrev_declaration,
    replace_abbrev_in_statement,
)


FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


def extract_final_channel_message(raw_output: str) -> str:
    matches = FINAL_CHANNEL_RE.findall(raw_output)
    assert matches, "No final-channel message found in raw_output"
    return matches[-1].strip()


def assert_eq(actual, expected, msg: str = "") -> None:
    if actual != expected:
        raise AssertionError(
            f"{msg}\nEXPECTED:\n{expected!r}\n\nACTUAL:\n{actual!r}"
        )


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_strip_comment_and_extract_abbrev_name() -> None:
    lean4_full_contents = dedent(
        """\
        import Mathlib

        open Set

        noncomputable abbrev putnam_2003_a3_solution : ℝ := sorry
        -- 2 * Real.sqrt 2 - 1
        /--
        Find the minimum value of $|\\sin x+\\cos x+\\tan x+\\cot x+\\sec x+\\csc x|$ for real numbers $x$.
        -/
        theorem putnam_2003_a3
            (f : ℝ → ℝ)
            (hf : ∀ x : ℝ, f x = |Real.sin x + Real.cos x + Real.tan x + 1 / Real.tan x + 1 / Real.cos x + 1 / Real.sin x|) :
            IsLeast (Set.range f) putnam_2003_a3_solution :=
          sorry
        """
    )

    ground_truth, stripped = extract_ground_truth_comment_and_strip_line(lean4_full_contents)

    assert_eq(ground_truth, "2 * Real.sqrt 2 - 1", "Failed to extract ground-truth comment")
    assert_true("-- 2 * Real.sqrt 2 - 1" not in stripped, "Comment line was not removed")
    assert_true("import Mathlib" in stripped, "Import line should be preserved")
    assert_true("open Set" in stripped, "open line should be preserved")

    abbrev_name = extract_abbrev_name_from_statement(stripped)
    assert_eq(abbrev_name, "putnam_2003_a3_solution", "Failed to extract abbrev name")

    print("[ok] test_strip_comment_and_extract_abbrev_name")


def test_extract_last_abbrev_declaration_from_plain_lean_block() -> None:
    text = dedent(
        """\
        ```lean4
        noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1
        ```
        """
    )

    decl = extract_last_abbrev_declaration(
        text,
        required_abbrev_name="putnam_2003_a3_solution",
    )
    assert_eq(
        decl,
        "noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1",
        "Failed to extract abbrev declaration from plain lean block",
    )

    rhs = extract_rhs_from_abbrev_declaration(decl)
    assert_eq(rhs, "2 * Real.sqrt 2 - 1", "Failed to extract RHS from abbrev declaration")

    print("[ok] test_extract_last_abbrev_declaration_from_plain_lean_block")


def test_extract_last_abbrev_declaration_from_final_channel_payload() -> None:
    raw_output = (
        "<|channel|>analysis<|message|>Let me formalize the answer.<|return|>\n"
        "<|channel|>final<|message|>```lean4\n"
        "noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1\n"
        "```<|return|>\n"
    )

    final_msg = extract_final_channel_message(raw_output)
    decl = extract_last_abbrev_declaration(
        final_msg,
        required_abbrev_name="putnam_2003_a3_solution",
    )

    assert_eq(
        decl,
        "noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1",
        "Failed to recover abbrev from final-channel payload",
    )

    print("[ok] test_extract_last_abbrev_declaration_from_final_channel_payload")


def test_extract_last_abbrev_declaration_stops_before_theorem() -> None:
    text = dedent(
        """\
        ```lean4
        noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1

        theorem putnam_2003_a3
            (f : ℝ → ℝ) :
            True :=
          sorry
        ```
        """
    )

    decl = extract_last_abbrev_declaration(
        text,
        required_abbrev_name="putnam_2003_a3_solution",
    )

    assert_eq(
        decl,
        "noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1",
        "Abbrev extraction incorrectly swallowed the theorem",
    )

    print("[ok] test_extract_last_abbrev_declaration_stops_before_theorem")


def test_replace_abbrev_in_statement() -> None:
    lean_statement = dedent(
        """\
        import Mathlib

        open Set

        noncomputable abbrev putnam_2003_a3_solution : ℝ := sorry
        /--
        Find the minimum value of something.
        -/
        theorem putnam_2003_a3
            (f : ℝ → ℝ) :
            IsLeast (Set.range f) putnam_2003_a3_solution :=
          sorry
        """
    )

    new_abbrev = "noncomputable abbrev putnam_2003_a3_solution : ℝ := 2 * Real.sqrt 2 - 1"

    replaced = replace_abbrev_in_statement(
        lean_statement,
        new_abbrev,
        required_abbrev_name="putnam_2003_a3_solution",
    )

    assert_true(new_abbrev in replaced, "Replacement abbrev not inserted")
    assert_true(":= sorry" in replaced, "The theorem sorry should remain untouched")
    assert_true("noncomputable abbrev putnam_2003_a3_solution : ℝ := sorry" not in replaced,
                "Old abbrev placeholder was not removed")

    print("[ok] test_replace_abbrev_in_statement")


def test_multiline_abbrev_body() -> None:
    text = dedent(
        """\
        ```lean4
        noncomputable abbrev putnam_2007_a1_solution : Set ℝ :=
          {2 / 3, 3 / 2, (13 + √601) / 12, (13 - √601) / 12}

        theorem putnam_2007_a1
            (P : (ℝ → ℝ) → Prop) :
            True :=
          sorry
        ```
        """
    )

    decl = extract_last_abbrev_declaration(
        text,
        required_abbrev_name="putnam_2007_a1_solution",
    )

    expected = dedent(
        """\
        noncomputable abbrev putnam_2007_a1_solution : Set ℝ :=
          {2 / 3, 3 / 2, (13 + √601) / 12, (13 - √601) / 12}
        """
    ).strip()

    assert_eq(decl, expected, "Failed to extract multiline abbrev declaration correctly")

    rhs = extract_rhs_from_abbrev_declaration(decl)
    assert_eq(
        rhs,
        "{2 / 3, 3 / 2, (13 + √601) / 12, (13 - √601) / 12}",
        "Failed to extract RHS from multiline abbrev declaration",
    )

    print("[ok] test_multiline_abbrev_body")


def main() -> None:
    tests = [
        test_strip_comment_and_extract_abbrev_name,
        test_extract_last_abbrev_declaration_from_plain_lean_block,
        test_extract_last_abbrev_declaration_from_final_channel_payload,
        test_extract_last_abbrev_declaration_stops_before_theorem,
        test_replace_abbrev_in_statement,
        test_multiline_abbrev_body,
    ]

    print("Running formalizer/scheduler parsing tests...\n")
    for test in tests:
        test()

    print("\nAll parsing tests passed.")


if __name__ == "__main__":
    main()
