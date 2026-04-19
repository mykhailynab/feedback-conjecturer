"""
Tests for the utility functions in conjecturing_agents.lean_regex.
"""
import pytest

from conjecturing_agents.lean_regex import (
    contains_theorem_signature,
    extract_abbrev_name_from_statement,
    extract_ground_truth_comment_and_strip_line,
    extract_last_abbrev_declaration,
    extract_lean_code_block,
    extract_lean_code_block_or_text,
    extract_preamble,
    extract_rhs_from_abbrev_declaration,
    extract_theorem_signature,
    negate_theorem_statement,
    remove_lean_comments,
    replace_abbrev_in_statement,
)

from .fixtures import (
    MULTILINE_DECL_ABBREV_DECL,
    MULTILINE_DECL_ABBREV_NAME,
    MULTILINE_DECL_SCAFFOLD,
    NONCOMPUTABLE_ABBREV_DECL,
    NONCOMPUTABLE_ABBREV_NAME,
    NONCOMPUTABLE_GT_ANSWER,
    NONCOMPUTABLE_SCAFFOLD,
    PROP_ABBREV_DECL,
    PROP_ABBREV_NAME,
    PROP_SCAFFOLD,
    SET_ABBREV_DECL,
    SET_ABBREV_NAME,
    SET_SCAFFOLD,
    SIMPLE_ABBREV_DECL,
    SIMPLE_ABBREV_NAME,
    SIMPLE_GT_ANSWER,
    SIMPLE_SCAFFOLD,
    SIMPLE_SCAFFOLD_WITH_GT_COMMENT,
)


# ---------------------------------------------------------------------------
# extract_lean_code_block
# ---------------------------------------------------------------------------

class TestExtractLeanCodeBlock:
    def test_lean4_block(self):
        text = "Here is the solution:\n```lean4\nabbrev foo : ℝ := 42\n```"
        result = extract_lean_code_block(text)
        assert result == "abbrev foo : ℝ := 42"

    def test_lean_block_fallback(self):
        text = "```lean\nabbrev foo : ℝ := 42\n```"
        result = extract_lean_code_block(text)
        assert result == "abbrev foo : ℝ := 42"

    def test_lean4_preferred_over_lean(self):
        text = "```lean\nabbrev foo : ℝ := 1\n```\n```lean4\nabbrev foo : ℝ := 2\n```"
        result = extract_lean_code_block(text)
        assert result == "abbrev foo : ℝ := 2"

    def test_returns_last_block(self):
        text = (
            "```lean4\nabbrev foo : ℝ := 1\n```\n"
            "Some text\n"
            "```lean4\nabbrev foo : ℝ := 99\n```"
        )
        result = extract_lean_code_block(text)
        assert result == "abbrev foo : ℝ := 99"

    def test_strips_whitespace(self):
        text = "```lean4\n  abbrev foo : ℝ := 42  \n```"
        result = extract_lean_code_block(text)
        assert result == "abbrev foo : ℝ := 42"

    def test_multiline_block(self):
        text = "```lean4\nabbrev foo : ℝ := 42\ntheorem bar : True := by trivial\n```"
        result = extract_lean_code_block(text)
        assert "abbrev foo" in result
        assert "theorem bar" in result

    def test_none_when_absent(self):
        assert extract_lean_code_block("no code block here") is None

    def test_none_on_empty(self):
        assert extract_lean_code_block("") is None

    def test_none_on_none_like_empty(self):
        assert extract_lean_code_block(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_lean_code_block_or_text
# ---------------------------------------------------------------------------

class TestExtractLeanCodeBlockOrText:
    def test_extracts_block_when_present(self):
        text = "Thinking...\n```lean4\nabbrev foo : ℝ := 1\n```"
        result = extract_lean_code_block_or_text(text)
        assert result == "abbrev foo : ℝ := 1"

    def test_returns_stripped_text_when_no_block(self):
        text = "  abbrev foo : ℝ := 1  "
        result = extract_lean_code_block_or_text(text)
        assert result == "abbrev foo : ℝ := 1"

    def test_empty_string(self):
        assert extract_lean_code_block_or_text("") == ""


# ---------------------------------------------------------------------------
# remove_lean_comments
# ---------------------------------------------------------------------------

class TestRemoveLeanComments:
    def test_removes_line_comment(self):
        result = remove_lean_comments("abbrev foo : ℝ := 1 -- comment")
        assert "--" not in result
        assert "abbrev foo" in result

    def test_removes_block_comment(self):
        result = remove_lean_comments("theorem foo /- inline -/ : P := by sorry")
        assert "/-" not in result
        assert "theorem foo" in result

    def test_removes_multiline_block_comment(self):
        text = "/--\nThis is a doc comment\n-/\ntheorem foo : P := by sorry"
        result = remove_lean_comments(text)
        assert "doc comment" not in result
        assert "theorem foo" in result

    def test_preserves_code(self):
        code = "abbrev foo : ℝ := 42"
        assert remove_lean_comments(code) == code

    def test_empty(self):
        assert remove_lean_comments("") == ""


# ---------------------------------------------------------------------------
# extract_abbrev_name_from_statement
# ---------------------------------------------------------------------------

class TestExtractAbbrevNameFromStatement:
    def test_simple_scaffold(self):
        assert extract_abbrev_name_from_statement(SIMPLE_SCAFFOLD) == SIMPLE_ABBREV_NAME

    def test_noncomputable_scaffold(self):
        assert (
            extract_abbrev_name_from_statement(NONCOMPUTABLE_SCAFFOLD)
            == NONCOMPUTABLE_ABBREV_NAME
        )

    def test_set_scaffold(self):
        assert extract_abbrev_name_from_statement(SET_SCAFFOLD) == SET_ABBREV_NAME

    def test_prop_scaffold(self):
        assert extract_abbrev_name_from_statement(PROP_SCAFFOLD) == PROP_ABBREV_NAME

    def test_from_declaration_string(self):
        assert extract_abbrev_name_from_statement(SIMPLE_ABBREV_DECL) == SIMPLE_ABBREV_NAME

    def test_from_noncomputable_declaration_string(self):
        assert (
            extract_abbrev_name_from_statement(NONCOMPUTABLE_ABBREV_DECL)
            == NONCOMPUTABLE_ABBREV_NAME
        )

    def test_returns_none_when_absent(self):
        assert extract_abbrev_name_from_statement("theorem foo : P := by sorry") is None

    def test_returns_none_on_empty(self):
        assert extract_abbrev_name_from_statement("") is None


# ---------------------------------------------------------------------------
# extract_rhs_from_abbrev_declaration
# ---------------------------------------------------------------------------

class TestExtractRhsFromAbbrevDeclaration:
    def test_simple_number(self):
        assert extract_rhs_from_abbrev_declaration(SIMPLE_ABBREV_DECL) == SIMPLE_GT_ANSWER

    def test_noncomputable(self):
        assert (
            extract_rhs_from_abbrev_declaration(NONCOMPUTABLE_ABBREV_DECL)
            == NONCOMPUTABLE_GT_ANSWER
        )

    def test_set_type(self):
        rhs = extract_rhs_from_abbrev_declaration(SET_ABBREV_DECL)
        assert rhs is not None
        assert "3, 4, 7, 8" in rhs

    def test_prop_value(self):
        assert extract_rhs_from_abbrev_declaration(PROP_ABBREV_DECL) == "True"

    def test_multiline_decl(self):
        rhs = extract_rhs_from_abbrev_declaration(MULTILINE_DECL_ABBREV_DECL)
        assert rhs is not None
        assert "(2 : ℝ) / 3" in rhs

    def test_sorry_placeholder(self):
        assert (
            extract_rhs_from_abbrev_declaration("abbrev foo_solution : ℝ := sorry")
            == "sorry"
        )

    def test_returns_none_without_assign(self):
        assert extract_rhs_from_abbrev_declaration("abbrev foo_solution : ℝ") is None


# ---------------------------------------------------------------------------
# extract_preamble
# ---------------------------------------------------------------------------

class TestExtractPreamble:
    def test_simple_scaffold_preamble(self):
        preamble = extract_preamble(SIMPLE_SCAFFOLD)
        assert "import Mathlib" in preamble
        assert "open Filter Topology Set Nat" in preamble
        # The abbrev line itself should not be in the preamble
        assert "abbrev" not in preamble

    def test_noncomputable_scaffold_preamble(self):
        preamble = extract_preamble(NONCOMPUTABLE_SCAFFOLD)
        assert "import Mathlib" in preamble
        assert "open" in preamble
        assert "abbrev" not in preamble

    def test_no_open_statements(self):
        scaffold = "import Mathlib\n\nabbrev foo : ℝ := sorry\n"
        preamble = extract_preamble(scaffold)
        assert "import Mathlib" in preamble
        assert "abbrev" not in preamble

    def test_empty_preamble(self):
        scaffold = "abbrev foo : ℝ := sorry\ntheorem bar : True := by sorry"
        preamble = extract_preamble(scaffold)
        assert preamble == ""


# ---------------------------------------------------------------------------
# extract_ground_truth_comment_and_strip_line
# ---------------------------------------------------------------------------

class TestExtractGroundTruthCommentAndStripLine:
    def test_simple(self):
        gt, stripped = extract_ground_truth_comment_and_strip_line(
            SIMPLE_SCAFFOLD_WITH_GT_COMMENT
        )
        assert gt == SIMPLE_GT_ANSWER
        assert "-- -1" not in stripped
        assert "abbrev putnam_2008_b2_solution" in stripped

    def test_stripped_scaffold_matches_without_comment(self):
        _, stripped = extract_ground_truth_comment_and_strip_line(
            SIMPLE_SCAFFOLD_WITH_GT_COMMENT
        )
        # After stripping, content should match SIMPLE_SCAFFOLD (modulo blank lines)
        assert "theorem putnam_2008_b2" in stripped
        assert "sorry" in stripped

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Expected exactly one"):
            extract_ground_truth_comment_and_strip_line(SIMPLE_SCAFFOLD)

    def test_two_matches_raises(self):
        double = (
            "abbrev foo : ℝ := sorry\n-- answer1\n"
            "abbrev bar : ℕ := sorry\n-- answer2\n"
        )
        with pytest.raises(ValueError, match="Expected exactly one"):
            extract_ground_truth_comment_and_strip_line(double)

    def test_noncomputable(self):
        scaffold_with_comment = (
            "noncomputable abbrev putnam_1986_a3_solution : ℝ := sorry\n"
            "-- π / 2\n"
            "theorem putnam_1986_a3 : True := by sorry\n"
        )
        gt, stripped = extract_ground_truth_comment_and_strip_line(scaffold_with_comment)
        assert gt == "π / 2"
        assert "-- π / 2" not in stripped


# ---------------------------------------------------------------------------
# replace_abbrev_in_statement
# ---------------------------------------------------------------------------

class TestReplaceAbbrevInStatement:
    def test_simple_replacement(self):
        result = replace_abbrev_in_statement(SIMPLE_SCAFFOLD, SIMPLE_ABBREV_DECL)
        assert "abbrev putnam_2008_b2_solution : ℝ := -1" in result
        assert "abbrev putnam_2008_b2_solution : ℝ := sorry" not in result

    def test_theorem_preserved(self):
        result = replace_abbrev_in_statement(SIMPLE_SCAFFOLD, SIMPLE_ABBREV_DECL)
        assert "theorem putnam_2008_b2" in result
        assert "Tendsto" in result

    def test_noncomputable(self):
        result = replace_abbrev_in_statement(
            NONCOMPUTABLE_SCAFFOLD, NONCOMPUTABLE_ABBREV_DECL
        )
        assert "noncomputable abbrev putnam_1986_a3_solution : ℝ := π / 2" in result
        assert ":= sorry" not in result.split("abbrev")[1].split("theorem")[0]

    def test_set_type(self):
        result = replace_abbrev_in_statement(SET_SCAFFOLD, SET_ABBREV_DECL)
        assert "3, 4, 7, 8" in result

    def test_prop_type(self):
        result = replace_abbrev_in_statement(PROP_SCAFFOLD, PROP_ABBREV_DECL)
        assert "abbrev putnam_1995_a5_solution : Prop := True" in result

    def test_multiline_decl(self):
        result = replace_abbrev_in_statement(
            MULTILINE_DECL_SCAFFOLD, MULTILINE_DECL_ABBREV_DECL
        )
        assert "(2 : ℝ) / 3" in result
        assert "theorem putnam_2007_a1" in result

    def test_wrong_name_raises(self):
        wrong_decl = "abbrev wrong_name : ℝ := 0"
        with pytest.raises(ValueError):
            replace_abbrev_in_statement(SIMPLE_SCAFFOLD, wrong_decl)

    def test_required_name_mismatch_raises(self):
        with pytest.raises(ValueError):
            replace_abbrev_in_statement(
                SIMPLE_SCAFFOLD,
                SIMPLE_ABBREV_DECL,
                required_abbrev_name="wrong_name",
            )

    def test_required_name_match_passes(self):
        result = replace_abbrev_in_statement(
            SIMPLE_SCAFFOLD,
            SIMPLE_ABBREV_DECL,
            required_abbrev_name=SIMPLE_ABBREV_NAME,
        )
        assert SIMPLE_GT_ANSWER in result


# ---------------------------------------------------------------------------
# extract_last_abbrev_declaration
# ---------------------------------------------------------------------------

class TestExtractLastAbbrevDeclaration:
    # --- Basic extraction from lean4 code blocks ---

    def test_from_lean4_block(self):
        text = f"Here is my answer:\n```lean4\n{SIMPLE_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert SIMPLE_GT_ANSWER in result

    def test_from_lean_block(self):
        text = f"```lean\n{SIMPLE_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(text)
        assert result is not None

    def test_prefers_code_block_over_raw(self):
        # Both a code block and raw text contain abbrevs; block should win
        text = (
            "abbrev wrong_name : ℝ := 0\n"
            f"```lean4\n{SIMPLE_ABBREV_DECL}\n```"
        )
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert SIMPLE_ABBREV_NAME in result

    # --- Last block wins ---

    def test_returns_last_abbrev_in_block(self):
        text = (
            "```lean4\n"
            "abbrev foo_solution : ℝ := 1\n"
            "abbrev foo_solution : ℝ := 99\n"
            "```"
        )
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert ":= 99" in result

    def test_returns_last_code_block(self):
        text = (
            "```lean4\nabbrev foo_solution : ℝ := 1\n```\n"
            "```lean4\nabbrev foo_solution : ℝ := 99\n```"
        )
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert ":= 99" in result

    # --- Sorry rejection ---

    def test_rejects_sorry(self):
        text = "```lean4\nabbrev foo_solution : ℝ := sorry\n```"
        assert extract_last_abbrev_declaration(text) is None

    def test_accepts_non_sorry(self):
        text = "```lean4\nabbrev foo_solution : ℝ := -1\n```"
        result = extract_last_abbrev_declaration(text)
        assert result is not None

    # --- required_abbrev_name filter ---

    def test_required_name_matches(self):
        text = f"```lean4\n{SIMPLE_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(
            text, required_abbrev_name=SIMPLE_ABBREV_NAME
        )
        assert result is not None

    def test_required_name_rejects_wrong_name(self):
        text = f"```lean4\n{SIMPLE_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(
            text, required_abbrev_name="totally_different_name"
        )
        assert result is None

    # --- Multi-line abbrev body ---

    def test_multiline_abbrev_body_extracted(self):
        text = f"```lean4\n{MULTILINE_DECL_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert "(2 : ℝ) / 3" in result

    # --- Stopped at next top-level declaration ---

    def test_stops_at_theorem(self):
        text = (
            "```lean4\n"
            f"{SIMPLE_ABBREV_DECL}\n"
            "theorem putnam_2008_b2 : True := by sorry\n"
            "```"
        )
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert "theorem" not in result

    # --- Noncomputable ---

    def test_noncomputable_abbrev(self):
        text = f"```lean4\n{NONCOMPUTABLE_ABBREV_DECL}\n```"
        result = extract_last_abbrev_declaration(text)
        assert result is not None
        assert NONCOMPUTABLE_GT_ANSWER in result

    # --- Raw text (no code block) ---

    def test_raw_text_fallback(self):
        result = extract_last_abbrev_declaration(SIMPLE_ABBREV_DECL)
        assert result is not None
        assert SIMPLE_GT_ANSWER in result

    # --- Edge cases ---

    def test_returns_none_on_empty(self):
        assert extract_last_abbrev_declaration("") is None

    def test_returns_none_on_none(self):
        assert extract_last_abbrev_declaration(None) is None  # type: ignore[arg-type]

    def test_returns_none_when_no_abbrev(self):
        text = "```lean4\ntheorem foo : True := by sorry\n```"
        assert extract_last_abbrev_declaration(text) is None


# ---------------------------------------------------------------------------
# negate_theorem_statement
# ---------------------------------------------------------------------------

class TestNegateTheoremStatement:
    def test_simple_scaffold(self):
        result = negate_theorem_statement(SIMPLE_SCAFFOLD)
        assert "theorem putnam_2008_b2_neg" in result
        assert "¬ (" in result
        assert "putnam_2008_b2_solution" in result
        assert ":= by sorry" in result
        # Preamble and abbrev are preserved
        assert "import Mathlib" in result
        assert "abbrev putnam_2008_b2_solution : ℝ := sorry" in result

    def test_proposition_wrapped(self):
        result = negate_theorem_statement(SIMPLE_SCAFFOLD)
        # The whole original proposition should appear inside ¬ (...)
        assert "Tendsto" in result

    def test_no_duplicate_theorem(self):
        result = negate_theorem_statement(SIMPLE_SCAFFOLD)
        # After negation the original theorem name should NOT appear (only _neg)
        assert "theorem putnam_2008_b2\n" not in result
        assert "theorem putnam_2008_b2 " not in result

    def test_noncomputable_scaffold(self):
        result = negate_theorem_statement(NONCOMPUTABLE_SCAFFOLD)
        assert "theorem putnam_1986_a3_neg" in result
        assert "¬ (" in result
        assert ":= by sorry" in result
        assert "abbrev putnam_1986_a3_solution" in result

    def test_with_args(self):
        scaffold = (
            "import Mathlib\n\n"
            "abbrev foo_solution : ℕ := sorry\n\n"
            "theorem foo (n : ℕ) (h : n > 0) : foo_solution = 7 := by sorry"
        )
        result = negate_theorem_statement(scaffold)
        assert "theorem foo_neg (n : ℕ) (h : n > 0)" in result
        assert "¬ (foo_solution = 7)" in result

    def test_set_scaffold(self):
        result = negate_theorem_statement(SET_SCAFFOLD)
        assert "theorem putnam_1985_a5_neg" in result
        assert "¬ (" in result

    def test_prop_scaffold(self):
        result = negate_theorem_statement(PROP_SCAFFOLD)
        assert "theorem putnam_1995_a5_neg" in result
        assert "¬ (" in result

    def test_multiline_theorem_args(self):
        # NONCOMPUTABLE_SCAFFOLD has theorem args split over multiple lines
        result = negate_theorem_statement(NONCOMPUTABLE_SCAFFOLD)
        assert "theorem putnam_1986_a3_neg" in result
        # Args should still be present
        assert "(cot : ℝ → ℝ)" in result

    def test_ends_with_by_sorry(self):
        result = negate_theorem_statement(SIMPLE_SCAFFOLD)
        assert result.rstrip().endswith(":= by sorry")

    def test_no_theorem_raises(self):
        with pytest.raises(ValueError, match="no theorem found"):
            negate_theorem_statement("abbrev foo : ℝ := sorry\n-- no theorem here")

    def test_round_trip_with_replace_abbrev(self):
        # After replacing the abbrev placeholder with a real value, negation
        # should still work correctly.
        proved_lean = replace_abbrev_in_statement(SIMPLE_SCAFFOLD, SIMPLE_ABBREV_DECL)
        negated = negate_theorem_statement(proved_lean)
        assert "theorem putnam_2008_b2_neg" in negated
        assert "abbrev putnam_2008_b2_solution : ℝ := -1" in negated
        assert "¬ (" in negated

    def test_name_not_doubled(self):
        # A theorem named "theorem_foo" should become "theorem_foo_neg", not
        # "theorem_theorem_foo_neg".
        scaffold = (
            "abbrev theorem_foo_solution : ℕ := sorry\n"
            "theorem theorem_foo : theorem_foo_solution = 1 := by sorry"
        )
        result = negate_theorem_statement(scaffold)
        assert "theorem theorem_foo_neg" in result


# ---------------------------------------------------------------------------
# extract_theorem_signature
# ---------------------------------------------------------------------------

class TestExtractTheoremSignature:
    def test_simple_tactic_mode(self):
        text = "theorem foo (n : ℕ) : n = n := by sorry"
        sig = extract_theorem_signature(text)
        assert sig == "theorem foo (n : ℕ) : n = n := by"

    def test_simple_scaffold(self):
        # SIMPLE_SCAFFOLD uses term-mode `:=\nsorry` (no `by`).
        sig = extract_theorem_signature(SIMPLE_SCAFFOLD)
        assert sig is not None
        assert sig.startswith("theorem putnam_2008_b2")
        assert sig.rstrip().endswith(":=")
        assert "Tendsto" in sig

    def test_multiline_args(self):
        # NONCOMPUTABLE_SCAFFOLD also uses term-mode `:=\nsorry`.
        sig = extract_theorem_signature(NONCOMPUTABLE_SCAFFOLD)
        assert sig is not None
        assert "theorem putnam_1986_a3" in sig
        assert "(cot : ℝ → ℝ)" in sig
        assert sig.rstrip().endswith(":=")

    def test_term_mode_sorry(self):
        text = "theorem foo : True := sorry"
        sig = extract_theorem_signature(text)
        assert sig is not None
        assert sig == "theorem foo : True :="

    def test_preamble_ignored(self):
        sig = extract_theorem_signature(SIMPLE_SCAFFOLD)
        assert sig is not None
        assert not sig.startswith("import")
        assert "abbrev" not in sig

    def test_no_theorem(self):
        assert extract_theorem_signature("abbrev foo : ℝ := sorry") is None

    def test_empty(self):
        assert extract_theorem_signature("") is None

    def test_none_input(self):
        assert extract_theorem_signature(None) is None  # type: ignore[arg-type]

    def test_set_scaffold(self):
        sig = extract_theorem_signature(SET_SCAFFOLD)
        assert sig is not None
        assert "putnam_1985_a5" in sig

    def test_prop_scaffold_no_args(self):
        # PROP_SCAFFOLD uses term-mode `:=\n  sorry`.
        sig = extract_theorem_signature(PROP_SCAFFOLD)
        assert sig is not None
        assert "putnam_1995_a5" in sig
        assert sig.rstrip().endswith(":=")


# ---------------------------------------------------------------------------
# contains_theorem_signature
# ---------------------------------------------------------------------------

class TestContainsTheoremSignature:
    def test_exact_match(self):
        stmt = SIMPLE_SCAFFOLD
        code = SIMPLE_SCAFFOLD.replace("sorry", "exact foo")
        assert contains_theorem_signature(stmt, code)

    def test_sorry_replaced_by_tactic_block(self):
        stmt = "theorem foo (n : ℕ) : n = n := by sorry"
        code = "theorem foo (n : ℕ) : n = n := by\n  rfl"
        assert contains_theorem_signature(stmt, code)

    def test_extra_defs_before_theorem(self):
        stmt = SIMPLE_SCAFFOLD
        code = (
            "import Mathlib\n\n"
            "open Filter Topology Set Nat\n\n"
            "abbrev putnam_2008_b2_solution : ℝ := sorry\n\n"
            "lemma helper : True := trivial\n\n"
            + "\n".join(
                line for line in SIMPLE_SCAFFOLD.splitlines()
                if line.strip().startswith("theorem")
                or "Tendsto" in line
                or "hF" in line
                or line.strip().startswith("(")
                or line.strip().startswith(":")
                or line.strip() == "sorry"
            ).replace("sorry", "exact foo")
        )
        assert contains_theorem_signature(stmt, code)

    def test_whitespace_normalized(self):
        # Multi-line args collapsed to single line should still match
        stmt = NONCOMPUTABLE_SCAFFOLD
        sig = extract_theorem_signature(stmt)
        assert sig is not None
        # Build a version with all whitespace collapsed
        collapsed_code = " ".join(sig.split()) + " exact foo"
        assert contains_theorem_signature(stmt, collapsed_code)

    def test_bare_tactic_block_rejected(self):
        stmt = SIMPLE_SCAFFOLD
        code = "  rfl\n  done"
        assert not contains_theorem_signature(stmt, code)

    def test_different_theorem_rejected(self):
        stmt = SIMPLE_SCAFFOLD
        code = "theorem different_name : True := by trivial"
        assert not contains_theorem_signature(stmt, code)

    def test_no_theorem_in_statement(self):
        stmt = "abbrev foo : ℝ := sorry"
        code = "abbrev foo : ℝ := 42"
        assert not contains_theorem_signature(stmt, code)

    def test_multiline_scaffold(self):
        stmt = NONCOMPUTABLE_SCAFFOLD
        code = NONCOMPUTABLE_SCAFFOLD.replace(
            "\nsorry", "\n  exact foo"
        )
        assert contains_theorem_signature(stmt, code)

    def test_set_scaffold(self):
        stmt = SET_SCAFFOLD
        code = SET_SCAFFOLD.replace("sorry", "exact foo")
        assert contains_theorem_signature(stmt, code)
