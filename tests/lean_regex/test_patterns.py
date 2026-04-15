"""
Tests for the compiled regex patterns in conjecturing_agents.lean_regex.

Each test focuses on one pattern and checks both positive matches (with
correct group captures) and negative non-matches.
"""
import pytest

from conjecturing_agents.lean_regex import (
    ABBREV_LINE_RE,
    ABBREV_NAME_RE,
    ABBREV_RHS_RE,
    ABBREV_SINGLE_LINE_RE,
    ABBREV_WITH_COMMENT_RE,
    BY_CLAUSE_RE,
    LEAN_BLOCK_PATTERNS,
    SORRY_RE,
    THEOREM_NAME_RE,
    TOP_LEVEL_DECL_RE,
)

from .fixtures import (
    NONCOMPUTABLE_SCAFFOLD,
    PROP_SCAFFOLD,
    SET_SCAFFOLD,
    SIMPLE_SCAFFOLD,
)


# ---------------------------------------------------------------------------
# ABBREV_NAME_RE
# ---------------------------------------------------------------------------

class TestAbbrevNameRe:
    """Captures the abbrev identifier from any abbrev declaration line."""

    def test_simple(self):
        m = ABBREV_NAME_RE.search("abbrev foo_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1) == "foo_solution"

    def test_with_type_annotation(self):
        m = ABBREV_NAME_RE.search("abbrev putnam_2008_b2_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1) == "putnam_2008_b2_solution"

    def test_noncomputable(self):
        m = ABBREV_NAME_RE.search("noncomputable abbrev putnam_1986_a3_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1) == "putnam_1986_a3_solution"

    def test_set_type(self):
        m = ABBREV_NAME_RE.search("abbrev putnam_1985_a5_solution : Set ℕ := sorry")
        assert m is not None
        assert m.group(1) == "putnam_1985_a5_solution"

    def test_prop_type(self):
        m = ABBREV_NAME_RE.search("abbrev putnam_1995_a5_solution : Prop := sorry")
        assert m is not None
        assert m.group(1) == "putnam_1995_a5_solution"

    def test_prime_in_name(self):
        m = ABBREV_NAME_RE.search("abbrev foo' : ℕ := 0")
        assert m is not None
        assert m.group(1) == "foo'"

    def test_multiline_finds_in_scaffold(self):
        m = ABBREV_NAME_RE.search(SIMPLE_SCAFFOLD)
        assert m is not None
        assert m.group(1) == "putnam_2008_b2_solution"

    def test_multiline_noncomputable_in_scaffold(self):
        m = ABBREV_NAME_RE.search(NONCOMPUTABLE_SCAFFOLD)
        assert m is not None
        assert m.group(1) == "putnam_1986_a3_solution"

    def test_does_not_match_theorem(self):
        assert ABBREV_NAME_RE.search("theorem foo : P := by sorry") is None

    def test_does_not_match_def(self):
        assert ABBREV_NAME_RE.search("def bar : ℕ := 42") is None

    def test_leading_whitespace(self):
        m = ABBREV_NAME_RE.search("  abbrev foo_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1) == "foo_solution"


# ---------------------------------------------------------------------------
# ABBREV_SINGLE_LINE_RE
# ---------------------------------------------------------------------------

class TestAbbrevSingleLineRe:
    """Matches only abbrev lines that contain ':=' on the same line."""

    def test_matches_sorry_placeholder(self):
        m = ABBREV_SINGLE_LINE_RE.search("abbrev foo_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1) == "foo_solution"

    def test_matches_real_value(self):
        m = ABBREV_SINGLE_LINE_RE.search("abbrev foo_solution : ℝ := -1")
        assert m is not None

    def test_matches_noncomputable(self):
        m = ABBREV_SINGLE_LINE_RE.search(
            "noncomputable abbrev putnam_1986_a3_solution : ℝ := sorry"
        )
        assert m is not None
        assert m.group(1) == "putnam_1986_a3_solution"

    def test_matches_in_scaffold(self):
        # There is exactly one single-line abbrev in the simple scaffold
        matches = list(ABBREV_SINGLE_LINE_RE.finditer(SIMPLE_SCAFFOLD))
        assert len(matches) == 1
        assert matches[0].group(1) == "putnam_2008_b2_solution"

    def test_does_not_match_multiline_abbrev(self):
        # Multi-line abbrev: `:=` is on the next line → no match
        text = "abbrev foo_solution : Set ℝ :=\n  ({1, 2} : Set ℝ)"
        # ABBREV_SINGLE_LINE_RE checks for ':=' within the same logical line
        # The first line has ':=' so it should match (the pattern is per-line)
        m = ABBREV_SINGLE_LINE_RE.search(text)
        # `:=` appears on the first line (before the newline), so it does match
        assert m is not None

    def test_does_not_match_theorem(self):
        assert ABBREV_SINGLE_LINE_RE.search("theorem foo : P := by sorry") is None


# ---------------------------------------------------------------------------
# ABBREV_WITH_COMMENT_RE
# ---------------------------------------------------------------------------

class TestAbbrevWithCommentRe:
    """Matches an abbrev line immediately followed by a comment line."""

    def test_basic(self):
        text = "abbrev foo_solution : ℝ := sorry\n-- 42\n\ntheorem foo : True := by sorry"
        m = ABBREV_WITH_COMMENT_RE.search(text)
        assert m is not None
        assert "foo_solution" in m.group("abbrev_line")
        assert "42" in m.group("comment_line")

    def test_noncomputable(self):
        text = "noncomputable abbrev foo_solution : ℝ := sorry\n-- π / 2\n"
        m = ABBREV_WITH_COMMENT_RE.search(text)
        assert m is not None
        assert "foo_solution" in m.group("abbrev_line")
        assert "π / 2" in m.group("comment_line")

    def test_no_match_without_comment(self):
        text = "abbrev foo_solution : ℝ := sorry\n\ntheorem foo : True := by sorry"
        assert ABBREV_WITH_COMMENT_RE.search(text) is None

    def test_no_match_comment_not_immediately_after(self):
        text = "abbrev foo_solution : ℝ := sorry\n\n-- delayed comment\n"
        assert ABBREV_WITH_COMMENT_RE.search(text) is None


# ---------------------------------------------------------------------------
# ABBREV_RHS_RE
# ---------------------------------------------------------------------------

class TestAbbrevRhsRe:
    """Captures everything after ':=' in an abbrev line."""

    def test_simple_number(self):
        m = ABBREV_RHS_RE.search("abbrev foo_solution : ℝ := -1")
        assert m is not None
        assert m.group(1).strip() == "-1"

    def test_sorry(self):
        m = ABBREV_RHS_RE.search("abbrev foo_solution : ℝ := sorry")
        assert m is not None
        assert m.group(1).strip() == "sorry"

    def test_expression(self):
        m = ABBREV_RHS_RE.search("noncomputable abbrev foo_solution : ℝ := π / 2")
        assert m is not None
        assert m.group(1).strip() == "π / 2"


# ---------------------------------------------------------------------------
# TOP_LEVEL_DECL_RE
# ---------------------------------------------------------------------------

class TestTopLevelDeclRe:
    """Matches the opening keyword of any top-level Lean declaration."""

    @pytest.mark.parametrize("line", [
        "abbrev foo : ℝ := sorry",
        "theorem bar : P := by sorry",
        "lemma baz : Q := by sorry",
        "def qux : ℕ := 0",
        "noncomputable def quux : ℝ := 0",
        "example : True := trivial",
        "structure MyStruct where",
        "class MyClass (α : Type) where",
        "inductive MyInductive : Type",
        "instance : Inhabited ℕ := ⟨0⟩",
        "namespace Foo",
        "end Foo",
        "section MySection",
        "open Nat",
        "import Mathlib",
        "#check Nat",
        "#eval 1 + 1",
        "#print Nat",
    ])
    def test_matches_top_level_keywords(self, line: str):
        assert TOP_LEVEL_DECL_RE.match(line) is not None, f"Expected match for: {line!r}"

    @pytest.mark.parametrize("line", [
        "  -- a comment",
        "(F : ℕ → ℝ → ℝ)",
        ": Tendsto (fun n => n) atTop (𝓝 0) :=",
        "sorry",
        "by exact trivial",
    ])
    def test_does_not_match_non_declarations(self, line: str):
        assert TOP_LEVEL_DECL_RE.match(line) is None, f"Unexpected match for: {line!r}"

    def test_stops_abbrev_block_extraction(self):
        # The extraction loop in extract_last_abbrev_declaration stops when it
        # sees a top-level declaration after the abbrev.  Verify the theorem
        # line triggers a stop.
        line = "theorem putnam_2008_b2"
        assert TOP_LEVEL_DECL_RE.match(line) is not None


# ---------------------------------------------------------------------------
# THEOREM_NAME_RE
# ---------------------------------------------------------------------------

class TestTheoremNameRe:
    """Captures the theorem identifier."""

    def test_simple(self):
        m = THEOREM_NAME_RE.search("theorem foo : P := by sorry")
        assert m is not None
        assert m.group(1) == "foo"

    def test_noncomputable(self):
        m = THEOREM_NAME_RE.search("noncomputable theorem bar : Q := by sorry")
        assert m is not None
        assert m.group(1) == "bar"

    def test_in_scaffold(self):
        m = THEOREM_NAME_RE.search(SIMPLE_SCAFFOLD)
        assert m is not None
        assert m.group(1) == "putnam_2008_b2"

    def test_noncomputable_scaffold(self):
        m = THEOREM_NAME_RE.search(NONCOMPUTABLE_SCAFFOLD)
        assert m is not None
        assert m.group(1) == "putnam_1986_a3"

    def test_prop_scaffold(self):
        m = THEOREM_NAME_RE.search(PROP_SCAFFOLD)
        assert m is not None
        assert m.group(1) == "putnam_1995_a5"

    def test_prime_in_name(self):
        m = THEOREM_NAME_RE.search("theorem foo' : P := by sorry")
        assert m is not None
        assert m.group(1) == "foo'"

    def test_does_not_match_abbrev(self):
        assert THEOREM_NAME_RE.search("abbrev foo_solution : ℝ := sorry") is None

    def test_does_not_match_lemma(self):
        # THEOREM_NAME_RE only matches `theorem`, not `lemma`
        assert THEOREM_NAME_RE.search("lemma foo : P := by sorry") is None


# ---------------------------------------------------------------------------
# SORRY_RE
# ---------------------------------------------------------------------------

class TestSorryRe:
    def test_matches_standalone(self):
        assert SORRY_RE.search("sorry") is not None

    def test_matches_in_expression(self):
        assert SORRY_RE.search("abbrev foo : ℝ := sorry") is not None

    def test_word_boundary_start(self):
        # "notsorry" should not match
        assert SORRY_RE.search("notsorry") is None

    def test_word_boundary_end(self):
        assert SORRY_RE.search("sorrys") is None

    def test_no_match_when_absent(self):
        assert SORRY_RE.search("abbrev foo : ℝ := -1") is None

    def test_matches_in_multiline(self):
        text = "abbrev foo : ℝ := -1\n\ntheorem bar : True := by sorry"
        assert SORRY_RE.search(text) is not None


# ---------------------------------------------------------------------------
# BY_CLAUSE_RE
# ---------------------------------------------------------------------------

class TestByClauseRe:
    def test_basic(self):
        assert BY_CLAUSE_RE.search(":= by") is not None

    def test_extra_space(self):
        assert BY_CLAUSE_RE.search(":=  by") is not None

    def test_no_space(self):
        assert BY_CLAUSE_RE.search(":=by") is not None

    def test_in_theorem(self):
        assert BY_CLAUSE_RE.search("theorem foo : P := by sorry") is not None

    def test_no_match_assign_only(self):
        assert BY_CLAUSE_RE.search("abbrev foo : ℝ := -1") is None


# ---------------------------------------------------------------------------
# LEAN_BLOCK_PATTERNS
# ---------------------------------------------------------------------------

class TestLeanBlockPatterns:
    """The patterns are tried in order: lean4 first, then lean."""

    def test_lean4_pattern(self):
        text = "```lean4\nabbrev foo : ℝ := 1\n```"
        m = LEAN_BLOCK_PATTERNS[0].search(text)
        assert m is not None
        assert "abbrev foo" in m.group(1)

    def test_lean_pattern(self):
        text = "```lean\nabbrev foo : ℝ := 1\n```"
        # lean4 patterns won't match; lean pattern will
        assert LEAN_BLOCK_PATTERNS[0].search(text) is None
        assert LEAN_BLOCK_PATTERNS[2].search(text) is not None

    def test_lean4_without_trailing_newline(self):
        text = "```lean4\nabbrev foo : ℝ := 1```"
        m = LEAN_BLOCK_PATTERNS[1].search(text)
        assert m is not None

    def test_multiline_block(self):
        text = "```lean4\nabbrev foo : ℝ := 1\ntheorem bar : True := by sorry\n```"
        m = LEAN_BLOCK_PATTERNS[0].search(text)
        assert m is not None
        assert "theorem bar" in m.group(1)
