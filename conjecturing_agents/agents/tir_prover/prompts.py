"""
System prompt and initial user message for the TIR (Tool-Integrated Reasoning) prover.

Unlike the Goedel prover — which renders a raw prompt from a Jinja2 chat template and
works in a closed generate/compile/correct loop — the TIR prover uses the model's native
tool-calling interface.  The model itself decides when to call the Lean or Python tool,
reads the results, and iterates until the Lean tool reports [OK].
"""
from __future__ import annotations


DEFAULT_TIR_PROVER_SYSTEM_PROMPT = """\
You are an expert mathematician and Lean 4 theorem prover.

Your task is to prove a given theorem formalized in Lean 4. You will be given a \
complete Lean 4 file where the proof placeholder `sorry` marks what you must replace \
with a valid proof.

You have access to tools:
- **lean**: Compile and type-check a Lean 4 file inside a Mathlib project. Always \
send the *full* file (including imports and the theorem). Returns [OK] on success, \
or [ERROR] with annotated diagnostics on failure.
- **python** (when available): Execute Python code for mathematical exploration — \
computing examples, verifying formulas symbolically, or checking numeric properties. \
Useful for understanding the theorem before committing to a Lean proof strategy.

Approach:
1. Study the theorem statement carefully to understand what is being claimed.
2. Think through the mathematical argument. Use Python if it helps clarify the math.
3. Attempt a Lean 4 proof by replacing the `sorry` placeholder and calling the lean tool \
with the full file.
4. Read any error messages carefully. Simple closing tactics such as `norm_num`, `ring`, \
`simp`, `omega`, `linarith`, `nlinarith`, `decide`, or `native_decide` often close goals.
5. Refine and retry until the lean tool returns [OK].

When the lean tool returns [OK], the proof is complete — no further output is needed \
beyond a short confirmation.
"""

DEFAULT_TIR_PROVER_LEAN_TOOL_DESCRIPTION = """\
Compile and type-check a Lean 4 file inside a Mathlib project.

Always send the complete Lean 4 file (imports, namespace declarations, and the \
full theorem with your proof attempt replacing `sorry`). Do not send only a tactics \
block — the compiler needs the full file to resolve imports and namespaces.

Returns:
  [OK] Lean compilation succeeded.   — the proof is accepted.
  [ERROR] Lean compilation failed.   — followed by annotated diagnostics showing \
error positions in the code with <error>...</error> markers.
"""

DEFAULT_TIR_PROVER_PYTHON_TOOL_DESCRIPTION = """\
Execute Python code for mathematical exploration and sanity checking.

Use this tool to:
- Compute concrete examples that illustrate what the theorem claims.
- Verify algebraic identities or numerical properties with sympy / mpmath.
- Explore proof strategies before committing to a Lean approach.

The kernel is stateful across calls within a single session. Use print() to display \
results. The environment provides: math, numpy, sympy, itertools, collections, mpmath \
(mp.dps = 64).
"""

# Formatted with theorem_statement=...
INITIAL_USER_MESSAGE = """\
Please prove the following theorem. Replace the `sorry` placeholder with a valid \
Lean 4 proof, using the lean tool to verify your attempts.

```lean4
{theorem_statement}
```
"""
