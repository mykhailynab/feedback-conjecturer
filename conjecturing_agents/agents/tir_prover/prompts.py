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

Approach:
1. Study the theorem statement carefully to understand what is being claimed.
2. Think through the mathematical argument. Use Python tool if it helps clarify the math.
3. Attempt a Lean 4 proof by replacing the `sorry` placeholder and calling the lean tool \
with the full file. You may also attempt to compile any proof section.
4. Read any error messages carefully. Simple closing tactics such as `norm_num`, `ring`, \
`simp`, `omega`, `linarith`, `nlinarith`, `decide`, or `native_decide` often close goals.
5. Refine and retry until the lean tool returns [OK] for all parts of the proof.

When ready, use the lean_final tool to submit the final proof file, replacing the `sorry` \
statement in the initial proof placeholder.
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
- Complex calculations that would be error-prone by hand
- Numerical verification of analytical results
- Generating examples or testing conjectures
- Brute-force verification for small cases
- Verify algebraic identities or numerical properties with sympy / mpmath.
- Explore proof strategies before committing to a Lean approach.

The environment is a stateful Jupyter notebook. Code persists between executions. Always use print() \
to display results. Write clear, well-commented code. \
Remember: Code should support your mathematical reasoning, not replace it. \
The environment provides: math, numpy, sympy, itertools, collections, mpmath \
(mp.dps = 64).

You have access to `math`, `numpy`, and `sympy` for:
# Symbolic Computation (sympy):
- Algebraic manipulation and simplification
- Solving equations and systems of equations
- Symbolic differentiation and integration
- Number theory functions (primes, divisors, modular arithmetic)
- Polynomial operations and factorization
- Working with mathematical expressions symbolically
# Numerical Computation (numpy):
- Array operations and linear algebra
- Efficient numerical calculations for large datasets
- Matrix operations and eigenvalue problems
- Statistical computations
# Mathematical Functions (math):
- Standard mathematical functions (trig, log, exp)
- Constants like pi and e
- Basic operations for single values
Best Practices:
- Use sympy for exact symbolic answers when possible
- Use numpy for numerical verification and large-scale computation
- Combine symbolic and numerical approaches: derive symbolically, verify numerically
- Document your computational strategy clearly
- Validate computational results against known cases or theoretical bounds
"""

# Formatted with theorem_statement=...
INITIAL_USER_MESSAGE = """\
Please prove the following theorem. Replace the `sorry` placeholder with a valid \
Lean 4 proof, using the lean tool to verify your attempts.

```lean4
{theorem_statement}
```
"""
