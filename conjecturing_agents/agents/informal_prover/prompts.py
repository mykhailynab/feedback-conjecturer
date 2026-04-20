"""
System prompt and user message template for the informal proof agent.

The informal proof agent takes a solver's output (problem, solution trace, answer,
and Lean 4 statement) and produces a structured informal certification proof.  The
output is later fed into the formal prover's prompt to guide its proof search.
"""
from __future__ import annotations


DEFAULT_INFORMAL_PROVER_SYSTEM_PROMPT = """\
You are an expert mathematician writing a rigorous certification proof.

You are given:
1. A math competition problem statement.
2. A solver's reasoning trace (which may contain exploratory steps, dead ends, \
and computational verification alongside the core argument).
3. The claimed answer.
4. A formal Lean 4 theorem statement showing what must ultimately be proved.

Your task is to produce a clear, structured informal proof that the claimed answer \
is correct.  This proof will later be used to guide a formal theorem prover, so it \
must be:

- **Certifying, not exploratory.**  The answer is already given.  Do not search for it.  \
Prove that it is correct.
- **Distilled and reorganized.**  Do not reproduce the solver's trace verbatim.  \
Extract the essential argument, fill in any gaps, and present it in logical order.
- **Rigorous.**  Every key claim must be justified.  State intermediate results \
explicitly and explain why each step follows.
- **Structured.**  Use clearly labeled sections (e.g., "Setup", "Key Claim", "Proof \
of Key Claim", "Verification", "Conclusion") so a reader (or a downstream prover) \
can follow the argument step by step.

The Lean 4 theorem statement is provided so you know what the formal proof must \
establish. Your output should be in natural-language mathematics, not Lean code.\
"""

# Python tool description — borrowed from the TIR prover agent.
DEFAULT_INFORMAL_PROVER_PYTHON_TOOL_DESCRIPTION = """\
Execute Python code for mathematical exploration and sanity checking.

Use this tool to:
- Compute concrete examples that illustrate what the theorem claims.
- Complex calculations that would be error-prone by hand
- Numerical verification of analytical results
- Generating examples or testing conjectures
- Brute-force verification for small cases
- Verify algebraic identities or numerical properties with sympy / mpmath.

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


INITIAL_USER_MESSAGE = """\
## Problem statement
{problem_statement}

## Solver's reasoning trace
{solution_trace}

## Claimed answer
{answer}

## Formal theorem statement (Lean 4, for reference)
```lean4
{lean_statement}
```

Please write a detailed informal certification proof that the claimed answer is correct.\
"""
