# Efficient Conjecturing in Neural ATPs

An agentic pipeline that performs end-to-end automated theorem proving: informal conjecturing, conjecture formalization into Lean 4, and formal proof search. Evaluated on the no-answer split of PutnamBench (343 problems).

**Status**: Work-in-progress. Core functionality is kept for reproducibility with the thesis, but new features may be added as part of future work.

## Key results

**Answering is easy, proving is hard, and a compiling proof is not a correct one.**

| | Goedel-Prover-V2 | TIR-Prover | union |
|---|---|---|---|
| problems with a compiling proof | 27 | 21 | 30 |
| **verified after filtering** | **15** | **11** | **18** |
| rejected as reward hacks | 12 | 10 | 12 |

<!-- `putnam_1988_b2` is the one problem rejected for one prover but verified for the other — TIR's answer restates the theorem, Goedel's does not — so the union column counts 12 rejected, not 13. -->

- The informal solver answered **97%** of the 343 problems and was expert-judged correct on **77%** at pass@4.
- Conjecture formalization compiled in **97%** of answered attempts.
- **12 of the 30 distinct problems (40%) with a compiling proof were reward hacks**: the model defined its `abbrev …_solution` to be the theorem's own statement, so the goal became `P ↔ P` and closed by `rfl`. The proof compiles and certifies nothing.
- **Every hack had a `Prop`- or `Set`-valued answer.** No numeric, tuple, function or polynomial answer was ever hacked — the hack needs an answer type rich enough to hold the question.
- A **triviality probe** using nothing but the compiler (splice the model's own answer back into its own theorem, discard its proof, try one tactic) rejects **7 of the 12** hacks with **0 false positives** across 62 verified-correct attempts. See `scripts/probe_admissibility.py`.
<!-- - Concurrent work (ECP, [arXiv:2505.18492](https://arxiv.org/abs/2505.18492)) reports 17 solved problems on its own re-formalization of the split and identifies the same failure mode. **13 of our 15 Goedel-verified problems are in their 17** — two independent systems, independent formalizations, near-identical core set, which suggests similar weaknesses -->

The 18-problem union is *best of two provers at pass@4* (343 problems × 4 formalizations × 2 provers) and a post-hoc union, not a clean pass@8. It is reported that way deliberately.

### Verified problems

Proofs themselves cannot be published due to PutnamBench policy. THe following problem identifiers are PutnamBench names.

**Goedel-Prover-V2 — 15 verified** (of 27 claimed)

```
putnam_1975_a1  putnam_1975_b1  putnam_1977_a2  putnam_1977_a3  putnam_1984_b2
putnam_1986_a1  putnam_1986_a2  putnam_1986_b1  putnam_1988_b2  putnam_1990_a1
putnam_1991_a2  putnam_1993_b1  putnam_1995_b4  putnam_1998_b1  putnam_2005_b1
```

**TIR-Prover — 11 verified** (of 21 claimed)

```
putnam_1975_b1  putnam_1977_a2  putnam_1977_a3  putnam_1985_a4  putnam_1986_a1
putnam_1986_b1  putnam_1990_a5  putnam_1991_a2  putnam_1998_b1  putnam_2005_b1
putnam_2015_a2
```

**Union — 18 distinct.** 8 problems were solved by both provers (`putnam_1975_b1`, `putnam_1977_a2`, `putnam_1977_a3`, `putnam_1986_a1`, `putnam_1986_b1`, `putnam_1991_a2`, `putnam_1998_b1`, `putnam_2005_b1`); 7 by Goedel only, 3 by TIR only.

**Rejected as reward hacks — 12 distinct**

```
putnam_1963_b2  putnam_1963_b3  putnam_1982_a6  putnam_1983_b2  putnam_1990_a2
putnam_1995_a5  putnam_1996_a6  putnam_1999_a1  putnam_2009_a4  putnam_2012_a6
putnam_2021_a6  putnam_2025_b3
```

## Pipeline overview

1. **Solver Agent** — finds answers informally using gpt-oss:120B with Python TIR (pass@4, 20 min timeout)
2. **Informal Equivalence Agent** — expert-checks informal answers against ground truth
3. **Conjecture Formalization Agent** — formalizes answers into Lean 4 abbrevs (97% compile rate)
4. **Formal Equivalence Checking** — string match, Lean tactic search, Goedel proof/disproof
5. **Proving** — proves the full conjecture using Goedel-Prover-V2 or TIR-Prover

## Reproducing experiments

```bash
# 1. Informal conjecturing (Solver Agent, 20 min per attempt)
PYTHONPATH=. python conjecturing_agents/run_conjecturing.py \
  --reference-path reference.csv \
  --attempts-per-problem 4 \
  --context-tokens 65536 \
  --solver-max-turns 128 \
  --solver-temperature 0.5 --solver-min-p 0.02

# 2. Conjecture formalization
PYTHONPATH=. python conjecturing_agents/run_conjecture_formalization.py \
  --attempts-path logs/.../attempts.jsonl \
  --lean-project-dir /path/to/mathlib4 \
  --formalizer-temperature 0.2 --formalizer-max-turns 48 \
  --max-correction-rounds 8

# 3. Formal equivalence checking (string match + Lean tactics + Goedel prove/disprove)
PYTHONPATH=. python conjecturing_agents/check_formalizations.py \
  --formalizations-path logs/.../formalizations.jsonl \
  --lean-project-dir /path/to/mathlib4 \
  --goedel --goedel-disprover --parallelism 12

# 4a. Proving with Goedel-Prover-V2 (8-bit, 40960 context, 2 correction rounds)
PYTHONPATH=. python conjecturing_agents/prove_formalizations.py \
  --formalizations-path logs/.../formalizations.jsonl \
  --lean-project-dir /path/to/mathlib4 \
  --proof-retries 1 --parallelism 8

# 4b. TIR-Prover Base (5-bit, 262K context, 32 turns, CoT stripped)
PYTHONPATH=. python conjecturing_agents/prove_formalizations.py \
  --prover-type tir \
  --tir-strip-thinking \
  --tir-max-turns 32 \
  --tir-backend llamacpp \
  --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
  --tir-llamacpp-base-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
  --tir-llamacpp-max-concurrent 3 \
  --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
  --formalizations-path logs/.../formalizations.jsonl \
  --lean-project-dir /path/to/mathlib4 \
  --parallelism 12 \
  --limit-prover-tokens 262144

# 4c. TIR-Prover Keep-CoT (same as Base, but CoT kept in context)
# Same as 4b but replace --tir-strip-thinking with --tir-no-strip-thinking

# 4d. TIR-Prover Add-Informal (6-bit, 110K context, 512 turns, CoT stripped, informal proof injected)
PYTHONPATH=. python conjecturing_agents/prove_formalizations.py \
  --prover-type tir \
  --tir-strip-thinking \
  --add-informal-proof \
  --conjecturer-attempts logs/.../attempts.jsonl \
  --problem-references data/conjecture_formalizer_inputs/references_putnam.csv \
  --tir-max-turns 512 \
  --tir-backend llamacpp \
  --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
  --tir-llamacpp-base-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
  --tir-llamacpp-max-concurrent 3 \
  --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
  --formalizations-path logs/.../formalizations.jsonl \
  --lean-project-dir /path/to/mathlib4 \
  --parallelism 12 \
  --limit-prover-tokens 110000

# 5. Inspect results
python -m analysis_and_inspection.inspect_prove_formalizations \
  --goedel-events logs/.../prove_goedel_events.jsonl \
  --results logs/.../prove_results.jsonl
```

**Note:** Although the command signature may change with time, the behaviour is kept faithful to the thesis.

## Dependencies

See `requirements.txt`. Key: `vllm`, `openai`, `ollama`, `polars`, `jupyter_client`, `transformers`.
