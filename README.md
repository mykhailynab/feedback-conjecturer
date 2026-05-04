# Efficient Conjecturing in Neural ATPs

A agentic pipeline that performs end-to-end automated theorem proving: informal conjecturing, conjecture formalization into Lean 4, and formal proof search. Evaluated on the no-answer split of PutnamBench (343 problems).

**Status**: Work-in-progress. Core functionality is kept for reproducibility with the thesis, but new features may be added as part of future work.

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

## Key results

- 27 no-answer PutnamBench problems proved end-to-end (state-of-the-art on no-answer split)
- Goedel-Prover-V2 proved 27/343 problems; TIR-Prover (Keep-CoT / Add-Informal) proved 21/343
- Informal solver achieved 77% expert-check accuracy at pass@4 (312/343 problems)
- Conjecture formalization compiled successfully in 97% of answered attempts

## Dependencies

See `requirements.txt`. Key: `vllm`, `openai`, `ollama`, `polars`, `jupyter_client`, `transformers`.
