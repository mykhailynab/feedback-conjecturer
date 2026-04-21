# prove_formalizations commands — Blackwell RTX PRO 600 WS (96 GB VRAM)

All commands use `screen` for detached execution and log to a file.
Run from `/workspace` with Ollama already serving the model.

---

## Goedel Prover V2 (Ollama backend)

```sh
screen -dmS prove_formalizations bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --parallelism 6 \
       --goedel-tokenizer-path tokenizers/goedel_prover_hf_tokenizer \
       --limit-prover-tokens 16000 \
       --continue &> prove_formalizations.log'
```

Resume with a larger budget (progressive token expansion):

```sh
screen -dmS prove_formalizations bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --parallelism 6 \
       --goedel-tokenizer-path tokenizers/goedel_prover_hf_tokenizer \
       --limit-prover-tokens 40960 \
       --continue &> prove_formalizations_40k.log'
```

---

## TIR prover — Qwen3.6 (Ollama chat API, Python + Lean tools)

```sh
screen -dmS prove_formalizations_tir bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend ollama \
       --tir-ollama-ollama-model qwen3.6 \
       --tir-ollama-tokenizer-path tokenizers/Qwen3.6-35B-A3B \
       --tir-ollama-top-k 20 \
       --tir-ollama-min-p 0.0 \
       --tir-ollama-presence-penalty 0.0 \
       --tir-ollama-repeat-penalty 1.0 \
       --parallelism 6 \
       --limit-prover-tokens 16000 \
       --continue &> prove_formalizations_tir_qwen36.log'
```

Resume with a larger budget:

```sh
screen -dmS prove_formalizations_tir bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend ollama \
       --tir-ollama-ollama-model qwen3.6 \
       --tir-ollama-tokenizer-path tokenizers/Qwen3.6-35B-A3B \
       --tir-ollama-top-k 20 \
       --tir-ollama-min-p 0.0 \
       --tir-ollama-presence-penalty 0.0 \
       --tir-ollama-repeat-penalty 1.0 \
       --parallelism 4 \
       --limit-prover-tokens 40960 \
       --continue &> prove_formalizations_tir_qwen36_40k.log'
```

---

## TIR prover — Qwen3.5 (Ollama chat API, Python + Lean tools)

```sh
screen -dmS prove_formalizations_tir bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend ollama \
       --tir-ollama-ollama-model qwen3.5 \
       --tir-ollama-tokenizer-path tokenizers/Qwen3.5-27B \
       --tir-ollama-top-k 20 \
       --tir-ollama-min-p 0.0 \
       --tir-ollama-presence-penalty 0.0 \
       --tir-ollama-repeat-penalty 1.0 \
       --parallelism 6 \
       --limit-prover-tokens 16000 \
       --continue &> prove_formalizations_tir_qwen35.log'
```

## To test locally:
### Ollama:
```sh
rm -rf logs/test_prove_formalizations && \
cp -r logs/stripped_formalizations_goedel_pass/ logs/test_prove_formalizations && \
  python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/test_prove_formalizations/formalizations.jsonl \
       --lean-project-dir /Users/mila/lean/mathlib4 \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend ollama \
       --tir-ollama-ollama-model qwen3.5 \
       --tir-ollama-tokenizer-path tokenizers/Qwen3.5-27B \
       --tir-ollama-top-k 20 \
       --tir-ollama-min-p 0.0 \
       --tir-ollama-presence-penalty 0.0 \
       --tir-ollama-repeat-penalty 1.0 \
       --parallelism 1 \
       --limit-prover-tokens 1000 \
       --max-records 1 \
       --print-agent-conv
```
### LLamacpp:
```sh
rm -rf logs/test_prove_formalizations && \                                 
cp -r logs/stripped_formalizations_goedel_pass/ logs/test_prove_formalizations && \
  python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/test_prove_formalizations/formalizations.jsonl \
       --lean-project-dir /Users/mila/lean/mathlib4 \
       --prover-type tir \
       --tir-temperature 0.6 \
       --tir-top-p 0.95 \
       --tir-backend llamacpp \
       --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
       --tir-llamacpp-client-timeout 960 \
       --tir-llamacpp-presence-penalty 0.0 \
       --tir-llamacpp-tokenizer-path tokenizers/Qwen3.5-27B \
       --parallelism 1 \
       --limit-prover-tokens 262144 \
       --tir-max-tokens 32768 \
       --max-records 1 \
       --print-agent-conv
```
NOTE: top-k, min-p, repeat-penalty missing (set on server level)

---

## TIR prover — Gemma 4 (Ollama chat API, Python + Lean tools)

```sh
screen -dmS prove_formalizations_tir bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --prover-type tir \
       --tir-backend ollama \
       --tir-ollama-ollama-model gemma4 \
       --tir-ollama-no-think \
       --tir-ollama-tokenizer-path tokenizers/gemma-4-31B-it \
       --parallelism 6 \
       --limit-prover-tokens 16000 \
       --continue &> prove_formalizations_tir_gemma4.log'
```

---

## Notes

- `--parallelism` controls concurrent workers. TIR uses a full Jupyter kernel per worker — use fewer workers (4) for large models (Qwen3.6 35B), more (6+) for smaller ones.
- `--limit-prover-tokens 0` (the default) removes the token budget — sessions run until proof, max turns, or timeout. Incomplete records from a prior limited run are still resumed from their saved conversation history; they just run without a cap.
- `--continue` without `--limit-prover-tokens` resumes incomplete sessions to completion (no budget cap) and re-runs other undecided records fresh.
- `--tir-ollama-no-think` disables extended thinking for models that don't support it (e.g. Gemma 4).
- Progressive budget workflow: start at 8000–16000 tokens, then re-run with `--continue` at 32000, then 40960. Each pass only re-processes incomplete/undecided records.
