# prove_formalizations commands — Blackwell RTX PRO 600 WS (96 GB VRAM)

All commands use `screen` for detached execution and log to a file.
Run from `/workspace` with Ollama already serving the model.

---

### New commands (actually ran, uncategorized)

only the correctly formalized, strip thinking, pass@4:
```sh
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/stripped_formalizations_goedel_pass/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 4 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp.log'
```
pass@1, strip thinking:
```sh
screen -dmS prove_formalizations_1 bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/stripped_formalizations_goedel_pass_1/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_pass1.log'
```
pass@1, no strip thinking:
```sh
mkdir logs/stripped_formalizations_goedel_pass_1_no_strip
cp logs/stripped_formalizations_goedel_pass/formalizations.jsonl logs/stripped_formalizations_goedel_pass_1_no_strip
screen -dmS prove_formalizations_1_strip bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-no-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/stripped_formalizations_goedel_pass_1_no_strip/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_pass1_no_strip.log'
screen -S monitor_tir_pass_no_strip tail -f prove_formalizations_tir_llamacpp_pass1_no_strip.log
```
pass@1, strip thinking, 1M tokens:
```sh
mkdir logs/stripped_formalizations_goedel_pass_1_strip_1M
cp logs/stripped_formalizations_goedel_pass/formalizations.jsonl logs/stripped_formalizations_goedel_pass_1_strip_1M
screen -dmS prove_formalizations_1 bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/stripped_formalizations_goedel_pass_1_strip_1M/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 1048576 \
        --continue &> prove_formalizations_tir_llamacpp_pass1_strip_1M.log'
rm prove_formalizations_tir_llamacpp_pass1_strip_1M.log
screen -S monitor_tir_pass_strip_1M tail -f prove_formalizations_tir_llamacpp_pass1_strip_1M.log
```

Whole dataset (1 proof attempt per attempt, strip thinking):
```sh
mkdir logs/tir_prover_pass1_all_attempts_strip
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/tir_prover_pass1_all_attempts_strip
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/tir_prover_pass1_all_attempts_strip/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_final_strip.log'
screen -dmS monitor_tir_final_strip tail -f prove_formalizations_tir_llamacpp_final_strip.log
```

Whole dataset (1 proof attempt per attempt, no strip):
```sh
mkdir logs/tir_prover_pass1_all_attempts_no_strip
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/tir_prover_pass1_all_attempts_no_strip
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-no-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/tir_prover_pass1_all_attempts_no_strip/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_final_no_strip.log'
screen -S monitor_tir_final tail -f prove_formalizations_tir_llamacpp_final_no_strip.log
```

Whole dataset (1 proof attempt per attempt, strip, 1M):
```sh
mkdir logs/tir_prover_pass1_all_attempts_strip_1M
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/tir_prover_pass1_all_attempts_strip_1M
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/tir_prover_pass1_all_attempts_strip_1M/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 1 \
        --limit-prover-tokens 1048576 \
        --continue &> prove_formalizations_tir_llamacpp_final_strip_1M.log'
screen -dmS monitor_tir_final tail -f prove_formalizations_tir_llamacpp_final_strip_1M.log
```

Whole dataset (4 proof attempt per attempt, strip thinking):
```sh
mkdir logs/tir_prover_pass4_all_attempts_strip
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/tir_prover_pass4_all_attempts_strip
screen -dmS prove_formalizations_pass4 bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-strip-thinking \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/tir_prover_pass4_all_attempts_strip/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --proof-retries 4 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_final_pass4_strip.log'
screen -dmS monitor_tir_final_pass4_strip tail -f prove_formalizations_tir_llamacpp_final_pass4_strip.log
```


Whole dataset, no strip thinking, add informal proof:
```sh
# on the machine: mkdir /workspace/logs/putnam_120b_tir_pass4_20min/
scp -i /Users/mila/.ssh/vastai -P 24409 logs/putnam_120b_tir_pass4_20min/attempts.jsonl root@ssh7.vast.ai:/workspace/logs/putnam_120b_tir_pass4_20min/attempts.jsonl
# on the machine: mkdir /workspace/data/conjecture_formalizer_inputs/
scp -i /Users/mila/.ssh/vastai -P 24409 data/conjecture_formalizer_inputs/references_putnam.csv root@ssh7.vast.ai:/workspace/data/conjecture_formalizer_inputs/references_putnam.csv

screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-no-strip-thinking \
        --add-informal-proof \
        --conjecturer-attempts logs/putnam_120b_tir_pass4_20min/attempts.jsonl \
        --problem-references data/conjecture_formalizer_inputs/references_putnam.csv \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 1 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 4 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_llamacpp_addinformal.log'

screen -S monitor_prove tail -f prove_formalizations_tir_llamacpp_addinformal.log
```

Whole dataset, strip thinking, no informal proof, fixed logging:
```sh
mkdir logs/prove_nocot_noinformal_fix_term
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/prove_nocot_noinformal_fix_term
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-strip-thinking \
        --tir-max-tokens 32768 \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 \
        --tir-llamacpp-max-concurrent 1 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/prove_nocot_noinformal_fix_term/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 4 \
        --limit-prover-tokens 262144 \
        --continue &> prove_formalizations_tir_nocot_llamacpp_noinformal_fix.log'

screen -S monitor_prove tail -f prove_formalizations_tir_nocot_llamacpp_noinformal_fix.log
```

Whole dataset, no strip thinking, no informal proof, fixed logging, hotfix tool calls, 2x max tokens, 16x turns, 3x time:
(2x4090)
```sh
mkdir logs/prove_cot_noinformal_v2_schema
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/prove_cot_noinformal_v2_schema
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-no-strip-thinking \
        --tir-max-tokens 32768 \
        --tir-max-turns 512 \
        --tir-timeout-seconds 1800 \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/prove_cot_noinformal_v2_schema/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 6 \
        --limit-prover-tokens 262144 \
        --continue \
        --continue-keep-inconclusive &> prove_formalizations_tir_cot_llamacpp_noinformal_v2.log'
screen -S monitor_prove tail -f prove_formalizations_tir_cot_llamacpp_noinformal_v2.log
```

Whole dataset, strip thinking, no informal proof, fixed logging, hotfix tool calls, 2x max tokens, 16x turns, 3x time:
(4x 6000)
```sh
mkdir logs/prove_nocot_noinformal_v2_schema
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/prove_nocot_noinformal_v2_schema
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-strip-thinking \
        --tir-max-tokens 32768 \
        --tir-max-turns 512 \
        --tir-timeout-seconds 1800 \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/prove_nocot_noinformal_v2_schema/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --limit-prover-tokens 262144 \
        --continue \
        --continue-keep-inconclusive &> prove_formalizations_tir_cot_llamacpp_noinformal_v2.log'
screen -S monitor_prove tail -f prove_formalizations_tir_cot_llamacpp_noinformal_v2.log
```

Whole dataset, strip thinking, add informal proof, fixed logging, hotfix tool calls, 2x max tokens, 16x turns, 3x time:
NOTE: Even better logging to debug stalling GPU
(4x 5000)
```sh
mkdir logs/prove_nocot_informal_v2_schema
cp logs/conjecture_formalization_logs_20mins/formalizations.jsonl logs/prove_nocot_informal_v2_schema
screen -dmS prove_formalizations bash -c \
     'source /root/.elan/env && \
      cd /workspace && \
      export PYTHONPATH=. && \
      python conjecturing_agents/prove_formalizations.py \
        --prover-type tir \
        --tir-strip-thinking \
        --add-informal-proof \
        --conjecturer-attempts logs/putnam_120b_tir_pass4_20min/attempts.jsonl \
        --problem-references data/conjecture_formalizer_inputs/references_putnam.csv \
        --tir-max-tokens 32768 \
        --tir-max-turns 512 \
        --tir-timeout-seconds 1800 \
        --tir-temperature 0.6 \
        --tir-top-p 0.95 \
        --tir-backend llamacpp \
        --tir-llamacpp-model unsloth/Qwen3.6-35B-A3B \
        --tir-llamacpp-base-urls  http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
        --tir-llamacpp-max-concurrent 3 \
        --tir-llamacpp-client-timeout 960 \
        --tir-llamacpp-presence-penalty 0.0 \
        --tir-llamacpp-tokenizer-path /workspace/tokenizers/Qwen3.5-27B \
        --formalizations-path logs/prove_nocot_informal_v2_schema/formalizations.jsonl \
        --lean-project-dir /workspace/mathlib4 \
        --parallelism 12 \
        --limit-prover-tokens 262144 \
        --continue \
        --continue-keep-inconclusive &> prove_formalizations_tir_nocot_llamacpp_informal_v2.log'
screen -S monitor_prove tail -f prove_formalizations_tir_nocot_llamacpp_informal_v2.log
```

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
mkdir logs/test_prove_formalizations && \
cp logs/stripped_formalizations_goedel_pass/formalizations.jsonl logs/test_prove_formalizations/ && \
  python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/test_prove_formalizations/formalizations.jsonl \
       --lean-project-dir /Users/mila/lean/mathlib4 \
       --prover-type tir \
       --add-informal-proof \
       --conjecturer-attempts logs/putnam_120b_tir_pass4_20min/attempts.jsonl  \
       --problem-references data/conjecture_formalizer_inputs/references_putnam.csv \
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
