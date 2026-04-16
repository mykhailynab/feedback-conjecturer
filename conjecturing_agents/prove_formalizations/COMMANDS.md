On the blackwell RTX PRO 600 WS (96 GB VRAM)
```sh
screen -dmS prove_formalizations bash -c '
     source /root/.elan/env &&
     cd /workspace &&
     export PYTHONPATH=. &&
     python conjecturing_agents/prove_formalizations.py \
       --formalizations-path logs/conjecture_formalization_logs_20mins/formalizations.jsonl \
       --lean-project-dir /workspace/mathlib4 \
       --parallelism 6 \
       --limit-prover-tokens 16000 \
       --continue &> prove_formalizations.log'
```