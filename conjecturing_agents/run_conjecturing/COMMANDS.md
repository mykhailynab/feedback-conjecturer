Run on the A100 server:
```
screen -dmS conjecturing bash -c 'PYTHONPATH=. python conjecturing_agents/run_conjecturing.py  \
    --backend ollama \
    --ollama-model qwen3.5:27b \
    --ollama-host http://localhost:11434 \
    --reference-path data/conjecture_formalizer_inputs/references_putnam.csv \
    --attempts-per-problem 4 \
    --agent-parallelism 4 \
    --log-dir logs/local_qwen35_conjecturing_pass4 \
    --solver-timeout-seconds 1200 --checker-timeout-seconds 600 &> conjecturing.log'
```
