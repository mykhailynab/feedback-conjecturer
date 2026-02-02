#!/usr/bin/env python3
"""
Run Goedel-Prover-V2-32B GGUF via Ollama, using a HuggingFace-style Jinja chat template.

Requirements:
  pip install ollama jinja2

Assumptions:
  - You created the model in Ollama as: `goedel-v2`
  - Your Modelfile is just: FROM /path/to/goedel.gguf
"""

import time
import ollama
from jinja2 import Environment

# --- Your exact template (paste as-is) ---
CHAT_TEMPLATE = open("ignore/goedel_template.jinja", "r").read()


def render_with_template(messages, tools=None, add_generation_prompt=True, enable_thinking=True) -> str:
    """
    Render the HF-style chat template into a single prompt string.
    We adapt dicts to objects with attribute access by wrapping them in a tiny proxy.
    """
    class Obj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    msg_objs = [Obj(m) for m in messages]
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(CHAT_TEMPLATE)
    return tmpl.render(
        tools=tools or [],
        messages=msg_objs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def main():
    # --- Your Lean statement + prompt ---
    formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
""".strip()

    prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

    messages = [
        {"role": "user", "content": prompt.format(formal_statement)}
    ]

    # --- Path A: emulate tokenizer.apply_chat_template(...) by rendering your Jinja template ---
    rendered_prompt = render_with_template(
        messages=messages,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=True,  # set False if you want an *empty* <think> block inserted
    )

    # Ollama generation options (tune as needed)
    # NOTE: num_ctx must be large enough for your prompt + the long output.
    # If your Ollama build/model can't do 32768 ctx, reduce num_predict and/or num_ctx.
    options = {
        "seed": 30,
        "temperature": 0.0,
        # "num_predict": 32768,
        # "num_ctx": 32768,
        # You can also add: "top_p": 0.95, "repeat_penalty": 1.05, ...
    }

    out_path = "goedel_output.txt"
    start = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for part in ollama.generate(
            model="goedel-v2",
            prompt=rendered_prompt,
            options=options,
            stream=True,
        ):
            chunk = part.get("response", "")
            if chunk:
                f.write(chunk)
                f.flush()

    elapsed = time.time() - start
    print(f"\nDone. Wrote output to: {out_path}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    # --- Path B (simpler): let Ollama apply its own chat formatting ---
    # If you DON'T actually need the custom <|im_start|>... template, use this instead.
    #
    # start = time.time()
    # resp = ollama.chat(
    #     model="goedel-v2",
    #     messages=messages,
    #     options=options,
    # )
    # print(resp["message"]["content"])
    # print("Elapsed seconds:", time.time() - start)


if __name__ == "__main__":
    main()
