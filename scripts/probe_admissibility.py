"""Standalone triviality/admissibility probe.

For every *proved* attempt in a prove_results.jsonl, splice the model's own
`abbrev …_solution` back into its own formalized theorem statement, replace
`:= sorry` with `:= by <tactic>`, and compile.  If a tactic closes the goal,
the answer restates the question and the "proof" is contentless.

This is the production `--only-nontrivial` check from extract_proofs.py,
generalised from a single tactic to a battery, and run *without* any
ground-truth equivalence filter so the probe can be evaluated on its own.

Writes one JSON record per (problem_id, attempt, tactic) to --output.
Resumable: re-run with the same --output to fill in what is missing.

Usage:
  PYTHONPATH=. python scripts/probe_admissibility.py \
      --results        logs/prove_formalizations_40K_20mins/prove_results.jsonl \
      --formalizations logs/prove_formalizations_40K_20mins/formalizations.jsonl \
      --lean-project-dir /Users/mila/lean/mathlib4 \
      --output         logs/admissibility_probe_goedel.jsonl \
      --parallelism 8
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# {name} is substituted with the attempt's own abbrev name.
TACTICS = {
    "rfl": "rfl",
    "simp_solution": "simp [{name}]",
    "norm_num_solution": "norm_num [{name}]",
    "simp_tauto": "simp only [{name}] <;> tauto",
    "aesop": "aesop",
}


def load_formalizations(path):
    from conjecturing_agents.lean_regex import extract_abbrev_name_from_statement

    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            status = r.get("conjecture_formalization_status") or r.get("status")
            if status != "success":
                continue
            lean_stmt = r.get("lean_statement_without_comment") or ""
            abbrev_decl = (
                r.get("formalized_abbrev_declaration")
                or r.get("final_abbrev_declaration")
                or ""
            )
            if not lean_stmt or not abbrev_decl:
                continue
            abbrev_name = extract_abbrev_name_from_statement(lean_stmt)
            if not abbrev_name:
                continue
            out[(r["problem_id"], r["attempt"])] = {
                "lean_statement": lean_stmt,
                "abbrev_decl": abbrev_decl,
                "abbrev_name": abbrev_name,
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--formalizations", required=True)
    p.add_argument("--lean-project-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parallelism", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument(
        "--tactics",
        nargs="+",
        default=list(TACTICS),
        choices=list(TACTICS),
        help="Subset of the battery to run",
    )
    args = p.parse_args()

    from conjecturing_agents.lean_regex import replace_abbrev_in_statement
    from conjecturing_agents.tool_calling_backends.lean4_compiler import (
        LeanCompilerBackend,
        LeanCompilerConfig,
    )

    lean_cfg = LeanCompilerConfig(
        project_dir=args.lean_project_dir,
        timeout_seconds=args.timeout,
        treat_sorry_warning_as_failure=True,
        treat_any_warning_as_failure=False,
        cleanup_source_file=True,
    )

    forms = load_formalizations(args.formalizations)

    proved = []
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("proved"):
                proved.append((r["problem_id"], r["attempt"]))
    print(f"{len(proved)} proved attempts, {len(forms)} usable formalizations")

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                done.add((d["problem_id"], d["attempt"], d["tactic"]))
        print(f"resuming: {len(done)} records already present")

    jobs = []
    missing_form = []
    for pid, att in proved:
        form = forms.get((pid, att))
        if form is None:
            missing_form.append((pid, att))
            continue
        for tname in args.tactics:
            if (pid, att, tname) not in done:
                jobs.append((pid, att, tname, form))
    if missing_form:
        print(f"WARNING: no formalization record for {len(missing_form)} attempts: {missing_form}")
    print(f"{len(jobs)} compilations to run")

    def run_one(pid, att, tname, form):
        name = form["abbrev_name"]
        tactic = TACTICS[tname].format(name=name)
        rec = {
            "problem_id": pid,
            "attempt": att,
            "tactic": tname,
            "tactic_text": tactic,
            "abbrev_name": name,
        }
        try:
            code = replace_abbrev_in_statement(form["lean_statement"], form["abbrev_decl"])
            code = re.sub(r":=\s*sorry\s*$", f":= by\n  {tactic}", code)
        except ValueError as e:
            rec.update(ok=False, error=f"splice failed: {e}")
            return rec
        res = LeanCompilerBackend(lean_cfg).compile_code(code)
        rec.update(
            ok=bool(res.ok),
            timed_out=bool(res.timed_out),
            oom=bool(res.oom),
            elapsed_ms=int(res.elapsed_ms),
            n_errors=len(res.json_errors),
        )
        return rec

    with open(out_path, "a") as out_f, ThreadPoolExecutor(args.parallelism) as pool:
        futs = [pool.submit(run_one, *j) for j in jobs]
        fired = 0
        pbar = tqdm(total=len(futs), desc="probe")
        for fut in as_completed(futs):
            rec = fut.result()
            fired += bool(rec.get("ok"))
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            pbar.set_postfix(fired=fired)
            pbar.update(1)
        pbar.close()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
