"""
Reads prove_results.jsonl from a prove_formalizations run and produces a
stripped formalizations.jsonl containing only the attempts that were
successfully proved by Goedel.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conjecturing_agents.tools import load_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default="logs/prove_formalizations_16K_20mins",
        help="Directory containing prove_results.jsonl and formalizations.jsonl.",
    )
    p.add_argument(
        "--output-dir",
        default="logs/stripped_formalizations_goedel_pass",
        help="Directory to write the filtered formalizations.jsonl.",
    )
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out_dir = pathlib.Path(args.output_dir)

    prove_results = load_jsonl(run_dir / "prove_results.jsonl")
    formalizations = load_jsonl(run_dir / "formalizations.jsonl")

    proved_keys = {
        (r["problem_id"], r["attempt"])
        for r in prove_results
        if r.get("proved")
    }

    print(f"Proved attempts : {len(proved_keys)}")
    print(f"Total attempts  : {len(prove_results)}")

    kept = [
        f for f in formalizations
        if (f["problem_id"], f["attempt"]) in proved_keys
    ]

    print(f"Matching formalizations rows kept: {len(kept)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "formalizations.jsonl"
    write_jsonl(out_path, kept)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
