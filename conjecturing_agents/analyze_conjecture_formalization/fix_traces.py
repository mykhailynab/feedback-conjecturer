#!/usr/bin/env python3
from __future__ import annotations

import re
import json
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from tqdm import tqdm

from conjecturing_agents.tool_calling_backends.lean4_compiler import (
    Lean4CompilerBackend,
    LeanCompilerConfig,
)


# ============================================================
# JSONL helpers
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[warn] Failed to parse JSON at {path}:{line_num}: {exc}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] Non-dict JSON object at {path}:{line_num}; skipping.")
                continue
            records.append(obj)
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Lean / abbrev extraction helpers
# ============================================================

_ABBREV_NAME_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev\s+([A-Za-z0-9_']+)\b",
    re.MULTILINE,
)

_SINGLE_LINE_ABBREV_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*abbrev\s+([A-Za-z0-9_']+)\b[^\n]*:=.*$",
    re.MULTILINE,
)

_TOP_LEVEL_DECL_RE = re.compile(
    r"^\s*(?:(?:noncomputable|unsafe|protected|private)\s+)*"
    r"(?:abbrev|theorem|lemma|def|example|structure|class|inductive|instance|"
    r"namespace|end|section|open|import|#check|#eval|#print)\b"
)

_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)

_LEAN_BLOCK_PATTERNS = [
    re.compile(r"```lean4\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```lean4\s*\n(.*?)```", re.DOTALL),
    re.compile(r"```lean\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"```lean\s*\n(.*?)```", re.DOTALL),
]


CompileCacheValue = Tuple[bool, Optional[str], Optional[str], Optional[str]]


def extract_abbrev_name_from_statement(lean_statement: str) -> Optional[str]:
    m = _ABBREV_NAME_RE.search(lean_statement or "")
    return m.group(1) if m else None


def extract_rhs_from_abbrev_declaration(abbrev_declaration: str) -> Optional[str]:
    if ":=" not in abbrev_declaration:
        return None
    return abbrev_declaration.split(":=", 1)[1].strip()


def replace_abbrev_in_statement(
    lean_statement: str,
    new_abbrev_declaration: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> str:
    current_abbrev_name = extract_abbrev_name_from_statement(lean_statement)
    if current_abbrev_name is None:
        raise ValueError("Could not find abbrev name in lean_statement")

    if required_abbrev_name is not None and current_abbrev_name != required_abbrev_name:
        raise ValueError(
            f"Scaffold abbrev name {current_abbrev_name!r} != required_abbrev_name {required_abbrev_name!r}"
        )

    new_abbrev_name = extract_abbrev_name_from_statement(new_abbrev_declaration)
    if new_abbrev_name is None:
        raise ValueError("Could not find abbrev name in new_abbrev_declaration")

    if new_abbrev_name != current_abbrev_name:
        raise ValueError(
            f"Generated abbrev name {new_abbrev_name!r} != scaffold abbrev name {current_abbrev_name!r}"
        )

    matches = list(_SINGLE_LINE_ABBREV_RE.finditer(lean_statement))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one abbrev placeholder in scaffold, found {len(matches)}")

    m = matches[0]
    replacement = new_abbrev_declaration.strip()
    return lean_statement[: m.start()] + replacement + lean_statement[m.end() :]


def extract_lean_code_block(text: str) -> Optional[str]:
    for pat in _LEAN_BLOCK_PATTERNS:
        matches = pat.findall(text or "")
        if matches:
            return matches[-1].strip()
    return None


def extract_final_channel_messages(raw_output: str) -> List[str]:
    if not raw_output:
        return []
    return [m.strip() for m in _FINAL_CHANNEL_RE.findall(raw_output) if m.strip()]


def extract_last_abbrev_declaration_from_text(
    text: str,
    *,
    required_abbrev_name: Optional[str] = None,
) -> Optional[str]:
    """
    Post-hoc fixed extraction that tolerates modifiers such as `noncomputable`
    in both the abbrev line and the next top-level declaration.
    """
    if not text:
        return None

    candidate_texts: List[str] = []
    code_block = extract_lean_code_block(text)
    if code_block:
        candidate_texts.append(code_block)
    stripped_text = text.strip()
    if stripped_text and stripped_text != code_block:
        candidate_texts.append(stripped_text)

    for candidate_text in candidate_texts:
        lines = candidate_text.splitlines()
        abbrev_indices = [
            i for i, line in enumerate(lines)
            if _ABBREV_NAME_RE.match(line)
        ]

        for idx in reversed(abbrev_indices):
            head = lines[idx]
            m_name = _ABBREV_NAME_RE.match(head)
            if not m_name:
                continue

            abbrev_name = m_name.group(1)
            if required_abbrev_name is not None and abbrev_name != required_abbrev_name:
                continue

            block_lines = [head]
            for j in range(idx + 1, len(lines)):
                line = lines[j]
                if _TOP_LEVEL_DECL_RE.match(line):
                    break
                if line.strip().startswith("```"):
                    break
                block_lines.append(line)

            decl = "\n".join(block_lines).strip()
            if ":=" not in decl:
                continue
            if re.search(r"\bsorry\b", decl):
                continue

            return decl

    return None


# ============================================================
# Repair logic
# ============================================================

def compile_candidate_abbrev(
    lean_backend: Lean4CompilerBackend,
    *,
    lean_statement_without_comment: str,
    required_abbrev_name: str,
    abbrev_declaration: str,
    compile_cache: Dict[str, CompileCacheValue],
    cache_stats: Dict[str, int],
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
      (ok, assembled_lean, relative_path, formatted_diagnostics)

    Uses compile_cache so identical assembled Lean files are not recompiled.
    """
    try:
        assembled_lean = replace_abbrev_in_statement(
            lean_statement_without_comment,
            abbrev_declaration,
            required_abbrev_name=required_abbrev_name,
        )
    except Exception as exc:
        return False, None, None, f"Failed to replace scaffold abbrev: {exc}"

    cached = compile_cache.get(assembled_lean)
    if cached is not None:
        cache_stats["hits"] += 1
        return cached
    cache_stats["misses"] += 1

    compile_result = lean_backend.compile_code(assembled_lean)
    if not compile_result.ok:
        diag = compile_result.formatted_diagnostics or compile_result.stderr or compile_result.stdout
        result: CompileCacheValue = (
            False,
            assembled_lean,
            compile_result.relative_path,
            diag,
        )
        compile_cache[assembled_lean] = result
        return result

    result = (
        True,
        assembled_lean,
        compile_result.relative_path,
        compile_result.formatted_diagnostics,
    )
    compile_cache[assembled_lean] = result
    return result

def try_repair_round(
    round_record: Dict[str, Any],
    *,
    lean_statement_without_comment: str,
    required_abbrev_name: str,
    lean_backend: Lean4CompilerBackend,
    compile_cache: Dict[str, CompileCacheValue],
    cache_stats: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    raw_output = str(round_record.get("raw_output") or "")
    if not raw_output:
        return None

    final_messages = extract_final_channel_messages(raw_output)
    if not final_messages:
        return None

    # Prefer later final messages first, since the original extraction also used the last one.
    for final_msg in reversed(final_messages):
        abbrev_declaration = extract_last_abbrev_declaration_from_text(
            final_msg,
            required_abbrev_name=required_abbrev_name,
        )
        if not abbrev_declaration:
            continue

        ok, assembled_lean, relative_path, diagnostics = compile_candidate_abbrev(
            lean_backend,
            lean_statement_without_comment=lean_statement_without_comment,
            required_abbrev_name=required_abbrev_name,
            abbrev_declaration=abbrev_declaration,
            compile_cache=compile_cache,
            cache_stats=cache_stats,
        )
        if not ok:
            continue

        repaired = deepcopy(round_record)
        repaired["abbrev_declaration"] = abbrev_declaration
        repaired["abbrev_name"] = extract_abbrev_name_from_statement(abbrev_declaration)
        repaired["rhs"] = extract_rhs_from_abbrev_declaration(abbrev_declaration)
        repaired["compile_ok"] = True
        repaired["compile_relative_path"] = relative_path
        repaired["compile_formatted_diagnostics"] = diagnostics or ""
        repaired["assembled_lean"] = assembled_lean
        repaired.setdefault("posthoc_fix", {})
        repaired["posthoc_fix"].update(
            {
                "fixed_extraction_from_raw_output": True,
                "fix_reason": "recovered_final_channel_abbrev_after_top_level_decl_regex_bug",
            }
        )
        return repaired

    return None


def choose_best_successful_round(rounds: List[Dict[str, Any]]) -> Optional[Tuple[int, Dict[str, Any]]]:
    for i, rr in enumerate(rounds):
        if rr.get("compile_ok") is True and rr.get("abbrev_declaration"):
            return i, rr
    return None


def repair_record(
    record: Dict[str, Any],
    *,
    lean_backend: Lean4CompilerBackend,
    compile_cache: Dict[str, CompileCacheValue],
    cache_stats: Dict[str, int],
    print_fixes: bool,
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns:
      (possibly modified record, did_change)
    """
    new_record = deepcopy(record)

    lean_statement_without_comment = new_record.get("lean_statement_without_comment")
    required_abbrev_name = new_record.get("required_abbrev_name")
    rounds = new_record.get("rounds", []) or []

    if not lean_statement_without_comment or not required_abbrev_name or not rounds:
        return new_record, False

    changed = False

    for i, rr in enumerate(rounds):
        # Only try to repair rounds where extraction likely failed before.
        if rr.get("abbrev_declaration"):
            continue

        repaired_round = try_repair_round(
            rr,
            lean_statement_without_comment=str(lean_statement_without_comment),
            required_abbrev_name=str(required_abbrev_name),
            lean_backend=lean_backend,
            compile_cache=compile_cache,
            cache_stats=cache_stats,
        )
        if repaired_round is not None:
            rounds[i] = repaired_round
            changed = True

            if print_fixes:
                print(
                    f"[fix] Recovered abbrev for id={new_record.get('problem_id')} "
                    f"attempt={new_record.get('attempt')} round={i + 1}"
                )

    if not changed:
        return new_record, False

    new_record["rounds"] = rounds

    best = choose_best_successful_round(rounds)
    if best is not None:
        best_idx, best_round = best
        new_record["status"] = "success"
        new_record["rounds_used"] = best_idx + 1
        new_record["final_abbrev_declaration"] = best_round.get("abbrev_declaration")
        new_record["final_compile_ok"] = True
        new_record["final_compile_relative_path"] = best_round.get("compile_relative_path")
        new_record["final_compile_formatted_diagnostics"] = best_round.get("compile_formatted_diagnostics", "")
        new_record.setdefault("posthoc_fix", {})
        new_record["posthoc_fix"].update(
            {
                "record_fixed": True,
                "fixed_success_round": best_idx + 1,
                "fix_reason": "recovered_final_channel_abbrev_after_top_level_decl_regex_bug",
            }
        )

    return new_record, True


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-hoc repair of formalizations.jsonl rows affected by the faulty "
            "_TOP_LEVEL_DECL_RE abbrev extraction bug."
        )
    )
    p.add_argument(
        "run_dir",
        help="Directory containing formalizations.jsonl",
    )
    p.add_argument(
        "--input-file",
        default="formalizations.jsonl",
        help="Input JSONL filename inside run_dir",
    )
    p.add_argument(
        "--output-file",
        default="formalizations_fixed.jsonl",
        help="Output JSONL filename inside run_dir",
    )
    p.add_argument(
        "--lean-project-dir",
        required=True,
        help="Path to a Lean project with Mathlib configured",
    )
    p.add_argument(
        "--lean-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for Lean compilation",
    )
    p.add_argument(
        "--lean-jobs",
        type=int,
        default=12,
        help="Parallel jobs passed to lake env lean -j",
    )
    p.add_argument(
        "--lean-workspace-subdir",
        default=".conjecturing_agents/lean_tool_runs_posthoc_fix",
        help="Workspace subdir inside the Lean project for temporary files",
    )
    p.add_argument(
        "--print-fixes",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / args.input_file
    output_path = run_dir / args.output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    lean_cfg = LeanCompilerConfig(
        project_dir=args.lean_project_dir,
        workspace_subdir=args.lean_workspace_subdir,
        timeout_seconds=args.lean_timeout_seconds,
        lean_jobs=args.lean_jobs,
        recipient_name="lean",
        tool_name="lean",
        treat_sorry_warning_as_failure=False,  # theorem sorry remains expected in this pipeline
        treat_any_warning_as_failure=False,
        auto_extract_code_block=True,
        cleanup_source_file=False,
    )
    lean_backend = Lean4CompilerBackend(lean_cfg)

    compile_cache: Dict[str, CompileCacheValue] = {}
    cache_stats = {
        "hits": 0,
        "misses": 0,
    }
    changed_count = 0
    success_fixed_count = 0
    fixed_records: List[Dict[str, Any]] = []

    progress = tqdm(records, desc="Post-hoc fixing", unit="record")
    for rec in progress:
        new_rec, changed = repair_record(
            rec,
            lean_backend=lean_backend,
            compile_cache=compile_cache,
            cache_stats=cache_stats,
            print_fixes=args.print_fixes,
        )
        fixed_records.append(new_rec)

        if changed:
            changed_count += 1
            if new_rec.get("status") == "success" and not rec.get("final_compile_ok"):
                success_fixed_count += 1

        progress.set_postfix(
            recovered=success_fixed_count,
            changed=changed_count,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_size=len(compile_cache),
        )

    write_jsonl(output_path, fixed_records)

    print()
    print(f"Wrote repaired records to: {output_path}")
    print(f"Rows modified: {changed_count}")
    print(f"Rows converted to success: {success_fixed_count}")
    print(f"Compile cache hits: {cache_stats['hits']}")
    print(f"Compile cache misses: {cache_stats['misses']}")
    print(f"Unique compiled strings: {len(compile_cache)}")


if __name__ == "__main__":
    main()