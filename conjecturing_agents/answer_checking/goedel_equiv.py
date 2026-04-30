"""
Heuristic 3: Goedel-prover equivalence check.

Builds a Lean theorem asserting ``proposed = gt_rhs``, then calls the
GoedelProverAgent to attempt a proof.  Uses ``theorem`` (not ``example``)
because the Goedel fine-tune was trained on theorem statements.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from conjecturing_agents.agents.goedel_prover import GoedelProverAgent
from conjecturing_agents.answer_checking.lean_equiv import extract_preamble
from conjecturing_agents.inference_backends.raw_base import EventLogger, RawBackend

from .result import CheckResult


def build_inequiv_theorem_statement(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
) -> str:
    """
    Assemble a self-contained Lean file with a theorem asserting *non*-equivalence.

    Structure::

        <preamble from scaffold>

        <proposed abbrev declaration>

        theorem check_<abbrev_name>_inequiv : <abbrev_name> ≠ (<gt_rhs>) := by sorry
    """
    preamble = extract_preamble(lean_statement)
    theorem_name = f"check_{abbrev_name}_inequiv"
    return (
        f"{preamble}\n\n"
        f"{proposed_abbrev_decl.strip()}\n\n"
        f"theorem {theorem_name} : {abbrev_name} ≠ ({gt_rhs}) := by sorry\n"
    )


def build_equiv_theorem_statement(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
) -> str:
    """
    Assemble a self-contained Lean file with a theorem asserting equivalence.

    Structure::

        <preamble from scaffold>

        <proposed abbrev declaration>

        theorem check_<abbrev_name>_equiv : <abbrev_name> = (<gt_rhs>) := by sorry
    """
    preamble = extract_preamble(lean_statement)
    theorem_name = f"check_{abbrev_name}_equiv"
    return (
        f"{preamble}\n\n"
        f"{proposed_abbrev_decl.strip()}\n\n"
        f"theorem {theorem_name} : {abbrev_name} = ({gt_rhs}) := by sorry\n"
    )


def check_goedel_equiv(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
    agent: GoedelProverAgent,
    backend: RawBackend,
    *,
    seed: int = 0,
    event_logger: Optional[EventLogger] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CheckResult:
    """
    Heuristic 3: use the Goedel prover to prove ``proposed = gt_rhs``.

    Returns:
        ``equivalent=True``  if the proof compiled successfully.
        ``equivalent=None``  if all rounds failed (inconclusive).
        ``equivalent=False`` is not returned by this heuristic (a failed proof
        attempt does not constitute a disproof).
    """
    theorem_stmt = build_equiv_theorem_statement(
        lean_statement,
        proposed_abbrev_decl,
        gt_rhs,
        abbrev_name,
    )

    result = agent.prove_theorem(
        theorem_stmt,
        backend,
        seed=seed,
        event_logger=event_logger,
        metadata=metadata,
    )

    details: Dict[str, Any] = {
        "termination_reason": result.termination_reason,
        "rounds_used": result.rounds_used,
        "elapsed_ms": result.elapsed_ms,
        "rounds": result.rounds,
    }

    if result.proved:
        return CheckResult(
            equivalent=True,
            method="goedel_prover",
            details=details,
        )

    return CheckResult(
        equivalent=None,
        method="goedel_prover",
        details=details,
    )


def check_goedel_inequiv(
    lean_statement: str,
    proposed_abbrev_decl: str,
    gt_rhs: str,
    abbrev_name: str,
    agent: GoedelProverAgent,
    backend: RawBackend,
    *,
    seed: int = 0,
    event_logger: Optional[EventLogger] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CheckResult:
    """
    Heuristic 4: use the Goedel prover to prove ``proposed ≠ gt_rhs``.

    Returns:
        ``equivalent=False`` if the disproof compiled successfully.
        ``equivalent=None``  if the disproof attempt failed (inconclusive).
    """
    theorem_stmt = build_inequiv_theorem_statement(
        lean_statement,
        proposed_abbrev_decl,
        gt_rhs,
        abbrev_name,
    )

    result = agent.prove_theorem(
        theorem_stmt,
        backend,
        seed=seed,
        event_logger=event_logger,
        metadata=metadata,
    )

    details: Dict[str, Any] = {
        "termination_reason": result.termination_reason,
        "rounds_used": result.rounds_used,
        "elapsed_ms": result.elapsed_ms,
        "rounds": result.rounds,
    }

    if result.proved:
        return CheckResult(
            equivalent=False,
            method="goedel_disprover",
            details=details,
        )

    return CheckResult(
        equivalent=None,
        method="goedel_disprover",
        details=details,
    )


__all__ = [
    "build_inequiv_theorem_statement",
    "build_equiv_theorem_statement",
    "check_goedel_inequiv",
    "check_goedel_equiv",
]
