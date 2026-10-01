"""Upgrade-feasibility analysis.

Given the same package index, root requirements, locks and solve options as
``/resolve`` plus one target package and 1-8 candidate versions, this module
answers *independently for every candidate: can the target be pinned to
exactly that version while the whole dependency closure still resolves?*

Semantics
---------
* Candidate versions must exist in the supplied index under the target
  package.  Package names and versions are normalized with the same PEP
  503/440 rules as the index itself, so ``Foo_Bar``/``foo-bar`` and
  ``1.0``/``1.0.0`` refer to the same entry.  Candidates that coincide
  after PEP 440 normalization are rejected as duplicate input.
* A baseline solve first runs on the unmodified roots, locks and options;
  it is the "original locks" resolution that successful upgrades are
  diffed against.  The baseline never short-circuits the analysis: each
  candidate is still solved on its own (a baseline that hits the search
  limit may become trivially solvable once the target is pinned, because
  pinning shrinks the search space).
* Every candidate is solved independently with a **temporary extra exact
  root requirement** (``==<version>``) on the target package.  The
  original root requirements all still apply, *every* lock (including a
  lock on the target itself) still participates, and the ordinary
  exhaustive backtracking runs against the complete dependency
  constraints — feasibility is recomputed by the solver, never inferred
  from the target's own version range or from the baseline graph.
* Each candidate owns a fresh, full ``max_attempts`` budget: one
  candidate conflicting or hitting the search limit never affects the
  others.
* Candidates are reported in input order, each with its own ``ok`` /
  ``conflict`` / ``undetermined`` status and attempt count.  Successful
  items carry the full resolution plus a version diff against the
  baseline (``null`` when the baseline itself did not resolve);
  conflicting items carry the structured conflict summary; hitting
  ``max_attempts`` is ``undetermined`` and is never treated as
  unsatisfiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .resolver import (
    PackageIndex,
    ResolutionInputError,
    RootEntry,
    SolveResult,
    parse_requirements,
    resolve,
)
from .serialize import conflict_summary


@dataclass(frozen=True)
class UpgradeTarget:
    """The validated target package and its candidate versions.

    ``versions`` keeps the input order and holds each candidate as the
    matching ``Version`` object from the index itself.
    """

    package: str  # canonical name
    versions: list[Version]


def parse_upgrade_target(
    index: PackageIndex, package: str, raw_candidates: list[str]
) -> UpgradeTarget:
    """Validate target package and candidate versions against the index.

    Every semantic problem (unknown package, non-PEP 440 version, version
    absent from the index, normalized duplicates) is collected and raised
    together as one :class:`ResolutionInputError` so the caller answers
    with a single 400.
    """
    canonical = canonicalize_name(package)
    if canonical not in index.versions:
        raise ResolutionInputError(
            f"target: package {package!r} is not present in the supplied index"
        )

    index_versions = [version for version, _record in index.versions[canonical]]
    available = ", ".join(str(v) for v, _ in sorted(index.versions[canonical], reverse=True))

    errors: list[str] = []
    seen: dict[str, str] = {}
    versions: list[Version] = []
    for raw in raw_candidates:
        try:
            parsed = Version(raw)
        except InvalidVersion as exc:
            errors.append(f"target: candidate {raw!r} is not a valid PEP 440 version: {exc}")
            continue
        matched = next((v for v in index_versions if v == parsed), None)
        if matched is None:
            errors.append(
                f"target: package {canonical!r} has no version {raw!r} in the "
                f"supplied index (available: {available or 'none'})"
            )
            continue
        normalized = str(matched)
        if normalized in seen:
            errors.append(
                f"target: candidate {raw!r} duplicates candidate "
                f"{seen[normalized]!r} after PEP 440 normalization"
            )
            continue
        seen[normalized] = raw
        versions.append(matched)

    if errors:
        raise ResolutionInputError("; ".join(errors))
    return UpgradeTarget(package=canonical, versions=versions)


def _changes(baseline: dict[str, str], upgraded: dict[str, str]) -> dict[str, Any]:
    """Diff two successful resolutions; every list is sorted by package name."""
    added = [
        {"package": name, "version": upgraded[name]}
        for name in sorted(upgraded.keys() - baseline.keys())
    ]
    removed = [
        {"package": name, "version": baseline[name]}
        for name in sorted(baseline.keys() - upgraded.keys())
    ]
    changed = [
        {"package": name, "from": baseline[name], "to": upgraded[name]}
        for name in sorted(baseline.keys() & upgraded.keys())
        if baseline[name] != upgraded[name]
    ]
    return {"added": added, "removed": removed, "changed": changed}


def _baseline_body(result: SolveResult, max_attempts: int) -> dict[str, Any]:
    """Render the unpinned baseline solve."""
    body: dict[str, Any] = {
        "status": result.status if result.status != "limit" else "undetermined",
        "attempts": result.attempts,
        "locks": result.lock_notes,
    }
    if result.status == "ok":
        body["resolved"] = result.chosen
        return body
    if result.status == "conflict":
        body["conflict"] = conflict_summary(result)
        return body
    body["error"] = {
        "code": "search_limit_exceeded",
        "message": (
            f"baseline resolution exceeded {max_attempts} candidate "
            "evaluations; the unpinned input may still be satisfiable"
        ),
    }
    return body


def _candidate_body(
    target: UpgradeTarget,
    version: Version,
    result: SolveResult,
    max_attempts: int,
    baseline_changes: Optional[dict[str, str]],
) -> dict[str, Any]:
    """Render one candidate's independent outcome."""
    body: dict[str, Any] = {
        "candidate": str(version),
        "attempts": result.attempts,
        "locks": result.lock_notes,
    }
    if result.status == "ok":
        body["status"] = "ok"
        body["resolved"] = result.chosen
        body["dependency_graph"] = result.edges
        body["changes"] = (
            _changes(baseline_changes, result.chosen)
            if baseline_changes is not None
            else None
        )
        return body
    if result.status == "conflict":
        body["status"] = "conflict"
        body["conflict"] = conflict_summary(result)
        return body
    body["status"] = "undetermined"
    body["error"] = {
        "code": "search_limit_exceeded",
        "message": (
            f"resolution with {target.package}=={version} pinned exceeded "
            f"{max_attempts} candidate evaluations; the candidate may still "
            "be installable"
        ),
    }
    return body


def run_upgrade_feasibility(
    index: PackageIndex,
    entries: list[RootEntry],
    locks: dict[str, Version],
    *,
    allow_prereleases: bool,
    marker_env: Optional[dict[str, str]],
    max_attempts: int,
    target_package: str,
    candidates: list[str],
) -> dict[str, Any]:
    """Solve the baseline and every pinned candidate; render the JSON body."""
    target = parse_upgrade_target(index, target_package, candidates)
    roots = [entry.dep for entry in entries]

    def solve(pin: Optional[Version]) -> SolveResult:
        extra_roots: list = []
        if pin is not None:
            # A temporary *root* requirement: it intersects the original
            # root constraints (contradictions surface as ordinary root
            # conflicts) while leaving every lock in force.
            extra_roots = parse_requirements(
                {target.package: f"=={pin}"},
                origin=f"upgrade candidate {target.package}=={pin}",
            )
        return resolve(
            index,
            roots + extra_roots,
            locks,
            allow_prereleases=allow_prereleases,
            marker_env=marker_env,
            max_attempts=max_attempts,
        )

    baseline = solve(None)

    results: list[dict[str, Any]] = []
    counts = {"ok": 0, "conflict": 0, "undetermined": 0}
    for version in target.versions:
        result = solve(version)
        status = result.status if result.status != "limit" else "undetermined"
        counts[status] += 1
        results.append(
            _candidate_body(
                target,
                version,
                result,
                max_attempts,
                baseline.chosen if baseline.status == "ok" else None,
            )
        )

    total_attempts = baseline.attempts + sum(item["attempts"] for item in results)
    return {
        "status": "ok",
        "attempts": total_attempts,
        "allow_prereleases": allow_prereleases,
        "target": {
            "package": target.package,
            "candidates": [str(version) for version in target.versions],
        },
        "summary": {
            "candidates": len(target.versions),
            "ok": counts["ok"],
            "conflict": counts["conflict"],
            "undetermined": counts["undetermined"],
        },
        "baseline": _baseline_body(baseline, max_attempts),
        "results": results,
    }
