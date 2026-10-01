"""End-to-end tests for ``POST /analyze/upgrade-feasibility``.

Each candidate gets a temporary extra exact-version root constraint while
the original roots, every lock and the other solve options stay in force;
each candidate is a full independent backtracking solve (full
``max_attempts`` budget), so feasibility reflects the whole dependency
closure rather than the target package's own version range.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _check(**payload):
    response = client.post("/analyze/upgrade-feasibility", json=payload)
    return response.status_code, response.json()


PACKAGES = {
    "app": {"1.0": {"dependencies": {"web": ""}}},
    "web": {
        "3.0": {"dependencies": {"common": ">=2.0"}},
        "2.0": {"dependencies": {"common": "<2.0"}},
        "1.0": {},
    },
    "common": {"1.5": {}, "2.0": {}},
}


def test_candidates_reported_in_input_order_with_independent_outcomes():
    status, body = _check(
        packages=PACKAGES,
        root={"app": ""},
        target="web",
        candidates=["2.0", "3.0"],
    )
    assert status == 200
    assert body["target"] == {"package": "web", "candidates": ["2.0", "3.0"]}
    assert body["summary"] == {
        "candidates": 2, "ok": 2, "conflict": 0, "undetermined": 0
    }

    baseline = body["baseline"]
    assert baseline["status"] == "ok"
    assert baseline["resolved"] == {"app": "1.0", "common": "2.0", "web": "3.0"}

    first, second = body["results"]
    # Input order is preserved, not version order.
    assert [item["candidate"] for item in body["results"]] == ["2.0", "3.0"]

    assert first["status"] == "ok"
    assert first["resolved"] == {"app": "1.0", "common": "1.5", "web": "2.0"}
    assert first["changes"] == {
        "added": [],
        "removed": [],
        "changed": [
            {"package": "common", "from": "2.0", "to": "1.5"},
            {"package": "web", "from": "3.0", "to": "2.0"},
        ],
    }
    edges = {(e["source"], e["target"]) for e in first["dependency_graph"]}
    assert edges == {("<root>", "app"), ("<root>", "web"), ("app", "web"), ("web", "common")}
    assert all(
        e["target_version"] == first["resolved"][e["target"]]
        for e in first["dependency_graph"]
    )

    # The candidate identical to the baseline resolution changes nothing.
    assert second["status"] == "ok"
    assert second["resolved"] == baseline["resolved"]
    assert second["changes"] == {"added": [], "removed": [], "changed": []}

    # attempts covers the baseline plus one independent solve per candidate.
    assert body["attempts"] == baseline["attempts"] + sum(
        item["attempts"] for item in body["results"]
    )


def test_conflict_candidate_carries_structured_summary():
    # The original root requires web>=3.0 (through app); pinning web==2.0
    # contradicts the full closure rather than the target's own range.
    packages = {
        "app": {"1.0": {"dependencies": {"web": ">=3.0"}}},
        "web": {"3.0": {}, "2.0": {}},
    }
    status, body = _check(
        packages=packages, root={"app": ""}, target="web", candidates=["2.0", "3.0"]
    )
    assert status == 200
    conflicting, feasible = body["results"]
    assert conflicting["candidate"] == "2.0"
    assert conflicting["status"] == "conflict"
    assert "resolved" not in conflicting
    assert "changes" not in conflicting
    terminals = conflicting["conflict"]["terminal_conflicts"]
    assert terminals[0]["package"] == "web"
    assert terminals[0]["reason"] == "unsatisfiable"
    specs = sorted(e["specifier"] for e in terminals[0]["active_constraints"])
    assert ">=3.0" in specs
    assert "==2.0" in specs
    # The temporary pin is presented as a <root> constraint with a full path.
    pin_constraint = next(
        e for e in terminals[0]["active_constraints"] if e["specifier"] == "==2.0"
    )
    assert pin_constraint["required_by"] == "<root>"
    assert [node["name"] for node in pin_constraint["path"]] == ["<root>"]

    assert feasible["status"] == "ok"
    assert feasible["resolved"] == {"app": "1.0", "web": "3.0"}
    assert body["summary"] == {
        "candidates": 2, "ok": 1, "conflict": 1, "undetermined": 0
    }


def test_deeper_transitive_constraints_participate_in_feasibility():
    # app -> web -> common: only web 2.0 is compatible with common==1.5
    # pinned by db.  Merely checking the pin against the root's own web
    # range would call 3.0 feasible; the full solve does not.
    packages = {
        "app": {"1.0": {"dependencies": {"web": "", "db": ""}}},
        "web": {
            "3.0": {"dependencies": {"common": ">=2.0"}},
            "2.0": {"dependencies": {"common": "<2.0"}},
        },
        "db": {"1.0": {"dependencies": {"common": "==1.5"}}},
        "common": {"1.5": {}, "2.0": {}},
    }
    status, body = _check(
        packages=packages,
        root={"app": ""},
        target="web",
        candidates=["3.0", "2.0"],
    )
    assert status == 200
    assert [item["status"] for item in body["results"]] == ["conflict", "ok"]
    terminals = body["results"][0]["conflict"]["terminal_conflicts"]
    leaf_packages = {leaf["package"] for leaf in terminals}
    assert "common" in leaf_packages


def test_locks_participate_including_a_lock_on_the_target():
    status, body = _check(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ">=1.0"},
        locks={"x": "1.0"},
        target="x",
        candidates=["2.0", "1.0"],
    )
    assert status == 200
    upgraded, kept = body["results"]
    # The lock does not veto the temporary pin; the chosen 2.0 is reported
    # against the original lock.
    assert upgraded["status"] == "ok"
    assert upgraded["resolved"]["x"] == "2.0"
    assert upgraded["locks"] == {"x": "unsatisfiable"}
    assert upgraded["changes"] == {
        "added": [],
        "removed": [],
        "changed": [{"package": "x", "from": "1.0", "to": "2.0"}],
    }
    # The matching candidate keeps the lock-first resolution.
    assert kept["status"] == "ok"
    assert kept["resolved"]["x"] == "1.0"
    assert kept["locks"] == {"x": "applied"}
    assert kept["changes"] == {"added": [], "removed": [], "changed": []}


def test_unrelated_locks_still_participate_in_each_solve():
    # The lock on the transitive ``common`` package is independent of the
    # target, but every candidate solve still honors it: web 3.0 accepts
    # any common so the locked 1.5 is applied first; web 2.0 requires
    # common>=2.0, so the same lock is evaluated and recorded as
    # unsatisfiable while the solve backtracks to 2.0.
    packages = {
        "app": {"1.0": {"dependencies": {"web": ""}}},
        "web": {
            "3.0": {"dependencies": {"common": ">=1.0"}},
            "2.0": {"dependencies": {"common": ">=2.0"}},
        },
        "common": {"1.5": {}, "2.0": {}},
    }
    status, body = _check(
        packages=packages,
        root={"app": ""},
        locks={"common": "1.5"},
        target="web",
        candidates=["3.0", "2.0"],
    )
    assert status == 200
    locked, unlocked = body["results"]
    assert locked["status"] == "ok"
    assert locked["resolved"] == {"app": "1.0", "common": "1.5", "web": "3.0"}
    assert locked["locks"] == {"common": "applied"}
    assert unlocked["status"] == "ok"
    assert unlocked["resolved"] == {"app": "1.0", "common": "2.0", "web": "2.0"}
    assert unlocked["locks"] == {"common": "unsatisfiable"}
    # The baseline solve applies the same lock.
    assert body["baseline"]["resolved"]["common"] == "1.5"


def test_target_outside_root_closure_is_pinned_as_root_requirement():
    # Nothing requires ``side`` from the roots; the pin pulls it in.
    packages = {
        "app": {"1.0": {}},
        "side": {"1.0": {}, "2.0": {"dependencies": {"ghost": ""}}},
    }
    status, body = _check(
        packages=packages, root={"app": ""}, target="side", candidates=["2.0", "1.0"]
    )
    assert status == 200
    broken, fine = body["results"]
    assert broken["status"] == "conflict"
    assert broken["conflict"]["terminal_conflicts"][0]["package"] == "ghost"
    assert fine["status"] == "ok"
    assert fine["resolved"]["side"] == "1.0"
    assert fine["changes"]["added"] == [{"package": "side", "version": "1.0"}]


def test_added_and_removed_packages_are_diffed_against_baseline():
    packages = {
        "app": {"1.0": {"dependencies": {"web": ""}}},
        "web": {
            "3.0": {"dependencies": {"common": ">=2.0"}},
            "2.0": {"dependencies": {"common": "<2.0", "extra": ""}},
        },
        "common": {"1.5": {}, "2.0": {}},
        "extra": {"1.0": {}},
    }
    status, body = _check(
        packages=packages, root={"app": ""}, target="web", candidates=["2.0"]
    )
    assert status == 200
    result = body["results"][0]
    assert result["changes"]["added"] == [{"package": "extra", "version": "1.0"}]
    assert result["changes"]["removed"] == []
    assert {"package": "common", "from": "2.0", "to": "1.5"} in result["changes"]["changed"]
    assert {"package": "web", "from": "3.0", "to": "2.0"} in result["changes"]["changed"]


def test_limit_candidate_is_undetermined_never_conflict():
    # p0 2.0/1.0 both fan out into failing subtrees; the pin forces the
    # search through them with the candidate's own budget.
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "3.0": {},
            "2.0": {"dependencies": {"p1": ""}},
            "1.0": {"dependencies": {"p1": ""}},
        },
        "p1": {
            "2.0": {"dependencies": {"missing": ""}},
            "1.0": {"dependencies": {"missing": ""}},
        },
    }
    status, body = _check(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["2.0", "3.0"],
        max_attempts=2,
    )
    assert status == 200
    undetermined, feasible = body["results"]
    assert undetermined["candidate"] == "2.0"
    assert undetermined["status"] == "undetermined"
    assert undetermined["error"]["code"] == "search_limit_exceeded"
    assert "p0==2.0" in undetermined["error"]["message"]
    assert "conflict" not in undetermined
    assert "resolved" not in undetermined
    # Each candidate owns a fresh, full budget: the next one still solves.
    assert feasible["status"] == "ok"
    assert feasible["attempts"] <= 2
    assert body["summary"] == {
        "candidates": 2, "ok": 1, "conflict": 0, "undetermined": 1
    }


def test_baseline_undetermined_does_not_block_candidates():
    # The unpinned search burns the budget testing 3.0/2.0 (both fail via
    # ``ghost``) before reaching 1.0; pinning 1.0 is trivially feasible.
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "3.0": {"dependencies": {"ghost": ""}},
            "2.0": {"dependencies": {"ghost": ""}},
            "1.0": {},
        },
    }
    status, body = _check(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["1.0"],
        max_attempts=2,
    )
    assert status == 200
    assert body["baseline"]["status"] == "undetermined"
    assert body["baseline"]["error"]["code"] == "search_limit_exceeded"
    result = body["results"][0]
    assert result["status"] == "ok"
    assert result["resolved"]["p0"] == "1.0"
    # No baseline solution exists, so no version diff can be computed.
    assert result["changes"] is None


def test_baseline_conflict_still_evaluates_every_candidate():
    packages = {
        "a": {"1.0": {"dependencies": {"x": ">=2.0"}}},
        "b": {"1.0": {"dependencies": {"x": "<2.0"}}},
        "x": {"1.0": {}, "2.0": {}},
    }
    status, body = _check(
        packages=packages, root=["a", "b"], target="x", candidates=["2.0"]
    )
    assert status == 200
    assert body["baseline"]["status"] == "conflict"
    terminals = body["baseline"]["conflict"]["terminal_conflicts"]
    assert terminals[0]["package"] == "x"
    result = body["results"][0]
    # Restricting an unsatisfiable input stays unsatisfiable; the
    # candidate's own conflict summary is still produced.
    assert result["status"] == "conflict"
    assert result["conflict"]["terminal_conflicts"][0]["package"] == "x"
    assert result["attempts"] >= 1
    assert "changes" not in result


def test_marker_env_applies_to_the_candidate_solves():
    packages = {
        "app": {
            "1.0": {"dependencies": ["dep>=2.0; python_version < '3.11'"]}
        },
        "dep": {"1.0": {}, "2.0": {}},
    }
    # Under 3.10 the transitive requirement dep>=2.0 makes pinning 1.0 fail.
    status, body = _check(
        packages=packages,
        root=["app"],
        target="dep",
        candidates=["1.0", "2.0"],
        marker_env={"python_version": "3.10"},
    )
    assert status == 200
    assert [item["status"] for item in body["results"]] == ["conflict", "ok"]

    # Under 3.11 the marker-gated requirement disappears; the pin alone
    # governs dep.
    status, body = _check(
        packages=packages,
        root=["app"],
        target="dep",
        candidates=["1.0"],
        marker_env={"python_version": "3.11"},
    )
    assert status == 200
    assert body["results"][0]["status"] == "ok"
    assert body["results"][0]["resolved"] == {"app": "1.0", "dep": "1.0"}


def test_prerelease_candidate_visible_only_with_flag_or_explicit_pin():
    packages = {"x": {"1.0": {}, "2.0rc1": {}}}
    # The exact pin itself relaxes pre-release exclusion for the target.
    status, body = _check(
        packages=packages, root={"x": ""}, target="x", candidates=["2.0rc1"]
    )
    assert status == 200
    result = body["results"][0]
    assert result["status"] == "ok"
    assert result["resolved"]["x"] == "2.0rc1"

    status, body = _check(
        packages=packages,
        root={"x": ""},
        target="x",
        candidates=["2.0rc1", "1.0"],
        allow_prereleases=True,
    )
    assert status == 200
    assert body["allow_prereleases"] is True
    assert [item["resolved"]["x"] for item in body["results"]] == ["2.0rc1", "1.0"]


def test_names_and_versions_are_normalized_like_the_index():
    status, body = _check(
        packages={"Foo_Bar": {"1.0.0": {}, "2.0": {}}},
        root={"foo-bar": ""},
        target="FOO-BAR",
        candidates=["1.0"],
    )
    assert status == 200
    assert body["target"] == {"package": "foo-bar", "candidates": ["1.0.0"]}
    assert body["results"][0]["candidate"] == "1.0.0"


def test_deterministic_results():
    payload = {
        "packages": PACKAGES,
        "root": {"app": ""},
        "target": "web",
        "candidates": ["2.0", "3.0"],
    }
    results = [
        client.post("/analyze/upgrade-feasibility", json=payload).json()
        for _ in range(3)
    ]
    assert all(r == results[0] for r in results)


def test_index_is_not_mutated_by_candidate_pins():
    payload = {
        "packages": PACKAGES,
        "root": {"app": ""},
        "target": "web",
        "candidates": ["2.0", "1.0"],
    }
    status, body = _check(**payload)
    assert status == 200
    followup = client.post(
        "/resolve", json={"packages": PACKAGES, "root": {"app": ""}}
    ).json()
    # Candidate pins were per-solve root constraints, not index edits.
    assert followup["resolved"]["web"] == "3.0"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_target_package_is_400():
    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="ghost",
        candidates=["1.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "ghost" in body["error"]["message"]


def test_unknown_candidate_version_is_400_and_lists_available():
    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["9.9"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "9.9" in body["error"]["message"]
    assert "1.0" in body["error"]["message"]


def test_non_pep440_candidate_is_400():
    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["not-a-version"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "PEP 440" in body["error"]["message"]


def test_candidates_duplicate_after_normalization_is_400():
    status, body = _check(
        packages={"x": {"1.0.0": {}, "2.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["1.0", "1.0.0", "2.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "duplicate" in body["error"]["message"]


def test_multiple_candidate_errors_are_reported_together():
    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["9.9", "nope"],
    )
    assert status == 400
    message = body["error"]["message"]
    assert "9.9" in message
    assert "nope" in message


def test_candidate_count_bounds_and_shape_are_422():
    base = {"packages": {"x": {"1.0": {}}}, "root": {"x": ""}, "target": "x"}
    assert client.post(
        "/analyze/upgrade-feasibility", json={**base, "candidates": []}
    ).status_code == 422
    assert client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "candidates": [f"{i}.0" for i in range(9)]},
    ).status_code == 422
    assert client.post(
        "/analyze/upgrade-feasibility", json=base
    ).status_code == 422
    assert client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "target": "", "candidates": ["1.0"]},
    ).status_code == 422
    assert client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "candidates": ["1.0"], "unexpected": 1},
    ).status_code == 422
    assert client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "candidates": [1.0]},
    ).status_code == 422


def test_resolve_input_validation_is_reused():
    candidate_request = {
        "root": {"x": ""},
        "target": "x",
        "candidates": ["1.0"],
    }
    status, body = _check(
        packages={"x": {"not-a-version": {}}}, **candidate_request
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ">>>1"},
        target="x",
        candidates=["1.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    status, body = _check(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        locks={"x": "not-a-version"},
        target="x",
        candidates=["1.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
