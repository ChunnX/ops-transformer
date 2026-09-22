# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import copy

import pytest

from validation.tiling_protocol import (
    EXECUTION_MODE, SEARCH_MODE, batch_path, candidate_request,
    request_digest, validate_batch_payload,
)


def _session():
    return {
        "schema_version": 1, "mode": "session", "run_id": "test-run",
        "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
        "world_size": 2, "case": {"id": "case", "performance": {"repeats": 3}},
        "plan_digest": "full-plan", "environment": {"ranks": [0, 1]},
        "batch_size": 2,
        "candidates": [
            {"candidate_id": str(index), "tiling": {"m0": index + 1}, "is_default": index == 0}
            for index in range(5)
        ],
    }


def _batch(session, index):
    items = session["candidates"][index * session["batch_size"]:(index + 1) * session["batch_size"]]
    return {
        "schema_version": 1, "execution_mode": EXECUTION_MODE, "search_mode": SEARCH_MODE,
        "accuracy_checked": False, "batch_index": index,
        "request_digest": request_digest(session),
        "environment_digest": request_digest(session["environment"]),
        "results": [dict(
            item, case_id=session["case"]["id"], execution_mode=EXECUTION_MODE,
            request_digest=request_digest(candidate_request(session, item)),
        ) for item in items],
    }


def test_final_partial_batch_covers_every_candidate_once():
    session = _session()
    records = [
        record
        for index in range(3)
        for record in validate_batch_payload(_batch(session, index), session, index)
    ]
    assert [record["candidate_id"] for record in records] == ["0", "1", "2", "3", "4"]
    assert len(validate_batch_payload(_batch(session, 2), session, 2)) == 1
    with pytest.raises(ValueError, match="complete candidate range"):
        validate_batch_payload(_batch(session, 3), session, 3)


@pytest.mark.parametrize("defect", [
    "session_digest", "environment_digest", "batch_index", "duplicate", "reordered",
    "missing", "extra", "tiling", "default", "candidate_digest", "protocol", "accuracy",
])
def test_batch_rejects_incomplete_or_tampered_checkpoint(defect):
    session = _session()
    payload = _batch(session, 0)
    if defect == "session_digest":
        payload["request_digest"] = "other-session"
    elif defect == "environment_digest":
        payload["environment_digest"] = "other-machine"
    elif defect == "batch_index":
        payload["batch_index"] = True
    elif defect == "duplicate":
        payload["results"][1] = copy.deepcopy(payload["results"][0])
    elif defect == "reordered":
        payload["results"].reverse()
    elif defect == "missing":
        payload["results"].pop()
    elif defect == "extra":
        payload["results"].append(copy.deepcopy(payload["results"][0]))
    elif defect == "tiling":
        payload["results"][0]["tiling"] = {"m0": 999}
    elif defect == "default":
        payload["results"][0]["is_default"] = 1
    elif defect == "candidate_digest":
        payload["results"][0]["request_digest"] = "other-candidate"
    elif defect == "protocol":
        payload["execution_mode"] = "isolated"
    elif defect == "accuracy":
        payload["accuracy_checked"] = True
    with pytest.raises(ValueError):
        validate_batch_payload(payload, session, 0)


def test_candidate_identity_survives_resume_subset_and_session_batch_reordering():
    session = _session()
    item = session["candidates"][3]
    resumed = copy.deepcopy(session)
    resumed["candidates"] = list(reversed(resumed["candidates"][3:]))
    assert request_digest(session) != request_digest(resumed)
    assert candidate_request(session, item) == candidate_request(resumed, item)
    resumed["plan_digest"] = "different-sampling-or-plan"
    assert candidate_request(session, item) != candidate_request(resumed, item)


def test_cached_digest_validation_and_batch_paths(tmp_path):
    session = _session()
    payload = _batch(session, 0)
    assert validate_batch_payload(
        payload, session, 0, session_digest=request_digest(session),
        environment_digest=request_digest(session["environment"]),
    ) == payload["results"]
    assert batch_path(tmp_path, 12) == tmp_path / "batch_000012.json"
