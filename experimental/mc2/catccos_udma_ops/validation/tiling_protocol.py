# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""CPU-only identities and checkpoint envelopes for persistent tiling search."""

import hashlib
import json
from pathlib import Path


EXECUTION_MODE = "persistent_batches_v1"
SEARCH_MODE = "performance_only"


def request_digest(request):
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def candidate_request(session, item):
    """Stable candidate identity across sessions, retries and resume subsets.

    The plan digest binds the full search, including sampling and batch size.
    Session membership must not change a completed candidate's identity.
    """
    return {
        "schema_version": 1, "mode": "candidate",
        "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
        "run_id": session["run_id"], "world_size": session["world_size"],
        "case": session["case"], "tiling": item["tiling"],
        "plan_digest": session["plan_digest"],
        "topology_label": session.get("topology_label", ""),
        "environment": session["environment"],
    }


def batch_path(root, index):
    return Path(root) / "batch_{:06d}.json".format(index)


def validate_batch_payload(payload, session, index, *, session_digest=None, environment_digest=None):
    """Check identity and complete ordered coverage; caller verifies timing scores.

    Callers processing many batches should cache both digests. Hashing the full
    session for every batch would turn checkpoint ingestion into quadratic work.
    """
    if (
        not isinstance(payload, dict) or payload.get("schema_version") != 1
        or payload.get("execution_mode") != EXECUTION_MODE
        or payload.get("search_mode") != SEARCH_MODE
        or payload.get("accuracy_checked") is not False
        or "accuracy" in payload
        or type(payload.get("batch_index")) is not int
        or payload["batch_index"] != index
        or payload.get("request_digest") != (session_digest or request_digest(session))
        or payload.get("environment_digest") != (environment_digest or request_digest(session["environment"]))
    ):
        raise ValueError("batch checkpoint does not match its session or execution protocol")
    size = session["batch_size"]
    expected = session["candidates"][index * size:(index + 1) * size]
    records = payload.get("results")
    if not expected or not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("batch checkpoint does not cover its complete candidate range")
    for record, item in zip(records, expected):
        if (
            not isinstance(record, dict)
            or record.get("case_id") != session["case"]["id"]
            or record.get("candidate_id") != item["candidate_id"]
            or record.get("tiling") != item["tiling"]
            or record.get("is_default") is not item["is_default"]
            or record.get("execution_mode") != EXECUTION_MODE
            or record.get("request_digest") != request_digest(candidate_request(session, item))
        ):
            raise ValueError("batch checkpoint contains a missing, reordered or mismatched candidate")
    return records
