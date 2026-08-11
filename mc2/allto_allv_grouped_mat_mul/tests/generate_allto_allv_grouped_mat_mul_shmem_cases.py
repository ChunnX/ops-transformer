#!/usr/bin/env python3
"""Generate deterministic SHMEM precision/performance case metadata.

This script intentionally allocates no tensors. It also materializes and validates
all-rank route tables in memory so malformed route patterns are caught offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEED = 20260810

TYPICAL_CASES = [
    ("source_reference", 4084, 2048, 4096, 2, 4, "uneven"),
    ("ep1_local_only", 128, 256, 256, 1, 4, "balanced"),
    ("ep2_balanced_small", 128, 256, 256, 2, 1, "balanced"),
    ("ep2_balanced_multi_expert", 512, 512, 1024, 2, 4, "balanced"),
    ("ep4_balanced", 1024, 1024, 2048, 4, 4, "balanced"),
    ("ep8_balanced", 2048, 2048, 4096, 8, 8, "balanced"),
    ("m0_128_family", 640, 1024, 4096, 2, 8, "uneven"),
    ("m0_256_family", 1280, 1024, 512, 4, 8, "uneven"),
    ("large_local_expert", 4096, 2048, 4096, 8, 64, "sparse"),
]

GENERAL_CASES = [
    ("tiny_one_token", 1, 16, 16, 2, 1, "single_route"),
    ("tiny_less_than_ep", 3, 32, 48, 4, 1, "sparse"),
    ("tiny_ep8", 8, 64, 64, 8, 1, "diagonal"),
    ("odd_ep3", 192, 256, 256, 3, 2, "uneven"),
    ("odd_ep5", 320, 256, 512, 5, 2, "uneven"),
    ("odd_ep7", 448, 512, 512, 7, 2, "uneven"),
    ("m_residue_1", 129, 256, 256, 2, 2, "uneven"),
    ("m_residue_63", 191, 256, 512, 2, 4, "uneven"),
    ("ep6_balanced", 384, 256, 512, 6, 2, "balanced"),
    ("k_residue_16", 256, 272, 512, 2, 4, "balanced"),
    ("k_residue_48", 384, 304, 768, 4, 4, "uneven"),
    ("k_less_than_k0", 96, 128, 256, 2, 2, "uneven"),
    ("n_residue_16", 256, 512, 272, 2, 4, "balanced"),
    ("n_residue_80", 512, 1024, 592, 4, 4, "uneven"),
    ("n_less_than_n0", 192, 512, 96, 2, 4, "uneven"),
    ("all_residue", 333, 560, 336, 4, 3, "uneven"),
    ("zero_expert_ep2", 256, 256, 512, 2, 8, "zero_expert"),
    ("zero_expert_ep4", 512, 512, 1024, 4, 8, "zero_expert"),
    ("zero_expert_ep8", 1024, 1024, 2048, 8, 8, "zero_expert"),
    ("zero_peer_payload", 512, 512, 1024, 4, 4, "zero_peer"),
    ("single_hot_peer", 512, 1024, 1024, 8, 4, "single_peer"),
    ("highly_uneven", 2047, 1024, 2048, 8, 8, "first_route_heavy"),
    ("one_local_expert_ep2", 512, 1024, 2048, 2, 1, "uneven"),
    ("one_local_expert_ep8", 1024, 1024, 2048, 8, 1, "uneven"),
    ("expert_63", 2016, 512, 1024, 4, 63, "sparse"),
    ("expert_64", 2048, 512, 1024, 4, 64, "sparse"),
    ("workspace_near_limit", 1230000, 256, 128, 2, 1, "balanced"),
]

PERFORMANCE_CASE_NAMES = {
    "source_reference",
    "ep1_local_only",
    "ep2_balanced_multi_expert",
    "ep4_balanced",
    "ep8_balanced",
    "m0_128_family",
    "m0_256_family",
    "all_residue",
    "zero_peer_payload",
    "highly_uneven",
}


def _partition(total: int, weights: list[int]) -> list[int]:
    weight_sum = sum(weights)
    if total < 0 or weight_sum <= 0:
        raise ValueError("invalid weighted partition")
    values = [total * weight // weight_sum for weight in weights]
    for index in range(total - sum(values)):
        values[index % len(values)] += 1
    return values


def make_send_counts(m: int, ep: int, experts: int, pattern: str, src_rank: int) -> list[int]:
    route_count = ep * experts
    weights = [1] * route_count
    if pattern == "single_route":
        weights = [1] + [0] * (route_count - 1)
    elif pattern == "sparse":
        weights = [1 if (index + src_rank) % 3 == 0 else 0 for index in range(route_count)]
    elif pattern == "diagonal":
        weights = [1 if index == (src_rank % ep) * experts else 0 for index in range(route_count)]
    elif pattern == "uneven":
        weights = [((index * 17 + src_rank * 13 + SEED) % 11) + 1 for index in range(route_count)]
    elif pattern == "zero_expert":
        weights = [0 if index % experts == 0 else ((index + src_rank) % 5) + 1 for index in range(route_count)]
    elif pattern == "zero_peer":
        zero_peer = (src_rank + 1) % ep
        weights = [0 if index // experts == zero_peer else 1 for index in range(route_count)]
    elif pattern == "single_peer":
        selected_peer = (src_rank + 1) % ep
        weights = [1 if index // experts == selected_peer else 0 for index in range(route_count)]
    elif pattern == "first_route_heavy":
        weights = [route_count * 16] + [1] * (route_count - 1)
    elif pattern != "balanced":
        raise ValueError(f"unknown route pattern: {pattern}")
    return _partition(m, weights)


def materialize_routes(m: int, ep: int, experts: int, pattern: str) -> tuple[list[list[int]], list[list[int]]]:
    all_send = [make_send_counts(m, ep, experts, pattern, src) for src in range(ep)]
    all_recv = [
        [all_send[src][dst * experts + expert] for src in range(ep) for expert in range(experts)]
        for dst in range(ep)
    ]
    assert all(sum(row) == m for row in all_send)
    assert all(len(row) == ep * experts for row in all_send + all_recv)
    for dst in range(ep):
        for src in range(ep):
            for expert in range(experts):
                assert all_recv[dst][src * experts + expert] == all_send[src][dst * experts + expert]
    return all_send, all_recv


def make_case(category: str, spec: tuple[str, int, int, int, int, int, str]) -> dict:
    name, m, k, n, ep, experts, pattern = spec
    materialize_routes(m, ep, experts, pattern)
    return {
        "case_id": name,
        "category": category,
        "route_pattern": pattern,
        "route_seed": SEED,
        "inputs": [
            {"name": "gmm_x", "type": "tensor", "required": True, "dtype": "bfloat16", "shape": [m, k]},
            {
                "name": "gmm_weight",
                "type": "tensor",
                "required": True,
                "dtype": "bfloat16",
                "shape": [experts, k, n],
            },
            {"name": "ep_world_size", "type": "attr", "required": True, "dtype": "int", "value": ep},
            {"name": "local_experts", "type": "attr", "required": True, "dtype": "int", "value": experts},
            {"name": "route_pattern", "type": "attr", "required": True, "dtype": "str", "value": pattern},
            {"name": "route_seed", "type": "attr", "required": True, "dtype": "int", "value": SEED},
        ],
    }


def render_jsonl(cases: list[dict]) -> str:
    return "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    precision_path = root / "precision" / "allto_allv_grouped_mat_mul_shmem_precision_cases.jsonl"
    performance_path = root / "performance" / "allto_allv_grouped_mat_mul_shmem_perf_cases.jsonl"
    cases = [make_case("typical", spec) for spec in TYPICAL_CASES]
    cases += [make_case("general", spec) for spec in GENERAL_CASES]
    assert len(cases) == 36
    perf_cases = [case for case in cases if case["case_id"] in PERFORMANCE_CASE_NAMES]
    assert len(perf_cases) == 10

    outputs = {precision_path: render_jsonl(cases), performance_path: render_jsonl(perf_cases)}
    if args.check:
        stale = [str(path) for path, expected in outputs.items() if not path.is_file() or path.read_text("utf-8") != expected]
        if stale:
            raise SystemExit("stale generated case files: " + ", ".join(stale))
    else:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {path} ({content.count(chr(10))} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
