#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Exact scalar reference for MegaMoe's 256-row segmented Wave V1.

The implemented V1 converts the routed experts into an ordered stream of
expert-local <=256-row chunks.  A normal Wave consumes a fixed number of
chunks derived only from the AIC count and the GMM1/GMM2 N tile counts.  Wave
boundaries may split a hot expert; expert boundaries only select weights and
address bases.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from typing import Sequence


TILE_M = 256
TILE_N = 256


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return 0 if value == 0 else (value - 1) // divisor + 1


def gmm1_n_tiles(hidden_dim: int, interleaved: bool) -> int:
    if hidden_dim <= 0 or hidden_dim % 2:
        raise ValueError("hidden_dim must be a positive even number")
    scheduler_n = hidden_dim if interleaved else hidden_dim // 2
    return ceil_div(scheduler_n, TILE_N)


def gmm2_n_tiles(h: int) -> int:
    if h <= 0:
        raise ValueError("h must be positive")
    return ceil_div(h, TILE_N)


def chunk_target(
    cores: int,
    gmm1_tiles: int,
    gmm2_tiles: int,
    gmm1_owners_per_core: int = 2,
) -> int:
    if min(cores, gmm1_tiles, gmm2_tiles, gmm1_owners_per_core) <= 0:
        raise ValueError("core, tile and producer-depth values must be positive")
    gmm1_target = ceil_div(cores * gmm1_owners_per_core, gmm1_tiles)
    gmm2_target = ceil_div(cores, gmm2_tiles)
    return max(gmm1_target, gmm2_target)


@dataclass(frozen=True)
class Cursor:
    expert: int
    row: int


@dataclass(frozen=True)
class Segment:
    expert: int
    row_begin: int
    row_count: int
    chunk_count: int
    is_expert_final: bool


@dataclass(frozen=True)
class Wave:
    index: int
    begin: Cursor
    end: Cursor
    segments: tuple[Segment, ...]
    chunk_count: int
    valid_rows: int
    gmm1_owners: int
    gmm2_owners: int
    completed_expert_end: int


def plan_waves(
    counts: Sequence[int],
    target_chunks: int,
    gmm1_tiles: int,
    gmm2_tiles: int,
) -> tuple[Wave, ...]:
    if any(count < 0 for count in counts):
        raise ValueError("expert counts must be non-negative")
    if min(target_chunks, gmm1_tiles, gmm2_tiles) <= 0:
        raise ValueError("target and tile counts must be positive")

    waves: list[Wave] = []
    cursor = Cursor(0, 0)
    while cursor.expert < len(counts):
        begin = cursor
        segments: list[Segment] = []
        chunks = 0

        while cursor.expert < len(counts) and chunks < target_chunks:
            expert_rows = counts[cursor.expert]
            if expert_rows == 0 or cursor.row >= expert_rows:
                cursor = Cursor(cursor.expert + 1, 0)
                continue

            chunks_left = target_chunks - chunks
            remaining_rows = expert_rows - cursor.row
            segment_rows = min(remaining_rows, chunks_left * TILE_M)
            segment_chunks = ceil_div(segment_rows, TILE_M)
            segment_end = cursor.row + segment_rows
            is_expert_final = segment_end == expert_rows
            segments.append(
                Segment(
                    cursor.expert,
                    cursor.row,
                    segment_rows,
                    segment_chunks,
                    is_expert_final,
                )
            )
            chunks += segment_chunks
            cursor = (
                Cursor(cursor.expert + 1, 0)
                if is_expert_final
                else Cursor(cursor.expert, segment_end)
            )

        if cursor == begin:
            raise AssertionError("segmented Wave planner did not make progress")
        waves.append(
            Wave(
                index=len(waves),
                begin=begin,
                end=cursor,
                segments=tuple(segments),
                chunk_count=chunks,
                valid_rows=sum(segment.row_count for segment in segments),
                gmm1_owners=chunks * gmm1_tiles,
                gmm2_owners=chunks * gmm2_tiles,
                completed_expert_end=cursor.expert,
            )
        )

    return tuple(waves)


def replay_gmm2_segments(
    counts: Sequence[int], begin: Cursor, end: Cursor
) -> tuple[Segment, ...]:
    """Mirror ProcessGmm2Wave's replay of a GMM1-selected cursor range."""

    cursor = begin
    segments: list[Segment] = []
    while cursor != end and cursor.expert < len(counts):
        expert_rows = counts[cursor.expert]
        if expert_rows == 0 or cursor.row >= expert_rows:
            cursor = Cursor(cursor.expert + 1, 0)
            continue
        segment_end = end.row if cursor.expert == end.expert else expert_rows
        segment_rows = segment_end - cursor.row
        is_expert_final = segment_end == expert_rows
        segments.append(
            Segment(
                cursor.expert,
                cursor.row,
                segment_rows,
                ceil_div(segment_rows, TILE_M),
                is_expert_final,
            )
        )
        cursor = (
            Cursor(cursor.expert + 1, 0)
            if is_expert_final
            else Cursor(cursor.expert, segment_end)
        )
    assert cursor == end
    return tuple(segments)


def validate_plan(
    counts: Sequence[int],
    cores: int,
    h: int,
    hidden_dim: int,
    interleaved: bool,
    gmm1_owners_per_core: int = 2,
) -> tuple[Wave, ...]:
    p1 = gmm1_n_tiles(hidden_dim, interleaved)
    p2 = gmm2_n_tiles(h)
    target = chunk_target(cores, p1, p2, gmm1_owners_per_core)
    waves = plan_waves(counts, target, p1, p2)

    if counts:
        assert waves
        assert waves[0].begin == Cursor(0, 0)
        assert waves[-1].end == Cursor(len(counts), 0)
    else:
        assert not waves

    covered_rows = [0] * len(counts)
    for wave_idx, wave in enumerate(waves):
        assert wave.index == wave_idx
        if wave_idx:
            assert waves[wave_idx - 1].end == wave.begin
        assert wave.chunk_count == sum(segment.chunk_count for segment in wave.segments)
        assert wave.valid_rows == sum(segment.row_count for segment in wave.segments)
        assert wave.gmm1_owners == wave.chunk_count * p1
        assert wave.gmm2_owners == wave.chunk_count * p2
        assert wave.completed_expert_end == wave.end.expert
        assert replay_gmm2_segments(counts, wave.begin, wave.end) == wave.segments
        if wave.end.expert != len(counts):
            assert wave.chunk_count == target
            assert wave.gmm1_owners >= cores * gmm1_owners_per_core
            assert wave.gmm2_owners >= cores
        else:
            assert wave.chunk_count <= target

        seen_experts: set[int] = set()
        for segment in wave.segments:
            assert segment.expert not in seen_experts
            seen_experts.add(segment.expert)
            assert segment.row_begin == covered_rows[segment.expert]
            assert 0 < segment.row_count <= target * TILE_M
            assert segment.chunk_count == ceil_div(segment.row_count, TILE_M)
            assert segment.row_begin % TILE_M == 0
            segment_chunk_end = segment.row_begin // TILE_M + segment.chunk_count
            assert segment_chunk_end <= ceil_div(counts[segment.expert], TILE_M)
            assert (
                segment_chunk_end * p1 <= ceil_div(counts[segment.expert], TILE_M) * p1
            )
            covered_rows[segment.expert] += segment.row_count
            assert segment.is_expert_final == (
                covered_rows[segment.expert] == counts[segment.expert]
            )

    assert covered_rows == list(counts)
    return waves


def run_self_test(random_cases: int) -> dict[str, int]:
    exhaustive = 0
    values = (0, 1, 255, 256, 257)
    for expert_count in range(1, 5):
        for counts in itertools.product(values, repeat=expert_count):
            for cores in (1, 4, 24):
                for h in (256, 4096):
                    for hidden_dim in (512, 3072, 8192):
                        for interleaved in (False, True):
                            for depth in (1, 2):
                                validate_plan(
                                    counts,
                                    cores,
                                    h,
                                    hidden_dim,
                                    interleaved,
                                    depth,
                                )
                                exhaustive += 1

    rng = random.Random(20260804)
    for _ in range(random_cases):
        expert_count = rng.randint(1, 64)
        counts = tuple(rng.randint(0, 8192) for _ in range(expert_count))
        cores = rng.randint(1, 64)
        h = rng.randint(1, 32) * TILE_N
        hidden_dim = rng.randint(1, 64) * TILE_N * 2
        interleaved = bool(rng.getrandbits(1))
        depth = rng.randint(1, 3)
        first = validate_plan(counts, cores, h, hidden_dim, interleaved, depth)

        # Dispatch and AIC/AIV0 replay this same deterministic cursor rule.
        second = validate_plan(counts, cores, h, hidden_dim, interleaved, depth)
        assert first == second

    return {"exhaustive_configs": exhaustive, "random_cases": random_cases}


def parse_counts(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="32,511,1,256,1024,0,255,257")
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--h", type=int, default=4096, help="GMM2 output width")
    parser.add_argument(
        "--hidden-dim", type=int, default=8192, help="GMM1 gate+up total N"
    )
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--gmm1-depth", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--random-cases", type=int, default=20000)
    args = parser.parse_args()

    if args.self_test:
        if args.random_cases < 0:
            raise ValueError("random case count must be non-negative")
        print(json.dumps(run_self_test(args.random_cases), indent=2, sort_keys=True))
        return

    counts = parse_counts(args.counts)
    p1 = gmm1_n_tiles(args.hidden_dim, args.interleaved)
    p2 = gmm2_n_tiles(args.h)
    target = chunk_target(args.cores, p1, p2, args.gmm1_depth)
    waves = validate_plan(
        counts,
        args.cores,
        args.h,
        args.hidden_dim,
        args.interleaved,
        args.gmm1_depth,
    )
    result = {
        "config": {
            "counts": list(counts),
            "cores": args.cores,
            "h": args.h,
            "hidden_dim": args.hidden_dim,
            "interleaved": args.interleaved,
            "gmm1_depth": args.gmm1_depth,
        },
        "derived": {
            "gmm1_n_tiles": p1,
            "gmm2_n_tiles": p2,
            "chunks_per_wave": target,
        },
        "waves": [asdict(wave) for wave in waves],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
