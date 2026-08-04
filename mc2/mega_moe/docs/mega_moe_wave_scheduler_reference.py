#!/usr/bin/env python3
"""Scalar reference model for the MegaMoe 256-row Wave scheduler.

This is a design/verification model, not a device implementation.  It searches
the practical scheduling family proposed by the design document:

* each routed expert is logically split into <=256-row chunks;
* the cyclic fast path assigns a chunk's N tiles to consecutive logical AICs
  and searches the chunk-to-chunk starting-core stride;
* weighted tail-heavy Waves may instead use an ordered/LPT least-load owner
  table;
* GMM1 and GMM2 use independent strides and cursors;
* Wave sizes are selected with exact dynamic programming for the provided
  expert-token sequence, under a bounded Wave-size policy.

The cost model is theoretical MAC work.  It deliberately distinguishes the
non-interleaved GMM1 owner (two gate/up MMAD calls with K=h), interleaved GMM1
(one MMAD with K=h), and GMM2 (one MMAD with K=n).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable, Sequence


TILE = 256


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def tile_widths(width: int) -> tuple[int, ...]:
    if width <= 0:
        raise ValueError("matrix width must be positive")
    full, tail = divmod(width, TILE)
    result = [TILE] * full
    if tail:
        result.append(tail)
    return tuple(result)


def flatten_expert_rows(expert_tokens: Iterable[int]) -> tuple[int, ...]:
    rows: list[int] = []
    for token_count in expert_tokens:
        if token_count < 0:
            raise ValueError("expert token count must be non-negative")
        full, tail = divmod(token_count, TILE)
        rows.extend([TILE] * full)
        if tail:
            rows.append(tail)
    return tuple(rows)


@dataclass(frozen=True)
class StageSpec:
    name: str
    owner_widths: tuple[int, ...]
    k: int
    calls_per_owner: int

    @property
    def owners_per_chunk(self) -> int:
        return len(self.owner_widths)

    @property
    def full_chunk_work(self) -> int:
        return sum(
            TILE * width * self.k * self.calls_per_owner for width in self.owner_widths
        )

    def owner_work(self, m_rows: int, owner_idx: int) -> int:
        return m_rows * self.owner_widths[owner_idx] * self.k * self.calls_per_owner


def make_stage_specs(h: int, n: int, mode: str) -> tuple[StageSpec, StageSpec]:
    if h <= 0 or n <= 0:
        raise ValueError("h and n must be positive")
    if mode == "non_interleaved":
        gmm1 = StageSpec("gmm1", tile_widths(n), h, 2)
    elif mode == "interleaved":
        gmm1 = StageSpec("gmm1", tile_widths(2 * n), h, 1)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    gmm2 = StageSpec("gmm2", tile_widths(h), n, 1)
    return gmm1, gmm2


@dataclass(frozen=True)
class StageMapping:
    policy: str
    stride: int
    loads: tuple[int, ...]
    makespan: int
    total_work: int
    idle_core_work: int
    utilization: float
    lower_bound: int
    lower_bound_gap: float


def make_stage_mapping(
    policy: str, stride: int, loads: Sequence[int], max_task_work: int
) -> StageMapping:
    frozen_loads = tuple(loads)
    cores = len(frozen_loads)
    total_work = sum(frozen_loads)
    makespan = max(frozen_loads, default=0)
    capacity = makespan * cores
    idle = capacity - total_work
    utilization = 1.0 if capacity == 0 else total_work / capacity
    average_bound = ceil_div(total_work, cores) if cores else 0
    lower_bound = max(max_task_work, average_bound)
    lower_bound_gap = 1.0 if lower_bound == 0 else makespan / lower_bound
    return StageMapping(
        policy,
        stride,
        frozen_loads,
        makespan,
        total_work,
        idle,
        utilization,
        lower_bound,
        lower_bound_gap,
    )


def map_stage(
    rows: Sequence[int], spec: StageSpec, cores: int, stride: int
) -> StageMapping:
    if cores <= 0:
        raise ValueError("cores must be positive")
    if not 0 <= stride < cores:
        raise ValueError("stride must be in [0, cores)")
    loads = [0] * cores
    max_task_work = 0
    for chunk_idx, m_rows in enumerate(rows):
        base_core = (chunk_idx * stride) % cores
        for owner_idx in range(spec.owners_per_chunk):
            core_idx = (base_core + owner_idx) % cores
            task_work = spec.owner_work(m_rows, owner_idx)
            loads[core_idx] += task_work
            max_task_work = max(max_task_work, task_work)
    return make_stage_mapping("cyclic", stride, loads, max_task_work)


def map_stage_ordered_least_load(
    rows: Sequence[int], spec: StageSpec, cores: int
) -> StageMapping:
    """Assign chunk-major tasks to the currently least-loaded logical AIC.

    Chunk-major traversal preserves early-chunk priority.  The rotating tie
    cursor prevents equal-load ties from permanently selecting low core IDs.
    This policy needs an owner table (or equivalent deterministic replay), but
    adapts to short expert tails better than a single cyclic stride.
    """

    if cores <= 0:
        raise ValueError("cores must be positive")
    loads = [0] * cores
    tie_cursor = 0
    max_task_work = 0
    for m_rows in rows:
        for owner_idx in range(spec.owners_per_chunk):
            task_work = spec.owner_work(m_rows, owner_idx)
            core_idx = min(
                range(cores), key=lambda idx: (loads[idx], (idx - tie_cursor) % cores)
            )
            loads[core_idx] += task_work
            tie_cursor = (core_idx + 1) % cores
            max_task_work = max(max_task_work, task_work)
    return make_stage_mapping("ordered_least_load", -1, loads, max_task_work)


def map_stage_lpt(rows: Sequence[int], spec: StageSpec, cores: int) -> StageMapping:
    """Longest-processing-time first oracle for weighted tail-heavy Waves.

    LPT is useful as a utilization reference and for a stage whose whole input
    window is already ready.  It may reorder chunks, so the final design must
    not use it ahead of an older ready-critical chunk without checking the
    Dispatch/SwiGLU dependency.
    """

    tasks: list[tuple[int, int, int]] = []
    for chunk_idx, m_rows in enumerate(rows):
        for owner_idx in range(spec.owners_per_chunk):
            tasks.append((spec.owner_work(m_rows, owner_idx), chunk_idx, owner_idx))
    tasks.sort(key=lambda item: (-item[0], item[1], item[2]))

    loads = [0] * cores
    tie_cursor = 0
    for task_work, _, _ in tasks:
        core_idx = min(
            range(cores), key=lambda idx: (loads[idx], (idx - tie_cursor) % cores)
        )
        loads[core_idx] += task_work
        tie_cursor = (core_idx + 1) % cores
    max_task_work = max((item[0] for item in tasks), default=0)
    return make_stage_mapping("lpt_least_load", -1, loads, max_task_work)


def best_stage_mapping(
    rows: Sequence[int], spec: StageSpec, cores: int
) -> StageMapping:
    candidates = [map_stage(rows, spec, cores, stride) for stride in range(cores)]
    candidates.append(map_stage_ordered_least_load(rows, spec, cores))
    candidates.append(map_stage_lpt(rows, spec, cores))
    current_stride = spec.owners_per_chunk % cores

    def rank(mapping: StageMapping) -> tuple[int, int, int, int, int]:
        # Exact makespan is the primary objective.  Then minimize idle/spread.
        # Prefer the current cursor stride on a complete tie to avoid needless
        # implementation changes.
        spread = max(mapping.loads, default=0) - min(mapping.loads, default=0)
        current_penalty = (
            0 if mapping.policy == "cyclic" and mapping.stride == current_stride else 1
        )
        table_penalty = 1 if mapping.policy == "ordered_least_load" else 0
        return (
            mapping.makespan,
            mapping.idle_core_work,
            spread,
            current_penalty,
            table_penalty,
        )

    return min(candidates, key=rank)


@dataclass(frozen=True)
class WaveCost:
    begin_chunk: int
    chunk_count: int
    rows: tuple[int, ...]
    gmm1: StageMapping
    gmm2: StageMapping
    makespan: int
    total_work: int
    idle_core_work: int
    utilization: float


def evaluate_wave(
    all_rows: Sequence[int],
    begin: int,
    count: int,
    gmm1: StageSpec,
    gmm2: StageSpec,
    cores: int,
) -> WaveCost:
    rows = tuple(all_rows[begin : begin + count])
    if len(rows) != count or count <= 0:
        raise ValueError("invalid Wave range")
    map1 = best_stage_mapping(rows, gmm1, cores)
    map2 = best_stage_mapping(rows, gmm2, cores)
    makespan = map1.makespan + map2.makespan
    total_work = map1.total_work + map2.total_work
    capacity = makespan * cores
    idle = capacity - total_work
    utilization = total_work / capacity
    return WaveCost(
        begin, count, rows, map1, map2, makespan, total_work, idle, utilization
    )


@dataclass(frozen=True)
class BaseWave:
    producer_depth: int
    q_min: int
    q1_align: int
    q2_align: int
    q_count_align: int
    q_exact_count_candidate: int
    q_selected: int
    selected_utilization: float


def choose_base_wave(
    gmm1: StageSpec, gmm2: StageSpec, cores: int, producer_depth: int
) -> BaseWave:
    p1 = gmm1.owners_per_chunk
    p2 = gmm2.owners_per_chunk
    q_min = max(ceil_div(producer_depth * cores, p1), ceil_div(cores, p2))
    q1_align = cores // math.gcd(cores, p1)
    q2_align = cores // math.gcd(cores, p2)
    q_count_align = math.lcm(q1_align, q2_align)
    q_exact = ceil_div(q_min, q_count_align) * q_count_align

    # Count alignment is sufficient for full-width N tiles.  Search farther to
    # account for N-tail weights and select the smallest exactly balanced Wave
    # if one exists.  Otherwise select maximum utilization, then the smaller Q.
    search_limit = max(q_exact, cores)
    best_q = q_min
    best_util = -1.0
    for q in range(q_min, search_limit + 1):
        rows = (TILE,) * q
        wave = evaluate_wave(rows, 0, q, gmm1, gmm2, cores)
        if wave.idle_core_work == 0:
            best_q = q
            best_util = wave.utilization
            break
        if wave.utilization > best_util:
            best_q = q
            best_util = wave.utilization

    return BaseWave(
        producer_depth,
        q_min,
        q1_align,
        q2_align,
        q_count_align,
        q_exact,
        best_q,
        best_util,
    )


def base_wave_frontier(
    gmm1: StageSpec, gmm2: StageSpec, cores: int, base: BaseWave
) -> list[dict[str, object]]:
    """Return the Q/utilization Pareto improvements from q_min to one core period."""

    limit = max(base.q_exact_count_candidate, cores)
    frontier: list[dict[str, object]] = []
    best_util = -1.0
    for q in range(base.q_min, limit + 1):
        candidate = evaluate_wave((TILE,) * q, 0, q, gmm1, gmm2, cores)
        if candidate.utilization > best_util + 1e-15:
            frontier.append(
                {
                    "q": q,
                    "tokens": q * TILE,
                    "utilization": candidate.utilization,
                    "gmm1_policy": candidate.gmm1.policy,
                    "gmm1_stride": candidate.gmm1.stride,
                    "gmm2_policy": candidate.gmm2.policy,
                    "gmm2_stride": candidate.gmm2.stride,
                }
            )
            best_util = candidate.utilization
        if candidate.idle_core_work == 0:
            break
    return frontier


@dataclass(frozen=True)
class PoolSplit:
    gmm1_cores: int
    gmm2_cores: int
    gmm1_work: int
    gmm2_work: int
    normalized_bottleneck: float


def choose_fixed_pool(gmm1: StageSpec, gmm2: StageSpec, cores: int) -> PoolSplit:
    if cores < 2:
        raise ValueError("fixed two-stage pools require at least two cores")
    work1 = gmm1.full_chunk_work
    work2 = gmm2.full_chunk_work
    best: PoolSplit | None = None
    for c1 in range(1, cores):
        c2 = cores - c1
        bottleneck = max(work1 / c1, work2 / c2)
        candidate = PoolSplit(c1, c2, work1, work2, bottleneck)
        if best is None or (
            candidate.normalized_bottleneck,
            abs(c1 - cores * work1 / (work1 + work2)),
        ) < (
            best.normalized_bottleneck,
            abs(best.gmm1_cores - cores * work1 / (work1 + work2)),
        ):
            best = candidate
    assert best is not None
    return best


@dataclass(frozen=True)
class WavePlan:
    wave_sizes: tuple[int, ...]
    waves: tuple[WaveCost, ...]
    makespan: int
    total_work: int
    idle_core_work: int
    utilization: float


def optimize_actual_waves(
    rows: Sequence[int],
    gmm1: StageSpec,
    gmm2: StageSpec,
    cores: int,
    q_min: int,
    q_max: int,
) -> WavePlan:
    """Exact DP for the bounded practical Wave family.

    Every non-final Wave must contain at least q_min chunks.  The final Wave may
    be smaller only when the whole remaining workload is smaller; DP is free to
    absorb such a tail into its predecessor because q_max is normally set to
    2*q_base-1.
    """

    total_chunks = len(rows)
    if total_chunks == 0:
        return WavePlan((), (), 0, 0, 0, 1.0)
    if q_max <= 0:
        raise ValueError("q_max must be positive")

    @lru_cache(maxsize=None)
    def wave(begin: int, count: int) -> WaveCost:
        return evaluate_wave(rows, begin, count, gmm1, gmm2, cores)

    # dp[end] = (total makespan, wave count, predecessor, last wave size)
    infinity = 1 << 120
    dp: list[tuple[int, int, int, int]] = [
        (infinity, infinity, -1, 0) for _ in range(total_chunks + 1)
    ]
    dp[0] = (0, 0, -1, 0)
    for begin in range(total_chunks):
        if dp[begin][0] == infinity:
            continue
        remaining = total_chunks - begin
        for count in range(1, min(q_max, remaining) + 1):
            end = begin + count
            if end != total_chunks and count < q_min:
                continue
            candidate = (
                dp[begin][0] + wave(begin, count).makespan,
                dp[begin][1] + 1,
                begin,
                count,
            )
            if candidate[:2] < dp[end][:2]:
                dp[end] = candidate

    if dp[total_chunks][0] == infinity:
        raise ValueError("q_max is too small to form a valid Wave plan")

    ranges: list[tuple[int, int]] = []
    end = total_chunks
    while end:
        _, _, begin, count = dp[end]
        ranges.append((begin, count))
        end = begin
    ranges.reverse()
    waves = tuple(wave(begin, count) for begin, count in ranges)
    makespan = sum(item.makespan for item in waves)
    total_work = sum(item.total_work for item in waves)
    capacity = makespan * cores
    return WavePlan(
        tuple(item.chunk_count for item in waves),
        waves,
        makespan,
        total_work,
        capacity - total_work,
        total_work / capacity,
    )


def mapping_summary(mapping: StageMapping) -> dict[str, object]:
    return {
        "policy": mapping.policy,
        "stride": mapping.stride,
        "min_core_load": min(mapping.loads, default=0),
        "max_core_load": max(mapping.loads, default=0),
        "idle_core_work": mapping.idle_core_work,
        "utilization": mapping.utilization,
        "lower_bound": mapping.lower_bound,
        "lower_bound_gap": mapping.lower_bound_gap,
    }


def wave_summary(wave: WaveCost) -> dict[str, object]:
    return {
        "begin_chunk": wave.begin_chunk,
        "chunk_count": wave.chunk_count,
        "rows": list(wave.rows),
        "utilization": wave.utilization,
        "idle_core_work": wave.idle_core_work,
        "gmm1": mapping_summary(wave.gmm1),
        "gmm2": mapping_summary(wave.gmm2),
    }


def run_self_test(case_count: int = 500) -> dict[str, int]:
    rng = random.Random(20260803)
    checked = 0
    for _ in range(case_count):
        cores = rng.randint(2, 64)
        h = rng.randint(1, 32) * 128
        n = rng.randint(1, 32) * 128
        chunks = rng.randint(1, 2 * cores)
        rows = tuple(rng.randint(1, TILE) for _ in range(chunks))
        non_g1, g2 = make_stage_specs(h, n, "non_interleaved")
        int_g1, int_g2 = make_stage_specs(h, n, "interleaved")

        # Both GMM1 layouts cover exactly M * 2n * h theoretical work.
        assert non_g1.full_chunk_work == int_g1.full_chunk_work
        assert g2 == int_g2

        for stage in (non_g1, int_g1, g2):
            best = best_stage_mapping(rows, stage, cores)
            current = map_stage(rows, stage, cores, stage.owners_per_chunk % cores)
            expected_work = sum(
                m * sum(stage.owner_widths) * stage.k * stage.calls_per_owner
                for m in rows
            )
            assert best.total_work == expected_work
            assert best.makespan <= current.makespan

        for mode in ("non_interleaved", "interleaved"):
            g1, stage2 = make_stage_specs(h, n, mode)
            base = choose_base_wave(g1, stage2, cores, 2)
            assert base.q_selected >= base.q_min
            assert base.q_exact_count_candidate * g1.owners_per_chunk % cores == 0
            assert base.q_exact_count_candidate * stage2.owners_per_chunk % cores == 0
        checked += 1
    return {"random_cases": checked}


def parse_expert_tokens(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--h", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument(
        "--mode", choices=("non_interleaved", "interleaved"), default="non_interleaved"
    )
    parser.add_argument("--producer-depth", type=int, default=2)
    parser.add_argument(
        "--chunks",
        type=int,
        default=24,
        help="full 256-row chunks when --expert-tokens is absent",
    )
    parser.add_argument(
        "--expert-tokens",
        default="",
        help="comma-separated actual expert token counts; overrides --chunks",
    )
    parser.add_argument("--q-max", type=int, default=0, help="0 selects 2*q_selected-1")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-cases", type=int, default=500)
    args = parser.parse_args()

    if args.self_test:
        if args.self_test_cases <= 0:
            raise ValueError("self-test case count must be positive")
        print(json.dumps(run_self_test(args.self_test_cases), indent=2, sort_keys=True))
        return

    if args.cores < 2:
        raise ValueError("cores must be at least 2")
    if args.producer_depth <= 0:
        raise ValueError("producer depth must be positive")

    expert_tokens = parse_expert_tokens(args.expert_tokens)
    rows = (
        flatten_expert_rows(expert_tokens) if expert_tokens else (TILE,) * args.chunks
    )
    gmm1, gmm2 = make_stage_specs(args.h, args.n, args.mode)
    base = choose_base_wave(gmm1, gmm2, args.cores, args.producer_depth)
    q_max = args.q_max or (2 * base.q_selected - 1)
    plan = optimize_actual_waves(rows, gmm1, gmm2, args.cores, base.q_min, q_max)
    pool = choose_fixed_pool(gmm1, gmm2, args.cores)

    result = {
        "config": {
            "cores": args.cores,
            "h": args.h,
            "n": args.n,
            "mode": args.mode,
            "expert_tokens": list(expert_tokens),
            "chunk_rows": list(rows),
        },
        "stages": {
            "gmm1": {
                "owners_per_chunk": gmm1.owners_per_chunk,
                "owner_widths": list(gmm1.owner_widths),
                "k": gmm1.k,
                "calls_per_owner": gmm1.calls_per_owner,
                "full_chunk_work": gmm1.full_chunk_work,
            },
            "gmm2": {
                "owners_per_chunk": gmm2.owners_per_chunk,
                "owner_widths": list(gmm2.owner_widths),
                "k": gmm2.k,
                "calls_per_owner": gmm2.calls_per_owner,
                "full_chunk_work": gmm2.full_chunk_work,
            },
        },
        "base_wave": asdict(base),
        "base_wave_frontier": base_wave_frontier(gmm1, gmm2, args.cores, base),
        "fixed_pool_comparison": asdict(pool),
        "actual_plan": {
            "q_max": q_max,
            "wave_sizes": list(plan.wave_sizes),
            "utilization": plan.utilization,
            "idle_core_work": plan.idle_core_work,
            "waves": [wave_summary(item) for item in plan.waves],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
