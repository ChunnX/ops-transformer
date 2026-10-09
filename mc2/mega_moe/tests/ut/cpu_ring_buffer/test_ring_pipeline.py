#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""CPU lifetime model for the non-deduplicated MTE wave pipeline.

The model uses reader obligations on physical storage, rather than treating
wave completion itself as proof of safety. An overwrite fails when any reader
of the old generation remains outstanding. Operations represent completed
GM transfers; publishing a watermark before an asynchronous transfer finishes
is deliberately outside this model's assumptions.
"""

from dataclasses import dataclass
import random
import unittest


GROUP_ROWS = 256


@dataclass(frozen=True)
class Group:
    group_id: int
    expert: int
    expert_row: int
    logical_row: int
    valid_rows: int


def route_groups(expert_counts):
    """Independent row-level reference: experts never share a padded group."""
    groups = []
    logical_start = 0
    for expert, count in enumerate(expert_counts):
        for row in range(0, count, GROUP_ROWS):
            groups.append(Group(len(groups), expert, row, logical_start + row,
                                min(GROUP_ROWS, count - row)))
        logical_start += count
    return groups


class LiveStorage:
    """Physical cells carry a generation and every required reader."""

    def __init__(self, capacity_groups):
        self.capacity = capacity_groups
        self.cells = {}
        self.reads = 0
        self.writes = 0

    def key(self, stage, group, column):
        return stage, group % self.capacity, column

    def ready(self, stage, group, column):
        cell = self.cells.get(self.key(stage, group, column))
        return cell is not None and cell[0] == group

    def write(self, stage, group, column, readers):
        key = self.key(stage, group, column)
        previous = self.cells.get(key)
        if previous is not None and previous[1]:
            raise AssertionError(
                f"live overwrite: {key}, old group {previous[0]}, "
                f"new group {group}, pending {sorted(previous[1])}")
        self.cells[key] = [group, set(readers)]
        self.writes += 1

    def read_done(self, stage, group, column, reader):
        key = self.key(stage, group, column)
        if not self.ready(stage, group, column):
            raise AssertionError(f"wrong generation read: {key}, group {group}")
        pending = self.cells[key][1]
        if reader not in pending:
            raise AssertionError(f"unexpected or repeated reader: {reader}")
        pending.remove(reader)
        self.reads += 1

    def assert_drained(self):
        pending = {key: cell for key, cell in self.cells.items() if cell[1]}
        if pending:
            raise AssertionError(f"unconsumed final data: {pending}")


@dataclass(frozen=True)
class Operation:
    kind: str
    group: int
    column: int = 0


class Pipeline:
    """Per-core AIV1, AIC and AIV0 with independently skewed progress.

    Iteration w dispatches w+1, runs GMM1/activation for w, and runs
    GMM2/combine for w-1 when lag is enabled. The initial iteration dispatches
    waves 0 and 1. Every participant publishes w+1 only after all its iteration
    operations finish. Iteration w>=2 requires every participant's watermark
    to be at least w-1, allowing one overlapping prior iteration.
    """

    def __init__(self, expert_counts, budget, cores, lag, banks=3,
                 n1=4, n2=3, watermark=True):
        self.groups = route_groups(expert_counts)
        self.budget = budget
        self.cores = cores
        self.lag = lag
        self.n1 = n1
        self.n2 = n2
        self.watermark = watermark
        self.wave_count = (len(self.groups) + budget - 1) // budget
        self.iterations = self.wave_count + int(lag)
        self.storage = LiveStorage(banks * budget)
        self.participants = [(role, core) for role in ("v1", "cube", "v0")
                             for core in range(cores)]
        self.iteration = {part: 0 for part in self.participants}
        self.published = {part: 0 for part in self.participants}
        self.progress = {part: 0 for part in self.participants}
        self.work = {part: [self.make_work(part, w)
                            for w in range(self.iterations)]
                     for part in self.participants}

    def wave_groups(self, wave):
        if not 0 <= wave < self.wave_count:
            return []
        return self.groups[wave * self.budget:(wave + 1) * self.budget]

    def owned(self, group, column, columns, core):
        return (group * columns + column) % self.cores == core

    def tasks(self, kind, wave, columns, core):
        return [Operation(kind, g.group_id, n)
                for g in self.wave_groups(wave) for n in range(columns)
                if self.owned(g.group_id, n, columns, core)]

    def make_work(self, part, wave):
        role, core = part
        output_wave = wave - int(self.lag)
        if role == "v1":
            startup = self.tasks("dispatch", 0, 1, core) if wave == 0 else []
            return (startup + self.tasks("dispatch", wave + 1, 1, core)
                    + self.tasks("activation", wave, self.n1, core))
        if role == "cube":
            return (self.tasks("gmm1", wave, self.n1, core)
                    + self.tasks("gmm2", output_wave, self.n2, core))
        return self.tasks("combine", output_wave, self.n2, core)

    def normalize(self, part):
        while self.iteration[part] < self.iterations:
            w = self.iteration[part]
            if self.watermark and w >= 2 and min(self.published.values()) < w - 1:
                return
            if self.progress[part] != len(self.work[part][w]):
                return
            self.published[part] = w + 1
            self.iteration[part] += 1
            self.progress[part] = 0

    def next_operation(self, part):
        self.normalize(part)
        w = self.iteration[part]
        if w >= self.iterations:
            return None
        if self.watermark and w >= 2 and min(self.published.values()) < w - 1:
            return None
        return self.work[part][w][self.progress[part]]

    def inputs_ready(self, op):
        if op.kind == "dispatch":
            return True
        if op.kind == "gmm1":
            return self.storage.ready("dispatch", op.group, 0)
        if op.kind == "activation":
            return self.storage.ready("gmm1", op.group, op.column)
        if op.kind == "gmm2":
            return all(self.storage.ready("activation", op.group, n)
                       for n in range(self.n1))
        return self.storage.ready("gmm2", op.group, op.column)

    def execute(self, op):
        g, n = op.group, op.column
        if op.kind == "dispatch":
            self.storage.write("dispatch", g, 0,
                               [("gmm1", g, k) for k in range(self.n1)])
        elif op.kind == "gmm1":
            self.storage.read_done("dispatch", g, 0, ("gmm1", g, n))
            self.storage.write("gmm1", g, n, [("activation", g, n)])
        elif op.kind == "activation":
            self.storage.read_done("gmm1", g, n, ("activation", g, n))
            self.storage.write("activation", g, n,
                               [("gmm2", g, k) for k in range(self.n2)])
        elif op.kind == "gmm2":
            for k in range(self.n1):
                self.storage.read_done("activation", g, k, ("gmm2", g, n))
            self.storage.write("gmm2", g, n, [("combine", g, n)])
        else:
            self.storage.read_done("gmm2", g, n, ("combine", g, n))

    def run(self, seed, policy="random"):
        rng = random.Random(seed)
        steps = 0
        while True:
            previous_progress = (tuple(self.iteration.values()), tuple(self.published.values()))
            ready = []
            for part in self.participants:
                op = self.next_operation(part)
                if op is not None and self.inputs_ready(op):
                    ready.append((part, op))
            if all(w == self.iterations for w in self.iteration.values()):
                break
            if not ready:
                # Empty-core iterations also publish progress. Their updates
                # may unblock a participant visited earlier in this scan.
                current_progress = (tuple(self.iteration.values()), tuple(self.published.values()))
                if current_progress != previous_progress:
                    continue
                raise AssertionError(f"pipeline stalled: {self.iteration}, {self.published}")
            if policy == "producer_first":
                priority = {"dispatch": 0, "activation": 1, "gmm1": 2,
                            "gmm2": 3, "combine": 4}
                ready.sort(key=lambda pair: (priority[pair[1].kind], pair[0]))
                part, op = ready[0]
            elif policy == "core_skew":
                ready.sort(key=lambda pair: (pair[0][1], -self.iteration[pair[0]]))
                part, op = ready[0]
            else:
                part, op = rng.choice(ready)
            self.execute(op)
            self.progress[part] += 1
            steps += 1
        self.storage.assert_drained()
        expected_writes = len(self.groups) * (1 + self.n1 * 2 + self.n2)
        if self.storage.writes != expected_writes:
            raise AssertionError(f"missing production: {self.storage.writes} != {expected_writes}")
        return steps


class RingPipelineTest(unittest.TestCase):
    def test_expert_tails_have_distinct_complete_groups(self):
        groups = route_groups([100, 300, 50])
        self.assertEqual([g.valid_rows for g in groups], [100, 256, 44, 50])
        self.assertEqual([g.logical_row for g in groups], [0, 100, 356, 400])
        self.assertEqual([g.group_id * GROUP_ROWS for g in groups], [0, 256, 512, 768])
        self.assertEqual([g.expert_row for g in groups], [0, 0, 256, 0])

    def test_logical_metadata_and_physical_rows_remain_separate(self):
        groups = route_groups([0, 1, 255, 256, 257, 511, 513, 0, 7])
        for capacity in (3, 6, 15):
            logical_rows = []
            for group in groups:
                physical = [(group.group_id % capacity) * GROUP_ROWS + r
                            for r in range(group.valid_rows)]
                self.assertTrue(all(0 <= row < capacity * GROUP_ROWS for row in physical))
                self.assertEqual(len(set(physical)), group.valid_rows)
                logical_rows.extend(group.logical_row + r for r in range(group.valid_rows))
            self.assertEqual(logical_rows, list(range(sum([0, 1, 255, 256, 257, 511, 513, 0, 7]))))

    def test_three_banks_survive_skewed_consumers(self):
        for budget in (1, 2, 5, 9):
            for cores in (1, 2, 4, 7):
                for lag in (False, True):
                    counts = [0, 100, 300, 50, 256 * budget * 8 + 17, 513, 1]
                    for policy in ("producer_first", "core_skew", "random"):
                        with self.subTest(budget=budget, cores=cores, lag=lag, policy=policy):
                            Pipeline(counts, budget, cores, lag).run(17, policy)

    def test_random_routing_and_per_core_advancement(self):
        rng = random.Random(90210)
        for seed in range(150):
            budget = rng.randint(1, 9)
            counts = [rng.randint(0, budget * GROUP_ROWS * 3)
                      for _ in range(rng.randint(2, 15))]
            counts.append(budget * GROUP_ROWS * 6 + 1)
            with self.subTest(seed=seed, budget=budget):
                Pipeline(counts, budget, rng.randint(1, 8), bool(seed % 2),
                         n1=rng.randint(1, 7), n2=rng.randint(1, 5)).run(seed)

    def test_two_banks_do_not_cover_the_lagged_pipeline(self):
        with self.assertRaisesRegex(AssertionError, "live overwrite"):
            Pipeline([256 * 12], 1, 4, True, banks=2).run(0, "producer_first")

    def test_watermark_is_needed_even_with_three_banks(self):
        with self.assertRaisesRegex(AssertionError, "live overwrite"):
            Pipeline([256 * 18], 2, 4, True, watermark=False).run(0, "core_skew")


if __name__ == "__main__":
    unittest.main()
