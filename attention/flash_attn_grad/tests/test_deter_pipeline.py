#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Host-only regression checks for the deterministic physical-round pipeline.

The deterministic index formula assigns work in *physical* rounds.  Empty
rounds may suppress a new MM task, but they must not delay an already fetched
vector/fix task.  Delaying such a task until the final drain can make two cores
write the same dK/dV target in the same FIX synchronization round.
"""

from collections import defaultdict, namedtuple
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "op_kernel" / "fag_deter_ondevice.py"
SPEC = spec_from_file_location("fag_deter_ondevice_pipeline_test", MODULE_PATH)
DETER = module_from_spec(SPEC)
SPEC.loader.exec_module(DETER)


Case = namedtuple(
    "Case",
    "name mask_mode batch n2 group m n cores s1_token s2_token",
)
Event = namedtuple("Event", "phase core round_id index")


CASES = (
    Case("dense_basic", 0, 1, 1, 1, 4, 5, 4, 0, 0),
    Case("dense_multi_head_gqa", 0, 2, 2, 3, 4, 6, 8, 0, 0),
    Case("causal", 3, 1, 1, 2, 6, 4, 8, 0, 0),
    Case("sparse_band", 1, 1, 2, 2, 6, 5, 8, 64, 256),
    # Negative padded token: p=1/q=3.  Several cores fetch in round 1 and
    # must retire during an otherwise empty physical round.
    Case("mask4_internal_empty", 4, 1, 4, 1, 2, 3, 8, -128, 129),
    # MHA q==0 boundary from the board regression. Valid work alternates
    # with empty rounds; last-hit lookahead executes only the final 20 tiles.
    Case("mask4_mha_q_zero", 4, 4, 32, 1, 3, 1, 36, 256, -256),
    # Regression from the board failure: core2/index8 and core3/index18 have
    # the same dKV target, but their physical fetch rounds are 5 and 1.
    Case("dense_g2_drain_collision", 0, 1, 1, 2, 2, 5, 36, 0, 0),
)


class ConstInfo:
    pass


def _make_const(case):
    const = ConstInfo()
    const.bSize = case.batch
    const.n2Size = case.n2
    const.gSize = case.group
    const.s1Outer = case.m
    const.s2Outer = case.n
    const.s1Size = case.m * 128
    const.s2Size = case.n * 128
    const.s1Token = case.s1_token
    const.s2Token = case.s2_token
    return const


def _round_schedule(case):
    const = _make_const(case)
    DETER.mask_mode = case.mask_mode
    max_round = DETER.deter_lin_max_round(const, case.cores)
    by_core = []
    for core_id in range(case.cores):
        by_core.append(
            [
                DETER._deter_lin_gidx_host(const, core_id, case.cores, round_id)
                for round_id in range(max_round)
            ]
        )
    return max_round, by_core


def _physical_trace(case, lag):
    """Reference F/V/X state machine keyed by physical, not valid-task, round."""
    max_round, schedule = _round_schedule(case)
    events = []
    for round_id in range(max_round + lag):
        for core_id in range(case.cores):
            if round_id < max_round:
                index = schedule[core_id][round_id]
                if index >= 0:
                    events.append(Event("F", core_id, round_id, index))

            fetch_round = round_id - lag
            if 0 <= fetch_round < max_round:
                index = schedule[core_id][fetch_round]
                if index >= 0:
                    events.append(Event("V", core_id, round_id, index))
                    events.append(Event("X", core_id, round_id, index))
    return max_round, schedule, events


def _legacy_skip_empty_fix_trace(case, lag):
    """Model the old bug: an empty current round skips pending fix work."""
    max_round, schedule = _round_schedule(case)
    events = []
    for core_id, indices in enumerate(schedule):
        pending = []
        for round_id, index in enumerate(indices):
            if index < 0:
                continue
            pending.append(index)
            if len(pending) > lag:
                events.append(Event("X", core_id, round_id, pending.pop(0)))
        for drain_id in range(lag):
            if pending:
                events.append(Event("X", core_id, max_round + drain_id, pending.pop(0)))
        if pending:
            raise AssertionError("legacy model left more than lag pending tasks")
    return max_round, events


def _dkv_target(case, index):
    """dK/dV accumulation target; G and s1 do not distinguish the target."""
    tiles_per_head = case.m * case.n
    head = index // tiles_per_head
    batch = head // (case.n2 * case.group)
    n2_idx = (head // case.group) % case.n2
    s2_idx = (index % tiles_per_head) // case.m
    return batch, n2_idx, s2_idx


def _phase_map(events, phase):
    return {
        (event.core, event.index): event.round_id
        for event in events
        if event.phase == phase
    }


class TestDeterPhysicalRoundPipeline(unittest.TestCase):
    def test_kernel_maps_each_physical_round_without_dynamic_scan(self):
        schedule_source = (ROOT / "op_kernel" / "fag_schedule.py").read_text(
            encoding="utf-8"
        )
        kernel_source = "\n".join(
            (ROOT / "op_kernel" / name).read_text(encoding="utf-8")
            for name in (
                "fag_kernel.py",
                "fag_kernel_norm.py",
                "fag_kernel_deter.py",
            )
        )
        self.assertNotIn("find_next_valid_deter", schedule_source)
        self.assertNotIn("find_next_valid_deter", kernel_source)
        # Cube/vector in preload3 and preload2 each map the current physical
        # slot and, for a valid tile, its immediate L0C neighbor.
        self.assertEqual(kernel_source.count("skip_empty_deter("), 8)

    def test_empty_physical_round_splits_l0c_group(self):
        case = next(item for item in CASES if item.name == "sparse_band")
        _max_round, schedule = _round_schedule(case)
        found_split = False
        for core in schedule:
            valid = [
                (round_id, index) for round_id, index in enumerate(core) if index >= 0
            ]
            for (prev_round, prev_index), (round_id, index) in zip(valid, valid[1:]):
                if (
                    _dkv_target(case, prev_index) == _dkv_target(case, index)
                    and round_id > prev_round + 1
                ):
                    self.assertTrue(
                        all(
                            core[gap_round] < 0
                            for gap_round in range(prev_round + 1, round_id)
                        )
                    )
                    found_split = True
        self.assertTrue(found_split)

    def test_mask4_q_zero_has_all_interleaved_rounds(self):
        case = next(item for item in CASES if item.name == "mask4_mha_q_zero")
        max_round, schedule = _round_schedule(case)
        self.assertEqual(max_round, 17)
        self.assertEqual(
            [
                sum(core[round_id] >= 0 for core in schedule)
                for round_id in range(max_round)
            ],
            [0, 36, 0, 36, 0, 36, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_preload2_and_preload3_keep_f_v_x_on_physical_rounds(self):
        for pipeline, lag in (("preload2", 1), ("preload3", 2)):
            for case in CASES:
                with self.subTest(pipeline=pipeline, case=case.name):
                    max_round, _schedule, events = _physical_trace(case, lag)
                    fetched = _phase_map(events, "F")
                    vectorized = _phase_map(events, "V")
                    fixed = _phase_map(events, "X")

                    # The tail drains exactly the tasks produced in F.  No
                    # valid task disappears or is duplicated in either stage.
                    self.assertEqual(fetched.keys(), vectorized.keys())
                    self.assertEqual(fetched.keys(), fixed.keys())
                    self.assertTrue(
                        all(round_id < max_round + lag for round_id in fixed.values())
                    )
                    for key, fetch_round in fetched.items():
                        self.assertEqual(vectorized[key], fetch_round + lag)
                        self.assertEqual(fixed[key], fetch_round + lag)

                    for core_id in range(case.cores):
                        counts = [
                            sum(
                                event.core == core_id and event.phase == phase
                                for event in events
                            )
                            for phase in ("F", "V", "X")
                        ]
                        self.assertEqual(counts[0], counts[1])
                        self.assertEqual(counts[0], counts[2])

    def test_mask4_pending_work_retires_during_internal_empty_round(self):
        case = next(item for item in CASES if item.name == "mask4_internal_empty")
        max_round, schedule = _round_schedule(case)
        self.assertEqual(max_round, 6)
        self.assertEqual(schedule[4], [-1, 2, -1, -1, -1, -1])

        for pipeline, lag in (("preload2", 1), ("preload3", 2)):
            with self.subTest(pipeline=pipeline):
                _max_round, _schedule, events = _physical_trace(case, lag)
                ready_round = 1 + lag
                self.assertEqual(schedule[4][ready_round], -1)
                self.assertIn(Event("V", 4, ready_round, 2), events)
                self.assertIn(Event("X", 4, ready_round, 2), events)

    def test_same_dkv_target_has_only_one_writer_per_epoch(self):
        for pipeline, lag in (("preload2", 1), ("preload3", 2)):
            for case in CASES:
                with self.subTest(pipeline=pipeline, case=case.name):
                    _max_round, _schedule, events = _physical_trace(case, lag)
                    writers = {}
                    for event in events:
                        if event.phase != "X":
                            continue
                        key = (event.round_id, _dkv_target(case, event.index))
                        self.assertNotIn(
                            key,
                            writers,
                            msg=(
                                f"{pipeline} duplicate dKV writer at {key}: "
                                f"{writers.get(key)} and {event}"
                            ),
                        )
                        writers[key] = event

    def test_dense_g2_case_would_collide_if_empty_rounds_skipped(self):
        case = next(item for item in CASES if item.name == "dense_g2_drain_collision")
        max_round, schedule = _round_schedule(case)
        self.assertEqual(max_round, 6)
        self.assertEqual(schedule[2][5], 8)
        self.assertEqual(schedule[3][1], 18)
        self.assertEqual(_dkv_target(case, 8), (0, 0, 4))
        self.assertEqual(_dkv_target(case, 18), (0, 0, 4))

        for pipeline, lag in (("preload2", 1), ("preload3", 2)):
            with self.subTest(pipeline=pipeline):
                _max_round, _schedule, physical = _physical_trace(case, lag)
                physical_rounds = _phase_map(physical, "X")
                self.assertEqual(physical_rounds[(2, 8)], 5 + lag)
                self.assertEqual(physical_rounds[(3, 18)], 1 + lag)
                self.assertNotEqual(physical_rounds[(2, 8)], physical_rounds[(3, 18)])

                _max_round, legacy = _legacy_skip_empty_fix_trace(case, lag)
                grouped = defaultdict(list)
                for event in legacy:
                    grouped[(event.round_id, _dkv_target(case, event.index))].append(
                        (event.core, event.index)
                    )
                collision_epoch = max_round + lag - 1
                self.assertEqual(
                    sorted(grouped[(collision_epoch, (0, 0, 4))]),
                    [(2, 8), (3, 18)],
                )

    def test_kernel_does_not_gate_the_pipeline_on_an_empty_fetch(self):
        source = "\n".join(
            (ROOT / "op_kernel" / name).read_text(encoding="utf-8")
            for name in (
                "fag_kernel.py",
                "fag_kernel_norm.py",
                "fag_kernel_deter.py",
            )
        )
        call_count = source.count("skip_deter_empty_pipe(")
        self.assertEqual(
            call_count,
            0,
            msg=(
                "an empty deterministic fetch must not gate pending V/X work; "
                f"found {call_count} skip_deter_empty_pipe call(s)"
            ),
        )


class TestDeterNormalFlowSplit(unittest.TestCase):
    """Compile-time path split: deter is dispatch only, loops stay specialized."""

    @classmethod
    def setUpClass(cls):
        cls.files = {
            name: (ROOT / "op_kernel" / name).read_text(encoding="utf-8")
            for name in (
                "fag_kernel.py",
                "fag_kernel_norm.py",
                "fag_kernel_deter.py",
            )
        }
        cls.source = "\n".join(cls.files.values())
        cls.trees = {name: ast.parse(src) for name, src in cls.files.items()}
        cls.funcs = {}
        cls.fn_nodes = {}
        cls.fn_file = {}
        for name, src in cls.files.items():
            for node in cls.trees[name].body:
                if isinstance(node, ast.FunctionDef):
                    cls.funcs[node.name] = ast.get_source_segment(src, node)
                    cls.fn_nodes[node.name] = node
                    cls.fn_file[node.name] = name

    LOOP_NAMES = (
        "_run_preload3_cube_normal",
        "_run_preload3_cube_deter",
        "_run_preload3_vector_normal",
        "_run_preload3_vector_deter",
        "_run_preload2_cube_bn2",
        "_run_preload2_vector_bn2",
        "_run_preload2_cube_gs_normal",
        "_run_preload2_vector_gs_normal",
        "_run_preload2_cube_gs_deter",
        "_run_preload2_vector_gs_deter",
    )

    def test_specialized_loop_functions_exist(self):
        for name in self.LOOP_NAMES:
            self.assertIn(name, self.funcs)

    def test_loops_live_in_split_modules(self):
        for name in (
            "_run_preload3_cube_normal",
            "_run_preload3_vector_normal",
            "_run_preload2_cube_bn2",
            "_run_preload2_vector_bn2",
            "_run_preload2_cube_gs_normal",
            "_run_preload2_vector_gs_normal",
        ):
            self.assertEqual(self.fn_file[name], "fag_kernel_norm.py", msg=name)
        for name in (
            "_run_preload3_cube_deter",
            "_run_preload3_vector_deter",
            "_run_preload2_cube_gs_deter",
            "_run_preload2_vector_gs_deter",
        ):
            self.assertEqual(self.fn_file[name], "fag_kernel_deter.py", msg=name)
        self.assertNotIn("def _run_preload", self.files["fag_kernel.py"])
        self.assertEqual(self.fn_file["flash_attn_grad_preload3"], "fag_kernel.py")
        self.assertEqual(self.fn_file["flash_attn_grad_preload2"], "fag_kernel.py")
        self.assertNotIn("skip_empty_deter(", self.files["fag_kernel_norm.py"])
        self.assertNotIn("fetch_work_gidx(", self.files["fag_kernel_deter.py"])

    def test_specialized_loops_take_make_tuple_bags(self):
        expected = ["constInfo", "sched", "gm", "buf", "runInfos"]
        for name in self.LOOP_NAMES:
            args = [arg.arg for arg in self.fn_nodes[name].args.args]
            self.assertEqual(args, expected, msg=name)
        self.assertGreaterEqual(self.source.count("pl.make_tuple("), 8)
        self.assertNotIn("class _ArgPack", self.source)
        self.assertNotIn("**kwargs", self.source)
        for name in self.LOOP_NAMES:
            src = self.funcs[name]
            self.assertIn(" = sched.", src, msg=name)
            self.assertIn(" = gm.", src, msg=name)
            self.assertIn(" = buf.", src, msg=name)
            self.assertNotIn("taskId = sched.", src, msg=name)
            self.assertNotIn("physicalRound = sched.", src, msg=name)
            self.assertNotIn("denseCursor = sched.", src, msg=name)

    def test_normal_loops_have_no_physical_clock_or_fix_tokens(self):
        forbidden = (
            "physicalRound",
            "readyRound",
            "SYNC_DQ_DETER_FIX_FLAG",
            "SYNC_DK_DETER_FIX_FLAG",
            "SYNC_DV_DETER_FIX_FLAG",
        )
        for name in (
            "_run_preload3_cube_normal",
            "_run_preload3_vector_normal",
            "_run_preload2_cube_gs_normal",
            "_run_preload2_vector_gs_normal",
        ):
            src = self.funcs[name]
            for token in forbidden:
                self.assertNotIn(token, src, msg=f"{name} contains {token}")

    def test_deter_loops_have_no_runtime_deter_branch(self):
        for name in (
            "_run_preload3_cube_deter",
            "_run_preload3_vector_deter",
            "_run_preload2_cube_gs_deter",
            "_run_preload2_vector_gs_deter",
        ):
            src = self.funcs[name]
            self.assertNotIn("if deter == 0", src)
            self.assertNotIn("if deter == 1", src)

    def test_bn2_loops_do_not_use_deter_fix_tokens(self):
        for name in ("_run_preload2_cube_bn2", "_run_preload2_vector_bn2"):
            src = self.funcs[name]
            for token in (
                "SYNC_DQ_DETER_FIX_FLAG",
                "SYNC_DK_DETER_FIX_FLAG",
                "SYNC_DV_DETER_FIX_FLAG",
                "physicalRound",
                "readyRound",
            ):
                self.assertNotIn(token, src, msg=f"{name} contains {token}")
            self.assertIn("SYNC_DETER_FIX_FLAG", src)

    def test_specialized_loops_have_no_global_barrier(self):
        for name, src in self.funcs.items():
            if not name.startswith("_run_preload"):
                continue
            self.assertNotIn("sync_all()", src, msg=name)
            self.assertNotIn("bar_all()", src, msg=name)


if __name__ == "__main__":
    unittest.main()
