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
"""Host-only control-flow regressions; these do not simulate device pipe timing.

Execute the actual normal-loop Python AST with instrumented compute/event stubs.
Check task retirement, fetch gating and event order independently of PyPTO.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


KERNEL = Path(__file__).parents[1] / "op_kernel"


def functions(filename):
    tree = ast.parse((KERNEL / filename).read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def execute(node, namespace):
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<kernel-control-flow>", "exec"),
        namespace,
    )
    return namespace[node.name]


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + "." + node.attr
    return ""


class Tile:
    def current(self):
        return self

    next = previous = current


class Buffers:
    def __getattr__(self, name):
        return Tile()


class TestNormalPipeline(unittest.TestCase):
    def trace_loop(self, side, indices, block_end, swizzle=0, sink=0, mask=4):
        events = []
        scan_count = 0
        valid = set(indices)

        def next_work(_const, _core, _cores, _start, end, local, cursor, count):
            nonlocal scan_count
            events.append(("fetch", local))
            if swizzle:
                return indices[local] if local < count else -1
            while cursor < end:
                scan_count += 1
                if cursor in valid:
                    return cursor
                cursor += 1
            return -1

        def fetch_work(
            const, _meta, _off, core, cores, start, end, local, cursor, count, _round
        ):
            index = next_work(const, core, cores, start, end, local, cursor, count)
            return index, index + 1 if not swizzle and index >= 0 else cursor

        def set_info(_const, info, task, index, _sub, _local):
            info.task = task
            info.index = index
            info.loadKV = 1
            info.dkvAcc = 0
            info.flushDkv = 1
            info.n2oIdx = 0
            info.goIdx = 0
            events.append(("assign", index))

        def record_compute(name):
            def record(_const, info, *_args):
                events.append((name, info.index))

            return record

        namespace = {
            "PRELOAD_TIMES": 3,
            "mask_mode": mask,
            "swizzle": swizzle,
            "next_work_gidx": next_work,
            "fetch_work_gidx": fetch_work,
            "set_run_info": set_info,
            "apply_l0c_s2": lambda *_args: None,
        }
        for name in (
            "iterate_mm_qk",
            "iterate_mm_dyv",
            "iterate_mm_pdy",
            "iterate_mm_dsk",
            "iterate_mm_dsq",
            "flush_dkv",
            "process_vec1",
            "process_vec2",
            "process_vec3",
            "process_vec4",
            "process_sink_partial",
        ):
            namespace[name] = record_compute(name)
        flags = (
            "SYNC_C1_TO_V2_FLAG",
            "SYNC_C2_TO_V2_FLAG",
            "SYNC_V2_TO_C1_FLAG",
            "SYNC_V2_TO_C2_FLAG",
        )
        for name in flags:
            namespace[name] = [name + "0", name + "1"]
        for name in (
            "SYNC_C5_TO_V4_FLAG",
            "SYNC_C4_TO_V3_FLAG",
            "SYNC_V4_TO_C5_FLAG",
            "SYNC_V3_TO_C3_FLAG",
        ):
            namespace[name] = name
        namespace["pl"] = SimpleNamespace(
            DT_INT64=int,
            const=lambda value, _dtype: value,
            get_block_idx=lambda: 0,
            getval=lambda *_args: 0,
            PipeType=SimpleNamespace(FIX="FIX", V="V", MTE1="MTE1", MTE3="MTE3"),
            system=SimpleNamespace(
                set_cross_core=lambda **kw: events.append(("set", kw["event_id"])),
                wait_cross_core=lambda **kw: events.append(("wait", kw["event_id"])),
            ),
        )
        node = functions("fag_kernel_norm.py")[f"_run_preload3_{side}_normal"]
        loop = execute(node, namespace)
        sched = SimpleNamespace(
            meta=None,
            fagOff=0,
            cBlockIdx=0,
            coreNum=1,
            blockStart=0,
            blockEnd=block_end,
            numBlocks=len(indices),
            subIdx=0,
            v1w=128,
        )
        loop(
            SimpleNamespace(sinkOptional=sink, gSize=1),
            sched,
            Buffers(),
            Buffers(),
            [SimpleNamespace() for _ in range(3)],
        )
        return events, scan_count

    def test_preload3_fetch_once_per_task_and_once_for_end(self):
        # Empty cores, short rings, internal holes and very long invalid tails.
        for side in ("cube", "vector"):
            for swizzle in (0, 1):
                for indices, end in (
                    ([], 0),
                    ([], 4096),
                    ([0], 4096),
                    ([0, 3], 8192),
                    ([1, 4, 9], 16384),
                    (list(range(16)), 16),
                ):
                    with self.subTest(
                        side=side, swizzle=swizzle, indices=indices, end=end
                    ):
                        events, scans = self.trace_loop(side, indices, end, swizzle)
                        self.assertEqual(
                            sum(e[0] == "fetch" for e in events), len(indices) + 1
                        )
                        if not swizzle:
                            self.assertEqual(scans, end)
                        self.assertEqual(
                            [e[1] for e in events if e[0] == "assign"], indices
                        )
                        phases = (
                            (
                                "iterate_mm_qk",
                                "iterate_mm_dyv",
                                "iterate_mm_dsk",
                                "iterate_mm_dsq",
                                "iterate_mm_pdy",
                            )
                            if side == "cube"
                            else (
                                "process_vec1",
                                "process_vec2",
                                "process_vec3",
                                "process_vec4",
                            )
                        )
                        for phase in phases:
                            self.assertEqual(
                                [e[1] for e in events if e[0] == phase], indices
                            )

    def test_v1_starts_before_next_fetch_and_retains_two_ahead(self):
        indices = list(range(8))
        events, _ = self.trace_loop("vector", indices, 1024)
        for task in range(len(indices) - 1):
            self.assertLess(
                events.index(("process_vec1", indices[task])),
                events.index(("fetch", task + 2)),
            )
            self.assertLess(
                events.index(("assign", indices[task + 1])),
                events.index(("process_vec1", indices[task])),
            )

    def test_sink_overlaps_cube_but_finishes_before_ub_release(self):
        for count in (0, 1, 2, 3, 8):
            for mask in (0, 3, 4):
                with self.subTest(count=count, mask=mask):
                    events, _ = self.trace_loop(
                        "vector", list(range(count)), 1024, sink=1, mask=mask
                    )
                    for index in range(count):
                        begin = events.index(("process_vec3", index))
                        finish = (
                            events.index(("process_vec3", index + 1))
                            if index + 1 < count
                            else len(events)
                        )
                        chunk = events[begin:finish]
                        ready = chunk.index(("set", "SYNC_V3_TO_C3_FLAG"))
                        partial = chunk.index(("process_sink_partial", index))
                        self.assertLess(ready, partial)
                        for pos, event in enumerate(chunk):
                            if event[0] == "set" and event[1].startswith(
                                "SYNC_V2_TO_C"
                            ):
                                self.assertLess(partial, pos)
                    self.assertEqual(
                        sum(e == ("set", "SYNC_V3_TO_C3_FLAG") for e in events), count
                    )
                    self.assertEqual(
                        sum(
                            e[0] == "set" and e[1].startswith("SYNC_V2_TO_C1")
                            for e in events
                        ),
                        max(count - 2, 0),
                    )
                    tail = events[-2:] if count else []
                    if count:
                        self.assertEqual(
                            tail,
                            [
                                ("wait", "SYNC_C5_TO_V4_FLAG"),
                                ("wait", "SYNC_C4_TO_V3_FLAG"),
                            ],
                        )

    def test_bn2_sink_read_still_precedes_cube_ready(self):
        node = functions("fag_kernel_norm.py")["_run_preload2_vector_bn2"]
        calls = sorted(
            (item for item in ast.walk(node) if isinstance(item, ast.Call)),
            key=lambda item: item.lineno,
        )
        partial = next(
            call.lineno
            for call in calls
            if dotted_name(call.func) == "process_sink_partial"
        )
        v3 = next(
            call.lineno
            for call in calls
            if dotted_name(call.func) == "process_vec3_bn2"
        )
        self.assertLess(partial, v3)
        for call in calls:
            if dotted_name(call.func) == "pl.system.set_cross_core":
                event = next(kw.value for kw in call.keywords if kw.arg == "event_id")
                if dotted_name(event) in ("SYNC_V3_TO_C3_FLAG", "SYNC_V4_TO_C5_FLAG"):
                    self.assertLess(partial, call.lineno)

    def test_bn2_barriers_only_for_preclear_or_sink(self):
        node = functions("fag_kernel.py")["flash_attn_grad_preload2"]
        bn2_test = ast.dump(ast.parse("template == Tmpl.BN2", mode="eval").body)
        sections = [
            item
            for item in node.body
            if isinstance(item, ast.If)
            and ast.dump(item.test) == bn2_test
            and any(
                isinstance(call, ast.Call)
                and dotted_name(call.func) == "pl.system.sync_all"
                for call in ast.walk(item)
            )
        ]
        self.assertEqual(len(sections), 2)
        # Evaluate the real pre/post branches; all data/compute operations are stubs.
        for sink in (0, 1):
            for clear in (0, 1):
                for aiv in (0, 1):
                    with self.subTest(sink=sink, clear=clear, aiv=aiv):
                        trace = []
                        namespace = dict(
                            template=1,
                            Tmpl=SimpleNamespace(BN2=1),
                            bn2_need_zero=clear,
                            HEAD_DIM_ALIGN=128,
                            meta=None,
                            fagOff=0,
                            dq=None,
                            dk=None,
                            dv=None,
                            tensor_dsink_ws=None,
                            tensor_dsink_out=None,
                            constInfo=SimpleNamespace(
                                sinkOptional=sink,
                                n2Size=2,
                                gSize=1,
                                bSize=1,
                                s1Size=128,
                                s2Size=128,
                                dSize=64,
                                dvSize=64,
                                aivCount=2,
                            ),
                            pl=SimpleNamespace(
                                DT_FP16="fp16",
                                make_tensor=lambda *_a, **_k: None,
                                get_block_idx=lambda: aiv,
                                section_vector=nullcontext,
                                system=SimpleNamespace(
                                    sync_all=lambda: trace.append("barrier")
                                ),
                            ),
                        )
                        for name in (
                            "make_pre_zero_buffer_f16",
                            "make_pre_zero_buffer_f32",
                            "make_sink_red_post_ub",
                            "make_sink_row_post_ub",
                        ):
                            namespace[name] = lambda *_args: Tile()
                        for name in (
                            "process_pre_from_meta",
                            "process_sink_zero",
                            "process_sink_reduce",
                        ):
                            namespace[name] = lambda *_args, label=name: trace.append(
                                label
                            )
                        module = ast.Module(body=sections, type_ignores=[])
                        exec(compile(module, "<bn2-pre-post>", "exec"), namespace)
                        self.assertEqual(
                            trace.count("barrier"), int(clear or sink) + sink
                        )
                        self.assertEqual(trace.count("process_pre_from_meta"), clear)
                        self.assertEqual(trace.count("process_sink_zero"), sink)
                        self.assertEqual(
                            trace.count("process_sink_reduce"), int(sink and aiv == 0)
                        )
                        if sink:
                            self.assertLess(
                                trace.index("process_sink_zero"), trace.index("barrier")
                            )
                            if aiv == 0:
                                self.assertEqual(
                                    trace[-2:], ["barrier", "process_sink_reduce"]
                                )


if __name__ == "__main__":
    unittest.main()
