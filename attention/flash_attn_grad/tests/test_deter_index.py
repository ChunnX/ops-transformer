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
"""Host-side generalized checks for non-swizzle deterministic partitioning."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "op_kernel" / "fag_deter_ondevice.py"
SPEC = spec_from_file_location("fag_deter_ondevice_under_test", MODULE_PATH)
DETER = module_from_spec(SPEC)
SPEC.loader.exec_module(DETER)


class ConstInfo:
    pass


INT32_MAX = 2147483647


def _causal_tokens(s1, s2):
    return INT32_MAX + s1 - s2, s2 - s1


CASES = (
    # mode, B, N2, G, M, N, cores, s1Token, s2Token
    (0, 1, 1, 1, 4, 5, 4, 0, 0),
    (0, 2, 3, 1, 5, 4, 8, 0, 0),
    (0, 2, 2, 3, 4, 6, 8, 0, 0),
    (0, 1, 3, 4, 2, 2, 24, 0, 0),
    (3, 1, 1, 1, 4, 4, 4, 0, 0),
    (3, 2, 2, 1, 5, 4, 8, 0, 0),
    (3, 1, 1, 2, 6, 4, 8, 0, 0),
    # 矩形 right-down causal：S1<S2，token 走 pad 校正后切 band。
    (3, 1, 4, 1, 1, 2, 4) + _causal_tokens(128, 256),
    (3, 1, 2, 1, 2, 4, 8) + _causal_tokens(256, 512),
    (3, 1, 2, 2, 1, 2, 8) + _causal_tokens(128, 256),
    (1, 1, 1, 1, 4, 6, 4, 128, 128),
    (1, 2, 1, 1, 5, 4, 8, 256, 64),
    (1, 1, 2, 2, 6, 5, 8, 64, 256),
    (1, 2, 2, 3, 4, 6, 8, 384, 128),
    # pad 后负 token：C++ 向零 ceil 得到 p=1,q=3，不能用 floor //。
    (4, 1, 4, 1, 2, 3, 8, -128, 129),
)


def make_const(case):
    mode, batch, n2, group, m, n, cores, s1_token, s2_token = case
    const = ConstInfo()
    const.bSize = batch
    const.n2Size = n2
    const.gSize = group
    const.s1Outer = m
    const.s2Outer = n
    const.s1Size = m * 128
    const.s2Size = n * 128
    const.s1Token = s1_token
    const.s2Token = s2_token
    return mode, cores, const


class TestDeterIndex(unittest.TestCase):
    def test_device_dispatch_matches_ascendc_formula(self):
        for case in CASES:
            mode, cores, const = make_const(case)
            DETER.mask_mode = mode
            max_round = DETER.deter_lin_max_round(const, cores)
            for round_id in range(max_round):
                for core_id in range(cores):
                    with self.subTest(case=case, round_id=round_id, core_id=core_id):
                        got = DETER._deter_lin_gidx_device(
                            const, core_id, cores, round_id
                        )
                        ref = DETER._deter_lin_gidx_host(
                            const, core_id, cores, round_id
                        )
                        self.assertEqual(got, ref)

    def test_each_valid_tile_is_emitted_exactly_once(self):
        for case in CASES:
            mode, cores, const = make_const(case)
            DETER.mask_mode = mode
            max_round = DETER.deter_lin_max_round(const, cores)
            emitted = []
            for round_id in range(max_round):
                for core_id in range(cores):
                    index = DETER._deter_lin_gidx_device(
                        const, core_id, cores, round_id
                    )
                    if index >= 0:
                        emitted.append(index)

            expected = []
            m = const.s1Outer
            n = const.s2Outer
            tile_count = const.bSize * const.n2Size * const.gSize * m * n
            for index in range(tile_count):
                s1_idx = index % m
                s2_idx = (index % (m * n)) // m
                if DETER._fused_valid(const, s1_idx, s2_idx):
                    expected.append(index)

            with self.subTest(case=case):
                self.assertEqual(sorted(emitted), expected)
                self.assertEqual(len(emitted), len(set(emitted)))

    def test_gqa_dense_selected_round_matches_ascendc_example(self):
        # baseRound=5; AscendC locality selection aligns it to G=3 -> 6.
        self.assertEqual(DETER.select_gqa_dense_round(24, 2, 1, 36, 3), 6)

    def test_ceil_floor_unchanged_and_tz_matches_cpp(self):
        self.assertEqual(DETER._ceil(-128, 128), -1)
        self.assertEqual(DETER._ceil(129, 128), 2)
        self.assertEqual(DETER._ceil_tz(-128, 128), 0)
        self.assertEqual(DETER._ceil_tz(-1, 128), 0)
        self.assertEqual(DETER._ceil_tz(0, 128), 0)
        self.assertEqual(DETER._ceil_tz(129, 128), 2)
        self.assertEqual(DETER._ceil_tz(128, 128), 1)

    def test_band_pq_negative_token_matches_ascendc(self):
        # 192×320, w0×1 → token=(-128,129)；ASC p=1,q=3，PTO floor 会得到 p=0。
        actual_m, actual_n, p, q, moff, noff = DETER.band_pq(2, 3, -128, 129)
        self.assertEqual((actual_m, actual_n, p, q, moff, noff), (2, 3, 1, 3, 0, 0))
        # 320×192, w1×0 → token=(129,-128)；floor 会得到 q=0。
        actual_m, actual_n, p, q, moff, noff = DETER.band_pq(3, 2, 129, -128)
        self.assertEqual((actual_m, actual_n, p, q, moff, noff), (3, 2, 3, 1, 0, 0))

    def test_rect_causal_128x256_emits_full_band(self):
        # B1N4 128×256：截成 min(s2o,s1o) 只发 4，band 路径应发 8。
        s1_token, s2_token = _causal_tokens(128, 256)
        const = ConstInfo()
        const.bSize = 1
        const.n2Size = 4
        const.gSize = 1
        const.s1Outer = 1
        const.s2Outer = 2
        const.s1Size = 128
        const.s2Size = 256
        const.s1Token = s1_token
        const.s2Token = s2_token
        DETER.mask_mode = 3
        emitted = []
        max_round = DETER.deter_lin_max_round(const, 4)
        for round_id in range(max_round):
            for core_id in range(4):
                index = DETER._deter_lin_gidx_device(const, core_id, 4, round_id)
                if index >= 0:
                    emitted.append(index)
        self.assertEqual(len(emitted), 8)
        self.assertEqual(sorted(emitted), list(range(8)))

    def test_band_192x320_w0x1_covers_all_valid_tiles(self):
        const = ConstInfo()
        const.bSize = 1
        const.n2Size = 4
        const.gSize = 1
        const.s1Outer = 2
        const.s2Outer = 3
        const.s1Size = 192
        const.s2Size = 320
        const.s1Token = -128
        const.s2Token = 129
        DETER.mask_mode = 4
        max_round = DETER.deter_lin_max_round(const, 8)
        emitted = []
        for round_id in range(max_round):
            for core_id in range(8):
                index = DETER._deter_lin_gidx_device(const, core_id, 8, round_id)
                if index >= 0:
                    emitted.append(index)
        expected = []
        for index in range(24):
            s1_idx = index % 2
            s2_idx = (index % 6) // 2
            if DETER._fused_valid(const, s1_idx, s2_idx):
                expected.append(index)
        self.assertEqual(sorted(emitted), expected)
        self.assertEqual(len(emitted), len(set(emitted)))
        self.assertGreater(len(expected), 0)

    def test_band_zero_pq_is_not_rejected(self):
        # p==0 是 ASC 合法边界；maxRound 不能因此退回 m*n 或把 gidx 全打成 -1。
        const = ConstInfo()
        const.bSize = 1
        const.n2Size = 1
        const.gSize = 1
        const.s1Outer = 4
        const.s2Outer = 4
        const.s1Size = 512
        const.s2Size = 512
        const.s1Token = -255
        const.s2Token = 128
        DETER.mask_mode = 4
        actual_m, actual_n, p, q, _mo, _no = DETER.band_pq(4, 4, -255, 128)
        self.assertEqual(p, 0)
        self.assertGreater(q, 0)
        max_round = DETER.deter_lin_max_round(const, 4)
        self.assertNotEqual(max_round, 16)
        self.assertGreater(max_round, 0)
        hits = 0
        for round_id in range(max_round):
            for core_id in range(4):
                if DETER._deter_lin_gidx_host(const, core_id, 4, round_id) >= 0:
                    hits += 1
        self.assertGreater(hits, 0)


if __name__ == "__main__":
    unittest.main()
