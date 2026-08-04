#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference checks for the FIA MXFP4 mode-6 PA_NZ parameter contract.

This is intentionally a host-only address/formula oracle.  It does not replace the
device numerical test: its job is to make the producer transform and the kernel's
page-segment source coordinates independently inspectable.
"""

import unittest


def make_logical_params(batch, heads, seq, groups, base):
    return [
        [
            [[base + 10000 * b + 1000 * n + 100 * s + 10 * g for g in range(groups)] for s in range(seq)]
            for n in range(heads)
        ]
        for b in range(batch)
    ]


def to_pa_nz(logical, block_table, page_count, block_size):
    batch = len(logical)
    heads = len(logical[0])
    sequence = len(logical[0][0])
    groups = len(logical[0][0][0])
    final = [[[[None for _ in range(block_size)] for _ in range(groups)] for _ in range(heads)]
             for _ in range(page_count)]
    for b in range(batch):
        for s in range(sequence):
            page = block_table[b][s // block_size]
            token_in_page = s % block_size
            for n in range(heads):
                for g in range(groups):
                    final[page][n][g][token_in_page] = logical[b][n][s][g]
    return final


def load_pa_nz_tile(final, block_table, batch_index, head_index, start, count, block_size):
    groups = len(final[0][head_index])
    aligned_count = (count + 31) // 32 * 32
    tile = [[None for _ in range(aligned_count)] for _ in range(groups)]
    copied = 0
    current = start
    while copied < count:
        logical_block = current // block_size
        token_in_page = current % block_size
        page = block_table[batch_index][logical_block]
        length = min(block_size - token_in_page, count - copied)
        for g in range(groups):
            tile[g][copied:copied + length] = final[page][head_index][g][token_in_page:token_in_page + length]
        copied += length
        current += length
    return tile


class Mxfp4PaNzReferenceTest(unittest.TestCase):
    def setUp(self):
        self.batch = 2
        self.heads = 2
        self.sequence = 16
        self.groups = 3
        self.pages = 4
        self.block_size = 8
        # Every logical page maps to a distinct, non-monotonic physical page.
        self.block_table = [[2, 0], [3, 1]]

    def test_three_stage_transform_preserves_every_coordinate(self):
        logical = make_logical_params(self.batch, self.heads, self.sequence, self.groups, base=0)
        final = to_pa_nz(logical, self.block_table, self.pages, self.block_size)
        for b in range(self.batch):
            for n in range(self.heads):
                for s in range(self.sequence):
                    page = self.block_table[b][s // self.block_size]
                    token_in_page = s % self.block_size
                    for g in range(self.groups):
                        self.assertEqual(final[page][n][g][token_in_page], logical[b][n][s][g])

    def test_cross_page_group_plane_load_uses_physical_page_coordinates(self):
        logical = make_logical_params(self.batch, self.heads, self.sequence, self.groups, base=0)
        final = to_pa_nz(logical, self.block_table, self.pages, self.block_size)
        # Start in physical page 3 at u=5, then cross into physical page 1.
        tile = load_pa_nz_tile(final, self.block_table, batch_index=1, head_index=1,
                               start=5, count=9, block_size=self.block_size)
        for g in range(self.groups):
            expected = [logical[1][1][s][g] for s in range(5, 14)]
            self.assertEqual(tile[g][:9], expected)
        self.assertEqual(tile[0][:3], [11500, 11600, 11700])
        self.assertEqual(tile[0][3:9], [11800, 11900, 12000, 12100, 12200, 12300])

    def test_scale_and_offset_share_coordinates_but_not_values(self):
        scale = make_logical_params(self.batch, self.heads, self.sequence, self.groups, base=100000)
        offset = make_logical_params(self.batch, self.heads, self.sequence, self.groups, base=-50000)
        scale_final = to_pa_nz(scale, self.block_table, self.pages, self.block_size)
        offset_final = to_pa_nz(offset, self.block_table, self.pages, self.block_size)
        scale_tile = load_pa_nz_tile(scale_final, self.block_table, 1, 0, 5, 9, self.block_size)
        offset_tile = load_pa_nz_tile(offset_final, self.block_table, 1, 0, 5, 9, self.block_size)
        for g in range(self.groups):
            for t in range(9):
                self.assertEqual(scale_tile[g][t] - offset_tile[g][t], 150000)

    def test_symmetric_and_asymmetric_formulae_are_distinct(self):
        decoded = [-1.0, 0.5, 2.0, -0.5]
        scales = [0.5, 2.0]
        offsets = [-0.25, 1.5]
        symmetric = [decoded[d] * scales[d // 2] for d in range(4)]
        asymmetric = [(decoded[d] + offsets[d // 2]) * scales[d // 2] for d in range(4)]
        self.assertEqual(symmetric, [-0.5, 0.25, 4.0, -1.0])
        self.assertEqual(asymmetric, [-0.625, 0.125, 7.0, 2.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
