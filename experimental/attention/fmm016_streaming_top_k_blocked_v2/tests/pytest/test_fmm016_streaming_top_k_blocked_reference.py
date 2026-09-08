#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 fk2323
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dependency-free CPU golden tests for the FMM016 router contract.

The NPU functional tests use the generated ACLNN wrapper in the deployment
environment. This file keeps the reference and tie/near-window contract
runnable on a clean checkout without torch, torch_npu, or a device.
"""

import math
import unittest


def reference_topk(q_index, k_index, index_weight, top_k, radius=2):
    """Compute the stable router result for [B,H,L,G,I] inputs."""
    batch = len(q_index)
    heads = len(q_index[0])
    leaves = len(q_index[0][0])
    groups = len(q_index[0][0][0])
    index_dim = len(q_index[0][0][0][0])
    padded_leaves = len(k_index[0][0][0])
    inv_sqrt = 1.0 / math.sqrt(index_dim)
    selected_index = []
    selected_valid = []
    for b in range(batch):
        batch_index = []
        batch_valid = []
        for h in range(heads):
            head_index = []
            head_valid = []
            for target in range(leaves):
                scores = []
                for source in range(padded_leaves):
                    if source >= leaves or abs(source - target) <= radius:
                        scores.append(float("-inf"))
                        continue
                    score = 0.0
                    for group in range(groups):
                        dot = 0.0
                        for dim in range(index_dim):
                            dot += q_index[b][h][target][group][dim] * k_index[b][h][dim][source]
                        score += max(dot * inv_sqrt, 0.0) * index_weight[b][h][target][group]
                    scores.append(score)
                order = sorted(range(padded_leaves), key=lambda source: (-scores[source], source))
                ids = order[:top_k]
                head_index.append(ids)
                head_valid.append([1 if source < leaves and scores[source] != float("-inf") else 0 for source in ids])
            batch_index.append(head_index)
            batch_valid.append(head_valid)
        selected_index.append(batch_index)
        selected_valid.append(batch_valid)
    return selected_index, selected_valid


def zero_case(batch, heads, leaves, groups, index_dim):
    q = [[[[[0.0 for _ in range(index_dim)] for _ in range(groups)]
           for _ in range(leaves)] for _ in range(heads)] for _ in range(batch)]
    k = [[[[0.0 for _ in range(leaves)] for _ in range(index_dim)]
          for _ in range(heads)] for _ in range(batch)]
    w = [[[[1.0 / groups for _ in range(groups)] for _ in range(leaves)]
          for _ in range(heads)] for _ in range(batch)]
    return q, k, w


class Fmm016ReferenceTest(unittest.TestCase):
    def test_near_window_and_stable_ties(self):
        q, k, w = zero_case(1, 1, 9, 2, 8)
        indices, valid = reference_topk(q, k, w, 4)
        self.assertEqual(indices[0][0][0], [3, 4, 5, 6])
        self.assertEqual(indices[0][0][4], [0, 1, 7, 8])
        self.assertEqual(indices[0][0][8], [0, 1, 2, 3])
        self.assertEqual(valid[0][0][0], [1, 1, 1, 1])

    def test_cross_tile_tie_order(self):
        q, k, w = zero_case(1, 1, 2050, 2, 8)
        indices, valid = reference_topk(q, k, w, 16)
        self.assertEqual(indices[0][0][0], list(range(3, 19)))
        self.assertEqual(indices[0][0][1024], list(range(16)))
        self.assertEqual(valid[0][0][0], [1] * 16)
        self.assertEqual(valid[0][0][1024], [1] * 16)


if __name__ == "__main__":
    unittest.main()
