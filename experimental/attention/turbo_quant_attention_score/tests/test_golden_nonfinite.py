# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The task-golden wrapper must not silently accept NaN differences."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_turbo_quant_attention_score import check_task_golden


class NonfiniteGoldenTests(unittest.TestCase):
    def test_rejects_nonfinite_at_finite_reference(self):
        expected = np.zeros((1, 1, 4), dtype=np.float32)
        for value in [float("nan"), float("inf"), float("-inf")]:
            with (
                self.subTest(value=value),
                patch(
                    "test_turbo_quant_attention_score.calc_expect_func",
                    return_value=[expected],
                ),
            ):
                actual = torch.zeros((1, 1, 4))
                actual[0, 0, 2] = value
                with self.assertRaises(AssertionError):
                    check_task_golden(actual, [], 3)

    def test_finite_and_masked_reference(self):
        expected = np.array([[[1.0, -np.inf, 0.0]]], dtype=np.float32)
        with patch(
            "test_turbo_quant_attention_score.calc_expect_func", return_value=[expected]
        ):
            result = check_task_golden(torch.from_numpy(expected.copy()), [], 3)
        self.assertEqual(result["task_golden_mixed_exceedances"], 0)


if __name__ == "__main__":
    unittest.main()
