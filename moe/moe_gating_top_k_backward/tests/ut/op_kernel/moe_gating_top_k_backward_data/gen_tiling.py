#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import sys


params_info = {
    "test_moe_gating_top_k_backward_case_00": [8, 1, 1, 64, 1, 1, 1, 1, 8, 192, 10, 2, 0, 1, 2178868143328329728],
    "test_moe_gating_top_k_backward_case_01": [16, 1, 1, 47, 1, 1, 1, 1, 16, 256, 8, 4, 0, 1, 2178868143328329728],
    "test_moe_gating_top_k_backward_case_02": [16, 1, 1, 35, 1, 1, 1, 1, 16, 384, 8, 2, 0, 1, 2178868143328329728]
}


def main():
    params_list = params_info[sys.argv[1]]

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
