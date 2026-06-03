# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools

import pytest

import tri_inverse_operator_single
from test_tri_inverse_paramset import ENABLED_PARAMS


PARAM_COMBINATIONS = []
for params in ENABLED_PARAMS:
    for combo in itertools.product(params["shape"], params["seed"]):
        PARAM_COMBINATIONS.append(combo)


@pytest.mark.ci
@pytest.mark.parametrize("shape, seed", PARAM_COMBINATIONS)
def test_tri_inverse(shape, seed):
    tri_inverse_operator_single.run_precision_test(shape, seed)
