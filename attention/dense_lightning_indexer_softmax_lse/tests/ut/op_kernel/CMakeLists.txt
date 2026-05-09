# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(dense_lightning_indexer_softmax_lse_tiling_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/dense_lightning_indexer_softmax_lse_tiling.cpp
)

if (UT_TEST_ALL OR OP_KERNEL_UT)
    AddOpTestCase(dense_lightning_indexer_softmax_lse "Ascend910B" "-DDTYPE_QUERY_INDEX=half -DDTYPE_KEY_INDEX=half -DDTYPE_WEIGHT=half -DDTYPE_SOFTMAX_MAX=float" "${dense_lightning_indexer_softmax_lse_tiling_files}")
endif()
