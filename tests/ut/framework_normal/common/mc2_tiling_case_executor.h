/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "mc2_hcom_topology_mocker.h"

inline void Mc2ExecuteTestCase(const gert::TilingContextPara& tilingContextPara,
                               const Mc2Hcom::MockValues&     hcomTopologyMockValues,
                               ge::graphStatus                expectResult = ge::GRAPH_FAILED,
                               uint64_t                       expectTilingKey = 0, 
                               const std::string&             expectTilingData = "",
                               const std::vector<size_t>&     expectWorkspaces = {},
                               uint64_t                       tilingDataReservedLen = 0)
{
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, expectTilingData, expectWorkspaces,
        tilingDataReservedLen);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_MC2_TILING_CASE_EXECUTOR_H