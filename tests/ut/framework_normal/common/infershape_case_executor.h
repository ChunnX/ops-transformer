/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H

#include "infer_shape_context_faker.h"

void ExecuteTestCase(gert::InfershapeContextPara&             infershapeContextPara, 
                     ge::graphStatus                          expectResult = ge::GRAPH_FAILED,
                     const std::vector<std::vector<int64_t>>& expectOutputShape = {});

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFERSHAPE_CASE_EXECUTOR_H