/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file empty_tensor_checker.cpp
 * \brief Empty tensor scenario checker for npu_flash_attn (特殊场景处理)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "empty_tensor_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

bool EmptyTensorChecker::IsEmptyTensorScenario(const FaTilingInfo &faInfo)
{
    const gert::StorageShape *queryShape = faInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = faInfo.opParamInfo.attnOut.shape;
    const gert::StorageShape *keyShape = faInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = faInfo.opParamInfo.value.shape;
    const gert::StorageShape *lseOutShape = faInfo.opParamInfo.lseOut.shape;

    uint64_t queryShapeSize = 0;
    uint64_t attentionOutShapeSize = 0;
    uint64_t keyShapeSize = 0;
    uint64_t valueShapeSize = 0;

    if (queryShape != nullptr) {
        queryShapeSize = queryShape->GetStorageShape().GetShapeSize();
    }
    if (attentionOutShape != nullptr) {
        attentionOutShapeSize = attentionOutShape->GetStorageShape().GetShapeSize();
    }
    if (keyShape != nullptr) {
        keyShapeSize = keyShape->GetStorageShape().GetShapeSize();
    }
    if (valueShape != nullptr) {
        valueShapeSize = valueShape->GetStorageShape().GetShapeSize();
    }

    bool lseOutNotNull = (lseOutShape != nullptr && lseOutShape->GetStorageShape().GetShapeSize() > 0);

    if (queryShapeSize == 0 && attentionOutShapeSize == 0) {
        return true;
    }

    if (queryShapeSize > 0 && attentionOutShapeSize > 0 && keyShapeSize == 0 && valueShapeSize == 0 && lseOutNotNull) {
        return true;
    }

    if (attentionOutShapeSize == 0 && !lseOutNotNull) {
        return true;
    }

    return false;
}

} // namespace optiling