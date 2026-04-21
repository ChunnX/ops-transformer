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
 * \file softmax_lse_checker.cpp
 * \brief Checker for softmax_lse parameter (文档约束: SoftmaxLSE参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "softmax_lse_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *lseOutDesc = faInfo.opParamInfo.lseOut.desc;
    if (lseOutDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lseOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(faInfo.opName, "lseOut dtype must be FP32, but got %s",
                        DataTypeToSerialString(lseOutDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    if (faInfo.softmaxLseFlag) {
        const gert::StorageShape *lseOutShape = faInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseOutShape == nullptr,
                    OP_LOGE(faInfo.opName, "lseOut must not be null when softmaxLSE is enabled."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::StorageShape *lseOutShape = faInfo.opParamInfo.lseOut.shape;
    if (lseOutShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t dimNum = lseOutShape->GetStorageShape().GetDimNum();

    if (faInfo.outLayout == FaLayout::TND) {
        OP_CHECK_IF(dimNum != 2,
                    OP_LOGE(faInfo.opName, "lseOut dim num must be 2 when output layout is TND, but got %u", dimNum),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(lseOutShape->GetStorageShape().GetDim(0) != faInfo.n1Size,
                    OP_LOGE(faInfo.opName, "lseOut dim0(%ld) must be equal to N_Q(%ld) when output layout is TND",
                            lseOutShape->GetStorageShape().GetDim(0), faInfo.n1Size),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(lseOutShape->GetStorageShape().GetDim(1) != faInfo.qTSize,
                    OP_LOGE(faInfo.opName, "lseOut dim1(%ld) must be equal to T_Q(%u) when output layout is TND",
                            lseOutShape->GetStorageShape().GetDim(1), faInfo.qTSize),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            dimNum != 3,
            OP_LOGE(faInfo.opName, "lseOut dim num must be 3 when output layout is not TND, but got %u", dimNum),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(lseOutShape->GetStorageShape().GetDim(0) != faInfo.bSize,
                    OP_LOGE(faInfo.opName, "lseOut dim0(%ld) must be equal to batch(%u) when output layout is not TND",
                            lseOutShape->GetStorageShape().GetDim(0), faInfo.bSize),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(lseOutShape->GetStorageShape().GetDim(1) != faInfo.n1Size,
                    OP_LOGE(faInfo.opName, "lseOut dim1(%ld) must be equal to N_Q(%ld) when output layout is not TND",
                            lseOutShape->GetStorageShape().GetDim(1), faInfo.n1Size),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(lseOutShape->GetStorageShape().GetDim(2) != faInfo.s1Size,
                    OP_LOGE(faInfo.opName, "lseOut dim2(%ld) must be equal to Q_S(%u) when output layout is not TND",
                            lseOutShape->GetStorageShape().GetDim(2), faInfo.s1Size),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling