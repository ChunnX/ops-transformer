/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file metadata_checker.cpp
 * \brief Checker for metadata parameter (文档: INT32, shape=(max_schedule_size,))
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_attn_tiling_constants.h"
#include "metadata_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus MetadataChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    auto &metadataTensor = faInfo.opParamInfo.metadata.tensor;
    if (metadataTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *metadataDesc = faInfo.opParamInfo.metadata.desc;
    OP_CHECK_IF(metadataDesc == nullptr,
        OP_LOGE(faInfo.opName, "metadata desc is null pointer!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(faInfo.opName, "metadata dtype must be INT32, but got %s",
                DataTypeToSerialString(metadataDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    uint32_t dimNum = metadataTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 1,
        OP_LOGE(faInfo.opName, "metadata dim num must be 1, but got %u", dimNum),
        return ge::GRAPH_FAILED);

    int64_t dim0 = metadataTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 <= 0,
        OP_LOGE(faInfo.opName, "metadata shape dim0(%ld) must be greater than 0", dim0),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling