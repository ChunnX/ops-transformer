/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_no_quant_950_checker.h"
#include "log/log.h"
#include "gmm/common/op_host/op_tiling/format_string_util.h"

using namespace gmm;

constexpr int64_t MAX_LENGTH = 1024UL;

aclnnStatus AclnnGroupedMatmulNoQuantDAV3510Checker::CheckGroupedMatmulGroupSizeNoQuantDAV3510() {
  if (CheckTensorListLength(gmmParams_.x) != ACLNN_SUCCESS) {
    OP_LOGE_FOR_INVALID_LISTSIZE("grouped_matmul", "x",
        std::to_string(gmmParams_.x != nullptr ? gmmParams_.x->Size() : 0).c_str(),
        std::to_string(MAX_LENGTH).c_str());
    return ACLNN_ERR_PARAM_INVALID;
  }
  if (CheckTensorListLength(gmmParams_.weight) != ACLNN_SUCCESS) {
    OP_LOGE_FOR_INVALID_LISTSIZE("grouped_matmul", "weight",
        std::to_string(gmmParams_.weight != nullptr ? gmmParams_.weight->Size() : 0).c_str(),
        std::to_string(MAX_LENGTH).c_str());
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulNoQuantDAV3510Checker::CheckTensorListLength(const aclTensorList *tensorList) const {
  size_t groupSize = 0;
  if (tensorList != nullptr) {
      groupSize = tensorList->Size();
  }
  if (groupSize > MAX_LENGTH) {
    OP_LOGE_FOR_INVALID_LISTSIZE("grouped_matmul", "tensorList",
        std::to_string(groupSize).c_str(),
        std::to_string(MAX_LENGTH).c_str());
    return ACLNN_ERR_PARAM_INVALID;
  }
  return ACLNN_SUCCESS;
}