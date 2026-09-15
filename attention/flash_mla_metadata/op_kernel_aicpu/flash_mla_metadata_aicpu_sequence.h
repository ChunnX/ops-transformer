/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_METADATA_AICPU_SEQUENCE_H
#define FLASH_MLA_METADATA_AICPU_SEQUENCE_H
#include "cpu_tensor.h"
#include "../../flash_mla/op_host/flash_mla_value_tiling.h"

namespace aicpu {
namespace flash_mla_metadata {
using FlashMla::MlaSeqView;

inline bool PrepareSeq(const Tensor *tensor, MlaSeqView &view, uint32_t &present, uint64_t &count)
{
    if (tensor == nullptr) {
        present = 0;
        view = {};
        count = 0;
        return true;
    }
    if (tensor->GetTensorShape() == nullptr) {
        return false;
    }
    if (tensor->GetTensorShape()->GetDims() == 0 && tensor->GetData() == nullptr) {
        present = 0;
        view = {};
        count = 0;
        return true;
    }
    if (tensor->GetTensorShape()->GetDims() != 1) {
        return false;
    }
    // ACLNN optional inputs and an explicit empty sequence share the absent representation.
    if (tensor->GetTensorShape()->GetDimSize(0) == 0) {
        present = 0;
        view = {};
        count = 0;
        return true;
    }
    if (tensor->GetDataType() != DT_INT64 || tensor->GetTensorShape()->GetDims() != 1 ||
        tensor->GetTensorShape()->GetDimSize(0) < 0 || tensor->GetData() == nullptr) {
        return false;
    }
    count = static_cast<uint64_t>(tensor->GetTensorShape()->GetDimSize(0));
    present = count > 0;
    view = {static_cast<const int64_t *>(tensor->GetData()), count};
    return true;
}
} // namespace flash_mla_metadata
} // namespace aicpu
#endif
