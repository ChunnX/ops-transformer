/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_combine_x_tiling_arch35.h
 * \brief MoeTutelCombineX tiling implementation for arch35 (Ascend 950).
 */
#ifndef MOE_TUTEL_COMBINE_X_TILING_ARCH35_H
#define MOE_TUTEL_COMBINE_X_TILING_ARCH35_H

#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "moe_tutel_combine_x_tiling.h"
#include "../op_kernel/arch35/moe_tutel_combine_x_tiling_struct.h"

namespace optiling {

class MoeTutelCombineXTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeTutelCombineXTilingArch35(gert::TilingContext *context);
    ~MoeTutelCombineXTilingArch35() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus DeriveCapacity();
    ge::graphStatus HandleEmptyTensor();
    ge::graphStatus FillAndSet(bool hasBatch);

    MoetutelcombinexTilingData td_ = {};
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;

    int64_t rankYGrad_ = 0;
    int64_t batchSize_ = 1;
    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t eC_ = 0;
    int64_t capacity_ = 0;
    int64_t E_ = 0;
    int64_t dtypeSize_ = 0;

    int64_t tokenTotal_ = 0;
    int64_t coreTokenBase_ = 0;
    int32_t bigCoreCnt_ = 0;
    int32_t usedCoreNum_ = 0;

    int64_t hChunk_ = 0;
    int64_t hChunkCnt_ = 0;
    int64_t hChunkTail_ = 0;
};

} // namespace optiling
#endif // MOE_TUTEL_COMBINE_X_TILING_ARCH35_H
