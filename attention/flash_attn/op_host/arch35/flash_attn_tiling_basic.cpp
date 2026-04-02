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
 * \file flash_attn_tiling_basic.cpp
 * \brief FlashAttn arch35
 */

#include "flash_attn_tiling_regbase.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FA {

class FlashAttnTilingBasic : public FlashAttnTilingRegbase {
public:
    explicit FlashAttnTilingBasic(gert::TilingContext *context)
        : FlashAttnTilingRegbase(context)
    {
        this->regbase = true;
    }
    ~FlashAttnTilingBasic() override = default;

protected:
    bool IsCapable() override
    {
        if (npuArch != NpuArch::DAV_3510) {
            OP_LOGD(opName, "FlashAttnTilingBasic: current npu arch is not dav-3510, skip.");
            return false;
        }
        return true;
    }

    void CalcDBasicBlock() override
    {
    }

    void CalcS1S2BasicBlock() override
    {
        // TODO: 基本块计算
        // 可以参照flash_attn_score_tiling_basic.cpp中的CalcS1S2BasicBlock实现
    }

    int64_t CalcTotalSize() override
    {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        return totalSize;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        // TODO: workspace计算
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // 可以参照flash_attn_score_tiling_basic.cpp中的GetWorkspaceSize实现
        workspaces[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {

         return GET_TPL_TILING_KEY(0, 0, 0, 128, 128, 64, 0,
                                  0, 0, 0, 1);
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttnTilingRegbase::PostTiling();
        return ge::GRAPH_SUCCESS;
    }

private:
    uint16_t dVType = static_cast<uint16_t>(FADTemplateType::NONALIGNED);
    bool regbase = true;
};

// 注册到arch35 (DAV_3510)，优先级83（与flash_attn_score保持一致）
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttn, FlashAttnTilingBasic,
                                    static_cast<int32_t>(NpuArch::DAV_3510), 83);

} // namespace FA
} // namespace optiling