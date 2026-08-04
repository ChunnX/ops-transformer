/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_APACE_TILING_BASE_H
#define ALLTO_ALL_MATMUL_APACE_TILING_BASE_H

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <ascendc/host_api/tiling/template_argument.h>
#include "securec.h"
#include "apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_mx_tiling_swat.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/allto_all_matmul_apace_tiling_key.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint64_t WS_BYTES = 4096;

class AlltoAllMatmulApaceTilingClass : public TilingBaseClass {
public:
    explicit AlltoAllMatmulApaceTilingClass(gert::TilingContext *context) : TilingBaseClass(context) {}
    ~AlltoAllMatmulApaceTilingClass() override = default;

protected:
    uint32_t usedCoreNum_ = 0; // 保存 tiling 实际使用的核数
    bool IsCapable() override { return true; }

    ge::graphStatus GetPlatformInfo() override
    {
        fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            return ge::GRAPH_FAILED;
        }
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        return ge::GRAPH_SUCCESS;
    }

    static constexpr size_t IDX_INPUT_X1 = 1;
    static constexpr size_t IDX_INPUT_X2 = 2;
    static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
    static constexpr size_t IDX_ATTR_PRECISION_MODE = 14;

    ge::graphStatus GetShapeAttrsInfo() override
    {
        auto *x1sh = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2sh = context_->GetInputShape(IDX_INPUT_X2);
        if (!x1sh || !x2sh)
            return ge::GRAPH_FAILED;
        auto &x1ss = x1sh->GetStorageShape();
        auto &x2ss = x2sh->GetStorageShape();
        if (x1ss.GetDimNum() < 2 || x2ss.GetDimNum() < 2)
            return ge::GRAPH_FAILED;
        // x1 shape: [M_total, Ka]; kernel expects M_per_rank = M_total / world_size
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr || *wsPtr <= 0)
            return ge::GRAPH_FAILED;
        worldSize_ = static_cast<uint64_t>(*wsPtr);
        m_ = x1ss.GetDim(0U) / worldSize_;
        k_ = x1ss.GetDim(1U); // Ka (per-rank K)
        n_ = x2ss.GetDim(0U); // N (always dim0, x2 shape = [N, K_total])

        auto *pm = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_PRECISION_MODE);
        precisionMode_ = pm ? static_cast<uint32_t>(*pm) : 0;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        if (m_ == 0 || k_ == 0)
            return ge::GRAPH_FAILED;
        uint64_t ka = k_; // Ka = per-rank K

        auto *rt = context_->GetRawTilingData();
        auto cap = rt->GetCapacity();
        if (!rt || cap < sizeof(allToAllMatmulTilingData))
            return ge::GRAPH_FAILED;
        memset_s(rt->GetData(), cap, 0, cap);

        auto *td = reinterpret_cast<allToAllMatmulTilingData *>(rt->GetData());

        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(m_, n_, k_, false, true, td->tileQbmmTilingData);

        auto &mm = td->tileQbmmTilingData;
        usedCoreNum_ = mm.usedCoreNum;
        auto nTile = (n_ + mm.baseN - 1) / mm.baseN;
        uint64_t headMSize = ((mm.usedCoreNum + nTile - 1) / nTile) * mm.baseM;
        uint64_t headTileCnt = m_ / headMSize;
        uint64_t tailMSize = m_ % headMSize;
        uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

        td->commTilingData.splitAxisTileSize = headMSize;
        td->commTilingData.splitAxisTileCnt = headTileCnt;
        td->commTilingData.splitAxisTailSize = tailMSize;
        td->commTilingData.splitAxisTailCnt = tailTileCnt;
        td->commTilingData.nonSplitAxisSize = ka;
        td->scaleCommTilingData.splitAxisTileSize = headMSize;
        td->scaleCommTilingData.splitAxisTileCnt = headTileCnt;
        td->scaleCommTilingData.splitAxisTailSize = tailMSize;
        td->scaleCommTilingData.splitAxisTailCnt = tailTileCnt;
        td->scaleCommTilingData.nonSplitAxisSize = ka / 32;

        OP_LOGI("AlltoAllMatmulApace",
                "commTilingData: splitAxisTileSize=%lu, splitAxisTileCnt=%lu, splitAxisTailSize=%lu, "
                "splitAxisTailCnt=%lu, nonSplitAxisSize=%lu",
                td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTileCnt,
                td->commTilingData.splitAxisTailSize, td->commTilingData.splitAxisTailCnt,
                td->commTilingData.nonSplitAxisSize);
        OP_LOGI("AlltoAllMatmulApace",
                "scaleCommTilingData: splitAxisTileSize=%lu, splitAxisTileCnt=%lu, splitAxisTailSize=%lu, "
                "splitAxisTailCnt=%lu, nonSplitAxisSize=%lu",
                td->scaleCommTilingData.splitAxisTileSize, td->scaleCommTilingData.splitAxisTileCnt,
                td->scaleCommTilingData.splitAxisTailSize, td->scaleCommTilingData.splitAxisTailCnt,
                td->scaleCommTilingData.nonSplitAxisSize);

        td->localMatmul = precisionMode_;

        OP_LOGI("AlltoAllMatmulApace", "precisionMode=%u", precisionMode_);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(MX_QUANT_MODE, 1 /*x2Transpose*/, DTYPE_BIAS_FP32, 0 /*isSmallK*/,
                                  ALL2ALL_COMM_TYPE_UDMA);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        platform_ascendc::PlatformAscendC ascendcPlatform(context_->GetPlatformInfo());
        workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
        auto *wsBuf = context_->GetWorkspaceSizes(1);
        if (wsBuf != nullptr) {
            wsBuf[0] = workspaceSize_;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        context_->GetRawTilingData()->SetDataSize(context_->GetRawTilingData()->GetCapacity());
        context_->SetBlockDim(usedCoreNum_);
        return ge::GRAPH_SUCCESS;
    }

private:
    uint64_t m_{0}, k_{0}, n_{0};
    uint64_t worldSize_{1};
    uint32_t precisionMode_{0};
};

} // namespace MC2Tiling

using AlltoAllMatmulApaceTilingClass = MC2Tiling::AlltoAllMatmulApaceTilingClass;
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmulApace, AlltoAllMatmulApaceTilingClass,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

#endif
