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
#include "apace/tiling/quant_matmul_mx_tiling_swat.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/allto_all_matmul_apace_tiling_key.h"
#include "../../../op_kernel/arch35/allto_all_matmul_apace_tiling_data.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint64_t WS_BYTES = 4096;

class AlltoAllMatmulApaceTilingClass : public TilingBaseClass {
public:
    explicit AlltoAllMatmulApaceTilingClass(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    ~AlltoAllMatmulApaceTilingClass() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

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

    ge::graphStatus GetShapeAttrsInfo() override
    {
        auto *x1t = context_->GetInputTensor(1);
        auto *x2t = context_->GetInputTensor(2);
        if (!x1t || !x2t)
            return ge::GRAPH_FAILED;
        auto x1sh = x1t->GetOriginShape();
        auto x2sh = x2t->GetOriginShape();
        if (x1sh.GetDimNum() < 2 || x2sh.GetDimNum() < 2)
            return ge::GRAPH_FAILED;
        m_ = x1sh.GetDim(0U);
        k_ = x1sh.GetDim(1U);
        n_ = x2sh.GetDim(0U);

        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(1); // world_size is attr index 1
        worldSize_ = wsPtr ? static_cast<uint64_t>(*wsPtr) : 4;
        auto *pm = context_->GetAttrs()->GetAttrPointer<int64_t>(14); // precision_mode is attr index 14
        precisionMode_ = pm ? static_cast<uint32_t>(*pm) : 0;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        if (m_ == 0 || k_ == 0)
            return ge::GRAPH_FAILED;
        uint64_t ka = k_ / worldSize_;
        if (ka == 0)
            return ge::GRAPH_FAILED;
        uint64_t headMSize = 512;
        uint64_t tileCnt = m_ / headMSize;

        auto *rt = context_->GetRawTilingData();
        auto cap = rt->GetCapacity();
        if (!rt || cap < sizeof(AlltoAllMatmulApaceTilingData))
            return ge::GRAPH_FAILED;
        memset_s(rt->GetData(), cap, 0, cap);

        auto *td = reinterpret_cast<AlltoAllMatmulApaceTilingData *>(rt->GetData());
        td->tilingData.commTilingData.splitAxisTileSize = headMSize;
        td->tilingData.commTilingData.splitAxisTileCnt = tileCnt;
        td->tilingData.commTilingData.splitAxisTailSize = 0;
        td->tilingData.commTilingData.splitAxisTailCnt = 0;
        td->tilingData.commTilingData.nonSplitAxisSize = ka;
        td->tilingData.scaleCommTilingData.splitAxisTileSize = headMSize;
        td->tilingData.scaleCommTilingData.splitAxisTileCnt = tileCnt;
        td->tilingData.scaleCommTilingData.splitAxisTailSize = 0;
        td->tilingData.scaleCommTilingData.splitAxisTailCnt = 0;
        td->tilingData.scaleCommTilingData.nonSplitAxisSize =
            ((ka + 64 - 1) / 64) * 2;  // CeilDiv(ka,64) * MXFP_MULTI_BASE_SIZE

        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(m_, n_, ka, false, true, td->tilingData.tileQbmmTilingData);

        td->tilingData.localMatmul = precisionMode_;

        auto &mm = td->tilingData.tileQbmmTilingData;
        auto &cm = td->tilingData.commTilingData;

        fprintf(stderr, "[HOST_TILING] m=%lu k=%lu n=%lu ka=%lu coreNum=%u baseM=%u baseK=%u stepK=%u size=%u\n",
                (unsigned long)m_, (unsigned long)k_, (unsigned long)n_, (unsigned long)ka, mm.usedCoreNum,
                (unsigned)mm.baseM, (unsigned)mm.baseK, (unsigned)mm.stepK,
                (unsigned)sizeof(AlltoAllMatmulApaceTilingData));
        fprintf(stderr, "[HOST_PM] precisionMode=%u\n", precisionMode_);
        fprintf(stderr, "[HOST_COMM] tileSz=%lu tileCnt=%lu tailSz=%lu tailCnt=%lu nonSplitSz=%lu\n",
                (unsigned long)cm.splitAxisTileSize, (unsigned long)cm.splitAxisTileCnt,
                (unsigned long)cm.splitAxisTailSize, (unsigned long)cm.splitAxisTailCnt,
                (unsigned long)cm.nonSplitAxisSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(MX_QUANT_MODE, 1 /*x2Transpose*/, DTYPE_BIAS_FP32, 0 /*isSmallK*/,
                                  ALL2ALL_COMM_TYPE_AICPU);
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
        context_->SetBlockDim(aicoreParams_.aicNum);
        return ge::GRAPH_SUCCESS;
    }

private:
    uint64_t m_{0}, k_{0}, n_{0}, worldSize_{4};
    uint32_t precisionMode_{0};
};

} // namespace MC2Tiling

using AlltoAllMatmulApaceTilingClass = MC2Tiling::AlltoAllMatmulApaceTilingClass;
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmulApace, AlltoAllMatmulApaceTilingClass,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

#endif
