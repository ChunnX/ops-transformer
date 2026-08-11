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
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/tiling/comm_tiling_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/allto_all_matmul_v2_tiling_key.h"

namespace MC2Tiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint64_t WS_BYTES = 4096;

class AlltoAllMatmulV2TilingClass : public TilingBaseClass {
public:
    explicit AlltoAllMatmulV2TilingClass(gert::TilingContext *context) : TilingBaseClass(context) {}
    ~AlltoAllMatmulV2TilingClass() override = default;

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
    static constexpr size_t IDX_INPUT_BIAS = 3;
    static constexpr size_t IDX_INPUT_X1_SCALE = 4;
    static constexpr size_t IDX_INPUT_X2_SCALE = 5;
    static constexpr size_t IDX_OUTPUT_Y = 0;
    static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
    static constexpr size_t IDX_ATTR_TRANSPOSE_X2 = 7;
    static constexpr size_t IDX_ATTR_ALLTOALL_OUT_FLAG = 9;
    static constexpr size_t IDX_ATTR_COMM_MODE = 10;
    static constexpr size_t IDX_ATTR_PRECISION_MODE = 11;
    static constexpr uint64_t MX_SCALE_ALIGN = 64;
    static constexpr uint64_t SCALE_LAST_DIM = 2;
    static constexpr uint64_t SCALE_DIM_NUM = 3;
    static constexpr int64_t MAX_INT32_VAL = 2147483647;
    static constexpr uint64_t K_MAX_VAL = 65535;

    ge::graphStatus CheckTensorAttrs()
    {
        const char *opName = context_->GetNodeName();

        auto *x1Desc = context_->GetInputDesc(IDX_INPUT_X1);
        auto *x2Desc = context_->GetInputDesc(IDX_INPUT_X2);
        auto *yDesc = context_->GetOutputDesc(IDX_OUTPUT_Y);
        if (!x1Desc || !x2Desc || !yDesc) {
            OP_LOGE(opName, "Required input/output tensor descriptors missing.");
            return ge::GRAPH_FAILED;
        }

        // ---- world_size validation ----
        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        if (!wsPtr) {
            OP_LOGE(opName, "world_size attr is required but missing.");
            return ge::GRAPH_FAILED;
        }
        if (*wsPtr <= 0) {
            OP_LOGE(opName, "world_size must be positive, got %ld.", (long)*wsPtr);
            return ge::GRAPH_FAILED;
        }
        uint64_t worldSize = static_cast<uint64_t>(*wsPtr);
        if (worldSize != 2 && worldSize != 4 && worldSize != 8 && worldSize != 16) {
            OP_LOGE(opName, "world_size must be 2/4/8/16, got %lu.", worldSize);
            return ge::GRAPH_FAILED;
        }

        // ---- format checks ----
        auto x1Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        auto x2Fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
        auto yFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
        if (x1Fmt != ge::FORMAT_ND || x2Fmt != ge::FORMAT_ND) {
            OP_LOGE(opName, "x1 and x2 format must be ND, got x1=%d x2=%d.",
                    static_cast<int>(x1Fmt), static_cast<int>(x2Fmt));
            return ge::GRAPH_FAILED;
        }
        if (yFmt != ge::FORMAT_ND) {
            OP_LOGE(opName, "y output format must be ND, got %d.", static_cast<int>(yFmt));
            return ge::GRAPH_FAILED;
        }

        // ---- x1/x2 dtype checks ----
        auto x1Dtype = x1Desc->GetDataType();
        auto x2Dtype = x2Desc->GetDataType();
        if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
            (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
            OP_LOGE(opName, "x1/x2 dtype must be FP8_E4M3/FP8_E5M2/FP4_E2M1, got x1_dtype=%d x2_dtype=%d.",
                    static_cast<int>(x1Dtype), static_cast<int>(x2Dtype));
            return ge::GRAPH_FAILED;
        }
        if ((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1)) {
            OP_LOGE(opName, "FP4 requires both x1 and x2 to be FP4_E2M1, got x1_dtype=%d x2_dtype=%d.",
                    static_cast<int>(x1Dtype), static_cast<int>(x2Dtype));
            return ge::GRAPH_FAILED;
        }

        // ---- y output dtype ----
        auto yDtype = yDesc->GetDataType();
        if (yDtype != ge::DT_BF16 && yDtype != ge::DT_FLOAT16) {
            OP_LOGE(opName, "y output must be BF16 or FP16, got y_dtype=%d.", static_cast<int>(yDtype));
            return ge::GRAPH_FAILED;
        }

        // ---- scale checks ----
        auto *x1ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X1_SCALE);
        auto *x2ScaleDesc = context_->GetOptionalInputDesc(IDX_INPUT_X2_SCALE);
        if (!x1ScaleDesc || !x2ScaleDesc) {
            OP_LOGE(opName, "x1_scale or x2_scale is required but missing.");
            return ge::GRAPH_FAILED;
        }
        auto x1ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleDesc->GetStorageFormat()));
        auto x2ScaleFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleDesc->GetStorageFormat()));
        if (x1ScaleFmt != ge::FORMAT_ND || x2ScaleFmt != ge::FORMAT_ND) {
            OP_LOGE(opName, "x1_scale and x2_scale format must be ND, got x1_scale_fmt=%d x2_scale_fmt=%d.",
                    static_cast<int>(x1ScaleFmt), static_cast<int>(x2ScaleFmt));
            return ge::GRAPH_FAILED;
        }
        auto x1ScaleDtype = x1ScaleDesc->GetDataType();
        auto x2ScaleDtype = x2ScaleDesc->GetDataType();
        if (x1ScaleDtype != ge::DT_FLOAT8_E8M0 || x2ScaleDtype != ge::DT_FLOAT8_E8M0) {
            OP_LOGE(opName, "x1_scale and x2_scale must be FP8_E8M0, got x1_scale_dtype=%d x2_scale_dtype=%d.",
                    static_cast<int>(x1ScaleDtype), static_cast<int>(x2ScaleDtype));
            return ge::GRAPH_FAILED;
        }

        // ---- bias checks ----
        auto *biasDesc = context_->GetOptionalInputDesc(IDX_INPUT_BIAS);
        if (biasDesc != nullptr) {
            auto biasDtype = biasDesc->GetDataType();
            if (biasDtype != ge::DT_FLOAT) {
                OP_LOGE(opName, "bias dtype must be FLOAT, got bias_dtype=%d.", static_cast<int>(biasDtype));
                return ge::GRAPH_FAILED;
            }
            auto biasFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()));
            if (biasFmt != ge::FORMAT_ND) {
                OP_LOGE(opName, "bias format must be ND, got bias_fmt=%d.", static_cast<int>(biasFmt));
                return ge::GRAPH_FAILED;
            }
        }

        // ---- attr checks ----
        auto *tx2 = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_TRANSPOSE_X2);
        if (!tx2 || !(*tx2)) {
            OP_LOGE(opName, "transpose_x2 must be true, got %d.", tx2 ? (int)*tx2 : -1);
            return ge::GRAPH_FAILED;
        }

        auto *commModePtr = context_->GetAttrs()->GetAttrPointer<char>(IDX_ATTR_COMM_MODE);
        if (!commModePtr) {
            OP_LOGE(opName, "comm_mode attr is required but missing.");
            return ge::GRAPH_FAILED;
        }
        if (std::strcmp(commModePtr, "urma") != 0) {
            OP_LOGE(opName, "comm_mode must be urma, got '%s'.", commModePtr);
            return ge::GRAPH_FAILED;
        }

        auto *a2aFlag = context_->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_ALLTOALL_OUT_FLAG);
        if (!a2aFlag) {
            OP_LOGE(opName, "alltoall_out_flag attr is required but missing.");
            return ge::GRAPH_FAILED;
        }
        if (*a2aFlag) {
            OP_LOGE(opName, "alltoall_out_flag must be false, got true.");
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckTensorShapes()
    {
        const char *opName = context_->GetNodeName();

        auto *x1Shape = context_->GetInputShape(IDX_INPUT_X1);
        auto *x2Shape = context_->GetInputShape(IDX_INPUT_X2);
        auto *yShape = context_->GetOutputShape(IDX_OUTPUT_Y);
        if (!x1Shape || !x2Shape || !yShape) {
            return ge::GRAPH_FAILED;
        }

        uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
        uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
        uint64_t yDimNum = yShape->GetStorageShape().GetDimNum();
        if (x1DimNum != 2 || x2DimNum != 2) {
            OP_LOGE(opName, "x1 and x2 must be 2D, got %luD/%luD.", x1DimNum, x2DimNum);
            return ge::GRAPH_FAILED;
        }
        if (yDimNum != 2) {
            OP_LOGE(opName, "y output must be 2D, got %luD.", yDimNum);
            return ge::GRAPH_FAILED;
        }

        auto *wsPtr = context_->GetAttrs()->GetAttrPointer<int64_t>(IDX_ATTR_WORLD_SIZE);
        uint64_t worldSize = static_cast<uint64_t>(wsPtr ? *wsPtr : 1);

        uint64_t x1M = x1Shape->GetStorageShape().GetDim(0U);
        uint64_t x1K = x1Shape->GetStorageShape().GetDim(1U);
        uint64_t x2N = x2Shape->GetStorageShape().GetDim(0U);
        uint64_t x2K = x2Shape->GetStorageShape().GetDim(1U);
        uint64_t yM = yShape->GetStorageShape().GetDim(0U);
        uint64_t yN = yShape->GetStorageShape().GetDim(1U);

        if (x1M == 0 || x1K == 0 || x2N == 0 || x2K == 0) {
            OP_LOGE(opName, "x1/x2 dimensions must be non-zero, got x1[%lu,%lu] x2[%lu,%lu].", x1M, x1K, x2N, x2K);
            return ge::GRAPH_FAILED;
        }
        if (x1M > static_cast<uint64_t>(MAX_INT32_VAL) || x2N > static_cast<uint64_t>(MAX_INT32_VAL)) {
            OP_LOGE(opName, "x1 dim0 and x2 dim0 must not exceed INT32_MAX, got M=%lu N=%lu.", x1M, x2N);
            return ge::GRAPH_FAILED;
        }
        if (x1K > K_MAX_VAL || x2K > K_MAX_VAL) {
            OP_LOGE(opName, "K dimension must not exceed %lu, got x1_k=%lu x2_k=%lu.", K_MAX_VAL, x1K, x2K);
            return ge::GRAPH_FAILED;
        }

        if (x1M % worldSize != 0) {
            OP_LOGE(opName, "x1 dim0 M=%lu must be divisible by world_size=%lu.", x1M, worldSize);
            return ge::GRAPH_FAILED;
        }
        if (x1K % MX_SCALE_ALIGN != 0 || x2K % MX_SCALE_ALIGN != 0) {
            OP_LOGE(opName,
                    "K dimension must be divisible by %lu, got x1_k=%lu x2_k=%lu.", MX_SCALE_ALIGN, x1K, x2K);
            return ge::GRAPH_FAILED;
        }
        if (x2K != x1K * worldSize) {
            OP_LOGE(opName, "K mismatch: x2.K=%lu != x1.K=%lu * world_size=%lu.", x2K, x1K, worldSize);
            return ge::GRAPH_FAILED;
        }

        if (yM != x1M / worldSize || yN != x2N) {
            OP_LOGE(opName, "y shape mismatch: expected[%lu,%lu] got[%lu,%lu].", x1M / worldSize, x2N, yM, yN);
            return ge::GRAPH_FAILED;
        }

        // ---- bias shape ----
        auto *biasShape = context_->GetOptionalInputShape(IDX_INPUT_BIAS);
        if (biasShape != nullptr) {
            if (biasShape->GetStorageShape().GetDimNum() != 1) {
                OP_LOGE(opName, "bias must be 1D, got %luD.",
                        biasShape->GetStorageShape().GetDimNum());
                return ge::GRAPH_FAILED;
            }
            uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0U);
            if (biasDim0 != x2N) {
                OP_LOGE(opName, "bias dim0 must equal N=%lu, got bias_dim0=%lu.", x2N, biasDim0);
                return ge::GRAPH_FAILED;
            }
        }

        // ---- scale shapes ----
        auto *x1ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X1_SCALE);
        auto *x2ScaleShape = context_->GetOptionalInputShape(IDX_INPUT_X2_SCALE);
        if (!x1ScaleShape || !x2ScaleShape) {
            return ge::GRAPH_FAILED;
        }

        uint64_t x1ScaleDimNum = x1ScaleShape->GetStorageShape().GetDimNum();
        uint64_t x2ScaleDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
        if (x1ScaleDimNum != SCALE_DIM_NUM || x2ScaleDimNum != SCALE_DIM_NUM) {
            OP_LOGE(opName, "x1_scale and x2_scale must be %luD, got %luD/%luD.", SCALE_DIM_NUM, x1ScaleDimNum,
                    x2ScaleDimNum);
            return ge::GRAPH_FAILED;
        }

        uint64_t x1ScaleM = x1ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x1ScaleK = x1ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x1ScaleLast = x1ScaleShape->GetStorageShape().GetDim(2U);
        uint64_t x2ScaleN = x2ScaleShape->GetStorageShape().GetDim(0U);
        uint64_t x2ScaleK = x2ScaleShape->GetStorageShape().GetDim(1U);
        uint64_t x2ScaleLast = x2ScaleShape->GetStorageShape().GetDim(2U);

        if (x1ScaleM != x1M || x1ScaleK != x1K / MX_SCALE_ALIGN || x1ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE(opName,
                    "x1_scale shape mismatch: expected[m=%lu,k=%lu,last=%lu] got[m=%lu,k=%lu,last=%lu].",
                    x1M, x1K / MX_SCALE_ALIGN, SCALE_LAST_DIM, x1ScaleM, x1ScaleK, x1ScaleLast);
            return ge::GRAPH_FAILED;
        }
        if (x2ScaleN != x2N || x2ScaleK != x2K / MX_SCALE_ALIGN || x2ScaleLast != SCALE_LAST_DIM) {
            OP_LOGE(opName,
                    "x2_scale shape mismatch: expected[n=%lu,k=%lu,last=%lu] got[n=%lu,k=%lu,last=%lu].",
                    x2N, x2K / MX_SCALE_ALIGN, SCALE_LAST_DIM, x2ScaleN, x2ScaleK, x2ScaleLast);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOpInputInfo()
    {
        if (CheckTensorAttrs() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CheckTensorShapes() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

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
        if (CheckOpInputInfo() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
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

        apace::CommTilingBase::GetCommTilingData(m_, n_, ka, mm, td->commTilingData, td->scaleCommTilingData);

        const char *opName = context_->GetNodeName();
        OP_LOGI(opName,
                "comm tiling: tileSize=%lu, tileCnt=%lu, tailSize=%lu, tailCnt=%lu, nonSplitSize=%lu",
                td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTileCnt,
                td->commTilingData.splitAxisTailSize, td->commTilingData.splitAxisTailCnt,
                td->commTilingData.nonSplitAxisSize);
        OP_LOGI(opName,
                "scale comm tiling: tileSize=%lu, tileCnt=%lu, tailSize=%lu, tailCnt=%lu, nonSplitSize=%lu",
                td->scaleCommTilingData.splitAxisTileSize, td->scaleCommTilingData.splitAxisTileCnt,
                td->scaleCommTilingData.splitAxisTailSize, td->scaleCommTilingData.splitAxisTailCnt,
                td->scaleCommTilingData.nonSplitAxisSize);

        td->localMatmul = precisionMode_;

        OP_LOGI(opName, "localMatmul precisionMode=%u", precisionMode_);

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

using AlltoAllMatmulV2TilingClass = MC2Tiling::AlltoAllMatmulV2TilingClass;
REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmulV2, AlltoAllMatmulV2TilingClass,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

#endif
