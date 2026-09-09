/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file pdist_tiling.cpp
 * \brief Host tiling for the pypto pdist kernel (op_kernel/pdist.py).
 *
 * Tiling/kernel correspondence follows DenseLILseV2Tiling in
 * attention/dense_lightning_indexer_softmax_lse_v2: the PdistTiling struct filled
 * via GetTilingData<PdistTiling>() comes from the pypto codegen header
 * (force-included by enable_pypto_kernel); the tiling key is encoded with
 * GET_TPL_TILING_KEY from the codegen PdistTilingKey_tilingkey.h, whose bit
 * layout (PBranch, bits 0-2, values 0-5) matches the Python TilingKeyField.
 */

#include "pdist_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace ge;

namespace optiling {

// Closed-form F(x) = sum_{v=0}^{x} floor(v/64) = 32q(q-1) + qr, where x+1 = 64q+r.
// Same formula as the kernel-side inline expansion and the Python host wrapper.
static int64_t PdistFloorSum(int64_t x)
{
    int64_t q = (x + 1) / 64;
    int64_t r = x + 1 - q * 64;
    return 32 * q * (q - 1) + q * r;
}

// p-class -> PBranch (kernel TilingKey value; must match op_kernel/pdist.py):
// 0: p=0 (Hamming) / 1: p=+inf (Chebyshev) / 2: p=2 / 3: p=1 / 4: p=0.5 / 5: general 0<p<inf
static uint64_t PdistBranchOf(double pVal, double &invP)
{
    invP = 0.0;
    if (pVal == 0.0) {
        return 0;
    }
    if (std::isinf(pVal)) {
        return 1;
    }
    if (pVal == 2.0) {
        return 2;
    }
    if (pVal == 1.0) {
        return 3;
    }
    if (pVal == 0.5) {
        return 4;
    }
    invP = 1.0 / pVal;
    return 5;
}

static ge::graphStatus PdistTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_PARAM_INVALID);

    auto socVersion = ascendcPlatform.GetSocVersion();
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion));
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    // Input x: [N, M], 2-D, fp32/fp16; output y: [N*(N-1)/2].
    auto xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "x shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "x must be 2-D [N, M], but got %zuD.",
                        xShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    int64_t nSize = xShape->GetStorageShape().GetDim(0);
    int64_t mSize = xShape->GetStorageShape().GetDim(1);

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "x desc is null."),
                return ge::GRAPH_PARAM_INVALID);
    auto xDType = xDesc->GetDataType();
    OP_CHECK_IF(xDType != ge::DT_FLOAT && xDType != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(), "x dtype must be float32 or float16, but got %d.",
                        static_cast<int32_t>(xDType)),
                return ge::GRAPH_PARAM_INVALID);

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_IF(outputShape == nullptr, OP_LOGE(context->GetNodeName(), "output shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(context->GetNodeName(), "output desc is null."),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(outputDesc->GetDataType() != xDType,
                OP_LOGE(context->GetNodeName(),
                        "y dtype must be the same as x dtype, but got x=%d, y=%d.",
                        static_cast<int32_t>(xDType), static_cast<int32_t>(outputDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(nSize < 2,
                OP_LOGE(context->GetNodeName(), "N must be >= 2 (at least one row pair), but got %ld.", nSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(mSize < 1, OP_LOGE(context->GetNodeName(), "M must be >= 1, but got %ld.", mSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(nSize > 65535,
                OP_LOGE(context->GetNodeName(), "N must be <= 65535, but got %ld.", nSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(mSize > 65535,
                OP_LOGE(context->GetNodeName(), "M must be <= 65535, but got %ld.", mSize),
                return ge::GRAPH_PARAM_INVALID);

    int64_t outputLen = nSize * (nSize - 1) / 2;
    OP_CHECK_IF(outputShape->GetStorageShape().GetDimNum() != 1 ||
                outputShape->GetStorageShape().GetDim(0) != outputLen,
                OP_LOGE(context->GetNodeName(),
                        "y must be 1-D [N*(N-1)/2] = [%ld], but got %zuD with dim0=%ld.",
                        outputLen, outputShape->GetStorageShape().GetDimNum(),
                        outputShape->GetStorageShape().GetDimNum() == 1
                            ? outputShape->GetStorageShape().GetDim(0) : -1),
                return ge::GRAPH_PARAM_INVALID);

    // Attr p (float, default 2.0): p >= 0 or +inf, NaN/negative rejected.
    double pVal = 2.0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto pPtr = attrs->GetAttrPointer<float>(0);
        if (pPtr != nullptr) {
            pVal = static_cast<double>(*pPtr);
        }
    }
    OP_CHECK_IF(std::isnan(pVal) || pVal < 0.0,
                OP_LOGE(context->GetNodeName(), "p must be >= 0 or +inf, but got %lf.", pVal),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(std::isinf(pVal) && pVal < 0.0,
                OP_LOGE(context->GetNodeName(), "p must be >= 0 or +inf, but got %lf.", pVal),
                return ge::GRAPH_PARAM_INVALID);

    double invP = 0.0;
    uint64_t branch = PdistBranchOf(pVal, invP);

    // Total 64-pair task blocks (kernel block-domain split unit).
    int64_t bSize = (nSize - 1) + PdistFloorSum(nSize - 2);
    OP_CHECK_IF(bSize < 1, OP_LOGE(context->GetNodeName(), "task block count must be positive, but got %ld.",
                                   bSize),
                return ge::GRAPH_PARAM_INVALID);

    // MIX launch (KERNEL_TYPE_MIX_AIC_1_2, the fixed pypto opc delivery form): blockDim counts
    // tsch blocks (each = 1 AIC + task-ration AIV cores). CalcTschBlockDim over the full chip
    // mirrors the reference pypto ops (dense_lightning_indexer_softmax_lse_v2 /
    // sparse_flash_mla_softmax_l1_norm); the kernel derives the vector-core count itself
    // (get_block_num() * get_subblock_num()) and cores whose [lo, hi) slice is empty idle.
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // PdistTiling comes from the pypto kernel (.py dataclass) via force-include.
    PdistTiling *tilingData = context->GetTilingData<PdistTiling>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context->GetNodeName(), "tiling data got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    tilingData->N = nSize;
    tilingData->M = mSize;
    tilingData->B = bSize;
    tilingData->p_val = static_cast<float>(pVal);
    tilingData->inv_p = static_cast<float>(invP);

    // TilingKey: PBranch (bits 0-2, values 0-5). Encoded with the ASCENDC macro over the
    // codegen-declared layout (PdistTilingKey_tilingkey.h), identical to the Python
    // TilingKeyField packing: value index placed at its bit offset.
    uint64_t tilingKey = GET_TPL_TILING_KEY(branch);
    context->SetTilingKey(tilingKey);

    // No extra workspace: only the libapi reserve.
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "Pdist tiling completed: blockDim=%u, aivNum=%u, N=%ld, M=%ld, B=%ld, p=%lf, branch=%lu, "
            "tilingKey=%lu, workspaceSize=%zu.",
            blockDim, aivNum, nSize, mSize, bSize, pVal, branch, tilingKey, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPdist([[maybe_unused]] gert::TilingParseContext *context)
{
    // No persistent compile state: platform info and shapes are re-read at every tiling.
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Pdist)
    .Tiling(PdistTilingFunc)
    .TilingParse<PdistCompileInfo>(TilingParseForPdist);

} // namespace optiling
