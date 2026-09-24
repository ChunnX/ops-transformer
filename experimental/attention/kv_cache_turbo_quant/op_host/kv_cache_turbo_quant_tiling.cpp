/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */
#include "kv_cache_turbo_quant_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

constexpr uint32_t DIMS_LIMIT = 3; // [num_tokens, num_kv_heads, head_dim]
constexpr uint32_t DIM_TOKENS = 0;
constexpr uint32_t DIM_HEADS = 1;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr size_t ATTR_OUT_DTYPE_INDEX = 1;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t KV_HEADS_MIN = 4;
constexpr int64_t KV_HEADS_MAX = 32;

// tilingkey = 200 + mse_bits（202/203/204）
constexpr uint64_t TILING_KEY_MSE2 = 202;
constexpr uint64_t TILING_KEY_MSE3 = 203;
constexpr uint64_t TILING_KEY_MSE4 = 204;
constexpr int64_t CUBE_TILE_V = 64;         // 与 kernel_kv_cache_turbo_quant_cube.h 一致
constexpr uint64_t WS_HEAD = 64 * 1024 * 1024;   // 避开框架清零区（与 kernel 侧一致，勿单独改）
constexpr size_t MAT_DIM_NUM = 2;                // rotation/qjl_matrix 均为 [128, 128]

static uint64_t GetTilingKeyByBits(int64_t bits)
{
    switch (bits) {
        case 2:
            return TILING_KEY_MSE2;
        case 4:
            return TILING_KEY_MSE4;
        default:
            return TILING_KEY_MSE3;
    }
}

ge::graphStatus KvCacheTurboQuantTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "TilingContext is nullptr!"),
                return ge::GRAPH_FAILED);

    // ---- shape: [num_tokens, num_kv_heads, 128] ----
    const gert::StorageShape *kvShape = context->GetInputShape(0);
    OP_CHECK_IF(kvShape == nullptr, OP_LOGE("KvCacheTurboQuant", "input shape is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape &shape = kvShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIMS_LIMIT || shape.GetDim(2) != HEAD_DIM,
                OP_LOGE("KvCacheTurboQuant", "kv_vectors must be 3D [num_tokens, num_kv_heads, 128]."),
                return ge::GRAPH_FAILED);
    const int64_t numTokens = shape.GetDim(DIM_TOKENS);
    const int64_t numHeads = shape.GetDim(DIM_HEADS);
    OP_CHECK_IF(numTokens <= 0, OP_LOGE("KvCacheTurboQuant", "num_tokens must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeads < KV_HEADS_MIN || numHeads > KV_HEADS_MAX,
                OP_LOGE("KvCacheTurboQuant", "num_kv_heads only supports 4~32 in MHA/GQA scenarios."),
                return ge::GRAPH_FAILED);
    const int64_t numVectors = numTokens * numHeads;

    // ---- 属性 mse_bits / out_dtype ----
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("KvCacheTurboQuant", "attrs is nullptr!"), return ge::GRAPH_FAILED);
    const int64_t *pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    const int64_t mseBits = (pBits != nullptr) ? *pBits : 3;
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4,
                OP_LOGE("KvCacheTurboQuant", "mse_bits only supports 2/3/4."), return ge::GRAPH_FAILED);
    const int64_t *pOutDtype = attrs->GetAttrPointer<int64_t>(ATTR_OUT_DTYPE_INDEX);
    const int64_t outDtype = (pOutDtype != nullptr) ? *pOutDtype : 0;
    OP_CHECK_IF(outDtype != 0 && outDtype != 1,
                OP_LOGE("KvCacheTurboQuant", "out_dtype only supports 0(bfloat16)/1(float16)."),
                return ge::GRAPH_FAILED);

    // ---- 矩阵元素类型：0=float32，1=bfloat16 ----
    const gert::StorageShape *matShape = context->GetInputShape(1);
    OP_CHECK_IF(matShape == nullptr, OP_LOGE("KvCacheTurboQuant", "rotation_matrix shape is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape &hShape = matShape->GetStorageShape();
    OP_CHECK_IF(hShape.GetDimNum() != MAT_DIM_NUM || hShape.GetDim(0) != HEAD_DIM || hShape.GetDim(1) != HEAD_DIM,
                OP_LOGE("KvCacheTurboQuant", "rotation_matrix must be [128, 128]."), return ge::GRAPH_FAILED);
    uint32_t matDtype = 0;
    auto hDesc = context->GetInputDesc(1);
    OP_CHECK_IF(hDesc == nullptr, OP_LOGE("KvCacheTurboQuant", "rotation_matrix desc is nullptr!"),
                return ge::GRAPH_FAILED);
    auto hDtype = hDesc->GetDataType();
    OP_CHECK_IF(hDtype != ge::DT_FLOAT && hDtype != ge::DT_BF16,
                OP_LOGE("KvCacheTurboQuant", "rotation_matrix only supports float32/bfloat16."),
                return ge::GRAPH_FAILED);
    matDtype = (hDtype == ge::DT_BF16) ? 1u : 0u;
    auto sDesc = context->GetInputDesc(2);
    OP_CHECK_IF(sDesc == nullptr, OP_LOGE("KvCacheTurboQuant", "qjl_matrix desc is nullptr!"),
                return ge::GRAPH_FAILED);
    auto sDtype = sDesc->GetDataType();
    OP_CHECK_IF(sDtype != hDtype,
                OP_LOGE("KvCacheTurboQuant", "rotation_matrix and qjl_matrix must have the same dtype."),
                return ge::GRAPH_FAILED);

    // ---- 分核 ----
    // MIX_AIC_1_2 核映射（compressor 同款）：blockDim = AIC 数；AIV blockIdx ∈ [0, 2*blockDim)，
    // aiCoreIdx = GetBlockIdx()/2。
    // 平台查询（PlatformAscendC/GetCoreNumAic）在本环境每次调用都是 ms 级开销，
    // 而 tiling 每次下发都会执行一次 → 这里只查一次并缓存（同进程同平台）。
    static uint32_t sAicNum = 0;
    if (sAicNum == 0) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t n = ascendcPlatform.GetCoreNumAic();
        sAicNum = (n == 0) ? 1 : n;
    }
    const uint32_t aicNum = sAicNum;
    int64_t tiles = (numVectors + CUBE_TILE_V - 1) / CUBE_TILE_V;
    uint32_t usedCoreNum = (tiles < static_cast<int64_t>(aicNum)) ? static_cast<uint32_t>(tiles) : aicNum;
    uint32_t blockDim = usedCoreNum;
    const uint32_t vecsPerCore = static_cast<uint32_t>(numVectors / usedCoreNum);
    const uint32_t formerCoreNum = static_cast<uint32_t>(numVectors % usedCoreNum);

    auto *td = context->GetTilingData<KvCacheTurboQuant::KvCacheTurboQuantTilingData>();
    OP_CHECK_IF(td == nullptr, OP_LOGE("KvCacheTurboQuant", "tiling data is nullptr!"), return ge::GRAPH_FAILED);
    td->numVectors = numVectors;
    td->usedCoreNum = usedCoreNum;
    td->formerCoreNum = formerCoreNum;
    td->vecsPerCore = vecsPerCore;
    td->mseBits = static_cast<uint32_t>(mseBits);
    td->matDtype = matDtype;
    td->outDtype = static_cast<uint32_t>(outDtype);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(GetTilingKeyByBits(mseBits));

    // MIX 内核 codegen 会在 AIC 上执行 matmul::clearWorkspace(workspace)，workspace 必须 >0；
    // v4 另需 u/r hi-lo 拆分与 y/proj 中转：
    //   head 1MB + 4×(128×128 bf16 权重) 128KB + numVectors×128×(4×bf16 + 2×fp32) = N×2048B
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        const size_t layout = WS_HEAD + 4 * HEAD_DIM * HEAD_DIM * 2 +
                              static_cast<size_t>(numVectors) * HEAD_DIM * (4 * 2 + 2 * 4);
        // 实测：框架把 kernel 的 workspace 指针垫在用户区之后 ~91MB，而 host 拿到的总量只比声明量多 75MB，
        // 因此声明量必须 > 布局尺寸，否则布局尾部（y/proj）落在分配区外（越界写会踩坏输出张量）。
        workspaces[0] = layout + 32 * 1024 * 1024;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForKvCacheTurboQuant(gert::TilingParseContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 平台查询（PlatformAscendC/GetCoreMemSize）在本环境是 ms 级开销，而 tiling 流程可能每次
    // 下发都会走到这里 → 只做一次。
    static bool sProvisioned = false;
    if (sProvisioned) {
        return ge::GRAPH_SUCCESS;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto compileInfo = context->GetCompiledInfo<KvCacheTurboQuantCompileInfo>();
    if (compileInfo != nullptr) {
        compileInfo->coreNum = ascendcPlatform.GetCoreNum();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo->ubSizePlatform = ubSize;
    }
    sProvisioned = true;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheTurboQuant)
    .Tiling(KvCacheTurboQuantTilingFunc)
    .TilingParse<KvCacheTurboQuantCompileInfo>(TilingPrepareForKvCacheTurboQuant);

} // namespace optiling
