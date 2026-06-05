/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog_v3.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologV3InputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    param.tokenX = ctx->GetRequiredInputTensor(TOKEN_X_INDEX_V3);
    // OPS_ERR_IF(param.tokenX == nullptr, OPS_LOG_E("aclnnfallback", "tokenX is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "tokenX is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter tokenX of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.tokenX == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "tokenX"),
                return GRAPH_FAILED);

    param.weightDq = ctx->GetRequiredInputTensor(WEIGHT_DQ_INDEX_V3);
    // OPS_ERR_IF(param.weightDq == nullptr, OPS_LOG_E("aclnnfallback", "weightDq is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "weightDq is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter weightDq of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.weightDq == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "weightDq"),
                return GRAPH_FAILED);

    param.weightUqQr = ctx->GetRequiredInputTensor(WEIGHT_UQ_QR_INDEX_V3);
    // OPS_ERR_IF(param.weightUqQr == nullptr, OPS_LOG_E("aclnnfallback", "weightUqQr is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "weightUqQr is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter weightUqQr of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.weightUqQr == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "weightUqQr"),
                return GRAPH_FAILED);

    param.weightUk = ctx->GetRequiredInputTensor(WEIGHT_UK_INDEX_V3);
    // OPS_ERR_IF(param.weightUk == nullptr, OPS_LOG_E("aclnnfallback", "weightUk is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "weightUk is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter weightUk of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.weightUk == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "weightUk"),
                return GRAPH_FAILED);

    param.weightDkvKr = ctx->GetRequiredInputTensor(WEIGHT_DKV_KR_INDEX_V3);
    // OPS_ERR_IF(param.weightDkvKr == nullptr, OPS_LOG_E("aclnnfallback", "weightDkvKr is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "weightDkvKr is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter weightDkvKr of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.weightDkvKr == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "weightDkvKr"),
                return GRAPH_FAILED);

    param.rmsnormGammaCq = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CQ_INDEX_V3);
    // OPS_ERR_IF(param.rmsnormGammaCq == nullptr, OPS_LOG_E("aclnnfallback", "rmsnormGammaCq is null"),
    //            return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "rmsnormGammaCq is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter rmsnormGammaCq of MlaPrologV3 is required, but it is
    // empty."
    OP_CHECK_IF(param.rmsnormGammaCq == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "rmsnormGammaCq"), return GRAPH_FAILED);

    param.rmsnormGammaCkv = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CKV_INDEX_V3);
    // OPS_ERR_IF(param.rmsnormGammaCkv == nullptr, OPS_LOG_E("aclnnfallback", "rmsnormGammaCkv is null"),
    //            return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "rmsnormGammaCkv is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter rmsnormGammaCkv of MlaPrologV3 is required, but it is
    // empty."
    OP_CHECK_IF(param.rmsnormGammaCkv == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "rmsnormGammaCkv"), return GRAPH_FAILED);

    param.ropeSin = ctx->GetRequiredInputTensor(ROPE_SIN_INDEX_V3);
    // OPS_ERR_IF(param.ropeSin == nullptr, OPS_LOG_E("aclnnfallback", "ropeSin is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "ropeSin is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter ropeSin of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.ropeSin == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "ropeSin"),
                return GRAPH_FAILED);

    param.ropeCos = ctx->GetRequiredInputTensor(ROPE_COS_INDEX_V3);
    // OPS_ERR_IF(param.ropeCos == nullptr, OPS_LOG_E("aclnnfallback", "ropeCos is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "ropeCos is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter ropeCos of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.ropeCos == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "ropeCos"),
                return GRAPH_FAILED);

    param.kvCache = ctx->GetRequiredInputTensor(KV_CACHE_INDEX_V3);
    // OPS_ERR_IF(param.kvCache == nullptr, OPS_LOG_E("aclnnfallback", "kvCache is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "kvCache is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter kvCache of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.kvCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "kvCache"),
                return GRAPH_FAILED);

    param.krCache = ctx->GetRequiredInputTensor(KR_CACHE_INDEX_V3);
    // OPS_ERR_IF(param.krCache == nullptr, OPS_LOG_E("aclnnfallback", "krCache is null"), return GRAPH_FAILED);
    // 原报错: OPS_LOG_E("aclnnfallback", "krCache is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter krCache of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.krCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "krCache"),
                return GRAPH_FAILED);

    param.cacheIndex = ctx->GetOptionalInputTensor(CACHE_INDEX_V3);
    param.dequantScaleX = ctx->GetOptionalInputTensor(DEQUANT_SCALE_X_INDEX_V3);
    param.dequantScaleWDq = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DQ_INDEX_V3);
    param.dequantScaleWUqQr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_UQ_QR_INDEX_V3);
    param.dequantScaleWDkvKr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DKV_KR_INDEX_V3);
    param.quantScaleCkv = ctx->GetOptionalInputTensor(QUANT_SCALE_CKV_INDEX_V3);
    param.quantScaleCkr = ctx->GetOptionalInputTensor(QUANT_SCALE_CKR_INDEX_V3);
    param.smoothScalesCq = ctx->GetOptionalInputTensor(SMOOTH_SCALES_CQ_INDEX_V3);
    param.actualAeqLen = ctx->GetOptionalInputTensor(ACTUAL_SEQ_LEN_INDEX_V3);
    param.kNopeClipAlpha = ctx->GetOptionalInputTensor(K_NOPE_CLIP_ALPHA_INDEX_V3);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3OutputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    param.query = ctx->GetOutputTensor(QUERY_INDEX);
    // OP_CHECK_IF(param.query == nullptr, OP_LOGE("aclnnfallback", "query is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "query is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter query of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.query == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "query"),
                return GRAPH_FAILED);

    param.queryRope = ctx->GetOutputTensor(QUERY_ROPE_INDEX);
    // OP_CHECK_IF(param.queryRope == nullptr, OP_LOGE("aclnnfallback", "queryRope is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "queryRope is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter queryRope of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.queryRope == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "queryRope"),
                return GRAPH_FAILED);

    param.kvCacheOut = ctx->GetOutputTensor(KV_CACHE_OUT_INDEX);
    // OP_CHECK_IF(param.kvCacheOut == nullptr, OP_LOGE("aclnnfallback", "kvCacheOut is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "kvCacheOut is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter kvCacheOut of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.kvCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "kvCacheOut"),
                return GRAPH_FAILED);

    param.krCacheOut = ctx->GetOutputTensor(KR_CACHE_OUT_INDEX);
    // OP_CHECK_IF(param.krCacheOut == nullptr, OP_LOGE("aclnnfallback", "krCacheOut is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "krCacheOut is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter krCacheOut of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.krCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "krCacheOut"),
                return GRAPH_FAILED);

    param.dequantScaleQNope = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NOPE_INDEX);
    // OP_CHECK_IF(param.dequantScaleQNope == nullptr, OP_LOGE("aclnnfallback", "dequantScaleQNope is null"),
    //             return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "dequantScaleQNope is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter dequantScaleQNope of MlaPrologV3 is required, but it is
    // empty."
    OP_CHECK_IF(param.dequantScaleQNope == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "dequantScaleQNope"), return GRAPH_FAILED);

    param.queryNorm = ctx->GetOutputTensor(QUERY_NORM_INDEX);
    // OP_CHECK_IF(param.queryNorm == nullptr, OP_LOGE("aclnnfallback", "queryNorm is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "queryNorm is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter queryNorm of MlaPrologV3 is required, but it is empty."
    OP_CHECK_IF(param.queryNorm == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "queryNorm"),
                return GRAPH_FAILED);

    param.dequantScaleQNorm = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NORM_INDEX);
    // OP_CHECK_IF(param.dequantScaleQNorm == nullptr, OP_LOGE("aclnnfallback", "dequantScaleQNorm is null"),
    //             return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "dequantScaleQNorm is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter dequantScaleQNorm of MlaPrologV3 is required, but it is
    // empty."
    OP_CHECK_IF(param.dequantScaleQNorm == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "dequantScaleQNorm"), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3Attr(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    auto attrs = ctx->GetAttrs();
    const double *getRmsnormEpsilonCq = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CQ_INDEX);
    const double *getRmsnormEpsilonCkv = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CKV_INDEX);
    const char *getCacheMode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);
    const bool *getQueryNormFlag = attrs->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const int64_t *getWeightQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_QUANT_MODE_INDEX);
    const int64_t *getKvCacheQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_KV_CACHE_QUANT_MODE_INDEX);
    const int64_t *getQueryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    const int64_t *getCkvkrRepoMode = attrs->GetAttrPointer<int64_t>(ATTR_CKVKR_REPO_MODE_INDEX);
    const int64_t *getQuantScaleRepoMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_SCALE_REPO_MODE_INDEX);
    const int64_t *getTileSize = attrs->GetAttrPointer<int64_t>(ATTR_TILE_SIZE_INDEX);
    const double *getQcQrScale = attrs->GetAttrPointer<double>(ATTR_QC_QR_SCALE_INDEX);
    const double *getKcScale = attrs->GetAttrPointer<double>(ATTR_KC_SCALE_INDEX);
    param.rmsnormEpsilonCq = *getRmsnormEpsilonCq;
    param.rmsnormEpsilonCkv = *getRmsnormEpsilonCkv;
    param.cacheMode = getCacheMode;
    param.queryNormFlag = *getQueryNormFlag;
    param.weightQuantMode = *getWeightQuantMode;
    param.kvCacheQuantMode = *getKvCacheQuantMode;
    param.queryQuantMode = *getQueryQuantMode;
    param.ckvkrRepoMode = *getCkvkrRepoMode;
    param.quantScaleRepoMode = *getQuantScaleRepoMode;
    param.tileSize = *getTileSize;
    param.qcQrScale = getQcQrScale ? *getQcQrScale : 1.0;
    param.kcScale = getKcScale ? *getKcScale : 1.0;
    return GRAPH_SUCCESS;
}

graphStatus MlaV3HostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    // OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "host_api_ctx is null") → 上报EZ9999
    // 新报错: OP_LOGE_WITH_INVALID_INPUT(EZ0004) → "Parameter host_api_ctx of MlaPrologV3 is required, but it is
    // empty."
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("aclnnMlaPrologV3WeightNz", "host_api_ctx"),
                return GRAPH_FAILED);

    MlaPrologV3FallBackParam param{};
    auto apiRet = GetMlaPrologV3InputTensor(host_api_ctx, param);
    // OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get input tesnor failed"),
    //             return GRAPH_FAILED);
    // 原报错: OP_LOGE_WITHOUT_REPORT("aclnnfallback", "Context get input tesnor failed") → 不报EZ但记日志
    // 新做法: 内部已通过EZ0004报了具体错误，外层直接返回，不加冗余日志
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    apiRet = GetMlaPrologV3OutputTensor(host_api_ctx, param);
    // OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get output tesnor failed"),
    //             return GRAPH_FAILED);
    // 原报错: OP_LOGE_WITHOUT_REPORT("aclnnfallback", "Context get output tesnor failed") → 不报EZ但记日志
    // 新做法: 内部已通过EZ0004报了具体错误，外层直接返回，不加冗余日志
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    GetMlaPrologV3Attr(host_api_ctx, param);

    apiRet = EXEC_OPAPI_CMD(aclnnMlaPrologV3WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk,
                            param.weightDkvKr, param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin,
                            param.ropeCos, param.kvCache, param.krCache, param.cacheIndex, // cacheIndex当前为可选参数
                            param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr,
                            param.dequantScaleWDkvKr, param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq,
                            param.actualAeqLen, param.kNopeClipAlpha, param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv,
                            param.cacheMode, param.weightQuantMode, param.kvCacheQuantMode, param.queryQuantMode,
                            param.ckvkrRepoMode, param.quantScaleRepoMode, param.tileSize, param.qcQrScale,
                            param.kcScale, param.query, param.queryRope, param.kvCacheOut, param.krCacheOut,
                            param.dequantScaleQNope, param.queryNorm, param.dequantScaleQNorm);

    // OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "ret failed:%u", apiRet), return GRAPH_FAILED);
    // 原报错: OP_LOGE("aclnnfallback", "ret failed:%u", apiRet) → 上报EZ9999
    // 新做法: aclnnMlaPrologV3WeightNz内部已通过EZ宏报了具体错误，外层直接返回
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaPrologV3).OpExecuteFunc(MlaV3HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
