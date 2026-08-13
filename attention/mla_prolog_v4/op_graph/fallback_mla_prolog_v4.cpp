/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog_v4.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologV4InputTensor(const OpExecuteContext *ctx, MlaPrologV4FallBackParam &param)
{
    param.tokenX = ctx->GetRequiredInputTensor(TOKEN_X_INDEX_V4);
    OP_CHECK_IF(param.tokenX == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "tokenX"), return GRAPH_FAILED);

    param.weightDq = ctx->GetRequiredInputTensor(WEIGHT_DQ_INDEX_V4);
    OP_CHECK_IF(param.weightDq == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "weightDq"), return GRAPH_FAILED);

    param.weightUqQr = ctx->GetRequiredInputTensor(WEIGHT_UQ_QR_INDEX_V4);
    OP_CHECK_IF(param.weightUqQr == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "weightUqQr"),
                return GRAPH_FAILED);

    param.weightUk = ctx->GetRequiredInputTensor(WEIGHT_UK_INDEX_V4);
    OP_CHECK_IF(param.weightUk == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "weightUk"), return GRAPH_FAILED);

    param.weightDkvKr = ctx->GetRequiredInputTensor(WEIGHT_DKV_KR_INDEX_V4);
    OP_CHECK_IF(param.weightDkvKr == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "weightDkvKr"),
                return GRAPH_FAILED);

    param.rmsnormGammaCq = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CQ_INDEX_V4);
    OP_CHECK_IF(param.rmsnormGammaCq == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "rmsnormGammaCq"),
                return GRAPH_FAILED);

    param.rmsnormGammaCkv = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CKV_INDEX_V4);
    OP_CHECK_IF(param.rmsnormGammaCkv == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "rmsnormGammaCkv"),
                return GRAPH_FAILED);

    param.ropeSin = ctx->GetRequiredInputTensor(ROPE_SIN_INDEX_V4);
    OP_CHECK_IF(param.ropeSin == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "ropeSin"), return GRAPH_FAILED);
    param.ropeCos = ctx->GetRequiredInputTensor(ROPE_COS_INDEX_V4);
    OP_CHECK_IF(param.ropeCos == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "ropeCos"), return GRAPH_FAILED);

    param.kvCache = ctx->GetRequiredInputTensor(KV_CACHE_INDEX_V4);
    OP_CHECK_IF(param.kvCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "kvCache"), return GRAPH_FAILED);

    param.krCache = ctx->GetRequiredInputTensor(KR_CACHE_INDEX_V4);
    OP_CHECK_IF(param.krCache == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "krCache"), return GRAPH_FAILED);

    param.cacheIndex = ctx->GetOptionalInputTensor(CACHE_INDEX_V4);
    param.dequantScaleX = ctx->GetOptionalInputTensor(DEQUANT_SCALE_X_INDEX_V4);
    param.dequantScaleWDq = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DQ_INDEX_V4);
    param.dequantScaleWUqQr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_UQ_QR_INDEX_V4);
    param.dequantScaleWDkvKr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DKV_KR_INDEX_V4);
    param.quantScaleCkv = ctx->GetOptionalInputTensor(QUANT_SCALE_CKV_INDEX_V4);
    param.quantScaleCkr = ctx->GetOptionalInputTensor(QUANT_SCALE_CKR_INDEX_V4);
    param.smoothScalesCq = ctx->GetOptionalInputTensor(SMOOTH_SCALES_CQ_INDEX_V4);
    param.actualAeqLen = ctx->GetOptionalInputTensor(ACTUAL_SEQ_LEN_INDEX_V4);
    param.kNopeClipAlpha = ctx->GetOptionalInputTensor(K_NOPE_CLIP_ALPHA_INDEX_V4);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV4OutputTensor(const OpExecuteContext *ctx, MlaPrologV4FallBackParam &param)
{
    param.query = ctx->GetOutputTensor(QUERY_INDEX);
    OP_CHECK_IF(param.query == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "query"), return GRAPH_FAILED);

    param.queryRope = ctx->GetOutputTensor(QUERY_ROPE_INDEX);
    OP_CHECK_IF(param.queryRope == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "queryRope"),
                return GRAPH_FAILED);

    param.kvCacheOut = ctx->GetOutputTensor(KV_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.kvCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "kvCacheOut"),
                return GRAPH_FAILED);

    param.krCacheOut = ctx->GetOutputTensor(KR_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.krCacheOut == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "krCacheOut"),
                return GRAPH_FAILED);

    param.dequantScaleQNope = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_IF(param.dequantScaleQNope == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "dequantScaleQNope"),
                return GRAPH_FAILED);

    param.queryNorm = ctx->GetOutputTensor(QUERY_NORM_INDEX);
    OP_CHECK_IF(param.queryNorm == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "queryNorm"),
                return GRAPH_FAILED);

    param.dequantScaleQNorm = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_IF(param.dequantScaleQNorm == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "dequantScaleQNorm"),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV4Attr(const OpExecuteContext *ctx, MlaPrologV4FallBackParam &param)
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
    const bool *getDoRope = attrs->GetAttrPointer<bool>(ATTR_DO_ROPE_INDEX);
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
    param.doRope = getDoRope ? *getDoRope : true;
    return GRAPH_SUCCESS;
}

graphStatus MlaV4HostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("MlaPrologV4", "host_api_ctx"),
                return GRAPH_FAILED);

    MlaPrologV4FallBackParam param{};
    auto apiRet = GetMlaPrologV4InputTensor(host_api_ctx, param);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    apiRet = GetMlaPrologV4OutputTensor(host_api_ctx, param);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    GetMlaPrologV4Attr(host_api_ctx, param);

    apiRet = EXEC_OPAPI_CMD(aclnnMlaPrologV4WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk,
                            param.weightDkvKr, param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin,
                            param.ropeCos, param.kvCache, param.krCache, param.cacheIndex, // cacheIndex当前为可选参数
                            param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr,
                            param.dequantScaleWDkvKr, param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq,
                            param.actualAeqLen, param.kNopeClipAlpha, param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv,
                            param.cacheMode, param.weightQuantMode, param.kvCacheQuantMode, param.queryQuantMode,
                            param.ckvkrRepoMode, param.quantScaleRepoMode, param.tileSize, param.qcQrScale,
                            param.kcScale, param.doRope, param.query, param.queryRope, param.kvCacheOut,
                            param.krCacheOut, param.dequantScaleQNope, param.queryNorm, param.dequantScaleQNorm);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaPrologV4).OpExecuteFunc(MlaV4HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
