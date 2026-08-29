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
 * \file mla_prolog_v5.cpp
 * \brief MlaPrologV5 L0 实现：构建 AiCore kernel launcher。
 *
 * 独立算子工程无源仓 aclnnInner 生成框架，此处手写 L0 层：
 * 输出 tensor 全部由调用方传入，无需 InferShape/AllocTensor。
 *
 * torch_extension（aclnn_common.h ConvertType）创建的 aclTensor 在 ND 格式下
 * storage shape 为扁平一维（元素总数），而算子 tiling/inferShape 校验读取
 * storage shape（要求逻辑多维）。此处对 ND 输入/输出做 shape 归一化：
 * storage shape 与 view shape 元素数一致但维度不一致时，以 view shape 作为
 * storage shape（ND 连续内存语义等价）。FRACTAL_NZ 等非基础格式张量的
 * storage shape 本即为分形布局，保持不变。
 */

#include "mla_prolog_v5.h"

#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MlaPrologV5);

namespace {

// ND 张量 storage shape 归一化：扁平 storage → view shape；torch 侧对 3/4/5 维
// 基础格式张量打 NCL/NCHW/NCDHW 标签（数据仍为连续 ND 布局），统一转为 ND。
// 详见文件头注释。
void NormalizeNdStorageShape(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    const int32_t primaryFormat = ge::GetPrimaryFormat(tensor->GetStorageFormat());
    const bool isBaseFormat = (primaryFormat == ge::FORMAT_ND || primaryFormat == ge::FORMAT_NCL ||
                               primaryFormat == ge::FORMAT_NCHW || primaryFormat == ge::FORMAT_NCDHW);
    if (!isBaseFormat) {
        return; // 非基础格式（如 FRACTAL_NZ）storage 布局合法，保持不变
    }
    aclTensor *mutableTensor = const_cast<aclTensor *>(tensor);
    const op::Shape &viewShape = tensor->GetViewShape();
    const op::Shape &storageShape = tensor->GetStorageShape();
    if (viewShape.GetDimNum() != storageShape.GetDimNum() && viewShape.GetShapeSize() == storageShape.GetShapeSize()) {
        mutableTensor->SetStorageShape(viewShape);
        mutableTensor->SetOriginalShape(viewShape);
    }
    if (primaryFormat != ge::FORMAT_ND) {
        mutableTensor->SetStorageFormat(op::Format::FORMAT_ND);
        mutableTensor->SetViewFormat(op::Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    }
}

} // namespace

aclnnStatus MlaPrologV5(const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr,
                        const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq,
                        const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos,
                        aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional,
                        const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional,
                        const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
                        const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional,
                        const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional,
                        const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv,
                        char *cacheModeOptional, bool queryNormFlag, int64_t weightQuantMode, int64_t kvCacheQuantMode,
                        int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize,
                        double qcQrScale, double kcScale, bool doRope, const aclTensor *queryOut,
                        const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOut,
                        const aclTensor *queryNormOut, const aclTensor *dequantScaleQNormOut, aclOpExecutor *executor)
{
    L0_DFX(MlaPrologV5, tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
           ropeCos, kvCacheRef, krCacheRef, cacheIndexOptional, dequantScaleXOptional, dequantScaleWDqOptional,
           dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional,
           smoothScalesCqOptional, actualSeqLenOptional, kNopeClipAlphaOptional, rmsnormEpsilonCq, rmsnormEpsilonCkv,
           cacheModeOptional, queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode,
           quantScaleRepoMode, tileSize, qcQrScale, kcScale, doRope, queryOut, queryRopeOut, dequantScaleQNopeOut,
           queryNormOut, dequantScaleQNormOut);

    // optional 输入未传入时保持 nullptr 透传（框架按"未实例化"处理，
    // tiling 校验据此走"该参数未传入"分支；与 opbuild 生成的 Inner 行为一致）

    // torch_extension 路径下 ND 张量 storage shape 扁平，归一化为 view shape
    NormalizeNdStorageShape(tokenX);
    NormalizeNdStorageShape(weightUk);
    NormalizeNdStorageShape(rmsnormGammaCq);
    NormalizeNdStorageShape(rmsnormGammaCkv);
    NormalizeNdStorageShape(ropeSin);
    NormalizeNdStorageShape(ropeCos);
    NormalizeNdStorageShape(kvCacheRef);
    NormalizeNdStorageShape(krCacheRef);
    NormalizeNdStorageShape(cacheIndexOptional);
    NormalizeNdStorageShape(dequantScaleXOptional);
    NormalizeNdStorageShape(dequantScaleWDqOptional);
    NormalizeNdStorageShape(dequantScaleWUqQrOptional);
    NormalizeNdStorageShape(dequantScaleWDkvKrOptional);
    NormalizeNdStorageShape(quantScaleCkvOptional);
    NormalizeNdStorageShape(quantScaleCkrOptional);
    NormalizeNdStorageShape(smoothScalesCqOptional);
    NormalizeNdStorageShape(actualSeqLenOptional);
    NormalizeNdStorageShape(kNopeClipAlphaOptional);
    NormalizeNdStorageShape(queryOut);
    NormalizeNdStorageShape(queryRopeOut);
    NormalizeNdStorageShape(dequantScaleQNopeOut);
    NormalizeNdStorageShape(queryNormOut);
    NormalizeNdStorageShape(dequantScaleQNormOut);

    // attr 类型对齐：def 中 rmsnorm_epsilon_*/qc_qr_scale/kc_scale 为 Float(4B)，
    // API 形参为 double，须显式转 float（与 opbuild 生成 Inner 的处理一致），
    // 否则按 float 解释 double 内存会得到错误数值（如 eps 变成巨大值导致输出全 0）。
    float epsCq = static_cast<float>(rmsnormEpsilonCq);
    float epsCkv = static_cast<float>(rmsnormEpsilonCkv);
    float qcQrScaleF = static_cast<float>(qcQrScale);
    float kcScaleF = static_cast<float>(kcScale);

    // OP_INPUT / OP_OUTPUT 顺序与 mla_prolog_v5_def.cpp 的注册顺序严格一致
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MlaPrologV5,
        OP_INPUT(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                 kvCacheRef, krCacheRef, cacheIndexOptional, dequantScaleXOptional, dequantScaleWDqOptional,
                 dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional,
                 smoothScalesCqOptional, actualSeqLenOptional, kNopeClipAlphaOptional),
        OP_OUTPUT(queryOut, queryRopeOut, kvCacheRef, krCacheRef, dequantScaleQNopeOut, queryNormOut,
                  dequantScaleQNormOut),
        OP_ATTR(epsCq, epsCkv, cacheModeOptional, queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode,
                ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScaleF, kcScaleF, doRope));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MlaPrologV5 add to launcher list failed.");
        return ret;
    }
    return ACLNN_SUCCESS;
}

} // namespace l0op
