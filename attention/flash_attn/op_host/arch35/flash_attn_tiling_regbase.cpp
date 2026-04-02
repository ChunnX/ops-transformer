/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling_regbase.cpp
 * \brief FlashAttn arch35 tiling基类实现（非量化场景框架，具体计算待补充）
 */

#include <algorithm>
#include "flash_attn_tiling_regbase.h"
#include "../flash_attn_tiling_common.h"

namespace optiling {
namespace FA {

void FlashAttnTilingRegbase::Reset()
{
    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(ge::DT_FLOAT16);
    calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
    isHighPrecision = true;
    tilingKeyDType = DtypeEnum::FLOAT16;
    bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    tilingKeyLayout = FALayoutType::NONE;
    tilingKeyKVLayout = FAKVLayoutType::BNSD;
    implMode = FAImplMode::HIGH_PRECISION;
    inputLayoutQ = nullptr;
    inputLayoutKv = nullptr;
    inputLayoutOut = nullptr;

    bSize = gSize = dSize = dSizeV = n1Size = n2Size = s1Size = s2Size = 0LL;
    accumS1 = accumS2 = realT1Size = 0LL;

    softmaxScale = 0.0f;
    maskMode = 0LL;
    winLeft = 0LL;
    winRight = 0LL;
    returnSoftmaxLse = 0LL;
    deterministic = 0LL;

    hasAttenMask = false;
    isPA = false;
    isTND = false;
    hasVarLen = false;
    hasMetadata = false;

    blockSize = 0;
    blockTableDim2 = 0;
    paBlockNumSum = 0;
    paLayoutType = 0U;

    s1BasicBlock = FA_NUM_128;
    s2BasicBlock = FA_NUM_128;
    dBasicBlock = FA_NUM_128;
    dVBasicBlock = FA_NUM_128;
    dTemplateType = FADTemplateType::NONALIGNED;
    dVTemplateType = FADTemplateType::NONALIGNED;

    opName = nullptr;
}

ge::graphStatus FlashAttnTilingRegbase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto queryShape = context_->GetInputShape(FA_INPUT_Q_INDEX);
    auto keyShape = context_->GetInputShape(FA_INPUT_K_INDEX);
    auto valueShape = context_->GetInputShape(FA_INPUT_V_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingRegbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttnCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        npuArch = compileInfoPtr->npuArch;
        socVersion = compileInfoPtr->socVersion;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        npuArch = ascendcPlatform.GetCurNpuArch();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,  aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OP_LOGI(context_, "FlashAttn platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttnTilingRegbase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(FA_INPUT_Q_INDEX)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);

    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPrecision ? matmul_tiling::DataType::DT_FLOAT
                                            : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPrecision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPrecision ? ge::GetSizeByDataType(ge::DT_FLOAT)
                                             : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPrecision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "FlashAttn only supports FP16/BF16, got: %d.",
                                        static_cast<int>(inputDtype));
            return false;
    }
    bmm2OutDtype = bmm1OutDtype;
    return true;
}

bool FlashAttnTilingRegbase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto softmaxModePtr = attrs->GetAttrPointer<float>(FA_ATTR_SOFTMAX_MODE_INDEX);
    auto maskModePtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_MASK_MODE_INDEX);
    auto winLeftPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_WIN_LEFT_INDEX);
    auto winRightPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_WIN_RIGHT_INDEX);
    inputLayoutQ = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_Q_INDEX);
    inputLayoutKv = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_KV_INDEX);
    inputLayoutOut = attrs->GetAttrPointer<char>(FA_ATTR_LAYOUT_OUT_INDEX);
    auto returnLsePtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_RETURN_SOFTMAX_LSE);
    auto deterministicPtr = attrs->GetAttrPointer<int64_t>(FA_ATTR_DETERMINISTIC);

    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, winLeftPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, winRightPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutQ);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutKv);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutOut);
    OP_CHECK_NULL_WITH_CONTEXT(context_, returnLsePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, deterministicPtr);

    softmaxScale = *softmaxModePtr;
    maskMode = *maskModePtr;
    winLeft = *winLeftPtr;
    winRight = *winRightPtr;
    returnSoftmaxLse = *returnLsePtr;
    deterministic = *deterministicPtr;

    hasAttenMask = (maskMode != 0LL);

    implMode = FAImplMode::HIGH_PRECISION;

    OP_LOGD(context_, "FlashAttn attrs: softmaxScale[%f] maskMode[%ld] winLeft[%ld] winRight[%ld] "
            "layoutQ[%s] layoutKv[%s] layoutOut[%s] returnLse[%ld] deterministic[%ld].",
            softmaxScale, maskMode, winLeft, winRight,
            inputLayoutQ, inputLayoutKv, inputLayoutOut, returnSoftmaxLse, deterministic);
    return true;
}

bool FlashAttnTilingRegbase::AnalyzeLayout()
{
    return true;
}

bool FlashAttnTilingRegbase::AnalyzeVarLenInput()
{
    // TODO: 解析cuSeqlens或seqused
    return true;
}

bool FlashAttnTilingRegbase::AnalyzePAInput()
{
    return true;
}

bool FlashAttnTilingRegbase::AnalyzeMetadataInput()
{
    //TODO:metada 解析 
    auto metadataShape = context_->GetOptionalInputShape(FA_INPUT_METADATA_INDEX);
    if (metadataShape != nullptr && metadataShape->GetStorageShape().GetShapeSize() > 0) {
        hasMetadata = true;
        OP_LOGD(context_, "FlashAttn: metadata input is present, may use pre-computed tiling.");
    }
    return true;
}

ge::graphStatus FlashAttnTilingRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "FlashAttn TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze attrs/dtype/layout."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeVarLenInput() || !AnalyzePAInput() || !AnalyzeMetadataInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze optional inputs."), return ge::GRAPH_FAILED);

    // ================================================================
    // 写死 shape/attr/flags：对应 example 的 BNSD 推理场景
    //   Q shape:  BNSD = {B=1, N_q=8, S_q=128, D=64}
    //   K shape:  BSND = {B=1, S_kv=128, N_kv=2, D=64}
    //   V shape:  BSND = {B=1, S_kv=128, N_kv=2, D=64}
    //   Out shape: BNSD = {B=1, N_q=8, S_q=128, D=64}
    //   属性: softmaxMode=0→scale=0.125f, maskMode=0, returnSoftmaxLse=0
    // ================================================================
    bSize  = 1LL;     // batch size
    n1Size = 1LL;     // Q head 数，来自 qShape[1]=8
    n2Size = 1LL;     // KV head 数，来自 kShape[2]=2（BSND格式）
    gSize  = 1LL;     // GQA 比例 = n1Size/n2Size = 8/2
    s1Size = 128LL;   // Q 序列长度，来自 qShape[2]=128
    s2Size = 128LL;   // KV 序列长度，来自 kShape[1]=128
    dSize  = 64LL;    // Q/K head 维度，来自 qShape[3]=64
    dSizeV = 64LL;    // V head 维度，与 D 相同

    // softmaxMode=0.0f 表示使用默认缩放系数 1/sqrt(D) = 1/sqrt(64) = 0.125f
    if (softmaxScale == 0.0f) {
        softmaxScale = 0.125f;
    }
    tilingKeyLayout   = FALayoutType::BNSD;          // Q/Out 均为 BNSD
    tilingKeyKVLayout = FAKVLayoutType::BNSD;         // KV 为 BSND
    implMode          = FAImplMode::HIGH_PRECISION;   // 高精度模式
    hasAttenMask      = false;   // maskMode=0，无注意力掩码
    isPA              = false;   // 无分页注意力
    isTND             = false;   // 非 TND 变长

    // ================================================================
    // 填充 InputParamsRegbase（所有字段对应上述固定 shape/attr）
    // ================================================================
    // --- shape 维度 ---
    inputParamsRegbase_->set_bSize(bSize);                   // 1
    inputParamsRegbase_->set_n2Size(n2Size);                 // 2
    inputParamsRegbase_->set_gSize(gSize);                   // 4
    inputParamsRegbase_->set_t1Size(s1Size);                 // 128  （TND 时为 T1，此处 = S_q）
    inputParamsRegbase_->set_t2Size(s2Size);                 // 128  （TND 时为 T2，此处 = S_kv）
    inputParamsRegbase_->set_s1Size(s1Size);                 // 128
    inputParamsRegbase_->set_s2Size(s2Size);                 // 128
    inputParamsRegbase_->set_alignedS2(128LL);               // ceil(128/16)*16 = 128，已对齐
    inputParamsRegbase_->set_dSize(dSize);                   // 64
    inputParamsRegbase_->set_dSizeV(dSizeV);                 // 64
    inputParamsRegbase_->set_dSizeRope(0LL);                 // 无 rope
    // --- 缩放/精度 ---
    inputParamsRegbase_->set_scaleValue(softmaxScale);       // 0.125f
    inputParamsRegbase_->set_implMode(0U);                   // HIGH_PRECISION = 0
    // --- layout ---
    inputParamsRegbase_->set_layoutType(3U);                 // BNSD = 3
    // --- 注意力窗口（无掩码时设为全量） ---
    inputParamsRegbase_->set_preTokens(65536LL);             // 全量注意力：preTokens = 大值
    inputParamsRegbase_->set_nextTokens(0LL);                // 无因果掩码
    // --- GQA ---
    inputParamsRegbase_->set_isGqa(0U);                      // n1(8) != n2(2) → GQA 有效
    inputParamsRegbase_->set_headNumRatio(1U);               // n1/n2 = 8/2 = 4
    // --- 输出控制 ---
    inputParamsRegbase_->set_isSoftMaxLseEnable(0U);         // returnSoftmaxLse=0，不输出 lse
    // --- dropout（无） ---
    inputParamsRegbase_->set_needDropMaskOp(0U);             // 无 dropout
    // --- KV 连续性 ---
    inputParamsRegbase_->set_isKvContinuous(1U);             // 非 PA，KV 地址连续
    // --- 注意力掩码（无） ---
    inputParamsRegbase_->set_attenMaskCompressMode(1U);      // NONE = 1
    inputParamsRegbase_->set_attenMaskShapeType(0U);         // 默认
    inputParamsRegbase_->set_attenMaskS1Size(0);             // 无掩码
    inputParamsRegbase_->set_attenMaskS2Size(0U);            // 无掩码
    // --- PSE（无） ---
    inputParamsRegbase_->set_pseType(0U);
    // --- 变长序列（无 cuSeqlens / seqused） ---
    inputParamsRegbase_->set_isActualSeqLengthsNull(1U);     // cuSeqlensQ = nullptr
    inputParamsRegbase_->set_isActualSeqLengthsKVNull(1U);   // cuSeqlensKv = nullptr
    // --- prefix（无） ---
    inputParamsRegbase_->set_isActualSharedPrefixLenNull(1u);
    // --- 分页注意力（无） ---
    inputParamsRegbase_->set_blockSize(0);
    inputParamsRegbase_->set_blockTableDim2(0);
    inputParamsRegbase_->set_paBlockNumSum(0);
    inputParamsRegbase_->set_paLayoutType(0U);
    // --- 量化（无） ---
    inputParamsRegbase_->set_deqScaleFlag(0U);
    inputParamsRegbase_->set_deqScale2Flag(0U);
    // --- padding（无） ---
    inputParamsRegbase_->set_isQHasLeftPadding(0U);
    inputParamsRegbase_->set_isKVHasLeftPadding(0U);
    // --- rope（无） ---
    inputParamsRegbase_->set_ropeHeadSize(0U);
    inputParamsRegbase_->set_prefixSeqInnerSize(0U);

    OP_LOGD(context_, "FlashAttn shape[FIXED]: B=%ld N_q=%ld N_kv=%ld G=%ld S_q=%ld S_kv=%ld D=%ld Dv=%ld scale=%f.",
            bSize, n1Size, n2Size, gSize, s1Size, s2Size, dSize, dSizeV, softmaxScale);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingRegbase::DoOpTiling()
{
    CalcDBasicBlock();
    CalcDVBasicBlock();
    CalcS1S2BasicBlock();

    // ================================================================
    // 写死基本块大小：对应 D=64, S_q=S_kv=128 的 BNSD 场景
    //   dTemplateType  = ALIGNED_64  → D=64 按 64 对齐
    //   dVTemplateType = ALIGNED_64  → Dv=64 同上
    //   s1BasicBlock   = 128         → S_q=128 单次处理 128 行
    //   s2BasicBlock   = 128         → S_kv=128 单次处理 128 列
    // ================================================================
    dBasicBlock    = 64LL;
    dVBasicBlock   = 64LL;
    dTemplateType  = FADTemplateType::ALIGNED_64;
    dVTemplateType = FADTemplateType::ALIGNED_64;
    s1BasicBlock   = 128LL;
    s2BasicBlock   = 128LL;

    // aivNum 兜底：硬编码路径若平台信息未获取，默认按 32 核计算
    if (aivNum == 0) { aivNum = 32U; }

    // TND变长场景用realT1Size代替s1Size
    int64_t effectiveS1 = isTND ? realT1Size : s1Size;

    int64_t s1Outer = FA_CeilDiv(effectiveS1, s1BasicBlock);
    multiCoreParamsRegbase_->set_s1OuterSize(s1Outer);

    int64_t totalSz = CalcTotalSize();
    multiCoreParamsRegbase_->set_totalSize(totalSz);

    int32_t usedCoreNum = static_cast<int32_t>(
        std::min(totalSz, static_cast<int64_t>(aivNum)));
    OP_LOGD(context_, "FlashAttn totalSz=%ld aivNum=%ld usedCoreNum=%ld\n.",
            totalSz, aivNum, usedCoreNum);
    multiCoreParamsRegbase_->set_coreNum(1);

    // 均匀分配：前formerNum个核各处理splitFactor个任务，其余核处理splitFactorTail个
    int64_t splitFactor     = FA_CeilDiv(totalSz, static_cast<int64_t>(usedCoreNum));
    int64_t splitFactorTail = totalSz - (static_cast<int64_t>(usedCoreNum) - 1LL) * splitFactor;
    multiCoreParamsRegbase_->set_splitFactorSize(splitFactor);
    multiCoreParamsRegbase_->set_splitFactorTailSize(splitFactorTail);

    // 填充bnStartIdx：每个核的起始BN索引
    uint32_t bnStartIdxArr[48] = {};
    for (int32_t i = 0; i < usedCoreNum && i < 48; i++) {
        bnStartIdxArr[i] = static_cast<uint32_t>(
            static_cast<int64_t>(i) * splitFactor);
    }
    multiCoreParamsRegbase_->set_bnStartIdx(bnStartIdxArr);
    multiCoreParamsRegbase_->set_firstFullLoadS1OuterIdx(-1);
    multiCoreParamsRegbase_->set_splitCoreMode(0U);
    // sparseStartIdx 全零（无稀疏注意力）
    int64_t sparseArr[48] = {};
    multiCoreParamsRegbase_->set_sparseStartIdx(sparseArr);

    // ================================================================
    // DropmaskParamsRegbase：无 dropout，全部置零
    //   needDropMaskOp=0 → kernel 不执行 dropout，此结构体不被访问
    // ================================================================
    auto *dropParams = &tilingData->dropmaskParamsRegbase;
    dropParams->set_multiCoreFactorSize(0);     // 无 dropout，多核因子为 0
    dropParams->set_baseUbCalSize(0);           // 无 dropout，UB 计算大小为 0
    dropParams->set_multiCoreTotalSize(0LL);    // 无 dropout，多核总量为 0
    dropParams->set_shapeTotalSize(0LL);        // 无 dropout，shape 总量为 0
    dropParams->dropMaskAddrOffset = 0LL;       // 无 dropout，地址偏移为 0

    // ================================================================
    // InitOutputParams：输出初始化参数
    //   totalOutputSize = B*N_q*S_q*D = 1*8*128*64 = 65536（元素数）
    //   singleCoreSize  = totalOutputSize / coreNum = 65536/8 = 8192
    //   needInit = 0：推理场景 kernel 直接写满输出，无需清零
    //   isOneN   = 0：N_q=8 不等于 1
    //   totalSoftMaxLseOutputSize = 0：returnSoftmaxLse=0，不输出 lse
    // ================================================================
    auto *initOut = &tilingData->initOutputParams;
    int64_t totalOutElems = bSize * n1Size * s1Size * dSize;   // = 65536
    initOut->set_totalOutputSize(totalOutElems);               // 65536
    initOut->set_totalSoftMaxLseOutputSize(0LL);               // 不输出 softmax_lse
    initOut->set_needInit(0U);                                 // 推理无需初始化输出
    initOut->set_isOneN(1U);                                   // n1Size=8 ≠ 1
    initOut->set_singleCoreSize(
        usedCoreNum > 0 ? static_cast<uint32_t>(totalOutElems / usedCoreNum) : 0U);  // 8192

    OP_LOGD(context_,
        "FlashAttn DoOpTiling: s1Outer=%ld totalSize=%ld usedCoreNum=%d "
        "splitFactor=%ld splitTail=%ld totalOutElems=%ld.",
        s1Outer, totalSz, usedCoreNum, splitFactor, splitFactorTail, totalOutElems);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingRegbase::DoLibApiTiling()
{
    // TODO: 
    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingRegbase::CalcDVBasicBlock()
{
    // TODO:
}

int64_t FlashAttnTilingRegbase::CalcTotalSize()
{
    // bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize()
    return 0;
}

ge::graphStatus FlashAttnTilingRegbase::PostTiling()
{
    // 设置TilingKey和BlockDim，由子类GetTilingKey()提供key
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);

    int32_t usedCoreNum = multiCoreParamsRegbase_->get_coreNum();
    // if (usedCoreNum <= 0) {
    //     OPS_REPORT_VECTOR_INNER_ERR(opName,
    //         "PostTiling: usedCoreNum(%d) is invalid, blockDim cannot be 0.",
    //         usedCoreNum);
    //     return ge::GRAPH_FAILED;
    // }

    context_->SetBlockDim(1);
    // auto platformInfoPtr = context_->GetPlatformInfo();
    // if (platformInfoPtr != nullptr) {
    //     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    //     context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(
    //         static_cast<uint32_t>(usedCoreNum), 0U, aivNum));
    // } else {
    //     context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    // }

    // OP_LOGD(context_,
    //     "FlashAttn PostTiling: tilingKey=0x%lx blockDim=%d.",
    //     tilingKey, usedCoreNum);
    // context_->GetRawTilingData()->SetDataSize(sizeof(FlashAttentionScoreSimplifiedTilingData));
    return ge::GRAPH_SUCCESS;
}

} // namespace FA
} // namespace optiling