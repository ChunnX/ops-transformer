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
 * \file apace_mx_quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief QuantGroupedMatMulAlltoAllv 的 apace hcomm 通路 tiling 实现
 *
 * 选择条件（全部满足才走 apace 通路，否则回退原通路）：
 *   1. 构建开关 ENABLE_APACE_QUANT_GMM_A2AV=1（构建时环境变量，默认关闭；
 *      未开启时 kernel 侧 apace 分支为兜底空实现，本类 IsCapable 恒为 false）；
 *   2. 运行时环境变量 MC2_QUANT_GMM_A2AV_APACE=1；
 *   3. MX 量化模式（gmmX/gmmWeight 量化模式均为 mx）；
 *   4. comm_mode 属性为 ccu（apace hcomm 通路当前仅支持 CCU 引擎）；
 *   5. gmmX 非空且 H1 为 64 的倍数（MX scale 对齐 + SWAT 引擎约束）。
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "apace_mx_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include "mc2_comm_utils.h"
#include <cstring>
#include <cstdlib>
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {
namespace Mc2GroupedMatmul {

constexpr char APACE_ENABLE_ENV[] = "MC2_QUANT_GMM_A2AV_APACE";
constexpr uint64_t APACE_WS_ALIGN_SIZE = 512UL;

// SWAT 引擎按实际量化 dtype 实例化（mm::DataType 为模板参数，须用模板函数分发）
template <mm::DataType dtypeA, mm::DataType dtypeB>
static void RunSwatEngine(fe::PlatFormInfos *platformInfoPtr, uint32_t aicNum, uint32_t aivNum, uint64_t m, uint64_t n,
                          uint64_t k, bool transB, QuantMatmulTilingData &mmTile)
{
    QuantMatmulTilingSwat<dtypeA, dtypeB, mm::BiasDataType::DT_FLOAT> tilingEngine;
    tilingEngine.SetPlatformInfoPtr(platformInfoPtr);
    tilingEngine.SetCoreLimit(aicNum, aivNum);
    tilingEngine.SetOptimizeEnable(false);
    tilingEngine.SetMTailAlignEnable(true);
    tilingEngine.SetAdjustBasicBlockEnable(false);
    tilingEngine.SetBiasInfo(false);
    tilingEngine.GetTilingData(m, n, k, false, transB, mmTile);
}

bool ApaceMxQuantGroupedMatmulAllToAllvTiling::IsApaceEnvEnabled() const
{
    const char *apaceEnable = getenv(APACE_ENABLE_ENV);
    return (apaceEnable != nullptr) && (strncmp(apaceEnable, "1", 1UL) == 0);
}

bool ApaceMxQuantGroupedMatmulAllToAllvTiling::IsCapable()
{
#ifndef ENABLE_APACE_QUANT_GMM_A2AV
    // apace 构建开关未开启时（默认），kernel 侧 apace 分支为兜底空实现，tiling 侧须同步不下发该 tiling key
    return false;
#endif
    if (!IsApaceEnvEnabled()) {
        return false;
    }
    if (GetQuantMode(context_, opName_) != QUANT_PAIR_MX) {
        OP_LOGD(opName_, "Apace path requires MX quant mode, skip.");
        return false;
    }
    // apace hcomm 通路当前仅支持 CCU 通信引擎
    uint8_t commMode = 0xFFU;
    if ((QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) || (commMode != Mc2Comm::COMM_MODE_CCU)) {
        OP_LOGD(opName_, "Apace path requires ccu comm mode, skip.");
        return false;
    }
    // MX scale 按 64 分组对齐，SWAT tiling 引擎要求 K 为 64 的倍数；空输入回退原通路
    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    if (gmmXStorageShape == nullptr || gmmXStorageShape->GetStorageShape().GetDimNum() != DIM_TWO) {
        return false;
    }
    const int64_t aTotal = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    const int64_t h1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    if (aTotal <= 0 || h1 <= 0 || (static_cast<uint64_t>(h1) % MX_SCALE_GROUP != 0UL)) {
        OP_LOGD(opName_, "Apace path requires non-empty gmmX and H1 multiple of 64, skip.");
        return false;
    }
    OP_LOGI(opName_, "ApaceMxQuantGroupedMatmulAllToAllvTiling capable.");
    return true;
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::RunSwatTiling(uint64_t m, uint64_t n, uint64_t k, bool transB,
                                                                        QuantMatmulTilingData &mmTile)
{
    auto *platformInfoPtr = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "platformInfo"),
                    return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();

    // mmX/mmWeight dtype 与 gmmX/gmmWeight 一致（CheckMxQuantDtypeConstraints 已校验）
    if (localParams_.gmmXDtype == ge::DT_FLOAT4_E2M1) {
        RunSwatEngine<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1>(platformInfoPtr, aicNum, aivNum, m, n,
                                                                                  k, transB, mmTile);
    } else if (localParams_.gmmXDtype == ge::DT_FLOAT8_E5M2 && localParams_.gmmWeightDtype == ge::DT_FLOAT8_E5M2) {
        RunSwatEngine<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E5M2>(platformInfoPtr, aicNum, aivNum, m, n,
                                                                                  k, transB, mmTile);
    } else if (localParams_.gmmXDtype == ge::DT_FLOAT8_E4M3FN && localParams_.gmmWeightDtype == ge::DT_FLOAT8_E5M2) {
        RunSwatEngine<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E5M2>(platformInfoPtr, aicNum, aivNum, m,
                                                                                    n, k, transB, mmTile);
    } else if (localParams_.gmmXDtype == ge::DT_FLOAT8_E5M2 && localParams_.gmmWeightDtype == ge::DT_FLOAT8_E4M3FN) {
        RunSwatEngine<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E4M3FN>(platformInfoPtr, aicNum, aivNum, m,
                                                                                    n, k, transB, mmTile);
    } else if (localParams_.gmmXDtype == ge::DT_FLOAT8_E4M3FN && localParams_.gmmWeightDtype == ge::DT_FLOAT8_E4M3FN) {
        RunSwatEngine<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN>(platformInfoPtr, aicNum, aivNum,
                                                                                      m, n, k, transB, mmTile);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "gmmX, gmmWeight",
            (Ops::Base::ToString(localParams_.gmmXDtype) + ", " + Ops::Base::ToString(localParams_.gmmWeightDtype))
                .c_str(),
            "Apace path unsupported dtype combination");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::SetApaceHcclTiling(
    Apace::hcommQuantGmmAlltoAllvTilingData *td)
{
    const uint32_t alltoAllvCmd = HCCL_CMD_ALLTOALLV;
    const std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    const uint32_t alltoAllvReduceType = 0U;

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "null", "not null"),
                    return ge::GRAPH_FAILED);

    auto outputDataType = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "outputDataType", Ops::Base::ToString(outputDataType).c_str(),
                                              "The dtype of outputDataType must be within the supported range."),
        return ge::GRAPH_FAILED);
    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);

    AscendC::Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
                                                  alltoAllvDstDataType, alltoAllvSrcDataType);
    hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(td->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(td->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::SetApaceTilingData(
    Apace::hcommQuantGmmAlltoAllvTilingData *td)
{
    const uint64_t expertNum = localParams_.ep * localParams_.epWorldSize;
    td->h1 = localParams_.H1;
    td->n1 = localParams_.N1;
    td->h2 = localParams_.H2;
    td->n2 = localParams_.N2;
    td->bs = localParams_.Bs;
    td->aTotal = localParams_.A;
    td->bsk = localParams_.BsK;
    td->e = static_cast<uint32_t>(localParams_.ep);
    td->epWorldSize = static_cast<uint32_t>(localParams_.epWorldSize);
    td->hasSharedMm = localParams_.hasSharedMm ? 1U : 0U;
    // 分阶段诊断开关（真机二分定位卡死域，所有 rank 须一致设置）：
    // 0=完整(默认) 1=仅通信 2=计算+barrier 3=计算+提交 4=仅GMM 5=仅SoftBarrier 6=仅Init 7=空kernel
    td->debugStage = 0U;
    if (const char *stageEnv = getenv("MC2_APACE_GMM_A2AV_DEBUG_STAGE"); stageEnv != nullptr) {
        td->debugStage = static_cast<uint32_t>(atoi(stageEnv));
        // ERROR 级别确保现场可见，用于确认开关已生效
        OP_LOGE(opName_,
                "Apace debugStage=%u生效 (1=仅通信 2=计算+barrier 3=计算+提交 4=仅GMM 5=仅barrier 6=仅Init 7=空kernel)",
                td->debugStage);
    }
    for (uint64_t i = 0; i < expertNum; i++) {
        td->sendCnt[i] = localTilingData_.taskTilingInfo.sendCnt[i];
        td->recvCnt[i] = localTilingData_.taskTilingInfo.recvCnt[i];
    }

    // 单组最大 token 数作为 GMM 的 M（与原通路 DoQuantGMMTiling 的 mMaxSize 口径一致）
    uint64_t maxM = 0;
    for (uint64_t expertIdx = 0; expertIdx < localParams_.ep; expertIdx++) {
        uint64_t mSize = 0;
        for (uint64_t i = 0; i < localParams_.epWorldSize; i++) {
            mSize += static_cast<uint64_t>(localTilingData_.taskTilingInfo.sendCnt[i * localParams_.ep + expertIdx]);
        }
        maxM = std::max(mSize, maxM);
    }
    td->maxM = maxM;

    // user workspace 布局（与 kernel 侧 InitBaseParams 寻址一致）：
    //   [sendBuf(A*N1*yDtypeSize)] [groupList(e*ep*2*8B)] [cTable(e*ep*8B)] [sharedGroupList(8B)]
    const uint64_t gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    const uint64_t sendBufSize =
        mc2tiling::AlignUp(localParams_.A * localParams_.N1 * gmmYDtypeSize, APACE_WS_ALIGN_SIZE);
    td->wsGroupListOffset = sendBufSize;
    td->wsCTableOffset = td->wsGroupListOffset + expertNum * 2UL * sizeof(int64_t);
    td->wsSharedGroupListOffset = td->wsCTableOffset + expertNum * sizeof(uint64_t);
    apaceWorkSpaceSize_ = mc2tiling::AlignUp(td->wsSharedGroupListOffset + sizeof(int64_t), APACE_WS_ALIGN_SIZE);

    // GMM tiling（m=maxM, n=N1, k=H1）与共享专家 MM tiling（m=BS, n=N2, k=H2）
    MC2_CHECK_LOG_RET(
        opName_, RunSwatTiling(maxM, localParams_.N1, localParams_.H1, localParams_.isGmmWeightTrans, td->gmmTile));
    usedCoreNum_ = td->gmmTile.usedCoreNum;
    if (localParams_.hasSharedMm) {
        MC2_CHECK_LOG_RET(opName_, RunSwatTiling(localParams_.Bs, localParams_.N2, localParams_.H2,
                                                 localParams_.isMmWeightTrans, td->sharedMmTile));
        usedCoreNum_ = std::max(usedCoreNum_, static_cast<uint32_t>(td->sharedMmTile.usedCoreNum));
    }
    OP_LOGI(opName_,
            "Apace tiling: e=%u ep=%u A=%lu BSK=%lu maxM=%lu H1=%lu N1=%lu hasSharedMm=%u usedCoreNum=%u "
            "wsSize=%lu",
            td->e, td->epWorldSize, td->aTotal, td->bsk, td->maxM, td->h1, td->n1, td->hasSharedMm, usedCoreNum_,
            apaceWorkSpaceSize_);

    return SetApaceHcclTiling(td);
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::DoOpTiling()
{
    // 复用基类全量校验（MX dtype/shape/attr/sendRecvCounts），计数写入 taskTilingInfo
    MC2_CHECK_LOG_RET(opName_, CheckAndSetInputOutputInfo());

    auto *rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "rawTilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rawTilingData->GetCapacity() < sizeof(Apace::hcommQuantGmmAlltoAllvTilingData),
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "rawTilingData"), return ge::GRAPH_FAILED);
    auto *td = reinterpret_cast<Apace::hcommQuantGmmAlltoAllvTilingData *>(rawTilingData->GetData());
    errno_t ret = memset_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), 0,
                           sizeof(Apace::hcommQuantGmmAlltoAllvTilingData));
    OP_TILING_CHECK(ret != EOK, OP_LOGE(opName_, "memset apace tiling data failed"), return ge::GRAPH_FAILED);
    MC2_CHECK_LOG_RET(opName_, SetApaceTilingData(td));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(libApiWorkSpaceSize_) + apaceWorkSpaceSize_;
    OP_LOGD(opName_, "Apace Workspaces[0] size=%zu", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

uint64_t ApaceMxQuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, TILINGKEY_TPL_APACE, true);
    OP_LOGD(opName_, "Apace GET_TPL_TILING_KEY: [%d,%d,%d,%d,1], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, TILINGKEY_TPL_APACE, tilingKey);
    return tilingKey;
}

ge::graphStatus ApaceMxQuantGroupedMatmulAllToAllvTiling::PostTiling()
{
    // tiling 数据只声明结构体大小（capacity-8），尾部 8 字节留给框架写入 DFX 指针（atomicIndex）
    context_->GetRawTilingData()->SetDataSize(sizeof(Apace::hcommQuantGmmAlltoAllvTilingData));
    // 与原通路完全对齐：不设置 SetScheduleMode（原通路为默认 0；独占模式与 MC2 通信域
    // 辅助 task 的调度交互未知，属未验证组合）
    // 必须全核启动（与原通路 aicCoreNum 口径一致）：kernel 逐批使用 mode-0 全核 FFTS barrier
    // （CrossCoreSetFlag<0>(8)+WaitFlag）及 SyncAll<false>，其计数域为调度域内全部 AIC 核，
    // 部分核启动（SWAT usedCoreNum）时未启动核永不 arrive → barrier 永不释放 → 卡死。
    // GMM 调度器在满核下自行分配 block，闲置核空跑 Run 循环并正常参与 barrier。
    context_->SetBlockDim(localParams_.aicCoreNum);
    OP_LOGI(opName_, "Apace blockDim=%lu (usedCoreNum=%u)", localParams_.aicCoreNum, usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

// 注册 tiling 类：apace 能力域为原 MX 通路(priority=1)的子集，必须先于其尝试，故占用 priority=0
// （原 TT 通路顺延至 priority=2）。IsCapable 不满足时自动回退原通路。
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatMulAlltoAllv, ApaceMxQuantGroupedMatmulAllToAllvTiling, 0);

} // namespace Mc2GroupedMatmul
} // namespace Mc2Tiling
