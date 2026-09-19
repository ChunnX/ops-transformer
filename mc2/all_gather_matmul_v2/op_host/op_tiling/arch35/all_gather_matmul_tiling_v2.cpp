/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul_tiling_v2.cpp
 * \brief
 */
#include <algorithm>
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "all_gather_matmul_tiling_v2.h"
#include "../../../op_kernel/all_gather_matmul_v2_apt_tiling_key.h"
#include "all_gather_comm_algo_table.h"
#include "all_gather_fit_balance_tiling.h"
#include "all_gather_hccl_utils.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_comm_algo_selector.h"
#include "mc2_comm_utils.h"
#include "mc2_log.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "register/op_def_registry.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace optiling {
static constexpr int64_t CMP_MAX_LEN = 7;
// 非量化恒走 APACE（R13 拍板，USING_APACE_IMPL 维度删除），v1 实现保留编译、不派发
// commTile 256 对齐封顶（方案 §5.1 方案 A + a2a AlignCutResForKernel 同款）
static constexpr uint64_t APACE_KERNEL_ALIGN_M = 256;
// stepK 截断上限（mmv3 basic 消费口径：kL1 = baseK × min(stepKa, stepKb, 4)）
static constexpr uint32_t APACE_STEPK_THRESHOLD = 4;
// L1 A/B 滚动 buffer 数（Blaze basic l1BufferNum 常规值 / SWAT DB_SIZE 同款；mmv3 depthA1 为 K 深度
// 语义非 buffer 数，跨语义直映射有 L1 越界风险——W2 核定结论）
static constexpr uint8_t APACE_L1_BUFFER_NUM = 2;

bool AllGatherMatmulTilingV2::IsCapable()
{
    if ((npuArch_ == Ops::Base::DAV_3510) && inputIsBf16Fp16_) {
        OP_LOGI(opName_, "Start with AllGatherMatmulTilingV2 tiling.");
        return true;
    }

    OP_LOGI(opName_, "Skip AllGatherMatmulTilingV2 tiling when input datatype is not fp16 or bf16.");
    return false;
}

ge::graphStatus AllGatherMatmulTilingV2::SetRawTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingData == nullptr), OP_LOGE(opName_, "Fail to get rawTilingData."),
                    return ge::GRAPH_FAILED);
    allGatherMatmulTilingDataV2_ = context_->GetTilingData<AllGatherMatmulTilingDataV2>();
    return ge::GRAPH_SUCCESS;
}

void AllGatherMatmulTilingV2::FillApaceMmTile(const Mc2MatMulV3TilingData& localTiling)
{
    // mmTile 字段映射（Mc2MatMulV3TilingData → QuantMatmulTilingData，方案 §3.4 映射表）：
    //   m/n/k 供日志/校验（problemShape 由 kernel 三区拼装）；baseM/N/K 为 Blaze 基础块规格（L0 级）；
    //   尾块因子（mTailTile/nTailTile/mBaseTailSplitCnt/nBaseTailSplitCnt/mTailMain/nTailMain）同名映射
    //   （Blaze scheduler 尾块拆分，UpdateTailTile 判定式同构）。
    auto& mmTile = hcommTilingDataSelf_.mmTile;
    const auto& tCube = localTiling.tCubeTiling;
    mmTile = {};
    mmTile.m = static_cast<uint32_t>(args_.mValue);
    mmTile.n = static_cast<uint32_t>(args_.nValue);
    mmTile.k = static_cast<uint32_t>(args_.kValue);
    mmTile.baseM = tCube.baseM;
    mmTile.baseN = tCube.baseN;
    mmTile.baseK = tCube.baseK;
    mmTile.scaleKL1 = 0; // 非量化不消费（N2 不读该字段）
    mmTile.mTailTile = localTiling.mTailCnt;
    mmTile.nTailTile = localTiling.nTailCnt;
    mmTile.mBaseTailSplitCnt = localTiling.mBaseTailSplitCnt;
    mmTile.nBaseTailSplitCnt = localTiling.nBaseTailSplitCnt;
    mmTile.mTailMain = localTiling.mTailMain;
    mmTile.nTailMain = localTiling.nTailMain;
    mmTile.usedCoreNum = tCube.usedCoreNum;
    // stepK/nBufferNum 语义核定（W2 收口）：Blaze FragL1Params{kL1 = stepK·baseK, l1BufNum = nBufferNum}
    // 与 mmv3 basic 消费口径对齐（matmul_v3_base_tiling_advanced.h GetTilingData(Mc2MatMulV3BasicTilingData&)）：
    //   kL1 = baseK × min(stepKa, stepKb, 4) ⇒ stepK = min(stepKa, stepKb) 截断 4；
    //   nBufferNum 为 L1 A/B 滚动 buffer 数，取 2（Blaze basic 常规值 / SWAT DB_SIZE 同款）。
    uint32_t stepK = std::min(tCube.stepKa, tCube.stepKb);
    stepK = std::min(stepK, APACE_STEPK_THRESHOLD);
    mmTile.stepK = static_cast<uint8_t>(stepK);
    mmTile.nBufferNum = APACE_L1_BUFFER_NUM;
    mmTile.dbL0c = static_cast<uint8_t>(tCube.dbL0C);
}

ge::graphStatus AllGatherMatmulTilingV2::DoApaceTiling()
{
    // commTile：cutRes（v1 fit-balance + CCU/AICPU 调整链产出，O4 必落地项）→ 256 对齐封顶
    // （a2a AlignCutResForKernel 同款：tileM 向上 256 对齐、≥m 时收缩为单块；
    //   CCU 候选集 {2048,1024,512,256} 天然 256 对齐——对齐为 no-op，AICPU 路径对齐后 tileCnt 只减不增）
    auto& rcsCfg = MutableRCSTilingData();
    uint64_t apaceTileM = tileMValue_;
    uint64_t apaceTileCnt = rcsCfg.tileCnt;
    uint64_t apaceTailM = rcsCfg.tailM;
    uint64_t apaceTailCnt = rcsCfg.tailCnt;
    if ((apaceTileM != 0) && (args_.mValue != 0)) {
        uint64_t alignedTileM = mc2tiling::AlignUp(apaceTileM, APACE_KERNEL_ALIGN_M);
        if (alignedTileM >= args_.mValue) {
            apaceTileM = args_.mValue;
            apaceTileCnt = 1;
            apaceTailM = 0;
            apaceTailCnt = 0;
        } else {
            apaceTileM = alignedTileM;
            apaceTileCnt = args_.mValue / alignedTileM;
            uint64_t remainder = args_.mValue % alignedTileM;
            apaceTailM = (remainder > 0) ? remainder : 0;
            apaceTailCnt = (remainder > 0) ? 1 : 0;
        }
    }
    auto& commTile = hcommTilingDataSelf_.commTile;
    commTile.splitAxisTileSize = apaceTileM;
    commTile.splitAxisTileCnt = apaceTileCnt;
    commTile.splitAxisTailSize = apaceTailM;
    commTile.splitAxisTailCnt = apaceTailCnt;
    commTile.nonSplitAxisSize = args_.kValue;

    // mmTile：v1 DoVersion2Tiling 的 local tiling 输出（M=per-rank m）→ QuantMatmulTilingData 字段映射
    FillApaceMmTile(MutableMC2MatmulV3LocalTilingData());

    // mc2InitTiling/mc2CcTiling：SetMc2Hcomm 产出直接写入 apace 结构（a2a L565 同款）
    hcommTilingDataSelf_.mc2InitTiling = allGatherMatmulTilingDataV2_->mc2InitTiling;
    hcommTilingDataSelf_.mc2CcTiling = allGatherMatmulTilingDataV2_->mc2CcTiling;

    // gatherLen：沿用 v1 GetStorageA 语义（is_gather_out=false 时 R·AlignUp(m·k·ds,512)，否则 0）
    hcommTilingDataSelf_.gatherLen = rcsCfg.gatherLen;

    OP_LOGI(opName_,
            "[APACE] commTile: tileM=%lu, tileCnt=%lu, tailM=%lu, tailCnt=%lu, k=%lu; mmTile: baseM=%u, baseN=%u, "
            "baseK=%u, stepK=%u, nBufferNum=%u, dbL0c=%u, mTailTile=%u, nTailTile=%u; gatherLen=%lu.",
            commTile.splitAxisTileSize, commTile.splitAxisTileCnt, commTile.splitAxisTailSize,
            commTile.splitAxisTailCnt, commTile.nonSplitAxisSize, hcommTilingDataSelf_.mmTile.baseM,
            hcommTilingDataSelf_.mmTile.baseN, hcommTilingDataSelf_.mmTile.baseK, hcommTilingDataSelf_.mmTile.stepK,
            hcommTilingDataSelf_.mmTile.nBufferNum, hcommTilingDataSelf_.mmTile.dbL0c,
            hcommTilingDataSelf_.mmTile.mTailTile, hcommTilingDataSelf_.mmTile.nTailTile,
            hcommTilingDataSelf_.gatherLen);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, SetRawTilingData());
    SetRcsTilingData(MutableRCSTilingData());
    DoSplitMTiling(MutableRCSTilingData());
    // 通信配置放在切分之后：单轮通信量依赖切分结果 tileMValue_（对齐 alltoall 先切分后配置的时序）
    OP_TILING_CHECK(SetMc2Hcomm(MutableRCSTilingData()) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Fail to set Mc2Hcomm."),
                    return ge::GRAPH_FAILED);
    MC2_CHECK_LOG_RET(opName_, DoVersion2Tiling());
    DoAllGatherTiling(MutableRCSTilingData(), MutableMC2MatmulV3TileTilingData().tCubeTiling,
                      MutableMC2MatmulV3TailTilingData().tCubeTiling, allGatherMatmulTilingDataV2_->debugMode,
                      allGatherMatmulTilingDataV2_->dataType);
    // 非量化路径：cutRes 与 mc2InitTiling/mc2CcTiling 映射 apace tiling 结构（R13 起恒 APACE）
    OP_TILING_CHECK(DoApaceTiling() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Fail to do apace tiling."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void Mc2PrintMMV3TilingData(const std::string& opName, Mc2MatMulV3TilingData& tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mBaseTailSplitCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSplitCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

void AllGatherMatmulTilingV2::PrintAllTilingData()
{
    if (MutableRCSTilingData().rankID != 0) {
        return;
    }
    PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3LocalTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TileTilingData());
    if (MutableRCSTilingData().tailM <= 0) {
        return;
    }
    OP_LOGD(opName_, "Matmul has tail.");
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TailTilingData());
}

ge::graphStatus AllGatherMatmulTilingV2::PostTiling()
{
    // 非量化路径（R13 起唯一路径）：输出 hcommAllGatherMatmulTilingData（a2a PostTiling 输出范式）
#if MC2_DFX_ENABLE
    hcommTilingDataSelf_.dumpInfo.workspaceLayout = workspaceLayout_;
#endif
    auto* outTilingData = context_->GetTilingData<Apace::hcommAllGatherMatmulTilingData>();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Fail to get hcomm tiling data from context"),
                    return ge::GRAPH_FAILED);
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((tilingBufCap < sizeof(hcommTilingDataSelf_)),
                    OP_LOGE(opName_, "TilingBuffer capacity is too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(hcommTilingDataSelf_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &hcommTilingDataSelf_, sizeof(hcommTilingDataSelf_));
    if (ret != EOK) {
        OP_LOGE(opName_, "PostTiling: memcpy_s hcomm tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    // SetDataSize 取 max（注册容量按最大结构申请；补零至注册容量，V2-B6 方向核对 + a2a capacity CHECK 双兜底）
    const size_t registeredMaxTilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2);
    const size_t apaceTilingSize = sizeof(Apace::hcommAllGatherMatmulTilingData);
    const size_t finalSize = (registeredMaxTilingSize > apaceTilingSize) ? registeredMaxTilingSize : apaceTilingSize;
    OP_TILING_CHECK((finalSize % sizeof(uint64_t) != 0),
                    OP_LOGE(opName_, "Tiling data size[%zu] not aligned to 8.", finalSize), return ge::GRAPH_FAILED);
    // 容量防御 CHECK（a2a capacity CHECK 范式）：补齐 tilingBufCap∈[sizeof(hcomm), finalSize)
    // 区间的防御缺口——SetDataSize 声称 finalSize 而实际容量不足时框架侧读取越界，显式拒绝。
    // 正常配置（注册容量按最大结构申请，finalSize=1848 ≤ cap）不触发。
    OP_TILING_CHECK(
        (tilingBufCap < finalSize),
        OP_LOGE(opName_, "TilingBuffer capacity %zu is smaller than final tiling size %zu.", tilingBufCap, finalSize),
        return ge::GRAPH_FAILED);
    if ((tilingBufCap >= finalSize) && (finalSize > apaceTilingSize)) {
        errno_t zeroRet = memset_s(reinterpret_cast<uint8_t*>(outTilingData) + apaceTilingSize,
                                   tilingBufCap - apaceTilingSize, 0, finalSize - apaceTilingSize);
        if (zeroRet != EOK) {
            OP_LOGE(opName_, "PostTiling: memset_s hcomm tiling padding failed, ret=%d.", zeroRet);
            return ge::GRAPH_FAILED;
        }
    }
    context_->GetRawTilingData()->SetDataSize(finalSize);
    OP_LOGI(opName_, "Final apace hcomm tiling data size=%zu (v2=%zu, apace=%zu), context capacity size=%zu.",
            finalSize, registeredMaxTilingSize, apaceTilingSize, tilingBufCap);
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

CutResult AllGatherMatmulTilingV2::GetTilingResult()
{
    AllGatherMMFitBalanceTiling tileFormulate(args_, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    CutResult result = tileFormulate.GetTiling();

    bool isCcuMode = (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0);
    if (isCcuMode) {
        CommSizeInfo commSize{args_.kValue, args_.inputDtypeSize, args_.rankDim};
        AdjustCutResultForCCU(result, args_.mValue, commSize, opName_);
    } else {
        AdjustCutResultForAICPU(result, args_.mValue, tileFormulate.matmulPerf_.GetBaseM(), opName_);
    }

    return result;
}

ge::graphStatus AllGatherMatmulTilingV2::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg,
                                                          Mc2MMRegisterCfg& registerCfg,
                                                          Mc2MatMulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim - 1);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (args_.nValue != 0)
        if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
            return ge::GRAPH_FAILED;
        }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoVersion2Tiling()
{
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platformInfo."), return ge::GRAPH_FAILED);

    // 获取 compileInfo
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init Compile Info!");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatForm = platform_ascendc::PlatformAscendC(platformInfo);

    // 根据芯片型号获取策略模板
    platform_ascendc::SocVersion socVersion = ascendcPlatForm.GetSocVersion();
    NpuArch npuArch = ascendcPlatForm.GetCurNpuArch();

    std::vector<int32_t> priorities;
    MC2_CHECK_LOG_RET(opName_, mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch, priorities, opName_));

    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 计算 local 块 tiling
    Mc2MatmulHelper::Mc2MatmulTilingCfg localTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                       reinterpret_cast<const void*>(&mmV3Args_));
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(localTilingCfg, registerCfg, MutableMC2MatmulV3LocalTilingData()));

    // 计算 tile 块 tiling
    mmV3Args_.mValue = tileMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tileCnt);
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                      reinterpret_cast<const void*>(&mmV3Args_), tileMValue_);
    tileTilingCfg.SetCommCnt(MutableRCSTilingData().tileCnt);
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MatmulV3TileTilingData()));
    MutableMC2MatmulV3TileTilingData().tCubeTiling.M = (tileMValue_ * (args_.rankDim - 1));

    if (tailMValue_ > 0) {
        // 计算 tail 块 tiling
        mmV3Args_.mValue = tailMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tailCnt);
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                          reinterpret_cast<const void*>(&mmV3Args_), tailMValue_);
        tailTilingCfg.SetCommCnt(MutableRCSTilingData().tailCnt);
        MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tailTilingCfg, registerCfg, MutableMC2MatmulV3TailTilingData()));
        MutableMC2MatmulV3TailTilingData().tCubeTiling.M = (tailMValue_ * (args_.rankDim - 1));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::SetMc2Hcomm(Mc2Tiling::RCSTiling& rcsCfg)
{
    // Set hccl comm engine with comm_mode
    uint8_t commEngine = Mc2Comm::ENGINE_AICPU;
    if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
        commEngine = Mc2Comm::ENGINE_CCU_SCHED;
    }
    OP_LOGD(opName_, "Tiling SetMc2Hcom commMode_: %s", commMode_);
    OP_LOGD(opName_, "Tiling SetMc2Hcom commEngine: %d", commEngine);
    // 单轮通信量 = 每轮切分的 tileM × K × dtypeSize（对齐 alltoall 口径），算法表为空时回退默认算法
    uint64_t commDataBytes = tileMValue_ * args_.kValue * args_.inputDtypeSize;
    OP_LOGI(opName_, "[SetMc2Hcomm] commDataBytes=%llu, tileM=%llu, kValue=%llu, rankDim=%u, commEngine=%u",
            commDataBytes, tileMValue_, args_.kValue, args_.rankDim, commEngine);
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry* algoEntries = Mc2Tiling::GetAllGatherCommAlgoTable(algoCount);
    std::string algConfig =
        Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(opName_, group_, commEngine, commDataBytes, args_.rankDim,
                                                     algoEntries, algoCount, Mc2Tiling::ALLGATHER_DEFAULT_ALGO_NAME);
    OP_LOGI(opName_, "[SetMc2Hcomm] selected algConfig=%s, group=%s", algConfig.c_str(), group_);
    Mc2CcTilingConfig mc2CcTilingConfig(
        group_, static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER), algConfig, 0,
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)),
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
    mc2CcTilingConfig.SetCommEngine(commEngine);
    uint8_t skipBufferWindowCopy = (allGatherMatmulTilingDataV2_->param.gatherLen == 0) ?
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::CheckInput()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(SCALE_INV1);
    OP_TILING_CHECK(
        x1ScaleShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1Scale", "not nullptr",
                                              "If the dtype of x1 and x2 is fp16 or bf16, x1Scale must be nullptr"),
        return ge::GRAPH_FAILED);
    auto x2ScaleShape = context_->GetOptionalInputShape(SCALE_INV2);
    OP_TILING_CHECK(
        x2ScaleShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x2Scale", "not nullptr",
                                              "If the dtype of x1 and x2 is fp16 or bf16, x2Scale must be nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::GetWorkspaceSize()
{
    // 非量化路径（R13 起唯一路径）：workspace = libApi + gatherLen（storageA_ 在 DAV_3510 下 = gatherLen；
    // 不消费 v1 的 biasLen 预留——非量化 FP16/BF16 无 bias 扩展区需求）
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed."), return ge::GRAPH_FAILED);
    workspaceSize_ = libApiWorkSpaceSize_ + storageA_;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "[APACE] workspaces[0] size=%zu, libApi=%u, gatherLen(storageA)=%u.", workspaces[0],
            libApiWorkSpaceSize_, storageA_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AllGatherMatmulTilingV2::GetTilingKey() const
{
    uint8_t outputType = (outputIsFp8_) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);

    // Non-A5 platform must use AICPU mode
    uint8_t commMode = Mc2Comm::COMM_MODE_AICPU;
    if (npuArch_ == Ops::Base::DAV_3510) {
        if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
            commMode = Mc2Comm::COMM_MODE_CCU;
        }
    }
    // tiling key 6 维（R13：USING_APACE_IMPL 维度删除，非量化恒 APACE）——非量化 key 回落旧编码值
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputIsBf16Fp16_, args_.isBTrans, outputType, TPL_DEFAULT_MODE,
                                                  SCALE_TYPE_NOT_IS_MX, commMode);
    OP_LOGD(opName_, "AllGatherMatmulV2(apace), inputIsBf16Fp16_, args_.isBTrans, outputType, commMode: [%d,%d,%u,%u]",
            inputIsBf16Fp16_, args_.isBTrans, outputType, commMode);
    OP_LOGD(opName_, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

AllGatherMatmulTilingV2::AllGatherMatmulTilingV2(gert::TilingContext* context)
    : AllGatherMatmulTilingBase(context),
      allGatherMatmulTilingDataV2_(&allGatherMatmulTilingDataV2Self_)
{}
// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherMatmulTilingV2,
                                   static_cast<int32_t>(Ops::Base::DAV_3510), 0);
} // namespace optiling
