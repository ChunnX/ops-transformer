/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_hcomm_impl.h
 * \brief AllGather + Matmul fusion kernel implementation (Hcomm/CCU variant, non-quant FP16/BF16)
 *
 * 底本 apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_hcomm_impl.h（去 scale 通道）：
 *   - 去 scaleHandle/scale 区/scaleK* 系列/WaitTile 的 scale 等待分支；
 *   - dataHead+dataTail 双句柄 repeat 批量保留（AllGather<true> 批量句柄 + repeat，
 *     handle 数与 tile 数解耦，repeat 上限 255 ≫ tileCnt≤16）；
 *   - GetHcclDataType 扩 FP16→HCCL_DATA_TYPE_FP16(3)、bfloat16_t→HCCL_DATA_TYPE_BFP16(11)；
 *   - Init 8→6 参（去 aScaleGM/bScaleGM）；模板 IsMxFp4→TransB（LayoutB 双布局透传给内核）；
 *   - HCCL 生命周期（InitV2/SetCcTilingV2/GetRankId/GetRankDim/Finalize）与 gatherLen 语义原样保留；
 *   - MIX_AIC_1_2 入口核型下 AIV 参与生命周期注册（InitV2 等全核执行，P10 回归 v1/底本契约），
 *     不参与通信提交（CommitAllGather）与计算（Run 首行守卫）。
 *
 * Init():
 *   全核(AIC+AIV): hccl_.InitV2()/SetCcTilingV2()/GetRankId()/GetRankDim()（生命周期注册）。
 *   AIC: 批量下发 AllGather<true>（dataHead + dataTail），
 *        GetCommPolicy().state_ = &commState_ 绑定 WaitPolicy。
 *
 * Run():
 *   AIC: 执行 FragmentTensor kernel，所有核 WaitTile 完成后再统一 Finalize → hccl_.Finalize。
 *        kernel 内逐 tile commPolicy_.WaitTile(dependTileIdx) → state_->hccl_.Wait(handle)。
 */

#pragma once

#include "include/tensor_api/tensor.h"
#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/kernel/matmul/batch_matmul/all_gather_bmm_kernel.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace Apace {

using namespace AscendC;

template <typename T>
constexpr uint64_t GetHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        return AscendC::HCCL_DATA_TYPE_FP16;
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <AscendC::HcclServerType ServerType>
struct HcommCommState {
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t tileCnt_{0};
};

template <AscendC::HcclServerType ServerType>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType> *state_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        // dependTileIdx=0 (HEAD): 本 rank 数据，无需等待通信
        if (tileIdx == 0) {
            return;
        }
        // dependTileIdx=1..tileCnt (MAIN): 等 data head（repeat=tileCnt，每次 Wait 消费一轮）
        if (tileIdx <= state_->tileCnt_) {
            state_->hccl_.Wait(state_->dataHeadHandle_);
        } else {
            // dependTileIdx>tileCnt (TAIL): 等 data tail
            state_->hccl_.Wait(state_->dataTailHandle_);
        }
    }
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
class AllGatherMatmulHcommImpl {
public:
    explicit __aicore__ inline AllGatherMatmulHcommImpl(Apace::hcommAllGatherMatmulTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
                                GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    using MatmulKernelImpl = AllGatherBmmKernel<AType, BType, CType, HcommCommWaitPolicy<ServerType>, TransB>;
    using KernelParams = typename MatmulKernelImpl::Params;

    MatmulKernelImpl matmulKernelImpl_;
    HcommCommState<ServerType> commState_;
    Mc2Kernel::OpStateDump opStateDump_;

private:
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CommitAllGather();
    __aicore__ inline void SetupKernelParams(KernelParams &params);

    Apace::hcommAllGatherMatmulTilingData *tilingData_;

    GM_ADDR aGM_{};
    GM_ADDR bGM_{};
    GM_ADDR biasGM_{};
    GM_ADDR cGM_{};
    GM_ADDR workspaceGM_{};
    GM_ADDR gatherDataAddr_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t kForComm_{}; // 通信用 K 元素数（非量化无 FP4 打包，恒 = k）
    uint64_t dataBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t dataRegionBytes_{};

    static constexpr uint32_t kPaddingLength = 16;
    static constexpr uint64_t dataDataType_ = GetHcclDataType<AType>();
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
__aicore__ inline void AllGatherMatmulHcommImpl<AType, BType, CType, ServerType, TransB>::InitBaseParams()
{
    const auto &ct = tilingData_->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = tilingData_->mmTile.k;
    n_ = tilingData_->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt + ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    dataBytesPerMRow_ = static_cast<uint64_t>(k_) * sizeof(AType);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
    kForComm_ = static_cast<uint64_t>(k_);
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
__aicore__ inline void AllGatherMatmulHcommImpl<AType, BType, CType, ServerType, TransB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM)
{
    // 【P10·契约对齐】HCCL 上下文注册改为全核执行（含 AIV）后再按核型分流：
    // v1（all_gather_matmul_fp16_bf16.h Init 全核 hccl_.InitV2/SetCcTilingV2）与 apace 底本
    // （all_gather_mx_matmul_hcomm_impl.h 无 AIV 守卫）均在全部 block（AIC+AIV）注册通信
    // 上下文；v2 原 AIV 首行早退为无先例偏离，本处回归 v1/底本契约。AIV 仍不参与通信提交
    // （CommitAllGather）与计算（Run 守卫不变），仅承担生命周期注册。
    // 注：该对齐未消除 L1-133 无 host 日志快路径下的 Wait 永等问题（507014，APACE 分支特有、
    // host 时序双稳；v1 同路径不受影响），该问题登记为 P10-R1 已知事项回传（见 3.4 报告 v1.5）。
    commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    commState_.hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(Apace::hcommAllGatherMatmulTilingData, mc2CcTiling)));
    rankId_ = commState_.hccl_.GetRankId();
    rankSize_ = commState_.hccl_.GetRankDim();
    if ASCEND_IS_AIV {
        return;
    }
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;

    InitBaseParams();

    // Gather buffer: data 用 gatherOut（当可用）或 workspace（去 scale 区，data 区起始即 workspace 头部）
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    if (tilingData_->gatherLen == 0 && gatherOut != 0) {
        gatherDataAddr_ = gatherOut;
    } else {
        gatherDataAddr_ = workspaceGM_;
    }

    // Submit AllGather (non-blocking <true>)
    CommitAllGather();

#if MC2_DFX_ENABLE
    opStateDump_.Init(workspaceGM_, &tilingData_->dumpInfo.workspaceLayout, tilingData_->mmTile.usedCoreNum);
#endif
    // Bind CommPolicy to CommState
    matmulKernelImpl_.GetCommPolicy().state_ = &commState_;
    commState_.tileCnt_ = tileCnt_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
__aicore__ inline void AllGatherMatmulHcommImpl<AType, BType, CType, ServerType, TransB>::CommitAllGather()
{
    // Data Head: 批量提交所有 head tile（repeat=tileCnt）；纯尾块路径（tileCnt=0）跳过
    // count/stride 用 kForComm_（非量化恒 = k），地址偏移用 dataBytesPerMRow_
    if (tileCnt_ > 0) {
        commState_.dataHeadHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_, gatherDataAddr_, static_cast<uint64_t>(tileM_) * kForComm_,
            static_cast<AscendC::HcclDataType>(dataDataType_), static_cast<uint64_t>(m_) * kForComm_,
            static_cast<uint8_t>(tileCnt_));
    }

    // Data Tail: 批量提交所有 tail tile（repeat=tailCnt）
    if (tailCnt_ > 0) {
        commState_.dataTailHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_ + headRows_ * dataBytesPerMRow_, gatherDataAddr_ + headRows_ * dataBytesPerMRow_,
            static_cast<uint64_t>(tailM_) * kForComm_, static_cast<AscendC::HcclDataType>(dataDataType_),
            static_cast<uint64_t>(m_) * kForComm_, static_cast<uint8_t>(tailCnt_));
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
__aicore__ inline void AllGatherMatmulHcommImpl<AType, BType, CType, ServerType, TransB>::SetupKernelParams(
    KernelParams &params)
{
    params.mmTile = &(tilingData_->mmTile);
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN, tilingData_->mmTile.baseK,
                         tilingData_->mmTile.dbL0c};
    params.fragParams = {tileCnt_,
                         tileM_,
                         tailCnt_,
                         tailM_,
                         paddedTailM_,
                         commTurn_,
                         headRows_,
                         rankId_,
                         rankSize_,
                         m_,
                         static_cast<uint64_t>(k_),
                         static_cast<uint64_t>(n_)};
    params.aGM = aGM_;
    params.bGM = bGM_;
    params.biasGM = biasGM_;
    params.isBias = (biasGM_ != nullptr); // bias optional 输入缺失时框架传 nullptr（同 gatherOut 判空惯例）
    params.cGM = cGM_;
    params.winDataBase = gatherDataAddr_;
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransB>
__aicore__ inline void AllGatherMatmulHcommImpl<AType, BType, CType, ServerType, TransB>::Run()
{
    // MIX_AIC_1_2 入口核型：AIV 空转
    if ASCEND_IS_AIV {
        return;
    }
    KernelParams params;
    SetupKernelParams(params);
    matmulKernelImpl_(params, opStateDump_);
    commState_.hccl_.Finalize();
}

} // namespace Apace
