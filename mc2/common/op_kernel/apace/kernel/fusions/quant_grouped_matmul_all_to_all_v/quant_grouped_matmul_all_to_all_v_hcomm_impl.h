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
 * \file quant_grouped_matmul_all_to_all_v_hcomm_impl.h
 * \brief apace 库：QuantGroupedMatMulAlltoAllv MX(FP4/FP8) 路径 AIC 核内通信+计算融合实现。
 *
 * kernel 以 MIX_AIC_1_2 任务类型部署，GMM 计算（复用 QuantGroupedMatmulMx /
 * Blaze GroupedMatmulMxKernel）与 AlltoAllV 通信均在 AIC 核执行，AIV 空转退出——
 * Blaze（cube 模板）与 AIV 侧 KFC 通信引擎在同一 kernel 实例中共存时，AIV 通信任务
 * 参数校验会失败，故通信改由 AIC 核内 Hccl 提交；AIV 误入 Blaze/Hccl 调用会死锁，
 * 接口保留 AIV 守卫。
 *
 * 工作流（后通信：GMM 计算 → AlltoAllV 发送）：
 *   1. AIC 核：hccl_.InitV2(GetHcclContext<0>(), &hcclInitTiling) + SetCcTilingV2(a2avCcTiling)
 *   2. AIC 核：GMM 计算（QuantGroupedMatmulMx）写出 gmmOutputGM（批内 rank-major 布局）
 *   3. AIC 核：核间同步后，核 0 下发 AlltoAllV（sendBuf=gmmOutputGM, recvBuf=yGM），
 *      核 0 Wait(handle)，其余核经 SyncAll 跟随
 *   4. End：核间同步后 hccl_.Finalize()
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__

#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/a2av_gmm_utils.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_comm_params.h"
#include "quant_grouped_matmul_all_to_all_v_impl.h"

namespace Apace {

using namespace AscendC;
using namespace MC2KernelTemplate;

template <int commMode>
struct HcclServerTypeSelector {
    using type = AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template <>
struct HcclServerTypeSelector<1> {
    using type = AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, int CommMode>
class QuantGmmA2avHcommImpl {
public:
    using HcclType = typename HcclServerTypeSelector<CommMode>::type;
    using ComputeOp = QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                           aTrans, bTrans, isLocal, false>;

    __aicore__ inline QuantGmmA2avHcommImpl() {}

    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM,
                                GM_ADDR gmmOutputGM, GM_ADDR yGM, GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, MC2_TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe)
    {
        gmmOutputGM_ = gmmOutputGM;
        yGM_ = yGM;
        tilingData_ = tilingData;
        hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->hcclA2avTiling.hcclInitTiling));
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, hcclA2avTiling)) +
                            static_cast<uint64_t>(offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling)));
        computeOp_.Init(xGM, weightGM, xScaleGM, weightScaleGM, gmmOutputGM, workspaceGM, tilingData, gmmTilingData,
                        gmmArrayAddrIn, tPipe);
    }

    __aicore__ inline void Process(uint32_t startExpertIdx, uint32_t expertNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        computeOp_.Process(startExpertIdx, expertNum);
        // AIC 核间同步：确保所有核完成 GMM 计算后才下发通信
        AscendC::SyncAll();
        if (AscendC::GetBlockIdx() == 0) {
            CommitAlltoAllV(startExpertIdx, expertNum);
            WaitAlltoAllV();
        }
        AscendC::SyncAll();
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            return;
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    __aicore__ inline AscendC::HcclDataType GetHcclDataType()
    {
        if constexpr (AscendC::IsSameType<yType, bfloat16_t>::value) {
            return AscendC::HCCL_DATA_TYPE_BFP16;
        } else {
            return AscendC::HCCL_DATA_TYPE_FP16;
        }
    }

    __aicore__ inline void CommitAlltoAllV(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const uint32_t e = static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
        const uint32_t rankDim = static_cast<uint32_t>(tilingData_->taskTilingInfo.epWorldSize);
        const uint64_t n1 = tilingData_->taskTilingInfo.N1;
        const int32_t *sendCounts = tilingData_->taskTilingInfo.sendCnt;
        const int32_t *recvCounts = tilingData_->taskTilingInfo.recvCnt;

        // 本批 send/recv 计数（元素单位，axis=N1）
        for (uint32_t i = 0U; i < rankDim; i++) {
            commParams_.sendCnt[i] = 0UL;
            commParams_.recvCnt[i] = 0UL;
            for (uint32_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                commParams_.sendCnt[i] += static_cast<uint64_t>(sendCounts[expertIdx + i * e]) * n1;
                commParams_.recvCnt[i] += static_cast<uint64_t>(recvCounts[expertIdx + i * e]) * n1;
            }
        }

        // send 偏移：GMM 输出为批间串接、批内 rank-major 布局
        // （本批起点 = 批前全部 token 累计；批内 rank 段首 = 本批起点 + 前 rank 段累计）
        uint64_t batchStartOffset = 0UL;
        for (uint32_t i = 0U; i < startExpertIdx; i++) {
            for (uint32_t j = 0U; j < rankDim; j++) {
                batchStartOffset += static_cast<uint64_t>(sendCounts[i + j * e]);
            }
        }
        batchStartOffset *= n1;
        commParams_.sendOffset[0] = batchStartOffset;
        uint64_t rankRunningOffset = 0UL;
        for (uint32_t i = 1U; i < rankDim; i++) {
            for (uint32_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
                rankRunningOffset += static_cast<uint64_t>(sendCounts[expertIdx + (i - 1) * e]) * n1;
            }
            commParams_.sendOffset[i] = batchStartOffset + rankRunningOffset;
        }

        // recv 偏移：y 按 recvCounts 源 rank 聚合顺序（rank-major）。
        // 索引 startExpertIdx + (i-1)*e + j 跨过 rank i-1 的本批后段专家与 rank i 的批前专家，
        // 累加后恰好折叠为 rank 段基 + 批前前缀（伸缩和恒等式）
        commParams_.recvOffset[0] = 0UL;
        for (uint32_t i = 0U; i < startExpertIdx; i++) {
            commParams_.recvOffset[0] += static_cast<uint64_t>(recvCounts[i]) * n1;
        }
        for (uint32_t i = 1U; i < rankDim; i++) {
            commParams_.recvOffset[i] = commParams_.recvOffset[i - 1];
            for (uint32_t j = 0U; j < e; j++) {
                commParams_.recvOffset[i] += static_cast<uint64_t>(recvCounts[startExpertIdx + (i - 1) * e + j]) * n1;
            }
        }

        alltoAllvHandle_ = hccl_.template AlltoAllV<true>(
            gmmOutputGM_, commParams_.sendCnt, commParams_.sendOffset, GetHcclDataType(), yGM_, commParams_.recvCnt,
            commParams_.recvOffset, GetHcclDataType(), static_cast<uint8_t>(1));
    }

    __aicore__ inline void WaitAlltoAllV()
    {
        if (alltoAllvHandle_ >= 0) {
            hccl_.Wait(alltoAllvHandle_);
        }
    }

    HcclType hccl_;
    ComputeOp computeOp_;
    A2avCommParams commParams_;
    AscendC::HcclHandle alltoAllvHandle_{0};
    GM_ADDR yGM_{nullptr};
    GM_ADDR gmmOutputGM_{nullptr};
    const TilingDataType *tilingData_{nullptr};
};

} // namespace Apace
#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
