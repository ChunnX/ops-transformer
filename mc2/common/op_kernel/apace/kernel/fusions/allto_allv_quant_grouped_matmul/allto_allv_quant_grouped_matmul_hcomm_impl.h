/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_hcomm_impl.h
 * \brief AllToAllv + MX 量化 GroupedMatMul 融合实现（hcomm / HCCL 集合通信版本）
 *
 * 面向 AlltoAllvQuantGroupedMatMul 算子的 apace 切换路径（参照 AlltoAllMatmul 的
 * “MX 量化切 apace”模式）：
 *   - 通信：AIC 核内通过 Hccl<>::AlltoAllV 下发变长通信（量化 x 与 scale 各一个句柄），
 *     通信输出落在 workspace 的 rank-major 接收区；
 *   - 计算：等待通信完成后，以“段式分组”直接消费 rank-major 布局——
 *     每个 (expert, srcRank) 段为独立计算分组（段内 A/scaleA/C 连续），
 *     免去通信后的独立 permute；
 *   - 计算 kernel 复用 apace 分组量化 Matmul（Blaze BlockMmad<GroupedMatmulWithScaleMx>）。
 *
 * 本实现为纯 AIC（cube-only）路径：若 kernel 以 MIX 模式启动，Init/Run 内部有
 * AIC 守卫，AIV 核直接返回。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/qgmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "allto_allv_quant_grouped_matmul_tiling_data.h"

namespace Apace {

template <typename T>
constexpr uint64_t GetA2avHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_INT8; // fp4 打包存储，按 int8 字节流通信
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <typename AType, typename BType, typename CType, bool TransB, AscendC::HcclServerType ServerType>
class AllToAllvQuantGmmHcommImpl {
public:
    explicit __aicore__ inline AllToAllvQuantGmmHcommImpl(hcommA2avQuantGmmTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline ~AllToAllvQuantGmmHcommImpl() = default;

    // Layout 定义：A/C 行主序；B 随 TransB；scale 布局与算子规格一致
    static constexpr int32_t SCALE_C0 = 2;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn,
                                                         AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using LayoutScaleA = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn,
                                                        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Std::conditional_t<
        TransB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using GmmKernel = Apace::Kernel::QuantGroupedMatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;

    static constexpr uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t xHcclDataType_ = GetA2avHcclDataType<AType>();

    /*!
     * \brief 初始化并下发通信任务（AIC only）
     *
     * \param[in] tilingGM tiling buffer 首地址：GET_TILING_DATA_WITH_STRUCT 得到的是本地拷贝，
     *            段表（segList/segAOff/segCOff）以 __gm__ 指针按偏移直接引用 tiling buffer
     */
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightGM,
                                GM_ADDR gmmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR userWorkspaceGM, GM_ADDR tilingGM)
    {
        if ASCEND_IS_AIV {
            return;
        }
        xGM_ = gmmxGM;
        xScaleGM_ = gmmxScaleGM;
        weightGM_ = gmmWeightGM;
        weightScaleGM_ = gmmWeightScaleGM;
        yGM_ = gmmyGM;
        dataRecvGM_ = userWorkspaceGM + tilingData_->dataRecvOffset;
        scaleRecvGM_ = userWorkspaceGM + tilingData_->scaleRecvOffset;
        segListGm_ = reinterpret_cast<__gm__ const uint32_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, segList));
        segAOffGm_ =
            reinterpret_cast<__gm__ const uint32_t *>(tilingGM + offsetof(hcommA2avQuantGmmTilingData, segAOff));
        segCOffGm_ =
            reinterpret_cast<__gm__ const uint32_t *>(tilingGM + offsetof(hcommA2avQuantGmmTilingData, segCOff));

        hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(hcommA2avQuantGmmTilingData, mc2CcTiling)));

        // AlltoAllV 计数/偏移表复制到本地（避免直接依赖 tiling GM 地址空间约束）
        const uint32_t rankSize = tilingData_->rankSize;
        for (uint32_t i = 0; i < rankSize; ++i) {
            dataSendCnt_[i] = tilingData_->dataSendCnt[i];
            dataSendOff_[i] = tilingData_->dataSendOff[i];
            dataRecvCnt_[i] = tilingData_->dataRecvCnt[i];
            dataRecvOff_[i] = tilingData_->dataRecvOff[i];
            scaleSendCnt_[i] = tilingData_->scaleSendCnt[i];
            scaleSendOff_[i] = tilingData_->scaleSendOff[i];
            scaleRecvCnt_[i] = tilingData_->scaleRecvCnt[i];
            scaleRecvOff_[i] = tilingData_->scaleRecvOff[i];
        }

        scaleHandle_ = hccl_.template AlltoAllV<true>(
            xScaleGM_, scaleSendCnt_, scaleSendOff_, static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_),
            scaleRecvGM_, scaleRecvCnt_, scaleRecvOff_, static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_));
        dataHandle_ = hccl_.template AlltoAllV<true>(
            xGM_, dataSendCnt_, dataSendOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_), dataRecvGM_,
            dataRecvCnt_, dataRecvOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_));
    }

    /*!
     * \brief 等待通信完成并执行段式分组量化 Matmul（AIC only）
     */
    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            return;
        }
        hccl_.Wait(scaleHandle_);
        hccl_.Wait(dataHandle_);

        typename GmmKernel::GMMTiling gmm;
        gmm.groupNum = tilingData_->segNum;
        gmm.n = tilingData_->n;
        gmm.k = tilingData_->k;
        gmm.baseM = tilingData_->baseM;
        gmm.baseN = tilingData_->baseN;
        gmm.baseK = tilingData_->baseK;
        gmm.kAL1 = tilingData_->kAL1;
        gmm.kBL1 = tilingData_->kBL1;
        gmm.scaleKAL1 = tilingData_->scaleKAL1;
        gmm.isBias = 0;
        gmm.dbL0C = 2;
        gmm.l1BufferStage = 2;

        typename GmmKernel::Params params;
        params.aBasePtr = reinterpret_cast<__gm__ AType *>(dataRecvGM_);
        params.scaleABasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(scaleRecvGM_);
        params.bBasePtr = reinterpret_cast<__gm__ BType *>(weightGM_);
        params.scaleBBasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(weightScaleGM_);
        params.cBasePtr = reinterpret_cast<__gm__ CType *>(yGM_);
        params.groupList = segListGm_;
        params.segAOff = segAOffGm_;
        params.segCOff = segCOffGm_;
        params.segW = tilingData_->rankSize;
        params.gmm = gmm;
        gmmKernel_(params);

        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    hcommA2avQuantGmmTilingData *tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    GmmKernel gmmKernel_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHandle_{0};

    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR dataRecvGM_{nullptr};
    GM_ADDR scaleRecvGM_{nullptr};
    // 段表 __gm__ 指针（引用 tiling buffer 原始地址，避免本地拷贝地址空间问题）
    __gm__ const uint32_t *segListGm_{nullptr};
    __gm__ const uint32_t *segAOffGm_{nullptr};
    __gm__ const uint32_t *segCOffGm_{nullptr};

    // AlltoAllV 本地参数表（元素计）
    uint64_t dataSendCnt_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataSendOff_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvCnt_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvOff_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendCnt_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendOff_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvCnt_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvOff_[A2AV_GMM_MAX_RANK_NUM] = {0};
};

} // namespace Apace
