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
 * \file matmul_all_reduce_empty_tensor_k_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H
#define MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../common.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_tiling_struct_ar35.h"

constexpr uint32_t EMPTY_TENSOR_BIAS_UB_FACTOR = 1;

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename YType, int commMode>
class MatmulAllReduceEmptyTensorKGeneral
{
public:
    __aicore__ inline MatmulAllReduceEmptyTensorKGeneral(MC2GmAddrs* addrs, MC2TilingHeader* tilDat, TPipe* tPipe2)
        : addrs_(addrs), tPipe2_(tPipe2)
    {
        param_ = &(tilDat->param);
        cOffset_ = (uint64_t)param_->rankN * (uint64_t)param_->rankM;
#if (defined MC2_WEIGHT_QUANT) || (defined WEIGHT_F8)
        biasFl_ =
            (((Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData*)tilDat)->tileMmASTiling.matmulTiling.isBias != 0U);
#else
        biasFl_ = (((Mc2Tiling::MatmulAllReduce910TilingDataA5*)tilDat)->mC2Mmv3TileTilingData.tCubeTiling.isBias != 0U);
#endif
    }

    __aicore__ inline void Init()
    {
        hccl_.InitV2(GetHcclContext<0>(), tilDat_);
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
        notifyFl_ = (GetBlockIdx() == 0);
        if (notifyFl_) {
            hcclHdl2_ = hccl_.template AllReduce(
                addrs_->cGM, addrs_->outputGM, cOffset_, HCCL_DATA_TYPE, AscendC::HCCL_REDUCE_SUM, 1U);
        }
    }

    __aicore__ inline void Process()
    {
        // InitOutput value to 0. Supports Dtype: half/int32_t/float/uint32_t/
        uint64_t cSz = cOffset_ * sizeof(YType);
        cSz /= sizeof(half);
        GlobalTensor<half> cGl;
        cGl.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(addrs_->cGM), cSz * sizeof(half));
        InitOutput<half>(cGl, cSz, static_cast<half>(0.0));
        if (biasFl_ || param_->isAdd) {
            SyncAll();
            if (biasFl_) {
                ProcessBias();
            }
            if (param_->isAdd) {
                ProcessAdd();
            }
        }

        SyncAll();
        if (notifyFl_) {
            hccl_.Commit(hcclHdl2_);
            hccl_.Wait(hcclHdl2_);
            hccl_.Finalize();
        }
    }

private:
    __aicore__ inline void ProcessBias()
    {
        // Init buffer
        tPipe2_->Reset();
        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe2_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        // DataCopy from bias to cGM 32B aligned
        // total cGM size
        uint64_t cSz = cOffset_ * sizeof(YType);
        GlobalTensor<YType> cGlBias;
        cGlBias.SetGlobalBuffer(reinterpret_cast<__gm__ YType*>(addrs_->cGM), cSz);
        // init ub for datacopy
        LocalTensor<YType> bias = tmpBuf.Get<YType>();
        GlobalTensor<YType> biasGlobal;
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ YType*>(addrs_->biasGM));
        TBuffAddr buffAddr;
        buffAddr.logicPos = (uint8_t)QuePosition::VECCALC;
        // max calc num
        // when n is large, split N to make sure that ub used by each datacopy intrinsic does not exceed the size of ub
        int64_t ubAlignCntSize = UB_ALIGN_SIZE / sizeof(YType);
        int64_t biasUbAlignSize = (param_->rankN / ubAlignCntSize) * ubAlignCntSize;
        int64_t maxTmpBufCount =
            TOTAL_UB_SIZE / EMPTY_TENSOR_BIAS_UB_FACTOR / sizeof(YType) / ubAlignCntSize * ubAlignCntSize;
        int64_t biasSize = param_->rankN;
        int64_t biasTailSize = biasSize - biasUbAlignSize;
        for (uint64_t offsetRankN = 0; offsetRankN < biasUbAlignSize; offsetRankN += maxTmpBufCount) {
            uint64_t calCount =
                (biasUbAlignSize - offsetRankN) > maxTmpBufCount ? maxTmpBufCount : (biasUbAlignSize - offsetRankN);
            DataCopy(bias, biasGlobal[offsetRankN], calCount);
            SyncAll();
            // offset M*N
            for (uint32_t idx = 0; idx < param_->rankM; ++i) {
                uint64_t offsetRankM = i * param_->rankN + offsetRankN;
                DataCopy(cGlBias[offsetRankM], bias, calCount);
                SyncAll();
            }
        }
        // DataCopy from bias to cGM 32B aligned tail
        if (biasTailSize > 0) {
            DataCopy(bias, biasGlobal[biasUbAlignSize], ubAlignCntSize);
            SyncAll();
            for (uint32_t idx = 0; idx < param_->rankM; ++i) {
                uint64_t offsetDst = i * param_->rankN + biasUbAlignSize;
                DataCopyPad(
                    cGlBias[offsetDst], bias, {1, static_cast<uint16_t>(biasTailSize * sizeof(YType)), 0, 0});
            }
            SyncAll();
        }
    }

    __aicore__ inline void ProcessAdd()
    {
        const size_t cOffset = param_->rankN * sizeof(YType);
        for (uint32_t idx = 0; idx < param_->rankM; ++i) {
            MatmulAllReduceAddX3Kernel<YType>(
                addrs_->cGM, addrs_->addGM, cOffset / sizeof(YType), param_->addX3UbCnt, tPipe2_);
            addrs_->cGM += cOffset;
            addrs_->addGM += cOffset;
            SyncAll();
        }
    }

    MC2GmAddrs* addrs_;
    Mc2Tiling::RCSTiling* param_;
    MC2TilingHeader* tilDat_;
    bool biasFl_{false};
    uint64_t cOffset_;
    TPipe* tPipe2_;
    bool notifyFl_{false};
    typename HcclTypeSelector<commMode>::type hccl_;
    AscendC::HcclHandle hcclHdl2_;
};

#if (defined MC2_WEIGHT_QUANT) || (defined WEIGHT_F8)
#define GET_TILING_DATA_FOR_EMPTY_TENSOR() \
    GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData, tilDat, tilingGM)
#else
#define GET_TILING_DATA_FOR_EMPTY_TENSOR() \
    GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::MatmulAllReduce910TilingDataA5, tilDat, tilingGM)
#endif

#define INVOKE_MC2_EMPTY_TENSOR_OP_IMPL(commMode)                                                      \
    do {                                                                                               \
        GET_TILING_DATA_FOR_EMPTY_TENSOR();                                                            \
        MC2GmAddrs addrs = {nullptr, nullptr, biasGM, addGM, cGM, nullptr, cGM};                       \
        MatmulAllReduceEmptyTensorKGeneral<DTYPE_Y, commMode> op(&addrs, (MC2TilingHeader*)&tilDat, &tPipe2); \
        op.Init();                                                                                     \
        op.Process();                                                                                  \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H