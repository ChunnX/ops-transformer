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
 * \file moe_tutel_combine_x_kernel.h
 * \brief MoeTutelCombineX arch35(Ascend950 / regbase) 主模板。
 *
 * x_grad[s, :] = sum_k valid(k,s) ? gates[k,s] * y_grad[indices[k,s], locations[k,s], :] : 0
 * valid(k,s)   = 0 <= indices[k,s] < E && 0 <= locations[k,s] < C
 *
 * 切分：token 独占输出（核间不重叠、无原子累加）→ 结果逐位确定。
 *       H 按 hFactor 分块（hFactor == hidden 时即整行全载），k 按 kFactor 分块，
 *       token 按 rowFactor 成块，路由片一次搬 [kFactor, rowFactor] 的二维片。
 */

#ifndef MOE_TUTEL_COMBINE_X_KERNEL_H_
#define MOE_TUTEL_COMBINE_X_KERNEL_H_

#include "moe_tutel_combine_x_base.h"

namespace MoeTutelCombineXRegbase {
using namespace AscendC;

template <typename T>
class MoeTutelCombineXKernel {
public:
    __aicore__ inline MoeTutelCombineXKernel() {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR xGrad,
                                const TilingT *tiling, TPipe *pipe)
    {
        expertNum_ = tiling->expertNum;
        capacity_ = tiling->capacity;
        hidden_ = tiling->hidden;
        topK_ = tiling->topK;
        samples_ = tiling->samples;
        usedCoreNum_ = tiling->usedCoreNum;
        rowOfFormerBlock_ = tiling->rowOfFormerBlock;
        rowOfTailBlock_ = tiling->rowOfTailBlock;
        rowFactor_ = tiling->rowFactor;
        kFactor_ = tiling->kFactor;
        hFactor_ = tiling->hFactor;
        hChunkAlignT_ = tiling->hChunkAlignT;
        hChunkAlignF32_ = tiling->hChunkAlignF32;
        routeStride_ = tiling->routeStride;

        yGradGm_.SetGlobalBuffer((__gm__ T *)yGrad);
        gatesGm_.SetGlobalBuffer((__gm__ T *)gates);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t *)indices);
        locationsGm_.SetGlobalBuffer((__gm__ int32_t *)locations);
        xGradGm_.SetGlobalBuffer((__gm__ T *)xGrad);

        // 与 op_host/arch35/moe_tutel_combine_x_tiling.cpp 的 CalcUbBytes() 逐项对应，改一处必须改另一处。
        pipe->InitBuffer(outQue_, BUFFER_NUM, rowFactor_ * hChunkAlignT_ * sizeof(T));
        pipe->InitBuffer(rowQue_, BUFFER_NUM, kFactor_ * hChunkAlignT_ * sizeof(T));
        pipe->InitBuffer(idxBuf_, kFactor_ * routeStride_ * sizeof(int32_t));
        pipe->InitBuffer(locBuf_, kFactor_ * routeStride_ * sizeof(int32_t));
        pipe->InitBuffer(gateTBuf_, kFactor_ * routeStride_ * sizeof(T));
        pipe->InitBuffer(gateCompactBuf_, Ops::Base::CeilAlign<int64_t>(kFactor_, FP32_PER_BLOCK) * sizeof(float));
        if constexpr (!IsSameType<T, float>::value) {
            pipe->InitBuffer(accBuf_, rowFactor_ * hChunkAlignF32_ * sizeof(float));
            pipe->InitBuffer(gateF32Buf_, kFactor_ * routeStride_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= usedCoreNum_) {
            return;
        }
        const int64_t tokenBase = blockIdx * rowOfFormerBlock_;
        const int64_t tokenCnt = (blockIdx == usedCoreNum_ - 1) ? rowOfTailBlock_ : rowOfFormerBlock_;
        // S==0 / H==0 时 tokenCnt 或 hidden_ 为 0，循环直接不进，输出没有元素。
        for (int64_t rowOuter = 0; rowOuter < tokenCnt; rowOuter += rowFactor_) {
            const int64_t curRow = (tokenCnt - rowOuter < rowFactor_) ? (tokenCnt - rowOuter) : rowFactor_;
            for (int64_t hOuter = 0; hOuter < hidden_; hOuter += hFactor_) {
                const int64_t curH = (hidden_ - hOuter < hFactor_) ? (hidden_ - hOuter) : hFactor_;
                ProcessBlock(tokenBase + rowOuter, curRow, hOuter, curH);
            }
        }
    }

private:
    __aicore__ inline void ProcessBlock(int64_t tokenStart, int64_t curRow, int64_t hOffset, int64_t curH)
    {
        LocalTensor<T> outLocal = outQue_.template AllocTensor<T>();
        LocalTensor<float> accLocal;
        int64_t accStride = 0;
        if constexpr (IsSameType<T, float>::value) {
            // fp32 时累加器就是输出缓冲本身，省掉一次 UB 搬运；两者对齐长度相同。
            accLocal = outLocal.template ReinterpretCast<float>();
            accStride = hChunkAlignT_;
        } else {
            accLocal = accBuf_.template Get<float>();
            accStride = hChunkAlignF32_;
        }
        // INV-2：累加器无条件从 fp32 0 起累加。
        Duplicate(accLocal, 0.0f, static_cast<int32_t>(curRow * accStride));

        for (int64_t kOuter = 0; kOuter < topK_; kOuter += kFactor_) {
            const int64_t curK = (topK_ - kOuter < kFactor_) ? (topK_ - kOuter) : kFactor_;
            LoadRouteTile(kOuter, tokenStart, curK, curRow);
            for (int64_t r = 0; r < curRow; r++) {
                GatherAndAccumulate(accLocal[r * accStride], curK, r, hOffset, curH);
            }
            // GatherAndAccumulate 用 Scalar 读取复用的 route TBuf。即使最后一行 validK==0、
            // 没有触发 S_V，也必须等所有 Scalar 读完成，才能让下一轮 MTE2 覆写 route TBuf。
            event_t sToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(sToMte2);
            WaitFlag<HardEvent::S_MTE2>(sToMte2);
        }

        if constexpr (!IsSameType<T, float>::value) {
            for (int64_t r = 0; r < curRow; r++) {
                // INV-7：输出 dtype 的 cast 全程只发生这一次。
                Cast(outLocal[r * hChunkAlignT_], accLocal[r * accStride], RoundMode::CAST_RINT,
                     static_cast<int32_t>(curH));
            }
        }
        outQue_.EnQue(outLocal);
        outLocal = outQue_.template DeQue<T>();

        // UB 行跨距 hChunkAlignT_ 覆盖完整 VL 尾读且是 32B 的整数倍，
        // srcStride 只需补 curH 与已分配行跨距的差额。
        const int64_t srcStrideBlock =
            (hChunkAlignT_ * static_cast<int64_t>(sizeof(T)) -
             Ops::Base::CeilAlign<int64_t>(curH * static_cast<int64_t>(sizeof(T)), UB_BLOCK_BYTE)) /
            UB_BLOCK_BYTE;
        const int64_t dstStrideByte = (hidden_ - curH) * static_cast<int64_t>(sizeof(T));
        CopyOutPad<T>(outLocal, xGradGm_[tokenStart * hidden_ + hOffset], static_cast<uint16_t>(curRow),
                      static_cast<uint32_t>(curH), srcStrideBlock, dstStrideByte);
        outQue_.FreeTensor(outLocal);
    }

    // 一次搬进 [curK, curRow] 的路由片；[K, S] 是列布局，一个 token 的 K 项跨步 S，
    // 按二维片搬可以把跨步开销摊到 curRow 个 token 上。
    __aicore__ inline void LoadRouteTile(int64_t kOffset, int64_t tokenStart, int64_t curK, int64_t curRow)
    {
        LocalTensor<int32_t> idxLocal = idxBuf_.template Get<int32_t>();
        LocalTensor<int32_t> locLocal = locBuf_.template Get<int32_t>();
        LocalTensor<T> gateTLocal = gateTBuf_.template Get<T>();

        const int64_t gmOffset = kOffset * samples_ + tokenStart;
        const int64_t srcStrideI32 = (samples_ - curRow) * static_cast<int64_t>(sizeof(int32_t));
        const int64_t dstStrideI32 =
            (routeStride_ * static_cast<int64_t>(sizeof(int32_t)) -
             Ops::Base::CeilAlign<int64_t>(curRow * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_BYTE)) /
            UB_BLOCK_BYTE;
        CopyInPad<int32_t>(indicesGm_[gmOffset], idxLocal, static_cast<uint16_t>(curK), static_cast<uint32_t>(curRow),
                           srcStrideI32, dstStrideI32);
        CopyInPad<int32_t>(locationsGm_[gmOffset], locLocal, static_cast<uint16_t>(curK), static_cast<uint32_t>(curRow),
                           srcStrideI32, dstStrideI32);

        const int64_t srcStrideT = (samples_ - curRow) * static_cast<int64_t>(sizeof(T));
        const int64_t dstStrideT =
            (routeStride_ * static_cast<int64_t>(sizeof(T)) -
             Ops::Base::CeilAlign<int64_t>(curRow * static_cast<int64_t>(sizeof(T)), UB_BLOCK_BYTE)) /
            UB_BLOCK_BYTE;
        CopyInPad<T>(gatesGm_[gmOffset], gateTLocal, static_cast<uint16_t>(curK), static_cast<uint32_t>(curRow),
                     srcStrideT, dstStrideT);

        event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        if constexpr (!IsSameType<T, float>::value) {
            LocalTensor<float> gateF32Local = gateF32Buf_.template Get<float>();
            Cast(gateF32Local, gateTLocal, RoundMode::CAST_NONE, static_cast<int32_t>(curK * routeStride_));
        }

        // 后面要用标量 GetValue 读路由片，等搬运与 Cast 都落地。
        event_t vToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        event_t mte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToS);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS);
    }

    // 取第 token 个 token 在当前 k 片内的有效项，压紧后 gather 并累加。
    __aicore__ inline void GatherAndAccumulate(const LocalTensor<float> &acc, int64_t curK, int64_t rowInBlock,
                                               int64_t hOffset, int64_t curH)
    {
        LocalTensor<int32_t> idxLocal = idxBuf_.template Get<int32_t>();
        LocalTensor<int32_t> locLocal = locBuf_.template Get<int32_t>();
        LocalTensor<float> gateCompact = gateCompactBuf_.template Get<float>();
        LocalTensor<T> rowLocal = rowQue_.template AllocTensor<T>();

        // gateCompact 只有一份，标量重写之前要等上一轮 AccumulateRows 读完（WAR）。
        event_t vToSCompact = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vToSCompact);
        WaitFlag<HardEvent::V_S>(vToSCompact);

        uint16_t validK = 0;
        for (int64_t kk = 0; kk < curK; kk++) {
            const int64_t routeIdx = kk * routeStride_ + rowInBlock;
            const int32_t expertId = idxLocal.GetValue(routeIdx);
            const int32_t location = locLocal.GetValue(routeIdx);
            // INV-3/INV-6：四侧边界全部满足才发搬运；无效路由贡献 0，且不读 y_grad。
            if (expertId >= 0 && static_cast<int64_t>(expertId) < expertNum_ && location >= 0 &&
                static_cast<int64_t>(location) < capacity_) {
                const int64_t gmOffset =
                    (static_cast<int64_t>(expertId) * capacity_ + static_cast<int64_t>(location)) * hidden_ + hOffset;
                CopyInPad<T>(yGradGm_[gmOffset], rowLocal[validK * hChunkAlignT_], 1, static_cast<uint32_t>(curH), 0,
                             0);
                if constexpr (IsSameType<T, float>::value) {
                    gateCompact.SetValue(validK, gateTBuf_.template Get<float>().GetValue(routeIdx));
                } else {
                    gateCompact.SetValue(validK, gateF32Buf_.template Get<float>().GetValue(routeIdx));
                }
                validK++;
            }
        }

        rowQue_.EnQue(rowLocal);
        rowLocal = rowQue_.template DeQue<T>();
        if (validK > 0) {
            // gateCompact 是标量写入，VF 读之前挡一次。
            event_t sToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(sToV);
            WaitFlag<HardEvent::S_V>(sToV);
            AccumulateRows<T>(acc, rowLocal, gateCompact, validK, static_cast<uint32_t>(curH),
                              static_cast<uint32_t>(hChunkAlignT_));
        }
        rowQue_.FreeTensor(rowLocal);
    }

private:
    static constexpr int64_t UB_BLOCK_BYTE = 32;
    static constexpr int64_t FP32_PER_BLOCK = 8;

    int64_t expertNum_{0};
    int64_t capacity_{0};
    int64_t hidden_{0};
    int64_t topK_{0};
    int64_t samples_{0};
    int64_t usedCoreNum_{0};
    int64_t rowOfFormerBlock_{0};
    int64_t rowOfTailBlock_{0};
    int64_t rowFactor_{1};
    int64_t kFactor_{1};
    int64_t hFactor_{0};
    int64_t hChunkAlignT_{0};
    int64_t hChunkAlignF32_{0};
    int64_t routeStride_{0};

    GlobalTensor<T> yGradGm_;
    GlobalTensor<T> gatesGm_;
    GlobalTensor<T> xGradGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> locationsGm_;

    // TQue 模板深度是逻辑队列深度；双缓冲由 InitBuffer 的 buffer 数量单独控制。
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECIN, 1> rowQue_;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> locBuf_;
    TBuf<TPosition::VECCALC> gateTBuf_;
    TBuf<TPosition::VECCALC> gateF32Buf_;
    TBuf<TPosition::VECCALC> gateCompactBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
};
} // namespace MoeTutelCombineXRegbase

#endif // MOE_TUTEL_COMBINE_X_KERNEL_H_
