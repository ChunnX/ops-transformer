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
 * \file chunk_gated_delta_rule_stage2_arch35.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
template <typename cType>
using StageTwoMTT = matmul::MatmulImpl<aT2, bT2, MatmulType<TPosition::GM, CubeFormat::ND, cType>>;
using StageTwoMT = StageTwoMTT<bfloat16_t>;
using StageTwoMTFp32C = StageTwoMTT<float>;

template <typename stateType>
struct StageTwoParams {
    GlobalTensor<bfloat16_t> qPrime;       // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> vInner;       // (Nv, Sp, Dv) BF16 C1 output / Stage3 Z
    GlobalTensor<bfloat16_t> yRes;         // (Nv, Sp, Dv) BF16 V2 output consumed by C3
    GlobalTensor<bfloat16_t> vInnerBf16;   // alias of vInner for stage3/CalStateNew
    GlobalTensor<float> gCum;              // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay;    // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> curStateBf16; // (Nv, Dv, Dk) BF16 state for AIC
    GlobalTensor<stateType> stateIn;       // (Nv, Dv, Dk) state input
    GlobalTensor<stateType> stateOut;      // (Nv, Dv, Dk) state output
    GlobalTensor<bfloat16_t> kg;
    GlobalTensor<bfloat16_t> out;
    GM_ADDR ws;
    StageTwoMT *mm1Bf16;
    StageTwoMTFp32C *mm1Fp32;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    int64_t stateStride1;
    bool useInitialState;
    bool isFirstGroup;
};

template <typename stateType = bfloat16_t, bool gOptional = false>
class Stage2 {
public:
    static constexpr bool kIsFp32 = std::is_same_v<stateType, float>;

    /**
     * 解析当前 chunk group 和状态布局。
     * 输入 StageTwoParams 包含 Q'、K_decay、Z0、K、G、S 及输出地址；输出为成员维度、对齐维度和 UB 配置。
     */
    __aicore__ inline void Init(StageTwoParams<stateType> *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curChunkSize_ = chunkSize_;
        useInitialState_ = sTP_->useInitialState;
        stateStride1_ = (useInitialState_) ? sTP_->stateStride1 : Dv_ * Dk_;
        isFirstGroup_ = sTP_->isFirstGroup;
        InitLocalBuffers();
    }

    /**
     * 初始化 Stage2 Vector 队列及 UB。
     * FP32 分支保留 stateFp32Ub[Dv,aligned(Dk)]，两种分支均分配 Z/Y 的 FP32 计算区和 G[C] 门控区。
     */
    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (kIsFp32) {
            uint64_t inSize = Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ * sizeof(float);
            uint64_t vInnerCastInSize = chunkSize_ * curDv_ * sizeof(bfloat16_t);
            uint64_t vInnerCastOutSize = chunkSize_ * curDv_ * sizeof(bfloat16_t);
            // RegBase 的 DataCopy 会在尾轮读取完整 256B 寄存器。输入队列容量必须按寄存器宽度
            // 上取整，避免 Dv>64 且 Dv*aligned(Dk) 非 256B 整倍数时越过队列物理边界。
            uint64_t inQueueSize = Std::max(inSize, vInnerCastInSize);
            inQueueSize = (inQueueSize + VECTOR_REG_WIDTH - 1) / VECTOR_REG_WIDTH * VECTOR_REG_WIDTH;
            pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueSize);
            uint64_t outSize = Std::max(inSize, vInnerCastOutSize);
            pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outSize);
            // FP32 状态在 UB 中保留一份：转成 BF16 镜像供 AIC 使用，同时原 FP32 副本乘门控后写回。
            // VF 尾轮仍会读取完整 256B，因此物理容量向上补齐到寄存器宽度。
            uint64_t stateBytes = Dv_ * curDk_ * sizeof(float);
            uint64_t stateRegBytes =
                (stateBytes + VECTOR_REG_WIDTH - 1) / VECTOR_REG_WIDTH * VECTOR_REG_WIDTH;
            uint64_t v2Bytes = static_cast<uint64_t>(chunkSize_) * curDv_ * sizeof(float);
            pipe_->InitBuffer(tmpBuff_, stateRegBytes + v2Bytes + chunkSize_ * sizeof(float));
            uint32_t buffOffset = 0;
            stateFp32Ub_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(Dv_ * curDk_), buffOffset);
            buffOffset += static_cast<uint32_t>(stateRegBytes);
            vCalcUb_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * curDv_), buffOffset);
            buffOffset += static_cast<uint32_t>(chunkSize_ * curDv_ * sizeof(float));
            gateUb_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        } else {
            uint64_t stateInBytes = Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ * sizeof(bfloat16_t);
            // PrepareGate 与 BF16 状态、Z0 共用输入队列，容量取三者最大值。
            uint64_t gateInBytes = static_cast<uint64_t>(chunkSize_) * sizeof(float);
            uint64_t vInnerInBytes = static_cast<uint64_t>(chunkSize_) * curDv_ * sizeof(bfloat16_t);
            uint64_t inQueueSize = Std::max(Std::max(stateInBytes, gateInBytes), vInnerInBytes);
            inQueueSize = (inQueueSize + VECTOR_REG_WIDTH - 1) / VECTOR_REG_WIDTH * VECTOR_REG_WIDTH;
            pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueSize);
            uint64_t outQueueSize = Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(bfloat16_t),
                                             (uint64_t)Dv_ * curDk_ * sizeof(bfloat16_t));
            outQueueSize = Std::max(outQueueSize, vInnerInBytes);
            pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outQueueSize);
            uint64_t v2Bytes = static_cast<uint64_t>(chunkSize_) * curDv_ * sizeof(float);
            pipe_->InitBuffer(tmpBuff_, v2Bytes + chunkSize_ * sizeof(float));
            uint32_t buffOffset = 0;
            vCalcUb_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * curDv_), buffOffset);
            buffOffset += static_cast<uint32_t>(chunkSize_ * curDv_ * sizeof(float));
            gateUb_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        }
    }

    /**
     * 准备 V1/V2 共用的门控。
     * 输入 gCum=G[0:L] FP32，其中 G_i=sum(g_0..g_i)；输出 gateUb=exp(G)、vCalcUb 前 L 项=G，
     * 返回 exp(G_{L-1})。无 g 时分别构造 1 和 0。
     */
    __aicore__ inline float PrepareGate(GlobalTensor<float> gCum)
    {
        if constexpr (gOptional) {
            CopyIn<float>(gCum, 1, curChunkSize_);
            auto gLocal = inQueue_.DeQue<float>();
            // UB 到 UB 按 32B block 搬运，尾块向上对齐以保留最后一个有效门控元素。
            int32_t gateAligned = Ceil(curChunkSize_, BLOCK_SIZE / sizeof(float)) *
                                  (BLOCK_SIZE / sizeof(float));
            DataCopy(gateUb_, gLocal, gateAligned);
            // 保留 log-gate G，V2 直接计算 exp(G_last-G_i)，避免 exp(G_last)/exp(G_i) 的不稳定除法。
            DataCopy(vCalcUb_, gLocal, gateAligned);
            inQueue_.FreeTensor(gLocal);
            Exp<float, 0, true>(gateUb_, gateUb_, gateAligned);
        } else {
            Duplicate(gateUb_, 1.0f, curChunkSize_);
            Duplicate(vCalcUb_, 0.0f, curChunkSize_);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        return gateUb_.GetValue(curChunkSize_ - 1);
    }

    /** MicroAPI：将 src[count] FP32 复制到 dst，尾轮通过 mask 仅写有效元素。 */
    static __simd_vf__ inline void CopyFp32Vf(__ubuf__ float *dst, __ubuf__ float *src, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        RegTensor<float> dataF32;
        MaskReg activeMask;

        const uint16_t repeatTimes =
            static_cast<uint16_t>((count + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);
        uint32_t remainingCount = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            activeMask = UpdateMask<float>(remainingCount);
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
            DataCopy(dataF32, src + offset);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dst + offset, dataF32, activeMask);
        }
    }

    /** MicroAPI：用标量 value 填充 dst[count] FP32，主要用于清零行对齐区。 */
    static __simd_vf__ inline void FillFp32Vf(__ubuf__ float *dst, float value, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        RegTensor<float> dataF32;
        MaskReg activeMask;

        Duplicate(dataF32, value);
        const uint16_t repeatTimes =
            static_cast<uint16_t>((count + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);
        uint32_t remainingCount = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            activeMask = UpdateMask<float>(remainingCount);
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
            DataCopy<float, StoreDist::DIST_NORM_B32>(dst + offset, dataF32, activeMask);
        }
    }

    /** MicroAPI：src[count] FP32 以 RINT 模式转为 dst[count] BF16，输出采用 B32 pack 布局。 */
    static __simd_vf__ inline void CastFp32ToBf16Vf(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        static constexpr CastTrait CAST_FP32_TO_BF16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        RegTensor<float> dataF32;
        RegTensor<bfloat16_t> dataB16;
        MaskReg activeMask;

        const uint16_t repeatTimes =
            static_cast<uint16_t>((count + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);
        uint32_t remainingCount = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            activeMask = UpdateMask<float>(remainingCount);
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
            DataCopy(dataF32, src + offset);
            Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(dataB16, dataF32, activeMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(dst + offset, dataB16, activeMask);
        }
    }

    /** MicroAPI：原地计算 state[i]=state[i]*gate，输入输出均为 FP32。 */
    static __simd_vf__ inline void ScaleFp32ByGateVf(__ubuf__ float *state, float gate, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        RegTensor<float> stateF32;
        RegTensor<float> decayF32;
        MaskReg activeMask;

        Duplicate(decayF32, gate);

        const uint16_t repeatTimes =
            static_cast<uint16_t>((count + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);
        uint32_t remainingCount = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            activeMask = UpdateMask<float>(remainingCount);
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
            DataCopy(stateF32, state + offset);
            Mul(stateF32, stateF32, decayF32, activeMask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(state + offset, stateF32, activeMask);
        }
    }

    /** MicroAPI：BF16 src 转 FP32 后乘 gate，再以 RINT 转回 BF16 dst。 */
    static __simd_vf__ inline void ScaleBf16ByGateVf(__ubuf__ bfloat16_t *dst, __ubuf__ bfloat16_t *src,
                                                     float gate, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        static constexpr CastTrait CAST_BF16_TO_FP32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait CAST_FP32_TO_BF16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        RegTensor<bfloat16_t> stateB16;
        RegTensor<bfloat16_t> outB16;
        RegTensor<float> stateF32;
        RegTensor<float> decayF32;
        MaskReg activeMask;
        Duplicate(decayF32, gate);

        const uint16_t repeatTimes =
            static_cast<uint16_t>((count + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);
        uint32_t remainingCount = count;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            activeMask = UpdateMask<float>(remainingCount);
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(stateB16, src + offset);
            Cast<float, bfloat16_t, CAST_BF16_TO_FP32>(stateF32, stateB16, activeMask);
            Mul(stateF32, stateF32, decayF32, activeMask);
            Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outB16, stateF32, activeMask);
            DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(dst + offset, outB16, activeMask);
        }
    }

    /**
     * MicroAPI：仅缩放 data[start:start+count]，公式 data_i=gate*data_i。
     * 首尾寄存器用范围 mask，保证 Dv 非 64 对齐时不改写相邻行或 padding。
     */
    static __simd_vf__ inline void ScaleFp32RangeByGateVf(__ubuf__ float *data, float gate,
                                                          uint32_t start, uint32_t count)
    {
        using namespace AscendC::MicroAPI;
        constexpr uint32_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        RegTensor<float> dataF32;
        RegTensor<float> gateF32;
        RegTensor<int32_t> laneIndex;
        MaskReg allMask = CreateMask<int32_t, MaskPattern::ALL>();
        MaskReg lowerMask;
        MaskReg rangeMask;

        Duplicate(gateF32, gate);
        Arange(laneIndex, static_cast<int32_t>(0));
        const uint32_t firstRepeat = start / FP32_ELEMS_PER_REPEAT;
        const uint32_t end = start + count;
        const uint32_t lastRepeat = (end - 1) / FP32_ELEMS_PER_REPEAT;
        for (uint32_t repeat = firstRepeat; repeat <= lastRepeat; ++repeat) {
            const uint32_t repeatStart = repeat * FP32_ELEMS_PER_REPEAT;
            const uint32_t localStart = start > repeatStart ? start - repeatStart : 0;
            const uint32_t repeatEnd = repeatStart + FP32_ELEMS_PER_REPEAT;
            const uint32_t localEnd = end < repeatEnd ? end - repeatStart : FP32_ELEMS_PER_REPEAT;
            Compares<int32_t, CMPMODE::GE>(lowerMask, laneIndex, static_cast<int32_t>(localStart), allMask);
            Compares<int32_t, CMPMODE::LT>(rangeMask, laneIndex, static_cast<int32_t>(localEnd), lowerMask);
            DataCopy(dataF32, data + repeatStart);
            Mul(dataF32, dataF32, gateF32, rangeMask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(data + repeatStart, dataF32, rangeMask);
        }
    }

    /**
     * 按 value head 分核，并在每个 head 内按时间顺序串行处理 chunk，以维持递归状态依赖。
     * 每个 chunk 的执行顺序为 V1 -> C1/C2 -> V2 -> C3。
     */
    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            curChunkSize_ = chunkSize_;
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if constexpr (kIsFp32) {
                        ProcessAivFp32(nvId, cId, length);
                    } else {
                        ProcessAivBf16(nvId, cId, length);
                    }
                }
                if ASCEND_IS_AIC {
                    ProcessAic(nvId, cId, length);
                }
            }
        }
    }

    /**
     * FP32 状态分支的 Vector 部分。
     * 输入 S[Dv,Dk] FP32；先生成未缩放 BF16(S) 供 C1/C2，再计算 V1：S_decay=exp(G_last)*S（FP32），
     * 写入 stateOut；等待 C1/C2 后执行 V2，输出 Y[L,Dv] BF16，最后等待 C3 完成。
     */
    __aicore__ inline void ProcessAivFp32(int64_t nvId, int64_t cId, int64_t length)
    {
        if (GetSubBlockIdx() == 0) {
            float lastGCumExp = PrepareGate(sTP_->gCum[nvId * Sp_ + length]);
            if (isFirstGroup_ && cId == 0) {
                CopyIn<float>(sTP_->stateIn[nvId * stateStride1_], Dv_, Dk_);
            } else {
                CopyIn<float>(sTP_->stateOut[nvId * Dv_ * Dk_], Dv_, Dk_);
            }
            auto stateIn = inQueue_.DeQue<float>();
            auto stateFp32Ptr = reinterpret_cast<__ubuf__ float *>(stateFp32Ub_.GetPhyAddr());
            auto stateInPtr = reinterpret_cast<__ubuf__ float *>(stateIn.GetPhyAddr());
            const uint32_t stateCount = static_cast<uint32_t>(Dv_ * curDk_);
            AscendC::VF_CALL<CopyFp32Vf>(stateFp32Ptr, stateInPtr, stateCount);
            inQueue_.FreeTensor(stateIn);

            // 在 FP32 状态副本施加 V1 门控前，先保留未缩放的 BF16(S) 供 C1/C2 使用。
            auto bf16State = outQueue_.AllocTensor<bfloat16_t>();
            auto bf16StatePtr = reinterpret_cast<__ubuf__ bfloat16_t *>(bf16State.GetPhyAddr());
            AscendC::VF_CALL<CastFp32ToBf16Vf>(bf16StatePtr, stateFp32Ptr, stateCount);
            outQueue_.EnQue(bf16State);
            CopyOut<bfloat16_t>(sTP_->curStateBf16[nvId * Dv_ * Dk_], Dv_, Dk_, false, curDk_);
            // V1：stateIn(FP32) -> *exp(G_last) -> stateOut(FP32)。
            ScaleFp32State(lastGCumExp);
            auto fp32Local = outQueue_.AllocTensor<float>();
            auto fp32LocalPtr = reinterpret_cast<__ubuf__ float *>(fp32Local.GetPhyAddr());
            AscendC::VF_CALL<CopyFp32Vf>(fp32LocalPtr, stateFp32Ptr, stateCount);
            outQueue_.EnQue(fp32Local);
            CopyOut<float>(sTP_->stateOut[nvId * Dv_ * Dk_], Dv_, Dk_, false, curDk_);
        }
        // BF16 状态镜像写出后，通知 AIC 执行 C1/C2。
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
        CrossCoreWaitFlag(0x2);
        if (GetSubBlockIdx() == 0) {
            ApplyV2(nvId, cId, length);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        CrossCoreWaitFlag(0x4);
    }

    /**
     * BF16 状态分支的 Vector 部分。
     * 输入 S[Dv,Dk] BF16；在 UB 中以 FP32 乘 exp(G_last) 后转回 BF16。为避免覆盖 C1/C2 的 S，
     * 等待 AIC 读取完成才写 stateOut；随后执行 V2 并等待 C3。
     */
    __aicore__ inline void ProcessAivBf16(int64_t nvId, int64_t cId, int64_t length)
    {
        auto curState = (cId == 0) ? sTP_->curStateBf16[nvId * stateStride1_] : sTP_->stateOut[nvId * Dv_ * Dk_];
        auto finalState = sTP_->stateOut[nvId * Dv_ * Dk_];
        if (GetSubBlockIdx() == 0) {
            float lastGCumExp = PrepareGate(sTP_->gCum[nvId * Sp_ + length]);
            CopyIn(curState, Dv_, Dk_);
            CalGCumExpBf16(lastGCumExp);
        }
        // C1/C2 读取原始 S，因此门控后的状态先留在 UB，待 AIC 读取完成后再覆盖 stateOut。
        CrossCoreWaitFlag(0x2);
        if (GetSubBlockIdx() == 0) {
            CopyOutState(finalState);
            ApplyV2(nvId, cId, length);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        CrossCoreWaitFlag(0x4);
    }

    /**
     * Stage2 的 Cube 部分。
     * C1：Z=Z0+K_decay@S^T，[L,Dv] BF16 原子累加；
     * C2：O_inter=Q'@S^T，[L,Dv] BF16；
     * C3：S_new=S_decay+Y^T@K，[Dv,Dk]，BF16 状态输出 BF16，FP32 状态输出/累加 FP32。
     */
    __aicore__ inline void ProcessAic(int64_t nvId, int64_t cId, int64_t length)
    {
        uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
        uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
        auto stateOut = sTP_->stateOut[nvId * Dv_ * Dk_];
        GlobalTensor<bfloat16_t> stateForAic;
        if constexpr (kIsFp32) {
            // FP32 V1 先发布未缩放 BF16(S)，AIC 等待该镜像写完。
            CrossCoreWaitFlag(0x3);
            stateForAic = sTP_->curStateBf16[nvId * Dv_ * Dk_];
        } else {
            // BF16 分支在 C1/C2 读取前保持 stateOut 中的原始 S 不变。
            stateForAic = (cId == 0) ? sTP_->stateIn[nvId * stateStride1_]
                                     : sTP_->stateOut[nvId * Dv_ * Dk_];
        }
        CalVPrime(sTP_->kCumdecay[mm_offset0], stateForAic, sTP_->vInner[mm_offset1]);
        CalAttnInter(sTP_->qPrime[mm_offset0], stateForAic, sTP_->out[nvId * Dv_ + cId * chunkSize_ * Nv_ * Dv_]);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
        CrossCoreWaitFlag(0x5);
        CalStateNew(sTP_->yRes[mm_offset1], sTP_->kg[mm_offset0], stateOut);
        SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
        WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
    }

    /** V1 的 FP32 实现：stateFp32Ub=exp(G_last)*stateFp32Ub；无 g 时系数为 1。 */
    __aicore__ inline void ScaleFp32State(float lastGCumExp)
    {
        auto statePtr = reinterpret_cast<__ubuf__ float *>(stateFp32Ub_.GetPhyAddr());
        AscendC::VF_CALL<ScaleFp32ByGateVf>(statePtr, lastGCumExp, static_cast<uint32_t>(Dv_ * curDk_));
    }

    /**
     * V1 的 BF16 实现：输入队列中的 S BF16 -> FP32 乘 exp(G_last) -> BF16(RINT)，
     * 输出保留在 outQueue，待 C1/C2 读取原始 GM 状态后写回。
     */
    __aicore__ inline void CalGCumExpBf16(float lastGCumExp)
    {
        auto stateIn = inQueue_.DeQue<bfloat16_t>();
        auto stateOut = outQueue_.AllocTensor<bfloat16_t>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        auto stateInPtr = reinterpret_cast<__ubuf__ bfloat16_t *>(stateIn.GetPhyAddr());
        auto stateOutPtr = reinterpret_cast<__ubuf__ bfloat16_t *>(stateOut.GetPhyAddr());
        AscendC::VF_CALL<ScaleBf16ByGateVf>(stateOutPtr, stateInPtr, lastGCumExp,
                                              static_cast<uint32_t>(Dv_ * curDk_));
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(stateOut);
        inQueue_.FreeTensor(stateIn);
    }

    /**
     * V2：输入 Z=vInner[L,Dv] BF16、G[L] FP32，计算 gamma_i=exp(G_last-G_i)，
     * Y_i=gamma_i*Z_i，并输出 yRes[L,Dv] BF16。vInner 保持不变供 Stage3 使用。
     * 每行通过 RegBase 范围 mask 缩放，有效覆盖 Dv 非 32B/256B 对齐的情况。
     */
    __aicore__ inline void ApplyV2(int64_t nvId, int64_t cId, int64_t length)
    {
        uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
        auto vIn = sTP_->vInner[mm_offset1];
        auto yRes = sTP_->yRes[mm_offset1];
        CopyIn(vIn, curChunkSize_, Dv_);
        auto vInLocal = inQueue_.DeQue<bfloat16_t>();
        auto yLocal = outQueue_.AllocTensor<bfloat16_t>();
        // gamma=exp(G_last-G_i)：先广播最后一个 log-gate，再做减法和指数。
        float lastLogGate = vCalcUb_.GetValue(curChunkSize_ - 1);
        Duplicate(gateUb_, lastLogGate, curChunkSize_);
        Sub(gateUb_, gateUb_, vCalcUb_, curChunkSize_);
        Exp<float, 0, true>(gateUb_, gateUb_, curChunkSize_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Cast(vCalcUb_, vInLocal, RoundMode::CAST_NONE, curChunkSize_ * curDv_);
        inQueue_.FreeTensor(vInLocal);
        auto vCalcPtr = reinterpret_cast<__ubuf__ float *>(vCalcUb_.GetPhyAddr());
        for (int32_t row = 0; row < curChunkSize_; ++row) {
            AscendC::VF_CALL<ScaleFp32RangeByGateVf>(vCalcPtr, gateUb_.GetValue(row),
                                                     static_cast<uint32_t>(row * curDv_),
                                                     static_cast<uint32_t>(Dv_));
        }
        PipeBarrier<PIPE_V>();
        Cast(yLocal, vCalcUb_, RoundMode::CAST_RINT, curChunkSize_ * curDv_);
        outQueue_.EnQue(yLocal);
        CopyOut(yRes, curChunkSize_, Dv_, false, curDv_);
    }

    /**
     * 通用 Cube 封装（紧凑矩阵）：C=A@B，可选转置；accum=1 表示原子累加到已有 C。
     * 输入 M/N/K 为当前实际形状，输出 dtype 由 CT 和 matmul 模板决定。
     */
    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA, GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K, int32_t accum)
    {
        mm->SetOrgShape(M, N, K);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    /**
     * 通用 Cube 封装（带 GM 原始行距）：Ka/Kb 描述 A/B 的原始末维 stride，支持 TND 输出和非紧凑状态。
     */
    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA, GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K, int64_t Ka, int64_t Kb,
                                     int32_t accum = 0)
    {
        mm->SetOrgShape(M, N, K, Ka, Kb);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    /** C2：输入 Q'[L,Dk]、S[Dv,Dk] BF16，输出 O_inter=Q'@S^T，[L,Dv] BF16。 */
    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> qPrime, GlobalTensor<bfloat16_t> state,
                                        GlobalTensor<bfloat16_t> out)
    {
        RunMatmul(sTP_->mm1Bf16, qPrime, false, state, true, out, curChunkSize_, Dv_, Dk_, Dk_, Nv_ * Dv_);
    }

    /**
     * C1：输入 K_decay[L,Dk]、S[Dv,Dk] BF16，原子累加到 Z0 得 Z=Z0+K_decay@S^T。
     * 输出 vPrime/Z 为 [L,Dv] BF16；FP32 状态分支同样使用 V1 生成的 BF16(S) 镜像。
     */
    template <typename vInnerType>
    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> kCumdecay, GlobalTensor<bfloat16_t> state,
                                     GlobalTensor<vInnerType> vPrime)
    {
        // C1 固定输出 BF16；FP32 精度只保留在递归 stateOut，Cube 使用 BF16 状态镜像。
        RunMatmul(sTP_->mm1Bf16, kCumdecay, false, state, true, vPrime, curChunkSize_, Dv_, Dk_, 1);
    }

    /**
     * C3：输入 Y[L,Dv]、K[L,Dk] BF16，原子累加 state=S_decay+Y^T@K。
     * BF16 分支使用 BF16 C，FP32 分支使用 BF16xBF16->FP32 C。
     */
    template <typename stateOutType>
    __aicore__ inline void CalStateNew(GlobalTensor<bfloat16_t> vInner, GlobalTensor<bfloat16_t> kg,
                                       GlobalTensor<stateOutType> state)
    {
        if constexpr (kIsFp32) {
            RunMatmul(sTP_->mm1Fp32, vInner, true, kg, false, state, Dv_, Dk_, curChunkSize_, 1);
        } else {
            RunMatmul(sTP_->mm1Bf16, vInner, true, kg, false, state, Dv_, Dk_, curChunkSize_, 1);
        }
    }

    /**
     * 将连续 GM FP32 张量转换为 GM BF16，输入 count 为逻辑元素数，尾部补齐后用 RegBase RINT cast。
     * 当前仅保留为 FP32->BF16 工具函数。
     */
    template <typename srcType, typename dstType>
    __aicore__ inline void CastGmToGm(GlobalTensor<srcType> srcGm, GlobalTensor<dstType> dstGm, int32_t count)
    {
        static_assert(std::is_same_v<srcType, float> && std::is_same_v<dstType, bfloat16_t>,
                      "Stage2 RegBase CastGmToGm only supports FP32 to BF16");
        int32_t paddedCount = Ceil(count, BLOCK_SIZE / sizeof(dstType)) * (BLOCK_SIZE / sizeof(dstType));
        int32_t srcAlign = BLOCK_SIZE / sizeof(srcType);
        int32_t srcPadded = Ceil(count, srcAlign) * srcAlign;
        int32_t padding = srcPadded - count;

        auto srcLocal = inQueue_.AllocTensor<srcType>();
        if (paddedCount > srcPadded) {
            auto srcPtr = reinterpret_cast<__ubuf__ float *>(srcLocal.GetPhyAddr());
            AscendC::VF_CALL<FillFp32Vf>(srcPtr, 0.0f, static_cast<uint32_t>(paddedCount));
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
        }
        DataCopyExtParams inParams{1, static_cast<uint32_t>(count * sizeof(srcType)), 0, 0, 0};
        DataCopyPadExtParams<srcType> inPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(srcLocal, srcGm, inParams, inPadParams);
        inQueue_.EnQue(srcLocal);

        srcLocal = inQueue_.DeQue<srcType>();
        auto dstLocal = outQueue_.AllocTensor<dstType>();
        auto srcPtr = reinterpret_cast<__ubuf__ float *>(srcLocal.GetPhyAddr());
        auto dstPtr = reinterpret_cast<__ubuf__ bfloat16_t *>(dstLocal.GetPhyAddr());
        AscendC::VF_CALL<CastFp32ToBf16Vf>(dstPtr, srcPtr, static_cast<uint32_t>(paddedCount));
        inQueue_.FreeTensor(srcLocal);
        outQueue_.EnQue(dstLocal);

        dstLocal = outQueue_.DeQue<dstType>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(count * sizeof(dstType)), 0, 0, 0};
        DataCopyPadExtParams<dstType> outPadParams{false, 0, 0, 0};
        DataCopyPad(dstGm, dstLocal, outParams);
        outQueue_.FreeTensor(dstLocal);
    }

    /**
     * 将 GM[row,col] 搬入 inQueue，并按计算行宽补齐。
     * FP32 状态按 curDk 行距布局，其他输入按各自 32B 对齐行宽布局；输出为队列 LocalTensor。
     */
    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        int padding;
        uint32_t dstStride = 0;
        if constexpr (kIsFp32 && std::is_same_v<inType, float>) {
            int32_t floatAligned = Ceil(col, BLOCK_FLOAT_NUM) * BLOCK_FLOAT_NUM;
            padding = floatAligned - col;
            if (floatAligned < curDk_) {
                uint32_t dataBlocks = static_cast<uint32_t>(floatAligned * sizeof(float) / BLOCK_SIZE);
                uint32_t rowBlocks = static_cast<uint32_t>(curDk_ * sizeof(float) / BLOCK_SIZE);
                dstStride = rowBlocks - dataBlocks;
                auto inLocalPtr = reinterpret_cast<__ubuf__ float *>(inLocal.GetPhyAddr());
                AscendC::VF_CALL<FillFp32Vf>(inLocalPtr, 0.0f, static_cast<uint32_t>(row * curDk_));
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            }
        } else {
            padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        }
        DataCopyExtParams inParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0), dstStride, 0};
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    /** 将 outQueue 中门控后的 BF16 状态 [Dv,aligned(Dk)] 紧凑写到 stateNew[Dv,Dk]。 */
    __aicore__ inline void CopyOutState(GlobalTensor<bfloat16_t> stateNew)
    {
        CopyOut<bfloat16_t>(stateNew, Dv_, Dk_, false);
    }

    /**
     * 将 outQueue 中 [row,srcCol] LocalTensor 的每行前 col 个元素写回紧凑 GM。
     * setAtomic=true 时使用原子加；srcCol 指定 UB 实际行宽，0 表示与 col 相同。
     */
    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false,
                                   int32_t srcCol = 0)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        if (srcCol > col) {
            uint32_t actualReadBytes = Ceil(col * sizeof(outType), BLOCK_SIZE) * BLOCK_SIZE;
            uint32_t srcRowBytes = srcCol * sizeof(outType);
            if (srcRowBytes >= actualReadBytes && (srcRowBytes - actualReadBytes) % BLOCK_SIZE == 0) {
                copyParams.srcStride = (srcRowBytes - actualReadBytes) / BLOCK_SIZE;
            } else {
                copyParams.srcStride = 0;
            }
        } else {
            copyParams.srcStride = static_cast<uint32_t>(0);
        }
        copyParams.dstStride = static_cast<uint32_t>(0);
        if (setAtomic) {
            SetAtomicAdd<outType>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageTwoParams<stateType> *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> stateFp32Ub_;
    LocalTensor<float> vCalcUb_;
    LocalTensor<float> gateUb_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int32_t curDv_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int64_t stateStride1_;
    bool useInitialState_;
    bool isFirstGroup_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H
