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
 * \file quant_block_sparse_attn_mx_block_vec.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 vector 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_attenmask.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/util_regbase_mx.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbasemx;

namespace BaseApi {
// V1 执行 softmax/P 量化，V2 更新并写回 attention output/LSE。
template <typename inputType, typename mmType, typename outputType, BSALayout layout, BSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool hasAtten, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxBlockVec {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;
    using OUTPUT_T = outputType;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_ATTEN = hasAtten;
    static constexpr bool IS_PA = isPa;
    static constexpr BSALayout LAYOUT = layout;
    static constexpr BSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t VEC_M_BASE = M_BASE >> 1U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == BSALayout::TND && KV_LAYOUT == BSALayout::PA_BNSD && IS_PA,
                  "MX vector currently only supports TND query and PA_BNSD KV");
    static_assert(M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX vector currently only supports S1=128, S2=512, D=128 and DV=128");
    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    static constexpr uint32_t dTemplateAlign64 = 128U;
    static constexpr uint32_t DN_VSELR_INDEX_SIZE = 256U;
    static constexpr uint32_t DN_VSELR_GROUP_NUM = 4U;
    static constexpr uint32_t DN_VSELR_GROUP_SIZE = DN_VSELR_INDEX_SIZE / DN_VSELR_GROUP_NUM;
    static constexpr uint32_t ATTEN_MASK_UB_SIZE = S2_SPLIT * VEC_M_BASE;
    static constexpr uint32_t MASK_COPY_BLOCK_BYTES = 32U;

    __aicore__ inline void Init(TPipe *pipe, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *attenMask,
                                const regbaseutil::AttenMaskInfo &attenMaskInfo)
    {
        if ASCEND_IS_AIV {
            tPipe_ = pipe;
            attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
            pScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)pScale);
            if constexpr (HAS_ATTEN) {
                attenMaskGm_.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
                attenMaskInfo_ = attenMaskInfo;
            }
            uint32_t negativeIntScalar = NEGATIVE_MIN_VALUE_FP32;
            negativeFloatScalar_ = *((MM_T *)&negativeIntScalar);
            // 输入 quantScale1 为 [1] e8m0；VF 另行生成 C2 使用的 PScale。
            pScaleValue_ = pScale != nullptr ? DecodeE8M0Scale(pScaleGm_.GetValue(0)) : 1.0f;
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // V2 累计输出：[64,128] fp32。
            tPipe_->InitBuffer(stage2OutBuf_, 64U * dTemplateAlign64 * sizeof(MM_T));
            // V1 DN packed P 使用双队列。
            tPipe_->InitBuffer(stage1OutQue_[0], 1, 16896U);
            tPipe_->InitBuffer(stage1OutQue_[1], 1, 16896U);
            // 单个 256-column subLoop 的 e8m0 PScale。
            tPipe_->InitBuffer(pScaleSubLoop0Que_, 1, 512U);
            // 三槽 softmax 状态对应 PRELOAD_N+1 环形流水。
            tPipe_->InitBuffer(softmaxSumBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxSumBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxSumBuf_[2], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[2], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[2], 256U);
            // online softmax update 临时状态。
            tPipe_->InitBuffer(preLoopMaxBuf_, 256U);
            tPipe_->InitBuffer(preLoopSumBuf_, 256U);
            tPipe_->InitBuffer(firstLoopSumBuf_, 256U);
            if constexpr (HAS_ATTEN) {
                // DN mask UB：[256,64]，按 sparse segment 拼接。
                tPipe_->InitBuffer(attenMaskInQue_, 1, ATTEN_MASK_UB_SIZE);
            }
            // DN VF lane 重排索引；当前 MX 仅实现 USE_DN=true。
            if constexpr (USE_DN) {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)],
                                   DN_VSELR_INDEX_SIZE);
                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (uint32_t i = 0U; i < DN_VSELR_GROUP_NUM; ++i) {
                    for (uint32_t j = 0U; j < DN_VSELR_GROUP_SIZE; ++j) {
                        vselrIndexesTensor.SetValue(i * DN_VSELR_GROUP_SIZE + j, i + (j << 2));
                    }
                }
            }

            // 同步 V2 cast 与 GM copy。
            mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    // V1：C1[128,256] -> P/PScale；两个 subLoop 写成 P[128,512]。
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo, uint32_t subLoop)
    {
        if (unlikely(runInfo.actVecMSize == 0U)) {
            bmm1ResBuf.WaitCrossCore();
            bmm1ResBuf.SetCrossCore();
            if (IsLastSubLoop(runInfo, subLoop)) {
                outputBuf.SetCrossCore();
            }
            return;
        }

        const uint32_t softmaxBufIdx = runInfo.mLoop % (PRELOAD_N + 1U);
        const uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1U);
        const uint32_t stage1Offset = runInfo.loop % DB;
        LocalTensor<float> sumUb = softmaxSumBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> maxUb = softmaxMaxBuf_[softmaxBufIdx].template Get<float>()[0];
        LocalTensor<float> preLoopMaxUb = preLoopMaxBuf_.template Get<float>()[0];
        LocalTensor<float> preLoopSumUb = preLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> firstLoopSumUb = firstLoopSumBuf_.template Get<float>()[0];
        LocalTensor<float> expUb = softmaxExpBuf_[expBufIdx].template Get<float>()[0];
        LocalTensor<MM_T> mmRes = bmm1ResBuf.template GetTensor<MM_T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        LocalTensor<SCALE_T> pScaleSubLoopTensor = pScaleSubLoop0Que_.template AllocTensor<SCALE_T>();

        const uint32_t s2CalcSize = GetSubLoopS2Size(runInfo, subLoop);
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_ATTEN) {
            attenMaskUb = CopyAttenMaskToUb(runInfo, subLoop, s2CalcSize);
            // zyhPrint
            // if (s2CalcSize != 0U && runInfo.loop == 0U && runInfo.mLoop == 0U) {
            //     constexpr uint32_t dumpRowsPerBlock = 64U;
            //     const uint32_t dumpBlockCount = (s2CalcSize + dumpRowsPerBlock - 1U) / dumpRowsPerBlock;
            //     for (uint32_t i = 0U; i < dumpBlockCount; ++i) {
            //         const uint32_t rowOffset = i * dumpRowsPerBlock;
            //         const uint32_t rows =
            //             rowOffset + dumpRowsPerBlock <= s2CalcSize ? dumpRowsPerBlock : s2CalcSize - rowOffset;
            //         uint32_t maskShape[] = {rows, VEC_M_BASE};
            //         AscendC::ShapeInfo maskShapeInfo(2, maskShape);
            //         AscendC::DumpTensor(attenMaskUb[rowOffset * VEC_M_BASE],
            //                             1000U + subLoop * 100U + constInfo.subBlockIdx * 10U + i,
            //                             rows * VEC_M_BASE, maskShapeInfo);
            //     }
            // }
        }
        // 先搬 mask，再等待 C1，以重叠 MTE2 与 BMM1。
        bmm1ResBuf.WaitCrossCore();
        // zyhPrint
        // if (s2CalcSize != 0U && runInfo.loop == 0U && runInfo.mLoop == 0U) {
        //     constexpr uint32_t dumpRowsPerBlock = 64U;
        //     const uint32_t dumpBlockCount = (s2CalcSize + dumpRowsPerBlock - 1U) / dumpRowsPerBlock;
        //     for (uint32_t i = 0U; i < dumpBlockCount; ++i) {
        //         const uint32_t rowOffset = i * dumpRowsPerBlock;
        //         const uint32_t rows =
        //             rowOffset + dumpRowsPerBlock <= s2CalcSize ? dumpRowsPerBlock : s2CalcSize - rowOffset;
        //         uint32_t mmResShape[] = {rows, VEC_M_BASE};
        //         AscendC::ShapeInfo mmResShapeInfo(2, mmResShape);
        //         AscendC::DumpTensor(mmRes[rowOffset * VEC_M_BASE],
        //                             2000U + subLoop * 100U + constInfo.subBlockIdx * 10U + i,
        //                             rows * VEC_M_BASE, mmResShapeInfo);
        //     }
        // }
        constexpr float descaleQK = 1.0f;
        // Q/K scale 已在 C1 随路消费，因此 descaleQK=1。
        if constexpr (USE_DN) {
            // isUpdate 表示是否合并上一 logical S2 tile，不表示 subLoop 0/1。
            const bool hasPreviousS2Loop = !runInfo.isFirstS2Loop;
            if (unlikely(!hasPreviousS2Loop)) {
                FaVectorApi::ProcessVec1VfDnMxfp8<MM_T, INPUT_T, false, HAS_ATTEN, S2_SPLIT>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, attenMaskUb, pScaleSubLoopTensor,
                    ((runInfo.actMSizeAlign32 >> 1U) + 63U) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2U,
                    s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<MM_T, INPUT_T, true, HAS_ATTEN, S2_SPLIT>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, attenMaskUb, pScaleSubLoopTensor,
                    ((runInfo.actMSizeAlign32 >> 1U) + 63U) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2U,
                    s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            }
        } else {
            static_assert(USE_DN, "MXFullQuantMode vector currently implements DN VF only; "
                                  "use_dn=false is reserved for future extension");
        }
        // zyhPrint
        // if (runInfo.actVecMSize != 0U && runInfo.loop == 0U && runInfo.mLoop == 0U) {
        //     uint32_t softmaxStateShape[] = {1U, runInfo.actVecMSize};
        //     AscendC::ShapeInfo softmaxStateShapeInfo(2, softmaxStateShape);
        //     AscendC::DumpTensor(maxUb, 3000U + subLoop * 100U + constInfo.subBlockIdx * 10U,
        //                         runInfo.actVecMSize, softmaxStateShapeInfo);
        //     AscendC::DumpTensor(sumUb, 4000U + subLoop * 100U + constInfo.subBlockIdx * 10U,
        //                         runInfo.actVecMSize, softmaxStateShapeInfo);
        // }
        // zyhPrint
        // if (runInfo.actVecMSize != 0U && runInfo.loop == 0U && runInfo.mLoop == 0U && subLoop == 0U) {
        //     Duplicate<float>(preLoopMaxUb, constInfo.scaleValue, runInfo.actVecMSize);
        //     uint32_t scaleShape[] = {1U, runInfo.actVecMSize};
        //     AscendC::ShapeInfo scaleShapeInfo(2, scaleShape);
        //     AscendC::DumpTensor(preLoopMaxUb, 6000U + constInfo.subBlockIdx * 10U, runInfo.actVecMSize,
        //                         scaleShapeInfo);
        // }
        bmm1ResBuf.SetCrossCore();
        if constexpr (HAS_ATTEN) {
            attenMaskInQue_.template FreeTensor(attenMaskUb);
        }

        // PScale 与 P 写入同一 L1 buffer，最后一个 subLoop 后通知 C2。
        CopyPScaleToL1(outputBuf, pScaleSubLoopTensor, runInfo, constInfo, subLoop);
        pScaleSubLoop0Que_.template FreeTensor(pScaleSubLoopTensor);

        stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        CopyPToL1(outputBuf, stage1CastTensor, constInfo, subLoop);
        stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (IsLastSubLoop(runInfo, subLoop)) {
            outputBuf.SetCrossCore();
        }
        if (unlikely(runInfo.isLastS2Loop && IsLastSubLoop(runInfo, subLoop))) {
            // 最后一个 S2/subLoop 后写 LSE。
            SoftmaxLseCopyOut(runInfo, constInfo, sumUb, maxUb);
        }
    }

    // V2：更新 C2[128,DV]，最后归一化并写回 bf16。
    __aicore__ inline void ProcessVec2(MxBmm2Buf &bmm2ResBuf, MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0U)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        const uint32_t vecMSize = runInfo.actVecMSize;
        const int64_t vec2CalcSize = static_cast<int64_t>(vecMSize) * dTemplateAlign64;
        LocalTensor<MM_T> vec2ResUb = stage2OutBuf_.template Get<MM_T>();
        LocalTensor<MM_T> mmRes = bmm2ResBuf.template GetTensor<MM_T>();
        // zyhPrint
        // if (runInfo.loop == 0U && runInfo.mLoop == 0U && constInfo.subBlockIdx == 0U) {
        //     uint32_t bmm2Shape[] = {vecMSize, dTemplateAlign64};
        //     AscendC::ShapeInfo bmm2ShapeInfo(2, bmm2Shape);
        //     AscendC::DumpTensor(mmRes, 500U + constInfo.subBlockIdx, static_cast<uint32_t>(vec2CalcSize),
        //                         bmm2ShapeInfo);
        // }
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            // 首个 S2 tile 直接作为累计值。
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<MM_T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1U)].template Get<MM_T>();
            if (!runInfo.isLastS2Loop) {
                // 中间 tile 保留 fp32 累计值。
                FlashUpdateNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                // 最后一个 tile 完成 update 与归一化。
                LocalTensor<float> sumUb = softmaxSumBuf_[runInfo.mLoop % (PRELOAD_N + 1U)].template Get<float>();
                FlashUpdateLastNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                // 单 tile 场景单独归一化。
                LocalTensor<float> sumUb = softmaxSumBuf_[runInfo.mLoop % (PRELOAD_N + 1U)].template Get<float>();
                LastDivNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, vec2ResUb, sumUb, vecMSize, static_cast<uint16_t>(dTemplateAlign64), 1.0f);
            }
            CopyOutAttentionOut(runInfo, constInfo, vec2ResUb);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

private:
    __aicore__ inline float DecodeE8M0Scale(uint8_t scale) const
    {
        // e8m0 字节对应 fp32 的 exponent bits[30:23]。
        uint32_t bits = static_cast<uint32_t>(scale) << 23;
        return *((float *)&bits);
    }

    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            return subLoop == 0U ? runInfo.actSingleLoopS2Size : 0U;
        }
        return subLoop == 0U ? S2_SPLIT : runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    __aicore__ inline bool IsLastSubLoop(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        return (runInfo.actSingleLoopS2Size <= S2_SPLIT) || (subLoop != 0U);
    }

    __aicore__ inline void BoolCopyInMxMask(const LocalTensor<uint8_t> &dstTensor, int64_t srcOffset,
                                            uint32_t s2Len, uint32_t qLen)
    {
        if (s2Len == 0U || qLen == 0U) {
            return;
        }

        // DN mask 按 GM row stride 搬到 [S2,64] UB。
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2Len;
        dataCopyParams.blockLen = CeilDiv(qLen, MASK_COPY_BLOCK_BYTES);
        dataCopyParams.dstStride = CeilDiv(VEC_M_BASE, MASK_COPY_BLOCK_BYTES) - dataCopyParams.blockLen;
        if (attenMaskInfo_.attenMaskS2Size % MASK_COPY_BLOCK_BYTES == 0 && qLen % MASK_COPY_BLOCK_BYTES == 0) {
            dataCopyParams.srcStride =
                (static_cast<uint32_t>(attenMaskInfo_.attenMaskS2Size) - qLen) / MASK_COPY_BLOCK_BYTES;
            DataCopy(dstTensor, attenMaskGm_[static_cast<uint64_t>(srcOffset)], dataCopyParams);
        } else {
            DataCopyExtParams dataCopyExtParams;
            dataCopyExtParams.blockCount = s2Len;
            dataCopyExtParams.blockLen = qLen;
            dataCopyExtParams.srcStride = static_cast<uint32_t>(attenMaskInfo_.attenMaskS2Size) - qLen;
            dataCopyExtParams.dstStride =
                CeilDiv(VEC_M_BASE, MASK_COPY_BLOCK_BYTES) - CeilDiv(qLen, MASK_COPY_BLOCK_BYTES);
            DataCopyPadExtParams<uint8_t> dataCopyPadParams;
            DataCopyPad(dstTensor, attenMaskGm_[static_cast<uint64_t>(srcOffset)], dataCopyExtParams,
                        dataCopyPadParams);
        }
    }

    __aicore__ inline int64_t ComputeCausalMaskOffset(const MxRunInfo &runInfo, int64_t segmentS2TokenOffset) const
    {
        const int64_t deltaN = static_cast<int64_t>(runInfo.actS1Size) - static_cast<int64_t>(runInfo.actS2Size);
        const int64_t deltaCausalOrNext = static_cast<int64_t>(runInfo.s1Idx) - segmentS2TokenOffset - deltaN;
        return regbaseutil::ComputeOffsetForCausal(
            deltaCausalOrNext, M_BASE, S2_SPLIT, static_cast<uint32_t>(attenMaskInfo_.attenMaskS2Size),
            static_cast<int64_t>(runInfo.vecMbaseIdx), USE_DN);
    }

    __aicore__ inline int64_t ComputeNoCompressMaskOffset(const MxRunInfo &runInfo,
                                                          int64_t segmentS2TokenOffset) const
    {
        const int64_t batchOffset = static_cast<int64_t>(runInfo.bIdx) * attenMaskInfo_.attenMaskS1Size *
                                    attenMaskInfo_.attenMaskS2Size;
        const int64_t qOffset = static_cast<int64_t>(runInfo.s1Idx + runInfo.vecMbaseIdx) *
                                attenMaskInfo_.attenMaskS2Size;
        return batchOffset + qOffset + segmentS2TokenOffset;
    }

    __aicore__ inline bool CopyAttenMaskSegment(LocalTensor<uint8_t> &attenMaskUb, const MxRunInfo &runInfo,
                                                uint32_t dstS2Offset, uint32_t segmentS2Len,
                                                int64_t sparseBlkIdx, int64_t segmentS2TokenOffset,
                                                bool needAtten)
    {
        // 返回值仅表示本段是否发起了 GM->UB 的 MTE2 搬运；Duplicate 写 UB 不需要 MTE2_V 同步。
        const uint32_t dstOffset = dstS2Offset * VEC_M_BASE;
        if (segmentS2Len == 0U) {
            return false;
        }
        if (sparseBlkIdx < 0) {
            Duplicate<uint8_t>(attenMaskUb[dstOffset], 0, segmentS2Len * VEC_M_BASE);
            return false;
        }

        // 完全可见段填 1，其余段搬 right-down causal mask。
        const bool isRightDownCausal =
            attenMaskInfo_.compressMode ==
            static_cast<uint8_t>(regbaseutil::AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
        if (isRightDownCausal && !needAtten) {
            Duplicate<uint8_t>(attenMaskUb[dstOffset], 1, segmentS2Len * VEC_M_BASE);
            return false;
        }

        const int64_t maskOffset = isRightDownCausal ? ComputeCausalMaskOffset(runInfo, segmentS2TokenOffset) :
                                                       ComputeNoCompressMaskOffset(runInfo, segmentS2TokenOffset);
        BoolCopyInMxMask(attenMaskUb[dstOffset], maskOffset, segmentS2Len, VEC_M_BASE);
        return true;
    }

    __aicore__ inline LocalTensor<uint8_t> CopyAttenMaskToUb(const MxRunInfo &runInfo, uint32_t subLoop,
                                                             uint32_t s2CalcSize)
    {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue_.template AllocTensor<uint8_t>();

        const uint32_t subLoopStart = subLoop * S2_SPLIT;
        const uint32_t subLoopEnd = subLoopStart + s2CalcSize;
        bool needMte2ToVSync = false;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockS2Len = runInfo.sparseBlockRealSize[i];
            const uint32_t blockTileStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockTileEnd = blockTileStart + blockS2Len;
            if (blockTileEnd <= subLoopStart || blockTileStart >= subLoopEnd) {
                continue;
            }
            const uint32_t overlapStart = blockTileStart > subLoopStart ? blockTileStart : subLoopStart;
            const uint32_t overlapEnd = blockTileEnd < subLoopEnd ? blockTileEnd : subLoopEnd;
            const uint32_t segmentS2Len = overlapEnd - overlapStart;
            const uint32_t segmentOffsetInBlock = overlapStart - blockTileStart;
            const int64_t segmentS2TokenOffset =
                runInfo.sparseBlockTokenOffset[i] + static_cast<int64_t>(segmentOffsetInBlock);
            const bool hasMte2Copy =
                CopyAttenMaskSegment(attenMaskUb, runInfo, overlapStart - subLoopStart, segmentS2Len,
                                     runInfo.sparseBlockIdx[i], segmentS2TokenOffset,
                                     runInfo.sparseBlockPartialMask[i]);
            needMte2ToVSync = needMte2ToVSync || hasMte2Copy;
        }
        if (needMte2ToVSync) {
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
        attenMaskInQue_.template EnQue(attenMaskUb);
        return attenMaskInQue_.template DeQue<uint8_t>();
    }

    __aicore__ inline void CopyPScaleToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         LocalTensor<SCALE_T> &pScaleSubLoopTensor, const MxRunInfo &runInfo,
                                         const MxConstInfo &constInfo, uint32_t subLoop)
    {
        // 将 subLoop PScale 整理到完整 P[128,512] 后方。
        constexpr uint64_t pScaleL1Offset = M_BASE * S2_BASE;
        LocalTensor<SCALE_T> pScaleL1Tensor = outputBuf.GetTensor<SCALE_T>(pScaleL1Offset);
        pScaleSubLoop0Que_.template EnQue(pScaleSubLoopTensor);
        pScaleSubLoop0Que_.template DeQue<SCALE_T>();
        constexpr uint64_t pScaleDataLen = (M_BASE >> 1U) * S2_SPLIT / MX_SCALE_GROUP;
        constexpr uint16_t pScaleDstStride = S2_SPLIT / MX_SCALE_GROUP / 2U - 1U;
        const uint64_t vecOffset = constInfo.subBlockIdx * pScaleDataLen;
        if (subLoop % 2U == 1U) {
            for (uint16_t i = 0U; i < 4U; ++i) {
                DataCopy(pScaleL1Tensor[vecOffset + i * 32U], pScaleSubLoopTensor, {4, 1, 0, pScaleDstStride});
            }
        } else if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            for (uint16_t i = 0U; i < 2U; ++i) {
                DataCopy(pScaleL1Tensor[vecOffset + i * 256U], pScaleSubLoopTensor, {1, 8, 0, 0});
            }
        }
    }

    __aicore__ inline void CopyPToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                    LocalTensor<INPUT_T> &stage1CastTensor, const MxConstInfo &constInfo,
                                    uint32_t subLoop)
    {
        // 两个 AIV、两个 subLoop 分别写入 P 的对应象限。
        LocalTensor<INPUT_T> pL1Tensor = outputBuf.GetTensor<INPUT_T>();
        const int64_t subLoopOffset = static_cast<int64_t>(S2_SPLIT) * M_BASE * subLoop;
        constexpr uint16_t elementSize = 32U;
        constexpr uint32_t singleProcessSOuterSize = M_BASE >> 1U;
        constexpr uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1U) / elementSize;
        constexpr uint32_t actVec0Align32 = M_BASE >> 1U;
        for (uint32_t i = 0U; i < actCopyCount; ++i) {
            uint32_t dstOffset = 32U * S2_SPLIT * i + subLoopOffset;
            if (constInfo.subBlockIdx == 1U) {
                dstOffset += S2_SPLIT * actVec0Align32;
            }
            uint32_t srcOffset = i * (65U << 5);
            DataCopy(pL1Tensor[dstOffset], stage1CastTensor[srcOffset], {4, S2_SPLIT / 4U, S2_SPLIT / 4U + 2U, 0});
        }
    }

    __aicore__ inline void CopyOutAttentionOut(MxRunInfo &runInfo, const MxConstInfo &constInfo,
                                               LocalTensor<MM_T> &vec2ResUb)
    {
        // 按 TND token/head stride 写回。
        LocalTensor<OUTPUT_T> attenOut;
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND,
             static_cast<int64_t>(runInfo.actVecMSize) * dTemplateAlign64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) >> 4;
        dataCopyParams.dstStride = constInfo.attentionOutStride;
        const uint64_t dstOffset =
            ((runInfo.queryTokenBase + runInfo.s1Idx + runInfo.vecMbaseIdx) * constInfo.realN2Size +
             runInfo.realN2Idx) * constInfo.dSizeV;
        DataCopyPad(attentionOutGm_[dstOffset], attenOut, dataCopyParams);
    }

    __aicore__ inline void SoftmaxLseCopyOut(MxRunInfo &runInfo, const MxConstInfo &constInfo,
                                             LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (!constInfo.isSoftmaxLseEnable || runInfo.actVecMSize == 0U) {
            return;
        }
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        LocalTensor<float> lseUb = softmaxExpBuf_[0].template Get<float>();
        ComputeLseOutputVF(lseUb, sumUb, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = constInfo.softmaxLseStride;
        const uint64_t dstOffset =
            (runInfo.queryTokenBase + runInfo.s1Idx + runInfo.vecMbaseIdx) * constInfo.realN2Size +
            runInfo.realN2Idx;
        DataCopyPad(softmaxLseGm_[dstOffset], lseUb, dataCopyParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<uint8_t> pScaleGm_;
    GlobalTensor<uint8_t> attenMaskGm_;
    regbaseutil::AttenMaskInfo attenMaskInfo_;
    TQue<QuePosition::VECIN, 1> attenMaskInQue_;
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECOUT, 1> pScaleSubLoop0Que_;
    TBuf<> stage2OutBuf_;
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1U];
    TBuf<> softmaxSumBuf_[PRELOAD_N + 1U];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1U];
    TBuf<> preLoopMaxBuf_;
    TBuf<> preLoopSumBuf_;
    TBuf<> firstLoopSumBuf_;
    TBuf<> vselrIndexesBuf_[4];
    TEventID mte3ToVId_[1];
    TEventID vToMte3Id_[1];
    MM_T negativeFloatScalar_ = 0.0f;
    float pScaleValue_ = 1.0f;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_VEC_H_
