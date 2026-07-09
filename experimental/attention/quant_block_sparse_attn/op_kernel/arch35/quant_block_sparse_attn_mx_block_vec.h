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
/*
 * MX vector block 主要负责两件事：
 *   V1：消费 fp32 C1 tile，执行 scale_value + softmax，然后使用输入 quantScale1
 *       将 P 量化为 fp8_e4m3fn，并额外生成 C2 用 P 反量化 scale。
 *   V2：消费 fp32 C2 tile，在多个 sparse S2 loop 间更新累计 attention output，
 *       最后 cast 为 bf16，并按需写回 softmax_lse。
 *
 * 当前 kernel 只打开 MX S2=512 计算 tile。一个 512 宽 logical S2 tile
 * 由多个 128/64-token sparse block 拼接而成，并拆成两个 256 列的 V1 subLoop。
 * V1 将两个 subLoop 都写入同一个 L1 P buffer，因此 C2 看到的是完整的
 * [128, 512] 左矩阵。
 */
template <bool useDn = true>
class QuantBlockSparseAttnMxBlockVec {
public:
    using INPUT_T = fp8_e4m3fn_t;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = float;
    using OUTPUT_T = bfloat16_t;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    static constexpr bool USE_DN = useDn;
    static constexpr uint32_t M_BASE = 128U;
    static constexpr uint32_t S2_BASE = 512U;
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = 128U;
    static constexpr uint32_t DV_BASE = 128U;
    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    static constexpr uint32_t dTemplateAlign64 = 128U;
    static constexpr uint32_t DN_VSELR_INDEX_SIZE = 256U;
    static constexpr uint32_t DN_VSELR_GROUP_NUM = 4U;
    static constexpr uint32_t DN_VSELR_GROUP_SIZE = DN_VSELR_INDEX_SIZE / DN_VSELR_GROUP_NUM;

    __aicore__ inline void Init(TPipe *pipe, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut)
    {
        if ASCEND_IS_AIV {
            tPipe_ = pipe;
            attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
            pScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)pScale);
            uint32_t negativeIntScalar = NEGATIVE_MIN_VALUE_FP32;
            negativeFloatScalar_ = *((MM_T *)&negativeIntScalar);
            // quantScale1 是输入传入的静态 P 量化 scale，shape 为 [1]，存储类型为 e8m0。
            // 这里解码成 float 后作为 pScaleValue_ 传给 VF；它参与 softmax 后 P 的量化。
            // 注意：后面的 pScaleSubLoopTensor 不是这个输入，而是 VF 为 C2 额外生成的
            // P 反量化 scale 临时 tensor。
            pScaleValue_ = pScale != nullptr ? DecodeE8M0Scale(pScaleGm_.GetValue(0)) : 1.0f;
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // vector 侧 UB 分配主要服务 V1/V2 两段：
            //   V1 参考 common MX 的 DN VF：消费 C1 fp32 score，输出 fp8 P 和 C2 用 P 反量化 scale；
            //   V2 参考 BSA FP8 的 flash-update 思路：缓存 fp32 O，跨 sparse S2 loop 做 update。
            //
            // stage2OutBuf_ 保存 V2 的 fp32 累计输出。每个 AIV subBlock 处理 64 行，
            // D 模板按 128 对齐，因此容量为 64 * 128 * sizeof(float)。
            tPipe_->InitBuffer(stage2OutBuf_, 64U * dTemplateAlign64 * sizeof(MM_T));
            // stage1OutQue_ 保存 V1 生成的 fp8 P 临时结果。设置两个队列槽是为了配合
            // PRELOAD_N 流水，避免当前 subLoop 的 P 尚未搬到 L1 时被下一轮覆盖。
            // 16896B 来自 DN VF 输出布局：64 行 * 256 列 fp8 为主体，再加上 VF 内部
            // 为 NZ-like/DN packed layout 保留的对齐和间隔空间。
            tPipe_->InitBuffer(stage1OutQue_[0], 1, 16896U);
            tPipe_->InitBuffer(stage1OutQue_[1], 1, 16896U);
            // pScaleSubLoop0Que_ 保存 V1 为单个 256-column subLoop 生成的 C2 用 P 反量化 scale。
            // 该 scale 是 e8m0，按 MX group 输出；512B 足够容纳当前 [64,256] 半 M tile
            // 的 scale 以及 VF 拷贝到 L1 前的对齐空间。
            tPipe_->InitBuffer(pScaleSubLoop0Que_, 1, 512U);
            // softmaxSum/Max/Exp 各分配 PRELOAD_N+1 份。原因是主流水中当前 task 做 V1，
            // PRELOAD_N 之前的 task 做 V2；三槽 ring buffer 可以让不同 task 的
            // softmax 统计量在 V1/V2 交错执行时互不覆盖。
            tPipe_->InitBuffer(softmaxSumBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxSumBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxSumBuf_[2], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxMaxBuf_[2], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[0], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[1], 256U);
            tPipe_->InitBuffer(softmaxExpBuf_[2], 256U);
            // preLoop/firstLoop buffer 用于 online softmax update。
            // 后续 logical S2 tile 到来时，需要用上一轮 max/sum 修正旧输出；
            // firstLoopSumBuf_ 处理首轮/尾轮路径差异，避免单 S2 tile 场景丢失归一化信息。
            tPipe_->InitBuffer(preLoopMaxBuf_, 256U);
            tPipe_->InitBuffer(preLoopSumBuf_, 256U);
            tPipe_->InitBuffer(firstLoopSumBuf_, 256U);
            // vselrIndexesBuf_ 是 DN VF 的 lane 重排索引表。它不是数据缓存，而是告诉 VF
            // 如何把 DN packed 的 P 重新选择成 C2 需要的顺序。参考 common MX
            // fia_block_vec_fullquant_mx.h 中 USE_DN=true 的 DN_INDEX 初始化逻辑。
            // use_dn 模板参数仍保留在 tilingkey/kernel 链路中，但当前 MX 路径只实现 DN VF；
            // 后续真正接入非 DN 时，再补对应 VF 和 GT_64/GT_0 索引表，避免现在留下误导性分支。
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

            // V2 最终写 GM 前需要在 vector cast 和 MTE3 copy 之间做同步。
            // 这里显式分配事件，避免 stage2OutBuf_ 还在被 MTE3 读时被下一轮 V 操作覆盖。
            mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
    }

    /*
     * V1 输入：
     *   - bmm1ResBuf：cube 输出的一块 fp32 C1 tile，当前 subLoop 下逻辑形状为 [128, 256]。
     *   - runInfo 对应的 softmax buffer：跨 sparse S2 loop 复用上一轮 max/sum。
     *
     * V1 输出：
     *   - outputBuf[0, 128*512)：fp8_e4m3fn P。subLoop 0 填充列 [0, 256)，
     *     subLoop 1 填充列 [256, 512)。
     *   - outputBuf[128*512, ...)：fp8_e8m0 P 反量化 scale。它不是输入 quantScale1，
     *     而是 V1 生成、C2 与 P 一起消费的左操作数 MX scale。
     */
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo, uint32_t subLoop)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0U)) {
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
        LocalTensor<uint8_t> emptyMask;

        const uint32_t s2CalcSize = GetSubLoopS2Size(runInfo, subLoop);
        constexpr float descaleQK = 1.0f;
        // VF 执行流程：C1 * scale_value -> mask/softmax -> 在线更新 sum/max
        // -> 使用输入 quantScale1 静态量化 P(fp8_e4m3fn) + 生成 C2 用 P 反量化 scale(fp8_e8m0)。
        // descaleQK 为 1.0，因为 Q/K 反量化 scale 已经在 BMM1 的 cube LoadData
        // MX 随路反量化过程中被消费。
        if constexpr (USE_DN) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                FaVectorApi::ProcessVec1VfDnMxfp8<MM_T, INPUT_T, false, false, S2_SPLIT>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, emptyMask, pScaleSubLoopTensor,
                    ((runInfo.actMSizeAlign32 >> 1U) + 63U) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2U,
                    s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<MM_T, INPUT_T, true, false, S2_SPLIT>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_, emptyMask, pScaleSubLoopTensor,
                    ((runInfo.actMSizeAlign32 >> 1U) + 63U) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2U,
                    s2CalcSize, static_cast<MM_T>(constInfo.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            }
        } else {
            static_assert(USE_DN, "MXFullQuantMode vector currently implements DN VF only; "
                                  "use_dn=false is reserved for future extension");
        }
        bmm1ResBuf.SetCrossCore();

        // 先拷贝 C2 用 P 反量化 scale，再拷贝 P。C2 只有在最后一个 subLoop 的
        // outputBuf.SetCrossCore() 之后才会读取；在此之前 P/scale 都只属于当前 V1 task。
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
            // 只有完成最后一个 S2 block 且完成最后一个 256-column subLoop 后，LSE 才完整。
            SoftmaxLseCopyOut(runInfo, constInfo, sumUb, maxUb);
        }
    }

    /*
     * V2 输入：
     *   - bmm2ResBuf：由完整 P[128,512] 生成的 fp32 C2 结果 [128, DV]。
     *
     * V2 输出：
     *   - 累计后的 attention result。中间 logical S2 tile 保持 fp32 留在 UB；
     *     最后一块会除以 softmax sum，cast 为 bf16，再按 TND layout 写回 GM。
     */
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
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            // 第一个 S2 tile 没有历史输出需要 update，因此 C2 结果直接作为累计值。
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<MM_T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1U)].template Get<MM_T>();
            if (!runInfo.isLastS2Loop) {
                // 中间 S2 tile：应用 online-softmax 修正因子 expUb，
                // 并将结果以 fp32 保留给下一个 S2 tile。
                FlashUpdateNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                // 最后一个 S2 tile：完成 update，并除以最终 softmax 分母。
                LocalTensor<float> sumUb = softmaxSumBuf_[runInfo.mLoop % (PRELOAD_N + 1U)].template Get<float>();
                FlashUpdateLastNew<MM_T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                // 只有一个 S2 tile 的退化场景不会进入 FlashUpdateLastNew，
                // 因此在这里对第一块 C2 结果做归一化。
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
        // e8m0 存储的是 fp32 指数字节。将其放到 bits[30:23] 后，
        // 即可得到对应的 2 的幂次 fp32 值。例如：0x7f -> 1.0f。
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

    __aicore__ inline void CopyPScaleToL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         LocalTensor<SCALE_T> &pScaleSubLoopTensor, const MxRunInfo &runInfo,
                                         const MxConstInfo &constInfo, uint32_t subLoop)
    {
        // VF 以紧凑 subLoop layout 输出 C2 用 P 反量化 scale。C2 期望该 scale 放在
        // 完整 [128,512] P tile 后方，并与 P 的 MX group 对齐。两种拷贝模式不同：
        //   512-token logical tile：等 subLoop 1 时一起整理两个 256 半块的 scale；
        //   tail <=256 logical tile：直接拷贝 subLoop 0 的 scale，并释放 P buffer。
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
        // stage1CastTensor 由 VF 按 DN 友好的 packed 顺序生成。这里将其拷贝到 BMM2
        // 读取的 L1 P tile 位置。AIV subBlock 0/1 各写 128 行 query 的一半；
        // subLoop 0/1 分别选择前/后 256 个 S2 列。
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
        // 输出 layout 为 TND。vecMbaseIdx 用于处理 AIV subBlock 的行拆分；
        // attentionOutStride 用于在连续 query token 之间跳过其他 head。
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
        // LSE = log(sum(exp(score - max))) + max。它与 attentionOut 使用相同的
        // TND token/head 寻址，但每个 query token/head 只有一个 float，而不是 DV 个元素。
        if (!constInfo.isSoftmaxLseEnable || runInfo.actVecMSize == 0U) {
            return;
        }
        LocalTensor<float> lseUb = softmaxExpBuf_[0].template Get<float>();
        ComputeLseOutputVF(lseUb, sumUb, maxUb, runInfo.actVecMSize);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.blockCount = runInfo.actVecMSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = constInfo.softmaxLseStride;
        const uint64_t dstOffset =
            (runInfo.queryTokenBase + runInfo.s1Idx + runInfo.vecMbaseIdx) * constInfo.realN2Size +
            runInfo.realN2Idx;
        DataCopyPad(softmaxLseGm_[dstOffset], lseUb, dataCopyParams);
    }

    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<uint8_t> pScaleGm_;
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
