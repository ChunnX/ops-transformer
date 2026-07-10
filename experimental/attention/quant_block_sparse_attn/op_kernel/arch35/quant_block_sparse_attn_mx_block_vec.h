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
            if constexpr (HAS_ATTEN) {
                // attenMaskInQue_ 保存当前 256-column V1 subLoop 使用的 bool mask。
                // 参考 BSA FP8 DN 路径的 attenMaskInQue[0]：
                //   - VF 的 DN mask 布局为 [S2, M/2]，每个 S2 token 一行，
                //     当前 AIV subBlock 只处理 64 个 query 行；
                //   - MX 的 logical S2 tile 是 512，但 V1 每次只处理一个 256-column subLoop，
                //     因此 UB 容量仍为 256 * 64 = 16384B；
                //   - 当 sparse block size 为 128/64 时，一个 subLoop 可能由多个
                //     task 内排序后的 sparse block 拼接，CopyAttenMaskToUb 会按 segment 填充。
                tPipe_->InitBuffer(attenMaskInQue_, 1, ATTEN_MASK_UB_SIZE);
            }
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
        }
        // 与 FP8 ProcessVec1Dn 的流水顺序一致：先发起 attenMask GM->UB 搬运，
        // 再等待 AIC 的 C1 结果。这样 mask 的 MTE2 搬运可以与 BMM1 计算重叠；
        // WaitCrossCore 之后 VF 才真正读取 mmRes，因此不会提前消费未完成的 C1。
        bmm1ResBuf.WaitCrossCore();
        constexpr float descaleQK = 1.0f;
        // VF 执行流程：C1 * scale_value -> mask/softmax -> 在线更新 sum/max
        // -> 使用输入 quantScale1 静态量化 P(fp8_e4m3fn) + 生成 C2 用 P 反量化 scale(fp8_e8m0)。
        // descaleQK 为 1.0，因为 Q/K 反量化 scale 已经在 BMM1 的 cube LoadData
        // MX 随路反量化过程中被消费。
        if constexpr (USE_DN) {
            // VF 的 isUpdate 表示是否需要与“上一轮 logical S2 tile”的 online-softmax 状态合并，
            // 不是表示当前 512 tile 内的 subLoop 0/1。第一轮 S2 tile 的两个 256 subLoop
            // 都没有历史 S2 状态，因此都走 NoUpdate；NoUpdate VF 内部会用 subLoop==1
            // 合并当前 512 tile 的前后两个 256 半块。第二轮及以后才都走 Update。
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
        bmm1ResBuf.SetCrossCore();
        if constexpr (HAS_ATTEN) {
            attenMaskInQue_.template FreeTensor(attenMaskUb);
        }

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

    __aicore__ inline void BoolCopyInMxMask(const LocalTensor<uint8_t> &dstTensor, int64_t srcOffset,
                                            uint32_t s2Len, uint32_t qLen)
    {
        if (s2Len == 0U || qLen == 0U) {
            return;
        }

        // 这是 BoolCopyInRegbase 的 MX 版小型复用：FP8 版本依赖 regbaseutil::ConstInfo，
        // 而 MX 路径使用独立的 MxConstInfo。实际 DN mask 搬运只需要 GM row stride
        // attenMaskS2Size 和 UB row stride VEC_M_BASE，因此在这里保留同样的数据搬运形态。
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

    __aicore__ inline void CopyAttenMaskSegment(LocalTensor<uint8_t> &attenMaskUb, const MxRunInfo &runInfo,
                                                uint32_t dstS2Offset, uint32_t segmentS2Len,
                                                int64_t sparseBlkIdx, int64_t segmentS2TokenOffset,
                                                bool needAtten)
    {
        const uint32_t dstOffset = dstS2Offset * VEC_M_BASE;
        if (segmentS2Len == 0U) {
            return;
        }
        if (sparseBlkIdx < 0) {
            Duplicate<uint8_t>(attenMaskUb[dstOffset], 0, segmentS2Len * VEC_M_BASE);
            return;
        }

        // MX 当前 host 只放开 sparseMode=3，也就是 right-down causal mask。
        // 对完全可见的 segment，直接填 1，避免多余 GM 搬运；跨 causal 边界的
        // segment 才按 FP8 DN 同款 offset 从 attenMask GM 搬到 UB。
        const bool isRightDownCausal =
            attenMaskInfo_.compressMode ==
            static_cast<uint8_t>(regbaseutil::AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
        if (isRightDownCausal && !needAtten) {
            Duplicate<uint8_t>(attenMaskUb[dstOffset], 1, segmentS2Len * VEC_M_BASE);
            return;
        }

        const int64_t maskOffset = isRightDownCausal ? ComputeCausalMaskOffset(runInfo, segmentS2TokenOffset) : 0;
        BoolCopyInMxMask(attenMaskUb[dstOffset], maskOffset, segmentS2Len, VEC_M_BASE);
    }

    __aicore__ inline LocalTensor<uint8_t> CopyAttenMaskToUb(const MxRunInfo &runInfo, uint32_t subLoop,
                                                             uint32_t s2CalcSize)
    {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue_.template AllocTensor<uint8_t>();
        // 与 FP8 AttenMaskCopyInDn 一致，只写当前拼接后有效的 S2 segment：
        // partial/full-mask segment 从 GM 搬运压缩 mask，完全可见 segment 填 1。
        // originN 之后的 C1 tail 会在 MX VF 内先写成最小值，因此不需要预清零整块
        // mask UB；这样也避免 vector Duplicate 与 MTE2 DataCopy 同时写同一块 UB。

        const uint32_t subLoopStart = subLoop * S2_SPLIT;
        const uint32_t subLoopEnd = subLoopStart + s2CalcSize;
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
            CopyAttenMaskSegment(attenMaskUb, runInfo, overlapStart - subLoopStart, segmentS2Len,
                                 runInfo.sparseBlockIdx[i], segmentS2TokenOffset,
                                 runInfo.sparseBlockPartialMask[i]);
        }
        attenMaskInQue_.template EnQue(attenMaskUb);
        return attenMaskInQue_.template DeQue<uint8_t>();
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
