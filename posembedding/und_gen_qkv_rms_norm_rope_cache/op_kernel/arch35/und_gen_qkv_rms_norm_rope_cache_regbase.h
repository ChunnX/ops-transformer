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
 * \file und_gen_qkv_rms_norm_rope_cache_regbase.h
 * \brief UndGenQkvRmsNormRopeCache regbase(DAV_3510 / arch35) 模板
 *
 * 切分（与 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h 一一对应）：
 *   多核：按输出 token 维 total 余数分配，前 formerCoreNum 个核处理 blockFactor 个
 *         token，其余核处理 tailBlockFactor 个；各核独立无同步。
 *   UB  ：只切 token 维，每次处理 ubFactor 个 token。CopyIn 按 tile 批量预取，
 *         计算与写出逐 token，计算中间量走 vreg 不落 UB。
 *
 * 每个 out_t 的计算流程：
 *   1) src_t = cat_indices[out_t]；isUnd = (src_t < undLen)
 *   2) 从 und_qkv / gen_qkv 取 [N, D] 行 -> Cast bf16->float32 -> 按 N 维拆 Q/K/V
 *   3) Q/K 做 RMSNorm（und/gen 两套权重按 isUnd 选择）
 *   4) 读 positions[0..2, out_t]，三轴 cos_sin 原样搬进 UB 的 [3, D] 窗口，
 *      VF 里用 Gather + axisLut 在寄存器内合并，再对 Q/K 做标准 RoPE；V 不参与 norm/rope
 *   5) Q Cast float32->bf16 写 q[out_t]；K/V Cast 后按 slot = slot_mapping[out_t]
 *      写入 k_cache/v_cache 的第 slot 行
 *
 * RMSNorm + MRoPE 融合在一个 VF 段里完成，见 und_gen_qkv_rms_norm_rope_cache_mrope_vf.h。
 * 中间量（fp32 的 x、合并后的 cos/sin、RoPE 交叉项）全部留在 vreg，不占 UB。
 */

#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_

#include "kernel_operator.h"
#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_PLATFORM_INCLUDED
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_PLATFORM_INCLUDED
#ifdef __CCE_UT_TEST__
#include "../../../kv_rms_norm_rope_cache/op_kernel/arch35/platform.h"
#include "../../../../third_party/opbase/include/op_common/op_kernel/platform_util.h"
#else
#include "op_kernel/platform_util.h"
#endif
#endif
#include "und_gen_qkv_rms_norm_rope_cache_mrope_vf.h"

namespace UndGenQkvRmsNormRopeCache {
using namespace AscendC;

constexpr static int64_t MROPE_AXIS_NUM = 3;
constexpr static int64_t DIGIT_TWO = 2;
constexpr static int64_t WEIGHT_NUM = 4;          // und_q / und_k / gen_q / gen_k
constexpr static int64_t BLOCK_ALIGN_BYTES = 32;
constexpr static int64_t BUFFER_NUM_DB = 2;
constexpr static int64_t BUFFER_NUM_SINGLE = 1;

// wInQue / wFp32Buf 内 4 个 gamma 的排布下标（bf16 与 float 两份用同一套下标）
constexpr static int64_t WEIGHT_IDX_UND_Q = 0;
constexpr static int64_t WEIGHT_IDX_UND_K = 1;
constexpr static int64_t WEIGHT_IDX_GEN_Q = 2;
constexpr static int64_t WEIGHT_IDX_GEN_K = 3;

// 标量写 UB 后被向量读，需要显式 S->V 同步；这里只用一个事件号
constexpr static event_t EVT_S_TO_V_INDEX_READY = EVENT_ID0;

/**
 * @brief UndGenQkvRmsNormRopeCache regbase 模板
 * @tparam T_QKV   und_qkv/gen_qkv 与 q 输出的数据类型（bfloat16_t）
 * @tparam T_CACHE k_cache/v_cache 的数据类型（bfloat16_t）
 */
template <typename T_QKV, typename T_CACHE>
class UndGenQkvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline UndGenQkvRmsNormRopeCacheRegbase(TPipe* pipe, const UndGenQkvRmsNormRopeCacheTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR undQkv, GM_ADDR undWeightsQ, GM_ADDR undWeightsK, GM_ADDR cosSinCache,
                                GM_ADDR kCache, GM_ADDR vCache, GM_ADDR slotMapping, GM_ADDR positions, GM_ADDR genQkv,
                                GM_ADDR genWeightsQ, GM_ADDR genWeightsK, GM_ADDR catIndices, GM_ADDR q)
    {
        ParseTiling();
        BindGlobalTensors(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping, positions, genQkv,
                          genWeightsQ, genWeightsK, catIndices, q);
        CalcCoreRange();
        InitUbBuffers();
    }

    __aicore__ inline void Process()
    {
        // 权重每核只搬一次，常驻到整个 tile 循环结束。走 VECIN 队列而不是裸 TBuf，
        // 是为了拿到 EnQue/DeQue 提供的 MTE2->V 同步，保证搬运落地后计算才读
        PrepareWeights();
        // axisLut 与 token 无关，每核只建一次
        BuildMropeGatherIndex();

        // 每次处理 ubFactor 个 token；尾块 token 数不足时按实际数量处理
        for (int64_t tileStart = coreStart_; tileStart < coreEnd_; tileStart += ubFactor_) {
            int64_t tokenNum = coreEnd_ - tileStart;
            if (tokenNum > ubFactor_) {
                tokenNum = ubFactor_;
            }
            CopyIn(tileStart, tokenNum);
            Compute(tileStart, tokenNum);
            CopyOut(tileStart, tokenNum);
        }

        wInQue_.FreeTensor(wBf16Local_);
    }

private:
    __aicore__ inline void ParseTiling()
    {
        totalTokens_ = tiling_->totalTokens;
        undLen_ = tiling_->undLen;
        numHead_ = tiling_->numHead;
        numHeadQ_ = tiling_->numHeadQ;
        numHeadK_ = tiling_->numHeadK;
        numHeadV_ = tiling_->numHeadV;
        headDim_ = tiling_->headDim;
        maxPos_ = tiling_->maxPos;
        blockNum_ = tiling_->blockNum;
        blockSize_ = tiling_->blockSize;
        hasGen_ = tiling_->hasGen != 0;
        hasCatIndices_ = tiling_->hasCatIndices != 0;
        mropeSection_[0] = tiling_->mropeSectionT;
        mropeSection_[1] = tiling_->mropeSectionH;
        mropeSection_[2] = tiling_->mropeSectionW;
        epsilon_ = tiling_->epsilon;
        reciprocal_ = tiling_->reciprocal;

        usedCoreNum_ = tiling_->usedCoreNum;
        formerCoreNum_ = tiling_->formerCoreNum;
        blockFactor_ = tiling_->blockFactor;
        tailBlockFactor_ = tiling_->tailBlockFactor;
        ubFactor_ = tiling_->ubFactor;

        headNumQK_ = numHeadQ_ + numHeadK_; // V 不参与 RMSNorm/RoPE
        halfDim_ = headDim_ / DIGIT_TWO;
        rowElems_ = numHead_ * headDim_;    // 单 token 的 QKV 元素数
        maxSlot_ = blockNum_ * blockSize_;  // slot_mapping 的合法上界（不含）
    }

    __aicore__ inline void BindGlobalTensors(GM_ADDR undQkv, GM_ADDR undWeightsQ, GM_ADDR undWeightsK,
                                             GM_ADDR cosSinCache, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR slotMapping,
                                             GM_ADDR positions, GM_ADDR genQkv, GM_ADDR genWeightsQ,
                                             GM_ADDR genWeightsK, GM_ADDR catIndices, GM_ADDR q)
    {
        undQkvGm_.SetGlobalBuffer((__gm__ T_QKV*)undQkv);
        undWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV*)undWeightsQ);
        undWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV*)undWeightsK);
        cosSinCacheGm_.SetGlobalBuffer((__gm__ float*)cosSinCache);
        kCacheGm_.SetGlobalBuffer((__gm__ T_CACHE*)kCache);
        vCacheGm_.SetGlobalBuffer((__gm__ T_CACHE*)vCache);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        positionsGm_.SetGlobalBuffer((__gm__ int64_t*)positions);
        if (hasGen_) {
            genQkvGm_.SetGlobalBuffer((__gm__ T_QKV*)genQkv);
            genWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV*)genWeightsQ);
            genWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV*)genWeightsK);
        }
        if (hasCatIndices_) {
            catIndicesGm_.SetGlobalBuffer((__gm__ int64_t*)catIndices);
        }
        qGm_.SetGlobalBuffer((__gm__ T_QKV*)q);
    }

    // 多核切分：前 formerCoreNum 个核多处理 1 个 token，与 host CalBlockTiling 一致
    __aicore__ inline void CalcCoreRange()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) { // SetBlockDim 已按 usedCoreNum 下发，这里只是兜底
            coreStart_ = 0;
            coreEnd_ = 0;
            return;
        }
        int64_t coreTokenNum;
        if (blockIdx < formerCoreNum_) {
            coreStart_ = blockIdx * blockFactor_;
            coreTokenNum = blockFactor_;
        } else {
            coreStart_ = formerCoreNum_ * blockFactor_ + (blockIdx - formerCoreNum_) * tailBlockFactor_;
            coreTokenNum = tailBlockFactor_;
        }
        coreEnd_ = coreStart_ + coreTokenNum;
    }

    // UB 划分严格对应 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h 的 "UB 划分" 注释，
    // 改这里必须同步改 host 的 CalUbTiling，否则 ubFactor 反推会与实际占用不符
    __aicore__ inline void InitUbBuffers()
    {
        pipe_->InitBuffer(qkvInQue_, BUFFER_NUM_DB, ubFactor_ * rowElems_ * sizeof(T_QKV));
        pipe_->InitBuffer(cosSinInQue_, BUFFER_NUM_DB, ubFactor_ * MROPE_AXIS_NUM * headDim_ * sizeof(float));
        pipe_->InitBuffer(outQue_, BUFFER_NUM_DB, ubFactor_ * rowElems_ * sizeof(T_QKV));
        pipe_->InitBuffer(wInQue_, BUFFER_NUM_SINGLE, WEIGHT_NUM * headDim_ * sizeof(T_QKV));

        // 计算区：4 份 gamma 的 float 版本 + gather 索引（与 token 无关），
        // 外加按 token 铺平的 gamma tile [ubFactor, 2, D]（随 ubFactor，单 buffer 即可，
        // 因为它在同一次 Compute 里写完就被消费掉）
        pipe_->InitBuffer(wFp32Buf_, WEIGHT_NUM * headDim_ * sizeof(float));
        pipe_->InitBuffer(gatherIdxBuf_, AlignUp(halfDim_ * sizeof(uint32_t), BLOCK_ALIGN_BYTES));
        pipe_->InitBuffer(gammaTileBuf_, DIGIT_TWO * ubFactor_ * headDim_ * sizeof(float));

        wFp32Local_ = wFp32Buf_.Get<float>();
        gatherIdxLocal_ = gatherIdxBuf_.Get<uint32_t>();
        gammaTileLocal_ = gammaTileBuf_.Get<float>();
    }

    // 4 个 gamma 一次性搬入并 Cast 成 float 常驻。不再按头展开成 [headNumQK, D]：
    // VF 里 gamma 只加载一次、段内所有 head 共用，展开纯属浪费 UB。
    __aicore__ inline void PrepareWeights()
    {
        LocalTensor<T_QKV> wBf16 = wInQue_.AllocTensor<T_QKV>();
        DataCopy(wBf16[WEIGHT_IDX_UND_Q * headDim_], undWeightsQGm_, headDim_);
        DataCopy(wBf16[WEIGHT_IDX_UND_K * headDim_], undWeightsKGm_, headDim_);
        if (hasGen_) {
            DataCopy(wBf16[WEIGHT_IDX_GEN_Q * headDim_], genWeightsQGm_, headDim_);
            DataCopy(wBf16[WEIGHT_IDX_GEN_K * headDim_], genWeightsKGm_, headDim_);
        }
        wInQue_.EnQue(wBf16);
        wBf16Local_ = wInQue_.DeQue<T_QKV>();

        const int64_t weightElems = (hasGen_ ? WEIGHT_NUM : DIGIT_TWO) * headDim_;
        Cast(wFp32Local_, wBf16Local_, RoundMode::CAST_NONE, weightElems);
        PipeBarrier<PIPE_V>();
    }

    // axisLut 展开成 gather 索引：gatherIndex[lane] = axis(lane) * D + lane。
    // 规则与竞品 _mrope / golden.py:mrope_axis_map 一致，注意 mropeSection_[0] 不参与判断，
    // T 轴既是 i%3==0 的归属轴，也是超出 3*section 之后的兜底轴。
    // mrope_section 为空时 host 填 [D/2, 0, 0]，此处 sectionH/W 均为 0，全部 lane 落 T 轴，
    // 自动退化成标准 RoPE，不需要额外分支。
    __aicore__ inline void BuildMropeGatherIndex()
    {
        for (int64_t lane = 0; lane < halfDim_; ++lane) {
            const int64_t repeat = lane / MROPE_AXIS_NUM;
            const int64_t axisInGroup = lane % MROPE_AXIS_NUM;
            int64_t axis = 0;
            if (axisInGroup == 1 && repeat < mropeSection_[1]) {
                axis = 1;
            } else if (axisInGroup == DIGIT_TWO && repeat < mropeSection_[DIGIT_TWO]) {
                axis = DIGIT_TWO;
            }
            gatherIdxLocal_.SetValue(lane, static_cast<uint32_t>(axis * headDim_ + lane));
        }
        // 标量写入的 UB 要被 VF 读，必须显式同步，否则 Gather 可能读到旧值
        SetFlag<HardEvent::S_V>(EVT_S_TO_V_INDEX_READY);
        WaitFlag<HardEvent::S_V>(EVT_S_TO_V_INDEX_READY);
    }

    // 取 out_t 对应的源 token 下标；越界一律回落到 0，避免非法 GM 访问
    __aicore__ inline int64_t GetSrcIndex(int64_t outT)
    {
        int64_t srcT = hasCatIndices_ ? catIndicesGm_.GetValue(outT) : outT;
        if (srcT < 0 || srcT >= totalTokens_) {
            srcT = 0;
        }
        return srcT;
    }

    // 逐 token 搬入：cat_indices 乱序导致各 token 的源行不连续，只能一 token 一笔；
    // DoubleBuffer 让本批 MTE2 与上一批的向量计算重叠
    // TODO(perf): 索引标量目前直接走 GM GetValue，每 token 5 次（cat/positions*3/slot）。
    //             后续可批量搬入 UB，需同步扩 host 侧 UB 预算。
    __aicore__ inline void CopyIn(int64_t tileStart, int64_t tokenNum)
    {
        LocalTensor<T_QKV> qkvLocal = qkvInQue_.AllocTensor<T_QKV>();
        for (int64_t i = 0; i < tokenNum; ++i) {
            int64_t srcT = GetSrcIndex(tileStart + i);
            if (srcT < undLen_) {
                DataCopy(qkvLocal[i * rowElems_], undQkvGm_[srcT * rowElems_], rowElems_);
            } else {
                DataCopy(qkvLocal[i * rowElems_], genQkvGm_[(srcT - undLen_) * rowElems_], rowElems_);
            }
        }
        qkvInQue_.EnQue(qkvLocal);

        // 三轴 cos_sin：每 token 按 positions[axis, out_t] 各取一行 [D]
        LocalTensor<float> cosSinLocal = cosSinInQue_.AllocTensor<float>();
        for (int64_t i = 0; i < tokenNum; ++i) {
            int64_t outT = tileStart + i;
            for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
                int64_t pos = positionsGm_.GetValue(axis * totalTokens_ + outT);
                if (pos < 0 || pos >= maxPos_) { // 运行期数据，host 校验不到，这里兜底
                    pos = 0;
                }
                DataCopy(cosSinLocal[(i * MROPE_AXIS_NUM + axis) * headDim_], cosSinCacheGm_[pos * headDim_],
                         headDim_);
            }
        }
        cosSinInQue_.EnQue(cosSinLocal);
    }

    // 整个 tile 只发一次 VF：Q/K 在 UB 上本就是同一行 token 内相邻的 head，
    // 合在一个 VF 里 token 循环只走一趟、cos/sin 只 Gather 一次。
    // und/gen 的差异靠 BuildGammaTile 预先按 token 铺平，不需要按 isUnd 切段
    // ——cat_indices 可以任意交错，切段在最坏情况下会退化成逐 token 调 VF。
    __aicore__ inline void Compute(int64_t tileStart, int64_t tokenNum)
    {
        LocalTensor<T_QKV> qkvLocal = qkvInQue_.DeQue<T_QKV>();
        LocalTensor<float> cosSinLocal = cosSinInQue_.DeQue<float>();
        LocalTensor<T_QKV> outLocal = outQue_.AllocTensor<T_QKV>();

        BuildGammaTile(tileStart, tokenNum);

        const int64_t qSegElems = numHeadQ_ * headDim_;
        const int64_t kSegElems = numHeadK_ * headDim_;
        const int64_t kSegBase = ubFactor_ * qSegElems;

        QkRmsNormMropeTileVF((__local_mem__ bfloat16_t*)qkvLocal.GetPhyAddr(),
                             (__local_mem__ float*)gammaTileLocal_.GetPhyAddr(),
                             (__local_mem__ float*)cosSinLocal.GetPhyAddr(),
                             (__local_mem__ uint32_t*)gatherIdxLocal_.GetPhyAddr(),
                             (__local_mem__ bfloat16_t*)outLocal.GetPhyAddr(),
                             (__local_mem__ bfloat16_t*)outLocal[kSegBase].GetPhyAddr(),
                             static_cast<uint16_t>(tokenNum), static_cast<uint16_t>(numHeadQ_),
                             static_cast<uint16_t>(numHeadK_), static_cast<uint32_t>(rowElems_),
                             static_cast<uint32_t>(DIGIT_TWO * headDim_),
                             static_cast<uint32_t>(MROPE_AXIS_NUM * headDim_), static_cast<uint32_t>(qSegElems),
                             static_cast<uint32_t>(kSegElems), static_cast<uint32_t>(headDim_),
                             static_cast<uint32_t>(halfDim_), epsilon_, reciprocal_);

        // V 不参与 norm/rope，且 T_QKV 与 T_CACHE 同为 bf16，纯搬运
        CopyVSegment(qkvLocal, outLocal, tokenNum);

        outQue_.EnQue(outLocal);
        qkvInQue_.FreeTensor(qkvLocal);
        cosSinInQue_.FreeTensor(cosSinLocal);
    }

    // 把 tile 内每个 token 该用的 gamma 铺成 [tokenNum, 2, D]：每 token 先 Q gamma 后 K gamma，
    // 与 VF 里 gammaRow / gammaRow + headStride 的取法对应。
    // 源是 Init 时就 Cast 好的 4 份常驻 float gamma，这里只是 UB->UB 选行，
    // 每 token 两笔 D*4B，相对 tile 内 (Hq+Hk) 个 head 的计算量可以忽略。
    // TODO(perf): 只有两种取值，可改成把 und/gen 两份 gamma 常驻在 vreg，
    //             VF 内按 token 的 isUnd 标志做 Select，省掉这里的 UB->UB 搬运。
    __aicore__ inline void BuildGammaTile(int64_t tileStart, int64_t tokenNum)
    {
        for (int64_t i = 0; i < tokenNum; ++i) {
            const bool isUnd = GetSrcIndex(tileStart + i) < undLen_;
            const int64_t qGammaIdx = isUnd ? WEIGHT_IDX_UND_Q : WEIGHT_IDX_GEN_Q;
            const int64_t kGammaIdx = isUnd ? WEIGHT_IDX_UND_K : WEIGHT_IDX_GEN_K;
            DataCopy(gammaTileLocal_[(i * DIGIT_TWO) * headDim_], wFp32Local_[qGammaIdx * headDim_], headDim_);
            DataCopy(gammaTileLocal_[(i * DIGIT_TWO + 1) * headDim_], wFp32Local_[kGammaIdx * headDim_], headDim_);
        }
        // UB->UB 搬运与随后 VF 的读之间要挡一下
        PipeBarrier<PIPE_V>();
    }

    // TODO(perf): V 段是 UB->UB 纯搬运，可以省掉 outLocal 的 v 段直接在 CopyOut 里
    //             从 qkvInQue 写 GM，代价是输入 buffer 生命周期要延到 CopyOut 之后。
    __aicore__ inline void CopyVSegment(const LocalTensor<T_QKV>& qkvLocal, const LocalTensor<T_QKV>& outLocal,
                                        int64_t tokenNum)
    {
        const int64_t vSegElems = numHeadV_ * headDim_;
        const int64_t vSegBase = ubFactor_ * headNumQK_ * headDim_;
        const int64_t vInOffset = headNumQK_ * headDim_;
        for (int64_t i = 0; i < tokenNum; ++i) {
            DataCopy(outLocal[vSegBase + i * vSegElems], qkvLocal[i * rowElems_ + vInOffset], vSegElems);
        }
    }

    // outLocal 按 [ubFactor 个 q | ubFactor 个 k | ubFactor 个 v] 分段：
    // q 的输出下标 out_t 连续，可一笔写出；k/v 按 slot_mapping 散写，只能逐 token
    __aicore__ inline void CopyOut(int64_t tileStart, int64_t tokenNum)
    {
        LocalTensor<T_QKV> outLocal = outQue_.DeQue<T_QKV>();

        const int64_t qSegElems = numHeadQ_ * headDim_;
        const int64_t kSegElems = numHeadK_ * headDim_;
        const int64_t vSegElems = numHeadV_ * headDim_;
        const int64_t kSegBase = ubFactor_ * qSegElems;
        const int64_t vSegBase = ubFactor_ * headNumQK_ * headDim_;

        DataCopy(qGm_[tileStart * qSegElems], outLocal[0], tokenNum * qSegElems);

        for (int64_t i = 0; i < tokenNum; ++i) {
            int64_t slot = slotMappingGm_.GetValue(tileStart + i);
            if (slot < 0 || slot >= maxSlot_) { // 运行期数据，越界则跳过写入而不是踩内存
                continue;
            }
            DataCopy(kCacheGm_[slot * kSegElems], outLocal[kSegBase + i * kSegElems], kSegElems);
            DataCopy(vCacheGm_[slot * vSegElems], outLocal[vSegBase + i * vSegElems], vSegElems);
        }

        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

private:
    TPipe* pipe_ = nullptr;
    const UndGenQkvRmsNormRopeCacheTilingData* tiling_ = nullptr;

    TQue<QuePosition::VECIN, BUFFER_NUM_DB> qkvInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_DB> cosSinInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_DB> outQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_SINGLE> wInQue_;

    TBuf<TPosition::VECCALC> wFp32Buf_;
    TBuf<TPosition::VECCALC> gatherIdxBuf_;
    TBuf<TPosition::VECCALC> gammaTileBuf_;

    // 常驻 LocalTensor：权重来自 wInQue_（Process 开头 DeQue、结束时 Free），
    // 其余是 TBuf 的视图，Init 时取一次，整个 Process 复用
    LocalTensor<T_QKV> wBf16Local_;
    LocalTensor<float> wFp32Local_;
    LocalTensor<uint32_t> gatherIdxLocal_;
    LocalTensor<float> gammaTileLocal_;

    GlobalTensor<T_QKV> undQkvGm_;
    GlobalTensor<T_QKV> genQkvGm_;
    GlobalTensor<T_QKV> undWeightsQGm_;
    GlobalTensor<T_QKV> undWeightsKGm_;
    GlobalTensor<T_QKV> genWeightsQGm_;
    GlobalTensor<T_QKV> genWeightsKGm_;
    GlobalTensor<float> cosSinCacheGm_;
    GlobalTensor<T_CACHE> kCacheGm_;
    GlobalTensor<T_CACHE> vCacheGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<int64_t> positionsGm_;
    GlobalTensor<int64_t> catIndicesGm_;
    GlobalTensor<T_QKV> qGm_;

    // shape / 属性
    int64_t totalTokens_ = 0;
    int64_t undLen_ = 0;
    int64_t numHead_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t headDim_ = 0;
    int64_t maxPos_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    int64_t mropeSection_[MROPE_AXIS_NUM] = {0, 0, 0};
    bool hasGen_ = false;
    bool hasCatIndices_ = false;
    float epsilon_ = 0.0f;
    float reciprocal_ = 0.0f;

    // 切分
    int64_t usedCoreNum_ = 0;
    int64_t formerCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreEnd_ = 0;

    // 派生量
    int64_t headNumQK_ = 0;
    int64_t halfDim_ = 0;
    int64_t rowElems_ = 0;
    int64_t maxSlot_ = 0;
};
} // namespace UndGenQkvRmsNormRopeCache

#endif // UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
