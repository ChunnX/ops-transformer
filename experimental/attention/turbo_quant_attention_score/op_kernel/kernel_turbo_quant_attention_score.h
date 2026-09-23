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
 * \file kernel_turbo_quant_attention_score.h
 * \brief TurboQuantAttentionScore kernel v3：向量化解包 + BlockReduceSum 批量点积。
 *
 * 语义（对每个 (q_token t, q_head h, kv 位置 i)，k = h / group）：
 *   score = norm[k,i] * <Hq[t,h], centroids[unpack(idx[k,i])]>
 *         + sqrt(pi/2)/128 * gamma[k,i] * <Sq[t,h], unpack_sign(qjl[k,i])>
 *   i >= seq_lens[b] 的位置输出 IEEE -inf（0xFF800000，与 golden 的 -torch.inf 位模式一致）
 *
 * v3 关键设计（v2 的 S 域逐元素解包是性能瓶颈，4096 key·dim 要 ~41k 次标量 UB 访问）：
 *   1. 维度重排消除 scatter：主码第 l lane、第 w 个打包字对应原维度 d=8w+l。
 *      把 yhat 平面按 d'=l*16+w 排布、query 在 LoadQuery 时一次性置换到同一顺序，
 *      点积结果与顺序无关 —— 于是 lane 提取变成对 [key][word] 连续平面的
 *      ShiftRight+And（整 tile 矢量），按 16 连续元素写回（无跨步 scatter）。
 *      QJL 同理按 d''=l*16+b（b=字节序）排布。
 *   2. S 域只做 24bit 字拼装：每 key 12 次 u32 GetValue + 16 次 SetValue（idx）、
 *      16+16（qjl），较逐 dim 写法省 ~5x 标量访问。
 *   3. 点积：Mul（逐 key 行 128 宽）+ BlockReduceSum（每 64 元素块归约，每 key 得 2 个
 *      部分和），S 域收割仅 2 次 GetValue/key/branch。
 *
 * 对齐纪律（沿用 kv_cache_turbo_quant A2 实测口径）：矢量指令 UB 访问 32B 对齐；
 * 单元素读写走 S 域；跨 pipe 边界显式 SetFlag/WaitFlag；TBuf 跨 item/tile 复用补齐同步。
 */
#ifndef KERNEL_TURBO_QUANT_ATTENTION_SCORE_H
#define KERNEL_TURBO_QUANT_ATTENTION_SCORE_H

#include "kernel_operator.h"
#include "turbo_quant_attention_score_tiling_data.h"

using namespace AscendC;

namespace TurboQuantAttentionScore {

constexpr int32_t HEAD_DIM = 128;
constexpr int32_t BLOCK_KEYS = 128;
constexpr int32_t QJL_BYTES = 16;
constexpr int32_t QJL_PITCH = 32; // 16B 行在 UB 内 32B 对齐
constexpr int32_t MAX_GROUP = 8;
constexpr int32_t MAX_KV_HEADS = 16;     // norm/gamma 平面 UB 常驻上限（官方 case 为 8）
constexpr int32_t MAX_BT_ENTRIES = 8192; // block_table 常驻 UB 上限（与 tiling 校验一致）
constexpr int32_t MAX_BATCH = 1024;
constexpr int32_t TILE_K = 16; // 每批处理的 key 数
// sqrt(pi/2)/128 的 fp32 最近值（golden 系数）
constexpr float QJL_COEF = 0.009791516697777345f;

// 质心表：d=128 单位球面坐标分布的 Lloyd-Max 最优质心（与 golden.py CENTROIDS 逐字一致）
template <int BITS>
struct CentTable;

template <>
struct CentTable<2> {
    static constexpr float CENT[4] = {-0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
};
template <>
struct CentTable<3> {
    static constexpr float CENT[8] = {-0.19020693f,   -0.1187859178f, -0.06682205945f, -0.02166347019f,
                                      0.02166347019f, 0.06682205945f, 0.1187859178f,   0.19020693f};
};
template <>
struct CentTable<4> {
    static constexpr float CENT[16] = {-0.2414890379f,  -0.1828317791f,  -0.1429702938f,  -0.1109927073f,
                                       -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
                                       0.01134236995f,  0.03428063914f,  0.05802082643f,  0.08325428516f,
                                       0.1109927073f,   0.1429702938f,   0.1828317791f,   0.2414890379f};
};

template <uint8_t BITS>
class KernelTurboQuantAttentionScore {
public:
    static constexpr int32_t CENT_CNT = 1 << BITS;
    static constexpr int32_t IDX_BYTES = HEAD_DIM * BITS / 8;        // 32/48/64
    static constexpr int32_t IDX_PITCH = (IDX_BYTES + 31) / 32 * 32; // UB 行 pitch：32/64/64
    static constexpr int32_t IDX_U32 = IDX_PITCH / 4;                // 每行 u32 槽位数
    static constexpr int32_t PLANE_ELEMS = TILE_K * HEAD_DIM;        // 2048
    static constexpr int32_t WORD_CNT = HEAD_DIM / 8;                // 每 key 打包字数 16

    __aicore__ inline KernelTurboQuantAttentionScore() {}

    __aicore__ inline void Init(GM_ADDR queryRotated, GM_ADDR queryQjl, GM_ADDR keyIdx, GM_ADDR keyQjl, GM_ADDR keyNorm,
                                GM_ADDR keyGamma, GM_ADDR blockTable, GM_ADDR seqLens, GM_ADDR attnScores,
                                const TurboQuantAttentionScoreTilingData *td, TPipe *pipe)
    {
        numTokens_ = td->numTokens;
        qHeads_ = td->numQHeads;
        kvHeads_ = td->numKvHeads;
        group_ = td->groupSize;
        batch_ = td->batch;
        maxBlocks_ = td->maxBlocksPerSeq;
        maxKvLen_ = td->maxKvLen;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t former = td->formerCoreNum;
        itemStart_ = static_cast<int64_t>(blockIdx) * td->itemsPerCore +
                     (blockIdx < former ? static_cast<int64_t>(blockIdx) : static_cast<int64_t>(former));
        itemCount_ = td->itemsPerCore + (blockIdx < former ? 1 : 0);

        qGm_.SetGlobalBuffer((__gm__ bfloat16_t *)queryRotated);
        qqGm_.SetGlobalBuffer((__gm__ bfloat16_t *)queryQjl);
        idxGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIdx);
        qjlGm_.SetGlobalBuffer((__gm__ uint8_t *)keyQjl);
        normGm_.SetGlobalBuffer((__gm__ bfloat16_t *)keyNorm);
        gamGm_.SetGlobalBuffer((__gm__ bfloat16_t *)keyGamma);
        btGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        slGm_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
        outGm_.SetGlobalBuffer((__gm__ float *)attnScores);

        pipe->InitBuffer(btBuf_, MAX_BT_ENTRIES * sizeof(int32_t));
        pipe->InitBuffer(slBuf_, MAX_BATCH * sizeof(int32_t));
        pipe->InitBuffer(qBf16Buf_, 2 * MAX_GROUP * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(qfBuf_, 2 * MAX_GROUP * HEAD_DIM * sizeof(float));
        pipe->InitBuffer(qPermBuf_, 2 * MAX_GROUP * HEAD_DIM * sizeof(float));
        pipe->InitBuffer(idxBuf_, BLOCK_KEYS * IDX_PITCH);
        pipe->InitBuffer(qjlBuf_, BLOCK_KEYS * QJL_PITCH);
        pipe->InitBuffer(normBf16Buf_, BLOCK_KEYS * MAX_KV_HEADS * sizeof(bfloat16_t));
        pipe->InitBuffer(gamBf16Buf_, BLOCK_KEYS * MAX_KV_HEADS * sizeof(bfloat16_t));
        pipe->InitBuffer(normFBuf_, BLOCK_KEYS * MAX_KV_HEADS * sizeof(float));
        pipe->InitBuffer(gamFBuf_, BLOCK_KEYS * MAX_KV_HEADS * sizeof(float));
        pipe->InitBuffer(wordPlaneBuf_, TILE_K * WORD_CNT * sizeof(uint32_t));
        pipe->InitBuffer(bytePlaneBuf_, TILE_K * WORD_CNT * sizeof(uint32_t));
        pipe->InitBuffer(scratchIBuf_, TILE_K * WORD_CNT * sizeof(uint32_t));
        pipe->InitBuffer(idxIntPlaneBuf_, PLANE_ELEMS * sizeof(uint32_t));
        pipe->InitBuffer(signIntPlaneBuf_, PLANE_ELEMS * sizeof(uint32_t));
        pipe->InitBuffer(idxPlaneBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(yhatPlaneBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(signPlaneBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(prodAPlaneBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(prodBPlaneBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(partABuf_, TILE_K * 2 * 8 * sizeof(float)); // BRS dst 预留两种布局余量
        pipe->InitBuffer(partBBuf_, TILE_K * 2 * 8 * sizeof(float));
        pipe->InitBuffer(dupBuf_, PLANE_ELEMS * sizeof(float));
        pipe->InitBuffer(maskBuf_, PLANE_ELEMS);
        pipe->InitBuffer(outBuf_, MAX_GROUP * BLOCK_KEYS * sizeof(float));

        btT_ = btBuf_.Get<int32_t>();
        slT_ = slBuf_.Get<int32_t>();
        qBf16T_ = qBf16Buf_.Get<bfloat16_t>();
        qfT_ = qfBuf_.Get<float>();
        qPermT_ = qPermBuf_.Get<float>();
        idxT_ = idxBuf_.Get<uint8_t>();
        idxU32T_ = idxBuf_.Get<uint8_t>().ReinterpretCast<uint32_t>();
        qjlT_ = qjlBuf_.Get<uint8_t>();
        normBf16T_ = normBf16Buf_.Get<bfloat16_t>();
        gamBf16T_ = gamBf16Buf_.Get<bfloat16_t>();
        normFT_ = normFBuf_.Get<float>();
        gamFT_ = gamFBuf_.Get<float>();
        wordPlaneUT_ = wordPlaneBuf_.Get<uint32_t>();
        bytePlaneUT_ = bytePlaneBuf_.Get<uint32_t>();
        scratchUT_ = scratchIBuf_.Get<uint32_t>();
        idxIntPlaneUT_ = idxIntPlaneBuf_.Get<uint32_t>();
        signIntPlaneUT_ = signIntPlaneBuf_.Get<uint32_t>();
        idxPlaneT_ = idxPlaneBuf_.Get<float>();
        yhatPlaneT_ = yhatPlaneBuf_.Get<float>();
        signPlaneT_ = signPlaneBuf_.Get<float>();
        prodAPlaneT_ = prodAPlaneBuf_.Get<float>();
        prodBPlaneT_ = prodBPlaneBuf_.Get<float>();
        partAT_ = partABuf_.Get<float>();
        partBT_ = partBBuf_.Get<float>();
        dupT_ = dupBuf_.Get<float>();
        maskT_ = maskBuf_.Get<uint8_t>();
        outT_ = outBuf_.Get<float>();

        uint32_t infBits = 0xFF800000u;
        negInf_ = *reinterpret_cast<float *>(&infBits);

        // block_table / seq_lens 常驻 UB（Init 一次）
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams btParams;
        btParams.blockCount = 1;
        btParams.blockLen = batch_ * maxBlocks_ * sizeof(int32_t);
        btParams.srcStride = 0;
        btParams.dstStride = 0;
        DataCopyPad(btT_, btGm_, btParams, padParams);
        DataCopyExtParams slParams;
        slParams.blockCount = 1;
        slParams.blockLen = batch_ * sizeof(int32_t);
        slParams.srcStride = 0;
        slParams.dstStride = 0;
        DataCopyPad(slT_, slGm_, slParams, padParams);
        SyncMte2V();
        SyncVS();
    }

    __aicore__ inline void Process()
    {
        const int64_t kbStride = static_cast<int64_t>(kvHeads_) * maxBlocks_; // item -> (t,k) 步长
        for (int64_t it = 0; it < itemCount_; it++) {
            int64_t item = itemStart_ + it;
            int64_t t = item / kbStride;
            int64_t rem = item % kbStride;
            int32_t k = static_cast<int32_t>(rem / maxBlocks_);
            int32_t lb = static_cast<int32_t>(rem % maxBlocks_);
            ProcessItem(t, k, lb);
        }
    }

private:
    __aicore__ inline void SyncVS()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
    }
    __aicore__ inline void SyncSV()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }
    __aicore__ inline void SyncSMte3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(e);
        WaitFlag<HardEvent::S_MTE3>(e);
    }
    __aicore__ inline void SyncMte2V()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void SyncVMte2()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(e);
        WaitFlag<HardEvent::V_MTE2>(e);
    }
    __aicore__ inline void SyncSMte2()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(e);
        WaitFlag<HardEvent::S_MTE2>(e);
    }
    __aicore__ inline void SyncMte3V()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e);
        WaitFlag<HardEvent::MTE3_V>(e);
    }

    // (t, k) 的 group 个 query 行搬入、转 fp32 并按解包维度序置换（每 (t,k) 一次）
    __aicore__ inline void LoadQuery(int64_t t, int32_t k)
    {
        int64_t tk = t * kvHeads_ + k;
        if (tk == lastTk_) {
            return;
        }
        lastTk_ = tk;
        SyncVMte2(); // 前一次 LoadQuery 的 Cast（V 读）完成后才能覆盖 qBf16T_
        const int32_t rows = group_ * HEAD_DIM;
        int64_t qOff = (t * qHeads_ + static_cast<int64_t>(k) * group_) * HEAD_DIM;
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(rows) * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(qBf16T_, qGm_[qOff], params, padParams);
        DataCopyPad(qBf16T_[MAX_GROUP * HEAD_DIM], qqGm_[qOff], params, padParams);
        SyncMte2V();
        Cast(qfT_, qBf16T_, RoundMode::CAST_NONE, rows);
        Cast(qfT_[MAX_GROUP * HEAD_DIM], qBf16T_[MAX_GROUP * HEAD_DIM], RoundMode::CAST_NONE, rows);
        SyncVS();
        // 置换到解包维度序：主码 d'=l*16+w（w=打包字序，l=字内 lane）；QJL d''=l*16+b（b=字节序）
        for (int32_t g = 0; g < group_; g++) {
            for (int32_t l = 0; l < 8; l++) {
                for (int32_t w = 0; w < WORD_CNT; w++) {
                    qPermT_.SetValue(g * HEAD_DIM + l * WORD_CNT + w, qfT_.GetValue(g * HEAD_DIM + w * 8 + l));
                }
                for (int32_t b = 0; b < WORD_CNT; b++) {
                    qPermT_.SetValue((MAX_GROUP + g) * HEAD_DIM + l * WORD_CNT + b,
                                     qfT_.GetValue((MAX_GROUP + g) * HEAD_DIM + b * 8 + l));
                }
            }
        }
        SyncSV();
    }

    // 物理块内当前 kv_head 的打包数据搬入（按行 stride 跳过其它 kv_head）
    __aicore__ inline void CopyInBlock(int32_t phys, int32_t k)
    {
        // 缓冲复用防护：上一 item 的 Cast（V 读 normBf16/gamBf16）与 S 域解包（S 读 idx/qjl）
        // 完成之后，本 item 的 MTE2 才能覆盖写
        SyncVMte2();
        SyncSMte2();
        DataCopyPadExtParams<uint8_t> padU8{false, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> padBf16{false, 0, 0, 0};
        const int64_t rowStrideIdx = static_cast<int64_t>(kvHeads_) * IDX_BYTES;
        const int64_t rowStrideQjl = static_cast<int64_t>(kvHeads_) * QJL_BYTES;

        DataCopyExtParams idxParams;
        idxParams.blockCount = BLOCK_KEYS;
        idxParams.blockLen = IDX_BYTES;
        idxParams.srcStride = static_cast<uint32_t>(rowStrideIdx - IDX_BYTES); // GM 侧单位：字节
        idxParams.dstStride = 0;                                               // UB 侧自动按 32B 对齐成行 pitch
        DataCopyPad(idxT_, idxGm_[(static_cast<int64_t>(phys) * BLOCK_KEYS * kvHeads_ + k) * IDX_BYTES], idxParams,
                    padU8);

        DataCopyExtParams qjlParams;
        qjlParams.blockCount = BLOCK_KEYS;
        qjlParams.blockLen = QJL_BYTES;
        qjlParams.srcStride = static_cast<uint32_t>(rowStrideQjl - QJL_BYTES);
        qjlParams.dstStride = 0;
        DataCopyPad(qjlT_, qjlGm_[(static_cast<int64_t>(phys) * BLOCK_KEYS * kvHeads_ + k) * QJL_BYTES], qjlParams,
                    padU8);

        // norm/gamma 整平面连续（[128, kvh] bf16）
        DataCopyExtParams ngParams;
        ngParams.blockCount = 1;
        ngParams.blockLen = static_cast<uint32_t>(BLOCK_KEYS * kvHeads_) * sizeof(bfloat16_t);
        ngParams.srcStride = 0;
        ngParams.dstStride = 0;
        const int64_t planeOff = static_cast<int64_t>(phys) * BLOCK_KEYS * kvHeads_;
        DataCopyPad(normBf16T_, normGm_[planeOff], ngParams, padBf16);
        DataCopyPad(gamBf16T_, gamGm_[planeOff], ngParams, padBf16);

        SyncMte2V();
        Cast(normFT_, normBf16T_, RoundMode::CAST_NONE, BLOCK_KEYS * kvHeads_);
        Cast(gamFT_, gamBf16T_, RoundMode::CAST_NONE, BLOCK_KEYS * kvHeads_);
        SyncVS();
    }

    // S 域拼装打包字平面：每 key 的 48B 读作 12 个 u32，拼出 16 个 24bit 小端字
    // （bits=2：8 个 u32 拆 16 个 16bit 字；bits=4：16 个 u32 直用）
    __aicore__ inline void BuildWordPlanes(int32_t keyStart, int32_t nKeys)
    {
        SyncVS(); // 上一 tile 的矢量读（ShiftRight 读 word/byte 平面）完成后才能覆盖写
        for (int32_t key = 0; key < nKeys; key++) {
            const int32_t rowU32 = (keyStart + key) * IDX_U32;
            if (BITS == 3) {
                uint32_t u[12];
                for (int32_t m = 0; m < 12; m++) {
                    u[m] = idxU32T_.GetValue(rowU32 + m);
                }
                for (int32_t w = 0; w < WORD_CNT; w++) {
                    const int32_t a = (3 * w) / 4;
                    const int32_t o = (3 * w) % 4 * 8; // 0/24/16/8
                    uint32_t word = u[a] >> o;
                    if (o > 8) { // 24bit 字跨 u32 边界
                        word |= u[a + 1] << (32 - o);
                    }
                    wordPlaneUT_.SetValue(key * WORD_CNT + w, static_cast<int32_t>(word & 0xFFFFFFu));
                }
            } else if (BITS == 2) {
                for (int32_t w = 0; w < WORD_CNT; w++) {
                    uint32_t u = idxU32T_.GetValue(rowU32 + w / 2);
                    wordPlaneUT_.SetValue(key * WORD_CNT + w, static_cast<int32_t>((u >> (16 * (w % 2))) & 0xFFFFu));
                }
            } else { // BITS == 4
                for (int32_t w = 0; w < WORD_CNT; w++) {
                    wordPlaneUT_.SetValue(key * WORD_CNT + w, static_cast<int32_t>(idxU32T_.GetValue(rowU32 + w)));
                }
            }
        }
        for (int32_t key = 0; key < nKeys; key++) {
            const int32_t srcBase = (keyStart + key) * QJL_PITCH;
            for (int32_t b = 0; b < QJL_BYTES; b++) {
                bytePlaneUT_.SetValue(key * WORD_CNT + b, static_cast<int32_t>(qjlT_.GetValue(srcBase + b)));
            }
        }
    }

    // 矢量化解包：lane 提取（右移+左移+右移取位，c220 的 And 仅支持 16bit，故走移位掩码）
    // → uint32 平面 reinterpret 为 int32 后 Cast 到 float
    __aicore__ inline void VectorUnpack(int32_t nKeys)
    {
        const int32_t wElems = nKeys * WORD_CNT;
        const UnaryRepeatParams laneParams{1, 1, 16, 2}; // dst 每 key 跨 128 元素，src 每 key 16 字
        for (int32_t l = 0; l < 8; l++) {
            ShiftRight<uint32_t>(scratchUT_, wordPlaneUT_, static_cast<uint32_t>(l * BITS), wElems);
            ShiftLeft<uint32_t>(scratchUT_, scratchUT_, static_cast<uint32_t>(32 - BITS), wElems);
            ShiftRight<uint32_t>(idxIntPlaneUT_[l * WORD_CNT], scratchUT_, static_cast<uint32_t>(32 - BITS),
                                 static_cast<uint64_t>(WORD_CNT), static_cast<uint8_t>(nKeys), laneParams);
        }
        Cast(idxPlaneT_, idxIntPlaneUT_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, nKeys * HEAD_DIM);

        for (int32_t l = 0; l < 8; l++) {
            ShiftRight<uint32_t>(scratchUT_, bytePlaneUT_, static_cast<uint32_t>(l), wElems);
            ShiftLeft<uint32_t>(scratchUT_, scratchUT_, 31u, wElems);
            ShiftRight<uint32_t>(signIntPlaneUT_[l * WORD_CNT], scratchUT_, 31u, static_cast<uint64_t>(WORD_CNT),
                                 static_cast<uint8_t>(nKeys), laneParams);
        }
        Cast(signPlaneT_, signIntPlaneUT_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, nKeys * HEAD_DIM);
        Muls(signPlaneT_, signPlaneT_, 2.0f, nKeys * HEAD_DIM);
        Adds(signPlaneT_, signPlaneT_, -1.0f, nKeys * HEAD_DIM);
    }

    // 质心查表（整 tile 平面）：yhat = centroids[idx]（Compare/Select 链，编码侧 kernel 同形态）
    __aicore__ inline void CentroidSelectTile(int32_t nKeys)
    {
        const int32_t elems = nKeys * HEAD_DIM;
        Duplicate(yhatPlaneT_, CentTable<BITS>::CENT[0], elems);
        for (int32_t c = 1; c < CENT_CNT; c++) {
            Duplicate(dupT_, static_cast<float>(c), elems);
            Compare<float>(maskT_, idxPlaneT_, dupT_, CMPMODE::EQ, elems);
            Duplicate(dupT_, CentTable<BITS>::CENT[c], elems);
            Select(yhatPlaneT_, maskT_, dupT_, yhatPlaneT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        }
    }

    // 归约树（仅到 off=8，保持 32B 对齐）：t[0..8) = 8 个部分和
    __aicore__ inline void TreeReduce8(LocalTensor<float> t)
    {
        Add(t, t, t[64], 64);
        Add(t, t, t[32], 32);
        Add(t, t, t[16], 16);
        Add(t, t, t[8], 8);
    }

    __aicore__ inline float SumPartials(LocalTensor<float> t)
    {
        float s = t.GetValue(0);
        for (int32_t i = 1; i < 8; i++) {
            s += t.GetValue(i);
        }
        return s;
    }

    // 整 tile 两路点积（逐 key 行 Mul + TreeReduce8）+ S 域收割写分
    __aicore__ inline void DotTile(int32_t keyStart, int32_t nKeys)
    {
        for (int32_t g = 0; g < group_; g++) {
            for (int32_t key = 0; key < nKeys; key++) {
                Mul(prodAPlaneT_[key * HEAD_DIM], qPermT_[g * HEAD_DIM], yhatPlaneT_[key * HEAD_DIM], HEAD_DIM);
                TreeReduce8(prodAPlaneT_[key * HEAD_DIM]);
                Mul(prodBPlaneT_[key * HEAD_DIM], qPermT_[(MAX_GROUP + g) * HEAD_DIM], signPlaneT_[key * HEAD_DIM],
                    HEAD_DIM);
                TreeReduce8(prodBPlaneT_[key * HEAD_DIM]);
            }
            SyncVS();
            for (int32_t key = 0; key < nKeys; key++) {
                const int32_t gkey = keyStart + key;
                const float normV = normFT_.GetValue(gkey * kvHeads_ + k_);
                const float gamV = gamFT_.GetValue(gkey * kvHeads_ + k_);
                const float main = SumPartials(prodAPlaneT_[key * HEAD_DIM]);
                const float corr = SumPartials(prodBPlaneT_[key * HEAD_DIM]);
                outT_.SetValue(g * BLOCK_KEYS + gkey, normV * main + QJL_COEF * gamV * corr);
            }
            SyncSV(); // S 收割完成后，下一 g/tile 的 Mul（V 写）才能覆盖 prod 平面
        }
    }

    __aicore__ inline void ProcessItem(int64_t t, int32_t k, int32_t lb)
    {
        k_ = k;
        const int32_t b = (batch_ == 1) ? 0 : static_cast<int32_t>(t);
        const int32_t phys = btT_.GetValue(b * maxBlocks_ + lb);
        const int32_t seqLen = slT_.GetValue(b);
        int32_t valid = seqLen - lb * BLOCK_KEYS;
        valid = valid < 0 ? 0 : (valid > BLOCK_KEYS ? BLOCK_KEYS : valid);

        LoadQuery(t, k);

        // 输出 tile 先填 -inf，仅写 valid 位置（golden masked_fill 语义）
        Duplicate(outT_, negInf_, group_ * BLOCK_KEYS);
        SyncVS();

        if (valid > 0) {
            CopyInBlock(phys, k);
            for (int32_t keyStart = 0; keyStart < valid; keyStart += TILE_K) {
                const int32_t nKeys = (valid - keyStart > TILE_K) ? TILE_K : (valid - keyStart);
                BuildWordPlanes(keyStart, nKeys);
                SyncSV();
                VectorUnpack(nKeys);
                CentroidSelectTile(nKeys);
                DotTile(keyStart, nKeys);
            }
        }

        SyncSMte3();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = BLOCK_KEYS * sizeof(float);
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        const int64_t outRowBase =
            (t * qHeads_ + static_cast<int64_t>(k) * group_) * maxKvLen_ + static_cast<int64_t>(lb) * BLOCK_KEYS;
        for (int32_t g = 0; g < group_; g++) {
            DataCopyPad(outGm_[outRowBase + static_cast<int64_t>(g) * maxKvLen_], outT_[g * BLOCK_KEYS], outParams);
        }
        SyncMte3V(); // MTE3 搬出完成后，下一 item 的 Duplicate(-inf)（V 写）才能覆盖 outT_
    }

private:
    GlobalTensor<bfloat16_t> qGm_;
    GlobalTensor<bfloat16_t> qqGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gamGm_;
    GlobalTensor<int32_t> btGm_;
    GlobalTensor<int32_t> slGm_;
    GlobalTensor<float> outGm_;

    TBuf<QuePosition::VECCALC> btBuf_;
    TBuf<QuePosition::VECCALC> slBuf_;
    TBuf<QuePosition::VECCALC> qBf16Buf_;
    TBuf<QuePosition::VECCALC> qfBuf_;
    TBuf<QuePosition::VECCALC> qPermBuf_;
    TBuf<QuePosition::VECCALC> idxBuf_;
    TBuf<QuePosition::VECCALC> qjlBuf_;
    TBuf<QuePosition::VECCALC> normBf16Buf_;
    TBuf<QuePosition::VECCALC> gamBf16Buf_;
    TBuf<QuePosition::VECCALC> normFBuf_;
    TBuf<QuePosition::VECCALC> gamFBuf_;
    TBuf<QuePosition::VECCALC> wordPlaneBuf_;
    TBuf<QuePosition::VECCALC> bytePlaneBuf_;
    TBuf<QuePosition::VECCALC> scratchIBuf_;
    TBuf<QuePosition::VECCALC> idxIntPlaneBuf_;
    TBuf<QuePosition::VECCALC> signIntPlaneBuf_;
    TBuf<QuePosition::VECCALC> idxPlaneBuf_;
    TBuf<QuePosition::VECCALC> yhatPlaneBuf_;
    TBuf<QuePosition::VECCALC> signPlaneBuf_;
    TBuf<QuePosition::VECCALC> prodAPlaneBuf_;
    TBuf<QuePosition::VECCALC> prodBPlaneBuf_;
    TBuf<QuePosition::VECCALC> partABuf_;
    TBuf<QuePosition::VECCALC> partBBuf_;
    TBuf<QuePosition::VECCALC> dupBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;

    LocalTensor<int32_t> btT_;
    LocalTensor<int32_t> slT_;
    LocalTensor<bfloat16_t> qBf16T_;
    LocalTensor<float> qfT_;
    LocalTensor<float> qPermT_;
    LocalTensor<uint8_t> idxT_;
    LocalTensor<uint32_t> idxU32T_;
    LocalTensor<uint8_t> qjlT_;
    LocalTensor<bfloat16_t> normBf16T_;
    LocalTensor<bfloat16_t> gamBf16T_;
    LocalTensor<float> normFT_;
    LocalTensor<float> gamFT_;
    LocalTensor<uint32_t> wordPlaneUT_;
    LocalTensor<uint32_t> bytePlaneUT_;
    LocalTensor<uint32_t> scratchUT_;
    LocalTensor<uint32_t> idxIntPlaneUT_;
    LocalTensor<uint32_t> signIntPlaneUT_;
    LocalTensor<float> idxPlaneT_;
    LocalTensor<float> yhatPlaneT_;
    LocalTensor<float> signPlaneT_;
    LocalTensor<float> prodAPlaneT_;
    LocalTensor<float> prodBPlaneT_;
    LocalTensor<float> partAT_;
    LocalTensor<float> partBT_;
    LocalTensor<float> dupT_;
    LocalTensor<uint8_t> maskT_;
    LocalTensor<float> outT_;

    int64_t numTokens_ = 0;
    int64_t itemStart_ = 0;
    int64_t itemCount_ = 0;
    int64_t lastTk_ = -1; // query 常驻缓存键（t * kvHeads + k）
    uint32_t qHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t group_ = 0;
    uint32_t batch_ = 0;
    uint32_t maxBlocks_ = 0;
    uint32_t maxKvLen_ = 0;
    int32_t k_ = 0; // 当前 item 的 kv_head（DotTile 收割用）
    float negInf_ = 0.0f;
};

} // namespace TurboQuantAttentionScore

#endif // KERNEL_TURBO_QUANT_ATTENTION_SCORE_H
