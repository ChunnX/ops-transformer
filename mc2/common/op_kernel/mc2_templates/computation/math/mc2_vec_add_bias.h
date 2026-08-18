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
 * \file mc2_vec_add_bias.h
 * \brief Vector Add Bias: 将 1D bias [N] 逐行广播加到 matmul 结果 [M, N] 上 (单卡, in-place)。
 *
 * 流水线: MTE2(GM→UB) → V(Cast↑→Add→Cast↓) → MTE3(UB→GM), half 双缓冲 pingpong。
 * 同步: 手动 SetFlag/WaitFlag (HardEvent), 无 EnQue/DeQue。
 *
 * 精度策略:
 *   MatmulDataType 为 half/bf16 时, Cast→float 后 Add<float>(in-place) 再 Cast 回, 避免 FP16 加法精度损失。
 *   MatmulDataType 为 float 时, 直接 Add<float>, 无需 Cast。
 *
 * 分核策略 (SplitModeVal):
 *   M_SPLIT  (1) — 按行分区: 每核负责 [mStart, mEnd) × 全 N
 *   N_SPLIT  (2) — 按列分区: 每核负责 全 M × [nStart, nEnd)
 *   MN_SPLIT (3) — 按 tile 索引分区: 每核分到一组 (mTile, nTile) 组合
 */

#ifndef MC2_VEC_ADD_BIAS_H
#define MC2_VEC_ADD_BIAS_H

#include <type_traits>

namespace MC2KernelTemplate {
using namespace AscendC;

struct MC2AddBiasContext {
    GM_ADDR matmulAddr = 0;    // matmul 结果 [M, N], row-major, in-place
    GM_ADDR biasAddr = 0;      // bias [N]
    uint64_t matmulOffset = 0; // 每个 task 之间的 matmul 偏移 (字节)
    uint64_t biasOffset = 0;   // 每个 task 之间的 bias 偏移 (字节, bias 为一维数组时恒为 0)
    uint64_t M = 0;            // matmul 结果行数
    uint64_t N = 0;            // matmul 结果列数 (= bias 长度)
};

template <typename MatmulDataType, typename BiasDataType = MatmulDataType, uint32_t SplitModeVal = 1>
class MC2VecAddBias {
public:
    enum SplitMode : uint32_t {
        M_SPLIT = 1,
        N_SPLIT = 2,
        MN_SPLIT = 3
    };
    static constexpr SplitMode MODE = static_cast<SplitMode>(SplitModeVal);

    struct TileRange {
        uint64_t start, step, count;
    };

    __aicore__ inline MC2VecAddBias(TPipe *tPipe)
        : tPipe_(tPipe){};
    __aicore__ inline MC2AddBiasContext *GetContextPtr();
    __aicore__ inline void Init() {};
    __aicore__ inline void Process(uint32_t taskIndex);

protected:
    __aicore__ inline void Init(GM_ADDR matmulAddr, GM_ADDR biasAddr);
    __aicore__ inline void Process();
    __aicore__ inline void Destroy();

    static constexpr uint32_t ELEMS_PER_BLOCK = 32 / sizeof(MatmulDataType);
    static constexpr uint32_t ELEMS_PER_REPEAT = 256 / sizeof(MatmulDataType);
    static constexpr uint32_t TILE_M = 64;
    static constexpr uint32_t TILE_N = 256;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t AIV_PER_AIC = 2;

    // FP32 计算常量
    static constexpr uint32_t FP32_ELEMS_PER_BLOCK = 32 / sizeof(float);
    static constexpr uint32_t FP32_ELEMS_PER_REPEAT = 256 / sizeof(float);

    // matmul 为 half/bf16 时需要 FP32 中间计算; float 直接计算
    static constexpr bool NEED_FP32_COMPUTE = !std::is_same<MatmulDataType, float>::value;
    // bias 非 float 时需要 Cast→float
    static constexpr bool NEED_BIAS_CAST = !std::is_same<BiasDataType, float>::value;

    MC2AddBiasContext context_;
    GlobalTensor<MatmulDataType> matmulGm_;
    GlobalTensor<BiasDataType> biasGm_;
    TPipe *tPipe_;
    TBuf<TPosition::VECCALC> matmulTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> biasTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> matmulFp32Buf_[BUFFER_NUM]; // FP32 计算缓冲 (双缓冲)
    TBuf<TPosition::VECCALC> biasFp32Buf_;               // bias Cast→float (NEED_BIAS_CAST 时)
    LocalTensor<MatmulDataType> matmulTile_[BUFFER_NUM];
    LocalTensor<BiasDataType> biasTile_[BUFFER_NUM];
    LocalTensor<float> matmulFp32_[BUFFER_NUM];
    AscendC::TEventID copyInEvent_[BUFFER_NUM];
    AscendC::TEventID computeEvent_[BUFFER_NUM];
    AscendC::TEventID copyOutEvent_[BUFFER_NUM];
    AscendC::TEventID biasInEvent_[BUFFER_NUM];

    // M_SPLIT / N_SPLIT: 数据维度分区范围 [start, end)
    uint32_t myMStart_ = 0, myMEnd_ = 0;
    uint32_t myNStart_ = 0, myNEnd_ = 0;
    // MN_SPLIT: tile 索引分区
    uint32_t tileMCnt_ = 0, tileNCnt_ = 0;
    uint32_t tailM_ = 0, tailN_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t nTileStart_ = 0, nTileEnd_ = 0;
    uint32_t myTileStart_ = 0, myTileEnd_ = 0;

    static __aicore__ inline uint32_t TileSize(uint32_t idx, uint32_t cnt, uint32_t tail, uint32_t full)
    {
        return (idx == cnt - 1) ? tail : full;
    }

    template <typename T>
    static __aicore__ inline DataCopyPadExtParams<T> PadParams()
    {
        return {true, 0, 0, static_cast<T>(0)};
    }

    __aicore__ inline void CalcSplit()
    {
        usedCoreNum_ = static_cast<uint32_t>(GetBlockNum()) * AIV_PER_AIC;

        if constexpr (MODE == SplitMode::MN_SPLIT) {
            tileMCnt_ = Ceil(context_.M, TILE_M);
            tileNCnt_ = Ceil(context_.N, TILE_N);
            if (tileMCnt_ == 0 || tileNCnt_ == 0) {
                usedCoreNum_ = 0;
                return;
            }
            tailM_ = context_.M - static_cast<uint64_t>(tileMCnt_ - 1) * TILE_M;
            tailN_ = context_.N - static_cast<uint64_t>(tileNCnt_ - 1) * TILE_N;
            uint64_t total64 = static_cast<uint64_t>(tileMCnt_) * static_cast<uint64_t>(tileNCnt_);
            uint32_t total = static_cast<uint32_t>(total64);
            if (total < usedCoreNum_) {
                usedCoreNum_ = total;
            }
            uint32_t per = Ceil(total, usedCoreNum_);
            myTileStart_ = static_cast<uint32_t>(GetBlockIdx()) * per;
            myTileEnd_ = (myTileStart_ + per > total) ? total : myTileStart_ + per;
            nTileStart_ = myTileStart_ / tileMCnt_;
            nTileEnd_ = (myTileEnd_ == 0) ? 0 : (myTileEnd_ - 1) / tileMCnt_ + 1;
        } else {
            if (context_.M == 0 || context_.N == 0) {
                usedCoreNum_ = 0;
                return;
            }
            uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());

            if constexpr (MODE == SplitMode::M_SPLIT) {
                uint32_t mPerCore = Ceil(static_cast<uint32_t>(context_.M), usedCoreNum_);
                myMStart_ = coreId * mPerCore;
                myMEnd_ =
                    (myMStart_ + mPerCore > context_.M) ? static_cast<uint32_t>(context_.M) : (myMStart_ + mPerCore);
                myNStart_ = 0;
                myNEnd_ = static_cast<uint32_t>(context_.N);
            } else {
                uint32_t nPerCore = Ceil(static_cast<uint32_t>(context_.N), usedCoreNum_);
                myNStart_ = coreId * nPerCore;
                myNEnd_ =
                    (myNStart_ + nPerCore > context_.N) ? static_cast<uint32_t>(context_.N) : (myNStart_ + nPerCore);
                myMStart_ = 0;
                myMEnd_ = static_cast<uint32_t>(context_.M);
            }
        }
    }

    // --- MN_SPLIT 辅助 ---

    __aicore__ inline TileRange GetMRange(uint32_t nTile) const
    {
        uint32_t base = nTile * tileMCnt_;
        uint32_t lo = (base < myTileStart_) ? (myTileStart_ - base) : 0;
        uint32_t hi = (base + tileMCnt_ > myTileEnd_) ? (myTileEnd_ - base) : tileMCnt_;
        return {lo, 1, (hi > lo) ? (hi - lo) : 0};
    }

    // --- Bias 加载 ---

    __aicore__ inline void LoadBias(uint32_t nStart, uint32_t len, uint32_t bufIdx)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(len * sizeof(BiasDataType)), 0, 0, 0};
        DataCopyPad(biasTile_[bufIdx], biasGm_[nStart], params, PadParams<BiasDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_[bufIdx]);
    }

    __aicore__ inline void LoadBiasTile(uint32_t nTile, uint32_t bufIdx)
    {
        uint32_t nStart = nTile * TILE_N;
        uint32_t len = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LoadBias(nStart, len, bufIdx);
    }

    // Cast bias 到 float (若类型不同), 整个 M 循环内复用同一份 bias
    // Cast 与后续 Add 同在 V-pipe, 硬件保序保证 Cast 先完成, 无需额外同步
    __aicore__ inline LocalTensor<float> PrepareBias(uint32_t biasBufIdx, uint32_t curTileN)
    {
        if constexpr (NEED_BIAS_CAST) {
            LocalTensor<float> biasFp32 = biasFp32Buf_.template Get<float>();
            Cast(biasFp32, biasTile_[biasBufIdx], RoundMode::CAST_NONE, (int32_t)curTileN);
            return biasFp32;
        } else {
            return biasTile_[biasBufIdx].template ReinterpretCast<float>();
        }
    }

    // --- MTE2 / V / MTE3 三级流水 ---

    // isPad=true 时 DataCopyPad 自动将每行 pad 到 32B 边界, dstStride=0 即可
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[bufIdx]);
        uint32_t blockLen = curTileN * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), blockLen,
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0, 0};
        DataCopyPad(matmulTile_[bufIdx], matmulGm_[offset], params, PadParams<MatmulDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_[bufIdx]);
    }

    // ubRowStride: UB 中每行实际占用的元素数 (含 32B padding), 用于行偏移计算
    __aicore__ inline void Compute(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                   uint32_t ubRowStride, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_[bufIdx]);
        if constexpr (NEED_FP32_COMPUTE) {
            ComputeFp32(bias, curTileM, curTileN, ubRowStride, bufIdx);
        } else {
            ComputeDirect(bias, curTileM, curTileN, ubRowStride, bufIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(computeEvent_[bufIdx]);
    }

    // FP32 路径: Cast↑(half→float) → Add<float>(in-place) → Cast↓(float→half)
    __aicore__ inline void ComputeFp32(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                       uint32_t ubRowStride, uint32_t bufIdx)
    {
        uint32_t ubRowStrideFp32 = Ceil(curTileN, FP32_ELEMS_PER_BLOCK) * FP32_ELEMS_PER_BLOCK;
        bool rowContiguous = (ubRowStride == curTileN);

        // Step 1: Cast half→float
        if (rowContiguous) {
            Cast<float, MatmulDataType>(matmulFp32_[bufIdx], matmulTile_[bufIdx], RoundMode::CAST_NONE,
                                        (int32_t)(curTileM * curTileN));
        } else {
            for (uint32_t r = 0; r < curTileM; r++) {
                Cast<float, MatmulDataType>(matmulFp32_[bufIdx][r * ubRowStrideFp32],
                                            matmulTile_[bufIdx][r * ubRowStride], RoundMode::CAST_NONE,
                                            (int32_t)curTileN);
            }
        }

        // Step 2: Add<float> 广播 (src1RepStride=0)
        uint32_t segCnt = Ceil(curTileN, FP32_ELEMS_PER_REPEAT);
        uint8_t repStrideFp32 = static_cast<uint8_t>(ubRowStrideFp32 / FP32_ELEMS_PER_BLOCK);
        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * FP32_ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > FP32_ELEMS_PER_REPEAT) ? FP32_ELEMS_PER_REPEAT : (curTileN - off);
            if (len == FP32_ELEMS_PER_REPEAT) {
                Add<float>(matmulFp32_[bufIdx][off], matmulFp32_[bufIdx][off], bias[off],
                           (uint64_t)FP32_ELEMS_PER_REPEAT, (uint8_t)curTileM,
                           BinaryRepeatParams(1, 1, 1, repStrideFp32, repStrideFp32, 0));
            } else {
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * ubRowStrideFp32 + off;
                    Add<float>(matmulFp32_[bufIdx][rowOff], matmulFp32_[bufIdx][rowOff], bias[off], (int32_t)len);
                }
            }
        }

        // Step 3: Cast float→half (CAST_ROUND 四舍五入)
        if (rowContiguous) {
            Cast<MatmulDataType, float>(matmulTile_[bufIdx], matmulFp32_[bufIdx], RoundMode::CAST_ROUND,
                                        (int32_t)(curTileM * curTileN));
        } else {
            for (uint32_t r = 0; r < curTileM; r++) {
                Cast<MatmulDataType, float>(matmulTile_[bufIdx][r * ubRowStride],
                                            matmulFp32_[bufIdx][r * ubRowStrideFp32], RoundMode::CAST_ROUND,
                                            (int32_t)curTileN);
            }
        }
    }

    // float 路径: 直接 Add<float> (matmul 已是 float, 无需 Cast)
    __aicore__ inline void ComputeDirect(const LocalTensor<float> &bias, uint32_t curTileM, uint32_t curTileN,
                                         uint32_t ubRowStride, uint32_t bufIdx)
    {
        // MatmulDataType == float 时 matmulTile_ 即为 float tensor
        LocalTensor<float> matmulFp32 = matmulTile_[bufIdx].template ReinterpretCast<float>();
        uint32_t segCnt = Ceil(curTileN, FP32_ELEMS_PER_REPEAT);
        uint8_t repStride = static_cast<uint8_t>(ubRowStride / FP32_ELEMS_PER_BLOCK);
        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * FP32_ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > FP32_ELEMS_PER_REPEAT) ? FP32_ELEMS_PER_REPEAT : (curTileN - off);
            if (len == FP32_ELEMS_PER_REPEAT) {
                Add<float>(matmulFp32[off], matmulFp32[off], bias[off], (uint64_t)FP32_ELEMS_PER_REPEAT,
                           (uint8_t)curTileM, BinaryRepeatParams(1, 1, 1, repStride, repStride, 0));
            } else {
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * ubRowStride + off;
                    Add<float>(matmulFp32[rowOff], matmulFp32[rowOff], bias[off], (int32_t)len);
                }
            }
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(computeEvent_[bufIdx]);
        uint32_t blockLen = curTileN * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), blockLen, 0,
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0};
        DataCopyPad(matmulGm_[offset], matmulTile_[bufIdx], params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[bufIdx]);
    }

    __aicore__ inline void ProcessMTile(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t ubRowStride,
                                        const LocalTensor<float> &bias, uint32_t bufIdx)
    {
        CopyIn(offset, curTileM, curTileN, bufIdx);
        Compute(bias, curTileM, curTileN, ubRowStride, bufIdx);
        CopyOut(offset, curTileM, curTileN, bufIdx);
    }

    // --- MN_SPLIT: tile 索引迭代 ---

    __aicore__ inline void ProcessNTile(uint32_t nTile, uint32_t biasBufIdx, uint32_t nextBiasBufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_[biasBufIdx]);
        if (nTile + 1 < nTileEnd_) {
            LoadBiasTile(nTile + 1, nextBiasBufIdx);
        }
        uint32_t curTileN = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LocalTensor<float> bias = PrepareBias(biasBufIdx, curTileN);
        uint32_t ubRowStride = Ceil(curTileN, ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
        TileRange mr = GetMRange(nTile);
        uint32_t matmulBufIdx = 0;
        for (uint64_t i = 0; i < mr.count; i++) {
            uint32_t mTile = static_cast<uint32_t>(i * mr.step + mr.start);
            if (mTile >= tileMCnt_) {
                break;
            }
            uint32_t curTileM = TileSize(mTile, tileMCnt_, tailM_, TILE_M);
            uint64_t offset =
                static_cast<uint64_t>(mTile) * TILE_M * context_.N + static_cast<uint64_t>(nTile) * TILE_N;
            ProcessMTile(offset, curTileM, curTileN, ubRowStride, bias, matmulBufIdx);
            matmulBufIdx ^= 1;
        }
    }

    // --- M_SPLIT / N_SPLIT: 数据维度迭代 ---

    __aicore__ inline void ProcessRange()
    {
        uint32_t biasBufIdx = 0;
        uint32_t firstN = (myNEnd_ - myNStart_ < TILE_N) ? (myNEnd_ - myNStart_) : TILE_N;
        LoadBias(myNStart_, firstN, biasBufIdx);

        for (uint32_t nPos = myNStart_; nPos < myNEnd_; nPos += TILE_N) {
            uint32_t curTileN = (nPos + TILE_N > myNEnd_) ? (myNEnd_ - nPos) : TILE_N;
            uint32_t nextBiasBufIdx = biasBufIdx ^ 1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_[biasBufIdx]);

            uint32_t nextNPos = nPos + TILE_N;
            if (nextNPos < myNEnd_) {
                uint32_t nextLen = (nextNPos + TILE_N > myNEnd_) ? (myNEnd_ - nextNPos) : TILE_N;
                LoadBias(nextNPos, nextLen, nextBiasBufIdx);
            }

            LocalTensor<float> bias = PrepareBias(biasBufIdx, curTileN);
            uint32_t ubRowStride = Ceil(curTileN, ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

            uint32_t matmulBufIdx = 0;
            for (uint32_t mPos = myMStart_; mPos < myMEnd_; mPos += TILE_M) {
                uint32_t curTileM = (mPos + TILE_M > myMEnd_) ? (myMEnd_ - mPos) : TILE_M;
                uint64_t offset = static_cast<uint64_t>(mPos) * context_.N + nPos;
                ProcessMTile(offset, curTileM, curTileN, ubRowStride, bias, matmulBufIdx);
                matmulBufIdx ^= 1;
            }

            biasBufIdx = nextBiasBufIdx;
        }
    }
};

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline MC2AddBiasContext *MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::GetContextPtr()
{
    return &context_;
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process(uint32_t taskIndex)
{
    Init(context_.matmulAddr + static_cast<uint64_t>(taskIndex) * context_.matmulOffset,
         context_.biasAddr + static_cast<uint64_t>(taskIndex) * context_.biasOffset);
    Process();
    Destroy();
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Init(GM_ADDR matmulAddr,
                                                                                       GM_ADDR biasAddr)
{
    tPipe_->Reset();
    CalcSplit();
    matmulGm_.SetGlobalBuffer((__gm__ MatmulDataType *)matmulAddr);
    biasGm_.SetGlobalBuffer((__gm__ BiasDataType *)biasAddr);

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        tPipe_->InitBuffer(matmulTileBuf_[i],
                           static_cast<uint32_t>(static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(MatmulDataType)));
        tPipe_->InitBuffer(biasTileBuf_[i],
                           static_cast<uint32_t>(static_cast<uint64_t>(TILE_N) * sizeof(BiasDataType)));
        matmulTile_[i] = matmulTileBuf_[i].template Get<MatmulDataType>();
        biasTile_[i] = biasTileBuf_[i].template Get<BiasDataType>();
    }
    if constexpr (NEED_FP32_COMPUTE) {
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            tPipe_->InitBuffer(matmulFp32Buf_[i],
                               static_cast<uint32_t>(static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(float)));
            matmulFp32_[i] = matmulFp32Buf_[i].template Get<float>();
        }
    }
    if constexpr (NEED_BIAS_CAST) {
        tPipe_->InitBuffer(biasFp32Buf_, static_cast<uint32_t>(static_cast<uint64_t>(TILE_N) * sizeof(float)));
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process()
{
    if (static_cast<uint32_t>(GetBlockIdx()) >= usedCoreNum_) {
        return;
    }

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        copyInEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
        computeEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::V_S>();
        copyOutEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE3_S>();
        biasInEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);
    }

    if constexpr (MODE == SplitMode::MN_SPLIT) {
        if (nTileStart_ < nTileEnd_) {
            uint32_t biasBufIdx = 0;
            LoadBiasTile(nTileStart_, biasBufIdx);
            for (uint32_t n = nTileStart_; n < nTileEnd_; n++) {
                uint32_t nextBiasBufIdx = biasBufIdx ^ 1;
                ProcessNTile(n, biasBufIdx, nextBiasBufIdx);
                biasBufIdx = nextBiasBufIdx;
            }
        }
    } else {
        if (myMStart_ < myMEnd_ && myNStart_ < myNEnd_) {
            ProcessRange();
        }
    }

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::V_S>(computeEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(biasInEvent_[i]);
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Destroy()
{}

#ifndef DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION
#define DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION(MatmulDataType, BiasDataType, SplitModeVal, AddBiasType) \
    using AddBiasType = MC2KernelTemplate::MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>
#endif

} // namespace MC2KernelTemplate

#endif
