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
 * \brief Vector Add Bias: add 1D bias [N] to matmul result [M, N] in-place (single card).
 * Buffer management reference fp8_dynamic_quant_pertoken.h: TBuf 在 Init 统一分配,
 * Destroy 统一释放; 手动 event 同步 MTE2→V→MTE3 流水, 避免异步搬运未完成时释放 buffer。
 *
 * 分核策略:
 *   M_SPLIT  — 数据维度分区: 每核分到 [mStart, mEnd) × 全 N, UB 内按 TILE_M × TILE_N 迭代
 *   N_SPLIT  — 数据维度分区: 每核分到 全 M × [nStart, nEnd), UB 内按 TILE_M × TILE_N 迭代
 *   MN_SPLIT — tile 索引分区: 每核分到一组 (mTile, nTile), 保持原有 tile 迭代逻辑
 */

#ifndef MC2_VEC_ADD_BIAS_H
#define MC2_VEC_ADD_BIAS_H

#include <type_traits>

namespace MC2KernelTemplate {
using namespace AscendC;

struct MC2AddBiasContext {
    GM_ADDR matmulAddr = 0;      // matmul result [M, N] row-major (in-place)
    GM_ADDR biasAddr = 0;        // bias [N]
    uint64_t matmulOffset = 0;   // per-task offset
    uint64_t biasOffset = 0;     // per-task offset (bias 为一维数组, 各 tile 共享, 偏移恒为 0)
    uint64_t M = 0;
    uint64_t N = 0;
};

// SplitModeVal 与 allto_all_matmul_tiling_key.h 宏定义对齐: 1=M_SPLIT, 2=N_SPLIT, 3=MN_SPLIT
template <typename MatmulDataType, typename BiasDataType = MatmulDataType,
          uint32_t SplitModeVal = 1>
class MC2VecAddBias {
public:
    enum SplitMode : uint32_t { M_SPLIT = 1, N_SPLIT = 2, MN_SPLIT = 3 };
    static constexpr SplitMode MODE = static_cast<SplitMode>(SplitModeVal);

    struct TileRange {
        uint64_t start, step, count;
    };

    __aicore__ inline MC2VecAddBias(TPipe *tPipe) : tPipe_(tPipe){};
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
    static constexpr uint32_t BUFFER_NUM = 2;    // 双缓冲
    static constexpr uint32_t AIV_PER_AIC = 2;

    static constexpr bool NEED_CAST = !std::is_same<MatmulDataType, BiasDataType>::value;
    static constexpr RoundMode CAST_MODE =
        std::is_same<MatmulDataType, half>::value || std::is_same<MatmulDataType, bfloat16_t>::value ?
            RoundMode::CAST_RINT : RoundMode::CAST_NONE;

    MC2AddBiasContext context_;
    GlobalTensor<MatmulDataType> matmulGm_;
    GlobalTensor<BiasDataType> biasGm_;
    TPipe *tPipe_;
    TBuf<TPosition::VECCALC> matmulTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> biasTileBuf_[BUFFER_NUM];
    TBuf<TPosition::VECCALC> biasCastBuf_;
    LocalTensor<MatmulDataType> matmulTile_[BUFFER_NUM];
    LocalTensor<BiasDataType> biasTile_[BUFFER_NUM];
    AscendC::TEventID copyInEvent_[BUFFER_NUM];   // MTE2: GM→UB 完成
    AscendC::TEventID computeEvent_[BUFFER_NUM];  // V:    Add 完成
    AscendC::TEventID copyOutEvent_[BUFFER_NUM];  // MTE3: UB→GM 完成
    AscendC::TEventID biasInEvent_[BUFFER_NUM];   // MTE2: bias 完成

    // M_SPLIT / N_SPLIT: 数据维度分区范围
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
            // MN_SPLIT: tile 索引分区 (保持原有逻辑)
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
            if (total < usedCoreNum_) usedCoreNum_ = total;
            uint32_t per = Ceil(total, usedCoreNum_);
            myTileStart_ = static_cast<uint32_t>(GetBlockIdx()) * per;
            myTileEnd_ = (myTileStart_ + per > total) ? total : myTileStart_ + per;
            nTileStart_ = myTileStart_ / tileMCnt_;
            nTileEnd_ = (myTileEnd_ == 0) ? 0 : (myTileEnd_ - 1) / tileMCnt_ + 1;
        } else {
            // M_SPLIT / N_SPLIT: 数据维度分区, 每核至少分到 1 行/列
            if (context_.M == 0 || context_.N == 0) {
                usedCoreNum_ = 0;
                return;
            }
            uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());

            if constexpr (MODE == SplitMode::M_SPLIT) {
                uint32_t mPerCore = Ceil(static_cast<uint32_t>(context_.M), usedCoreNum_);
                myMStart_ = coreId * mPerCore;
                myMEnd_ = (myMStart_ + mPerCore > context_.M) ?
                    static_cast<uint32_t>(context_.M) : (myMStart_ + mPerCore);
                myNStart_ = 0;
                myNEnd_ = static_cast<uint32_t>(context_.N);
            } else {
                // N_SPLIT
                uint32_t nPerCore = Ceil(static_cast<uint32_t>(context_.N), usedCoreNum_);
                myNStart_ = coreId * nPerCore;
                myNEnd_ = (myNStart_ + nPerCore > context_.N) ?
                    static_cast<uint32_t>(context_.N) : (myNStart_ + nPerCore);
                myMStart_ = 0;
                myMEnd_ = static_cast<uint32_t>(context_.M);
            }
        }
    }

    // --- MN_SPLIT: tile 索引迭代 ---

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
        DataCopyExtParams params{1, static_cast<uint32_t>(static_cast<uint64_t>(len) * sizeof(BiasDataType)), 0, 0, 0};
        DataCopyPad(biasTile_[bufIdx], biasGm_[nStart], params, PadParams<BiasDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_[bufIdx]);
    }

    __aicore__ inline void LoadBiasTile(uint32_t nTile, uint32_t bufIdx)
    {
        uint32_t nStart = nTile * TILE_N;
        uint32_t len = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LoadBias(nStart, len, bufIdx);
    }

    // Cast(可选) + 返回 Add 使用的 bias tensor, 整个 M 循环复用
    // NEED_CAST=false 时 BiasDataType==MatmulDataType, if constexpr 保证类型安全
    // Cast 与后续 Add 同在 V-pipe, 硬件保序保证 Cast 先完成; 无需额外同步
    __aicore__ inline LocalTensor<MatmulDataType> PrepareBias(uint32_t biasBufIdx, uint32_t curTileN)
    {
        if constexpr (NEED_CAST) {
            LocalTensor<MatmulDataType> dst = biasCastBuf_.template Get<MatmulDataType>();
            Cast(dst, biasTile_[biasBufIdx], CAST_MODE, (int32_t)curTileN);
            return dst;
        } else {
            // BiasDataType 与 MatmulDataType 相同, ReinterpretCast 保证返回类型一致
            return biasTile_[biasBufIdx].template ReinterpretCast<MatmulDataType>();
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[bufIdx]);  // 等上次 MTE3 完成再复用 buffer
        uint64_t blockLen = static_cast<uint64_t>(curTileN) * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), static_cast<uint32_t>(blockLen),
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0, 0};
        DataCopyPad(matmulTile_[bufIdx], matmulGm_[offset], params, PadParams<MatmulDataType>());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_[bufIdx]);
    }

    __aicore__ inline void Compute(const LocalTensor<MatmulDataType> &bias, uint32_t curTileM, uint32_t curTileN,
                                   uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(copyInEvent_[bufIdx]);  // 等 MTE2 搬入完成
        uint32_t segCnt = Ceil(curTileN, ELEMS_PER_REPEAT);
        uint8_t repStride = curTileN / ELEMS_PER_BLOCK;
        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > ELEMS_PER_REPEAT) ? ELEMS_PER_REPEAT : (curTileN - off);
            if (len == ELEMS_PER_REPEAT) {  // repeat 模式: src1RepStride=0 → bias 广播
                Add<MatmulDataType>(matmulTile_[bufIdx][off], matmulTile_[bufIdx][off], bias[off],
                                    (uint64_t)ELEMS_PER_REPEAT, (uint8_t)curTileM,
                                    BinaryRepeatParams(1, 1, 1, repStride, repStride, 0));
            } else {  // 尾 segment: count 模式逐行
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * curTileN + off;
                    Add(matmulTile_[bufIdx][rowOff], matmulTile_[bufIdx][rowOff], bias[off], (int32_t)len);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(computeEvent_[bufIdx]);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curTileM, uint32_t curTileN, uint32_t bufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(computeEvent_[bufIdx]);  // 等 V 计算完成
        uint64_t blockLen = static_cast<uint64_t>(curTileN) * sizeof(MatmulDataType);
        DataCopyExtParams params{static_cast<uint16_t>(curTileM), static_cast<uint32_t>(blockLen), 0,
                                 static_cast<int64_t>((context_.N - curTileN) * sizeof(MatmulDataType)), 0};
        DataCopyPad(matmulGm_[offset], matmulTile_[bufIdx], params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[bufIdx]);
    }

    __aicore__ inline void ProcessMTile(uint64_t offset, uint32_t curTileM, uint32_t curTileN,
                                        const LocalTensor<MatmulDataType> &bias, uint32_t bufIdx)
    {
        CopyIn(offset, curTileM, curTileN, bufIdx);
        Compute(bias, curTileM, curTileN, bufIdx);
        CopyOut(offset, curTileM, curTileN, bufIdx);
    }

    // --- MN_SPLIT: tile 索引迭代 ---

    __aicore__ inline void ProcessNTile(uint32_t nTile, uint32_t biasBufIdx, uint32_t nextBiasBufIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(biasInEvent_[biasBufIdx]);  // 等当前 bias 搬入完成
        if (nTile + 1 < nTileEnd_) {
            LoadBiasTile(nTile + 1, nextBiasBufIdx);  // 预取下一个 bias (MTE2 与 V 并行)
        }
        uint32_t curTileN = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LocalTensor<MatmulDataType> bias = PrepareBias(biasBufIdx, curTileN);
        TileRange mr = GetMRange(nTile);
        uint32_t matmulBufIdx = 0;
        for (uint64_t i = 0; i < mr.count; i++) {
            uint32_t mTile = static_cast<uint32_t>(i * mr.step + mr.start);
            if (mTile >= tileMCnt_) break;
            uint32_t curTileM = TileSize(mTile, tileMCnt_, tailM_, TILE_M);
            uint64_t offset = static_cast<uint64_t>(mTile) * TILE_M * context_.N +
                              static_cast<uint64_t>(nTile) * TILE_N;
            ProcessMTile(offset, curTileM, curTileN, bias, matmulBufIdx);
            matmulBufIdx ^= 1;  // pingpong
        }
    }

    // --- M_SPLIT / N_SPLIT: 数据维度迭代 ---

    __aicore__ inline void ProcessRange()
    {
        if (static_cast<uint32_t>(GetBlockIdx()) >= usedCoreNum_) return;
        if (myMStart_ >= myMEnd_ || myNStart_ >= myNEnd_) return;

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

            LocalTensor<MatmulDataType> bias = PrepareBias(biasBufIdx, curTileN);

            uint32_t matmulBufIdx = 0;
            for (uint32_t mPos = myMStart_; mPos < myMEnd_; mPos += TILE_M) {
                uint32_t curTileM = (mPos + TILE_M > myMEnd_) ? (myMEnd_ - mPos) : TILE_M;
                uint64_t offset = static_cast<uint64_t>(mPos) * context_.N + nPos;
                ProcessMTile(offset, curTileM, curTileN, bias, matmulBufIdx);
                matmulBufIdx ^= 1;  // pingpong
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
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Init(GM_ADDR matmulAddr, GM_ADDR biasAddr)
{
    tPipe_->Reset();
    CalcSplit();
    matmulGm_.SetGlobalBuffer((__gm__ MatmulDataType *)matmulAddr);
    biasGm_.SetGlobalBuffer((__gm__ BiasDataType *)biasAddr);

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        tPipe_->InitBuffer(matmulTileBuf_[i], static_cast<uint32_t>(
            static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(MatmulDataType)));
        tPipe_->InitBuffer(biasTileBuf_[i], static_cast<uint32_t>(
            static_cast<uint64_t>(TILE_N) * sizeof(BiasDataType)));
        matmulTile_[i] = matmulTileBuf_[i].template Get<MatmulDataType>();
        biasTile_[i] = biasTileBuf_[i].template Get<BiasDataType>();
    }
    if constexpr (NEED_CAST) {
        tPipe_->InitBuffer(biasCastBuf_, static_cast<uint32_t>(
            static_cast<uint64_t>(TILE_N) * sizeof(MatmulDataType)));
    }

    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        copyInEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
        computeEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::V_S>();
        copyOutEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE3_S>();
        biasInEvent_[i] = tPipe_->template AllocEventID<AscendC::HardEvent::MTE2_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);  // 首轮占位
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process()
{
    if (static_cast<uint32_t>(GetBlockIdx()) >= usedCoreNum_) return;

    if constexpr (MODE == SplitMode::MN_SPLIT) {
        uint32_t biasBufIdx = 0;
        LoadBiasTile(nTileStart_, biasBufIdx);
        for (uint32_t n = nTileStart_; n < nTileEnd_; n++) {
            uint32_t nextBiasBufIdx = biasBufIdx ^ 1;
            ProcessNTile(n, biasBufIdx, nextBiasBufIdx);
            biasBufIdx = nextBiasBufIdx;
        }
    } else {
        ProcessRange();
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Destroy()
{
    for (uint32_t i = 0; i < BUFFER_NUM; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(copyInEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::V_S>(computeEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_S>(copyOutEvent_[i]);
        tPipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_S>(biasInEvent_[i]);
    }
}

#ifndef DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION
#define DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION(MatmulDataType, BiasDataType, SplitModeVal, AddBiasType) \
    using AddBiasType = MC2KernelTemplate::MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>
#endif

} // namespace MC2KernelTemplate

#endif
