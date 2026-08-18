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
 *
 * Core loop: for n { cast bias; for m { CopyIn; Add; CopyOut } }
 *
 * SplitMode strategies (all keep the same for-n-for-m structure):
 * - M_SPLIT  (default): all N-tiles per core, subset of M-tiles (stride=usedCoreNum).
 * - N_SPLIT : subset of N-tiles per core, all M-tiles.
 * - MN_SPLIT: both axes split. Linear expansion (N-outer, M-inner) evenly divided
 *   across cores. Each core gets a contiguous segment → minimal N-tile crossing.
 *
 * Optimizations:
 * 1. Add repeat mode: src1RepeatStride=0 broadcasts bias across rows.
 * 2. bias double-buffer (TQue<2>): prefetch next N-tile bias.
 * 3. Type conversion: Cast bias before M loop to avoid V-pipe conflict with Add.
 *
 * Note: current implementation assumes a single contiguous [M, N] matmul block.
 * Extension for non-contiguous layouts (row stride / multi-block) is reserved —
 * add a block-iteration layer in Process() when needed.
 */

#ifndef MC2_VEC_ADD_BIAS_H
#define MC2_VEC_ADD_BIAS_H

#include <type_traits>

namespace MC2KernelTemplate {
using namespace AscendC;

struct MC2AddBiasContext {
    GM_ADDR matmulAddr;      // matmul result GM address [M, N], row-major (in-place)
    GM_ADDR biasAddr;        // bias GM address [N]
    uint64_t matmulOffset;   // per-task offset for matmul (multi-round)
    uint64_t biasOffset;     // per-task offset for bias (multi-round)
    uint64_t M;              // matmul result rows
    uint64_t N;              // matmul result columns (= bias length)
};

// 模板参数 SplitModeVal 使用 uint32_t 传递 (0=M_SPLIT, 1=N_SPLIT, 2=MN_SPLIT)
// 类内通过 enum SplitMode 转换, enum 完全封装在类内, 不污染命名空间
template <typename MatmulDataType, typename BiasDataType = MatmulDataType,
          uint32_t SplitModeVal = 0>
class MC2VecAddBias {
public:
    // 分核模式枚举 (仅类内使用, 不暴露到命名空间)
    enum SplitMode : uint32_t {
        M_SPLIT  = 0,
        N_SPLIT  = 1,
        MN_SPLIT = 2,
    };

    // 编译期常量: 模板参数值转换为 enum
    static constexpr SplitMode MODE = static_cast<SplitMode>(SplitModeVal);

    // M-tile 循环范围: mTile = start + i * step, i = 0..count-1
    struct TileRange {
        uint64_t start;
        uint64_t step;
        uint64_t count;
    };

    __aicore__ inline MC2VecAddBias(TPipe *tPipe) : tPipe_(tPipe){};
    __aicore__ inline MC2AddBiasContext *GetContextPtr();
    __aicore__ inline void Init() {};
    __aicore__ inline void Process(uint32_t taskIndex);

protected:
    __aicore__ inline void Init(GM_ADDR matmulAddr, GM_ADDR biasAddr);
    __aicore__ inline void Process();
    __aicore__ inline void Destroy();

    // --- 常量 ---
    static constexpr uint32_t BLOCK_BYTES = 32;
    static constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(MatmulDataType);
    static constexpr uint32_t ELEMS_PER_REPEAT = 256 / sizeof(MatmulDataType);
    static constexpr uint32_t TILE_M = (sizeof(MatmulDataType) <= 2) ? 128 : 64;
    static constexpr uint32_t TILE_N = 256;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t AIV_PER_AIC = 2;

    static constexpr bool NEED_CAST = !std::is_same<MatmulDataType, BiasDataType>::value;
    static constexpr RoundMode CAST_MODE =
        (std::is_same<MatmulDataType, half>::value ||
         std::is_same<MatmulDataType, bfloat16_t>::value)
            ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;

    // --- 成员 ---
    MC2AddBiasContext context_;
    GlobalTensor<MatmulDataType> matmulGm_;
    GlobalTensor<BiasDataType> biasGm_;
    TPipe *tPipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matmulQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> biasQueue_;
    TBuf<TPosition::VECCALC> biasCastBuf_;

    uint32_t tileMCnt_ = 0, tileNCnt_ = 0;
    uint32_t tailM_ = 0, tailN_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t nTileStart_ = 0, nTileEnd_ = 0;
    uint32_t mTilesPerCore_ = 0, nTilesPerCore_ = 0;
    uint32_t myTileStart_ = 0, myTileEnd_ = 0;

    // ================================================================
    // 辅助工具
    // ================================================================

    static __aicore__ inline uint32_t TileSize(uint32_t idx, uint32_t cnt,
                                                uint32_t tail, uint32_t full) {
        return (idx == cnt - 1) ? tail : full;
    }

    template <typename T>
    static __aicore__ inline DataCopyPadExtParams<T> PadParams() {
        return {true, 0, 0, *reinterpret_cast<T *>(uint8_t(0))};
    }

    // ================================================================
    // 分核计算
    // ================================================================

    __aicore__ inline void CalcSplit()
    {
        tileMCnt_ = Ceil(context_.M, TILE_M);
        tileNCnt_ = Ceil(context_.N, TILE_N);
        tailM_ = context_.M - static_cast<uint64_t>(tileMCnt_ - 1) * TILE_M;
        tailN_ = context_.N - static_cast<uint64_t>(tileNCnt_ - 1) * TILE_N;
        usedCoreNum_ = static_cast<uint32_t>(GetBlockNum()) * AIV_PER_AIC;

        if constexpr (MODE == SplitMode::N_SPLIT) {
            if (tileNCnt_ < usedCoreNum_) usedCoreNum_ = tileNCnt_;
            nTilesPerCore_ = Ceil(tileNCnt_, usedCoreNum_);
        } else if constexpr (MODE == SplitMode::MN_SPLIT) {
            uint32_t total = tileMCnt_ * tileNCnt_;
            if (total < usedCoreNum_) usedCoreNum_ = total;
            uint32_t per = Ceil(total, usedCoreNum_);
            myTileStart_ = static_cast<uint32_t>(GetBlockIdx()) * per;
            myTileEnd_ = (myTileStart_ + per > total) ? total : myTileStart_ + per;
        } else {
            if (tileMCnt_ < usedCoreNum_) usedCoreNum_ = tileMCnt_;
            mTilesPerCore_ = Ceil(tileMCnt_, usedCoreNum_);
        }

        CalcNRange();
    }

    __aicore__ inline void CalcNRange()
    {
        if constexpr (MODE == SplitMode::N_SPLIT) {
            nTileStart_ = static_cast<uint32_t>(GetBlockIdx()) * nTilesPerCore_;
            nTileEnd_ = (nTileStart_ + nTilesPerCore_ > tileNCnt_) ?
                        tileNCnt_ : nTileStart_ + nTilesPerCore_;
        } else if constexpr (MODE == SplitMode::MN_SPLIT) {
            nTileStart_ = myTileStart_ / tileMCnt_;
            nTileEnd_ = (myTileEnd_ == 0) ? 0 : (myTileEnd_ - 1) / tileMCnt_ + 1;
        } else {
            nTileStart_ = 0;
            nTileEnd_ = tileNCnt_;
        }
    }

    __aicore__ inline TileRange GetMRange(uint32_t nTile) const
    {
        if constexpr (MODE == SplitMode::M_SPLIT) {
            return {static_cast<uint64_t>(GetBlockIdx()), usedCoreNum_, mTilesPerCore_};
        } else if constexpr (MODE == SplitMode::N_SPLIT) {
            return {0, 1, tileMCnt_};
        } else {
            uint32_t base = nTile * tileMCnt_;
            uint32_t lo = (base < myTileStart_) ? (myTileStart_ - base) : 0;
            uint32_t hi = (base + tileMCnt_ > myTileEnd_) ? (myTileEnd_ - base) : tileMCnt_;
            return {lo, 1, (hi > lo) ? (hi - lo) : 0};
        }
    }

    // ================================================================
    // bias 准备: 载入 → (可选)Cast → 返回可计算的 tensor
    // ================================================================

    __aicore__ inline void LoadBiasTile(uint32_t nTile)
    {
        LocalTensor<BiasDataType> buf = biasQueue_.template AllocTensor<BiasDataType>();
        uint32_t len = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        uint64_t biasGmOffset = static_cast<uint64_t>(nTile) * TILE_N;
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(static_cast<uint64_t>(len) * sizeof(BiasDataType));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(buf, biasGm_[biasGmOffset], params, PadParams<BiasDataType>());
        biasQueue_.EnQue(buf);
    }

    __aicore__ inline LocalTensor<MatmulDataType> PrepareBias(
        LocalTensor<BiasDataType> &biasUb, uint32_t curTileN)
    {
        if constexpr (NEED_CAST) {
            LocalTensor<MatmulDataType> dst = biasCastBuf_.template Get<MatmulDataType>();
            Cast<MatmulDataType, BiasDataType>(dst, biasUb, CAST_MODE, (int32_t)curTileN);
            PipeBarrier<PIPE_V>();
            return dst;
        } else {
            return biasUb;
        }
    }

    // ================================================================
    // 单个子块流水: CopyIn → Compute → CopyOut
    // ================================================================

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curTileM, uint32_t curTileN)
    {
        LocalTensor<MatmulDataType> buf = matmulQueue_.template AllocTensor<MatmulDataType>();
        uint64_t blockLen = static_cast<uint64_t>(curTileN) * sizeof(MatmulDataType);
        uint64_t srcStride = (context_.N - curTileN) * sizeof(MatmulDataType);
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(curTileM);
        params.blockLen = static_cast<uint32_t>(blockLen);
        params.srcStride = static_cast<int64_t>(srcStride);
        params.dstStride = 0;
        DataCopyPad(buf, matmulGm_[offset], params, PadParams<MatmulDataType>());
        matmulQueue_.EnQue(buf);
    }

    __aicore__ inline void Compute(const LocalTensor<MatmulDataType> &bias,
                                   uint32_t curTileM, uint32_t curTileN)
    {
        LocalTensor<MatmulDataType> buf = matmulQueue_.template DeQue<MatmulDataType>();
        uint32_t segCnt = Ceil(curTileN, ELEMS_PER_REPEAT);
        uint8_t repStride = curTileN / ELEMS_PER_BLOCK;

        for (uint32_t s = 0; s < segCnt; s++) {
            uint32_t off = s * ELEMS_PER_REPEAT;
            uint32_t len = (curTileN - off > ELEMS_PER_REPEAT) ? ELEMS_PER_REPEAT : (curTileN - off);

            if (len == ELEMS_PER_REPEAT) {
                // repeat 模式: src1RepStride=0 → bias 行间广播
                Add<MatmulDataType>(buf[off], buf[off], bias[off],
                                    (uint64_t)ELEMS_PER_REPEAT, (uint8_t)curTileM,
                                    BinaryRepeatParams(1, 1, 1, repStride, repStride, 0));
            } else {
                // 尾 segment: count 模式逐行
                for (uint32_t r = 0; r < curTileM; r++) {
                    uint64_t rowOff = static_cast<uint64_t>(r) * curTileN + off;
                    Add(buf[rowOff], buf[rowOff], bias[off], (int32_t)len);
                }
            }
        }
        // 不需要 PipeBarrier<PIPE_V>: EnQue 自动设置 V→MTE3 事件,
        // 硬件保证所有 V 指令完成后才 fire, CopyOut 的 DeQue 会等待该事件
        matmulQueue_.EnQue(buf);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curTileM, uint32_t curTileN)
    {
        LocalTensor<MatmulDataType> buf = matmulQueue_.template DeQue<MatmulDataType>();
        uint64_t blockLen = static_cast<uint64_t>(curTileN) * sizeof(MatmulDataType);
        uint64_t dstStride = (context_.N - curTileN) * sizeof(MatmulDataType);
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(curTileM);
        params.blockLen = static_cast<uint32_t>(blockLen);
        params.srcStride = 0;
        params.dstStride = static_cast<int64_t>(dstStride);
        DataCopyPad(matmulGm_[offset], buf, params);
        matmulQueue_.FreeTensor(buf);
    }

    __aicore__ inline void ProcessMTile(uint32_t mTile, uint32_t nTile,
                                        const LocalTensor<MatmulDataType> &bias, uint32_t curTileN)
    {
        uint32_t curTileM = TileSize(mTile, tileMCnt_, tailM_, TILE_M);
        uint64_t offset = static_cast<uint64_t>(mTile) * TILE_M * context_.N +
                          static_cast<uint64_t>(nTile) * TILE_N;
        CopyIn(offset, curTileM, curTileN);
        Compute(bias, curTileM, curTileN);
        CopyOut(offset, curTileM, curTileN);
    }

    __aicore__ inline void ProcessNTile(uint32_t nTile)
    {
        LocalTensor<BiasDataType> biasUb = biasQueue_.template DeQue<BiasDataType>();

        // 预取下一个 bias tile (MTE2 与 V 并行)
        if (nTile + 1 < nTileEnd_) {
            LoadBiasTile(nTile + 1);
        }

        uint32_t curTileN = TileSize(nTile, tileNCnt_, tailN_, TILE_N);
        LocalTensor<MatmulDataType> bias = PrepareBias(biasUb, curTileN);

        TileRange mr = GetMRange(nTile);
        for (uint64_t i = 0; i < mr.count; i++) {
            uint32_t mTile = static_cast<uint32_t>(i * mr.step + mr.start);
            if (mTile >= tileMCnt_) break;
            ProcessMTile(mTile, nTile, bias, curTileN);
        }

        biasQueue_.FreeTensor(biasUb);
    }
};

// ================================================================
// 模板外定义
// ================================================================

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

    tPipe_->InitBuffer(matmulQueue_, BUFFER_NUM, static_cast<uint32_t>(
        static_cast<uint64_t>(TILE_M) * TILE_N * sizeof(MatmulDataType)));
    tPipe_->InitBuffer(biasQueue_, BUFFER_NUM, static_cast<uint32_t>(
        static_cast<uint64_t>(TILE_N) * sizeof(BiasDataType)));
    if constexpr (NEED_CAST) {
        tPipe_->InitBuffer(biasCastBuf_, static_cast<uint32_t>(
            static_cast<uint64_t>(TILE_N) * sizeof(MatmulDataType)));
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Process()
{
    if (static_cast<uint32_t>(GetBlockIdx()) >= usedCoreNum_) return;

    LoadBiasTile(nTileStart_);
    for (uint32_t n = nTileStart_; n < nTileEnd_; n++) {
        ProcessNTile(n);
    }
}

template <typename MatmulDataType, typename BiasDataType, uint32_t SplitModeVal>
__aicore__ inline void MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>::Destroy()
{
    matmulQueue_.FreeAllEvent();
    biasQueue_.FreeAllEvent();
}

// 使用 vec_add_bias 算子作为计算节点的计算实现
// SplitModeVal: 0=M_SPLIT, 1=N_SPLIT, 2=MN_SPLIT
#ifndef DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION
#define DEFINE_MC2_ADD_BIAS_FOR_MATH_COMPUTATION(MatmulDataType, BiasDataType, SplitModeVal, AddBiasType) \
    using AddBiasType = MC2KernelTemplate::MC2VecAddBias<MatmulDataType, BiasDataType, SplitModeVal>
#endif
} // namespace MC2KernelTemplate

#endif