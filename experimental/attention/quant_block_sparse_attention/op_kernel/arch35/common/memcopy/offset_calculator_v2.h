/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file offset_calculator_v2.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_V2_H
#define OFFSET_CALCULATOR_V2_H

#include "gm_layout.h"
#include "parser.h"

using AscendC::GlobalTensor;

// ----------------------------------------------GmLayoutParams--------------------------------
enum class FormatCategory {
    GM_Q_OUT_TND = 0,
    GM_KV_TND = 1,
    GM_KV_PA_BNBD = 2,
    GM_ANTIQ_BnNBs = 3,
};

template <GmFormat FORMAT>
struct GmLayoutParams {};

template <>
struct GmLayoutParams<GmFormat::TNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};
template <>
struct GmLayoutParams<GmFormat::TND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnBsND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnNBsD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnNBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnNBs;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY, typename ACTLEN_T, bool WITH_ZERO_HEAD = false>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT, typename ACTLEN_T, bool WITH_ZERO_HEAD>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND, ACTLEN_T, WITH_ZERO_HEAD> {
    GmLayout<FORMAT> gmLayout;
    using SeqLensQParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, WITH_ZERO_HEAD>;
    SeqLensQParserType actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, const SeqLensQParserType& parser)
    {
        actualSeqLensQParser = parser;
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensQParser.GetTBase(bIdx) + s1Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};
template <GmFormat FORMAT, typename ACTLEN_T, bool WITH_ZERO_HEAD>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND, ACTLEN_T, WITH_ZERO_HEAD> {
    GmLayout<FORMAT> gmLayout;
    using SeqLensKVParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, WITH_ZERO_HEAD>;
    SeqLensKVParserType actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV,
                                uint32_t actualLenKVDims)
    {
        actualSeqLensKVParser.Init(actualSeqLengthsGmKV, actualLenKVDims);
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline void Init(uint32_t n2, uint32_t d, const SeqLensKVParserType& parser)
    {
        actualSeqLensKVParser = parser;
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensKVParser.GetTBase(bIdx) + s2Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};
template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);

        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};
template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnNBs, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n, uint32_t blockSize, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n, blockSize);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t nIdx, uint32_t sIdx)
    {
        uint64_t blockIdxInBatch = sIdx / GetStrideBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = sIdx % GetStrideBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset = blockIdx * GetStrideBlockNum() + nIdx * GetStrideN() + bsIdx * GetStrideBlockSize();

        return offset;
    }

    // Get Stride

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimN()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }
};

template <GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY, ACTLEN_T, WITH_ZERO_HEAD> {
};

template <typename Q_T, GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT, ACTLEN_T, WITH_ZERO_HEAD> offsetCalculator;
};
#endif