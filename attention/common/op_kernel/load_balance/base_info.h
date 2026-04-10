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
 * \file base_info.h
 * \brief 
 */

#ifndef BASE_INFO_H
#define BASE_INFO_H

#include <cstdint>
#include <vector>
#include "load_balance_common.h"
#include <unordered_map>

namespace load_balance {

/**
 * This interface provides basic shape info for Balancer
 * This interface should be implemented by Operator itself
 */
class IBaseInfo {
public:
    IBaseInfo() = default;
    virtual ~IBaseInfo() = default;

    virtual uint32_t GetBatchSize() const = 0;
    virtual uint32_t GetGroupSize() const = 0;
    virtual uint32_t GetQueryHeadNum() const = 0;
    virtual uint32_t GetKvHeadNum() const = 0;
    virtual uint32_t GetHeadDim() const = 0;
    virtual uint32_t GetQuerySeqSize() const = 0;
    virtual uint32_t GetQuerySeqSize(uint32_t batchIdx) const = 0;
    virtual uint32_t GetKvSeqSize() const = 0;
    virtual uint32_t GetKvSeqSize(uint32_t batchIdx) const = 0;
    virtual SparseMode GetSparseMode() const = 0;
    virtual int64_t GetPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size) const = 0;
    virtual int64_t GetNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size) const = 0;
    virtual bool GetIsS1G() const = 0;
    virtual Layout GetQueryLayout() const = 0;
    virtual Layout GetKvLayout() const = 0;
    virtual DataType GetQueryDataType() const = 0;
    virtual DataType GetKvDataType() const = 0;
};

/**
 * BaseInfo represents as a standard IBaseInfo for convenience
 */
class BaseInfo : public IBaseInfo {
public:
    BaseInfo() = default;
    ~BaseInfo() override = default;

    uint32_t GetBatchSize() const override
    {
        return batchSize;
    }

    uint32_t GetGroupSize() const override
    {
        if (kvHeadNum == 0) {                                                                                                                                                        
            return 1;                                                                                                                                                                
        }
        return queryHeadNum / kvHeadNum;
    };

    uint32_t GetQueryHeadNum() const override
    {
        return queryHeadNum;
    }

    uint32_t GetKvHeadNum() const override
    {
        return kvHeadNum;
    }

    uint32_t GetHeadDim() const override
    {
        return headDim;
    }

    uint32_t GetQuerySeqSize() const override
    {
        return querySeqSize;
    }

    uint32_t GetQuerySeqSize(uint32_t batchIdx) const override
    {
        if (actualQuerySeqSize.empty()) {
            return querySeqSize;
        }

        if (actualQuerySeqSize.size() == 1U) {
            return static_cast<uint32_t>(actualQuerySeqSize[0]);
        }

        if (!isCumulativeQuerySeq) {
            return static_cast<uint32_t>(actualQuerySeqSize[batchIdx]);
        }

        return (batchIdx == 0) ? static_cast<uint32_t>(actualQuerySeqSize[batchIdx]) :
               static_cast<uint32_t>(actualQuerySeqSize[batchIdx] - actualQuerySeqSize[batchIdx - 1U]);
    }

    uint32_t GetKvSeqSize() const override
    {
        return kvSeqSize;
    }

    uint32_t GetKvSeqSize(uint32_t batchIdx) const override
    {
        if (actualKvSeqSize.empty()) {
            return kvSeqSize;
        }

        if (actualKvSeqSize.size() == 1U) {
            return static_cast<uint32_t>(actualKvSeqSize[0]);
        }

        if (!isCumulativeKvSeq) {
            return static_cast<uint32_t>(actualKvSeqSize[batchIdx]);
        }

        return (batchIdx == 0) ? static_cast<uint32_t>(actualKvSeqSize[batchIdx]) :
               static_cast<uint32_t>(actualKvSeqSize[batchIdx] - actualKvSeqSize[batchIdx - 1U]);
    }

    SparseMode GetSparseMode() const override
    {
        if (!attenMaskFlag) {
            return SparseMode::BUTT;
        }

        if (sparseMode > static_cast<uint32_t>(SparseMode::BUTT)) {
            return SparseMode::BUTT;
        }
        return static_cast<SparseMode>(sparseMode);
    }

    int64_t GetPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size) const override
    {
        auto mode = GetSparseMode();
        switch (mode) {
            case SparseMode::BAND:
                return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken;
            default:
                return preToken;
        }
    }

    int64_t GetNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size) const override
    {
        auto mode = GetSparseMode();
        switch (mode) {
            case SparseMode::DEFAULT_MASK:
            case SparseMode::ALL_MASK:
            case SparseMode::LEFT_UP_CAUSAL:
                return nextToken;
            case SparseMode::RIGHT_DOWN_CAUSAL:
                return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
            case SparseMode::BAND:
                return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken;
            default:
                return nextToken;
        }
    }

    bool GetIsS1G() const override
    {
        return (layoutKv == Layout::TND || layoutKv == Layout::BSH || layoutKv == Layout::BSND);
    }

    Layout GetQueryLayout() const override
    {
        return layoutQuery;
    }

    Layout GetKvLayout() const override
    {
        return layoutKv;
    }

    DataType GetQueryDataType() const override
    {
        return queryType;
    }

    DataType GetKvDataType() const override
    {
        return kvType;
    }
public:
    uint32_t batchSize { 0U };
    uint32_t queryHeadNum { 0U };
    uint32_t querySeqSize { 0U };
    uint32_t kvHeadNum { 0U };
    uint32_t kvSeqSize { 0U };
    uint32_t headDim { 64U };
    bool attenMaskFlag { false };
    uint32_t sparseMode { 0U };
    uint32_t preToken { 0U };
    uint32_t nextToken { 0U };
    bool isCumulativeQuerySeq { false };
    bool isCumulativeKvSeq { false };
    std::vector<int64_t> actualQuerySeqSize {};
    std::vector<int64_t> actualKvSeqSize {};
    Layout layoutQuery { Layout::BSND };
    Layout layoutKv { Layout::BSND };
    DataType queryType { DataType::FP16 };
    DataType kvType { DataType::FP16 };
};

}
#endif //BASE_INFO_H
