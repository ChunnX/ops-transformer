/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_base.h
 * \brief Common GM binding, zero initialization, metadata loading and route admission.
 */

#ifndef MOE_TUTEL_DISPATCH_BASE_H_
#define MOE_TUTEL_DISPATCH_BASE_H_

#include "moe_tutel_dispatch_common.h"

namespace MoeTutelDispatchNs {

template <typename TByte>
class MoeTutelDispatchBase {
public:
    __aicore__ inline MoeTutelDispatchBase(TPipe *pipe, const MoeTutelDispatchTilingData *__restrict tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

protected:
    __aicore__ inline void InitCommon(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y)
    {
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        samples_ = tiling_->samples;
        hidden_ = tiling_->hidden;
        expertE_ = tiling_->expertE;
        capacity_ = tiling_->capacity;
        routeNum_ = tiling_->routeNum;
        yElemCnt_ = tiling_->yElemCnt;
        usedCoreNum_ = tiling_->usedCoreNum;
        routeBatch_ = tiling_->routeBatch;
        rowBytes_ = hidden_ * static_cast<int64_t>(sizeof(TByte));
        alignedH_ = CeilAlignInt64(rowBytes_, BLOCK_SIZE) / static_cast<int64_t>(sizeof(TByte));

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(x), samples_ * hidden_);
        gatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(gates), routeNum_);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), routeNum_);
        locationsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations), routeNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(y), yElemCnt_);

        PartitionRange(yElemCnt_, usedCoreNum_, blockIdx_, initStart_, initLen_);
        PartitionRange(routeNum_, usedCoreNum_, blockIdx_, routeStart_, routeLen_);
        routeEnd_ = routeStart_ + routeLen_;
    }

    __aicore__ inline void InitRouteBuffers()
    {
        const uint32_t indexBytes =
            static_cast<uint32_t>(CeilAlignInt64(routeBatch_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_SIZE));
        const uint32_t gateBytes =
            static_cast<uint32_t>(CeilAlignInt64(routeBatch_ * static_cast<int64_t>(sizeof(TByte)), BLOCK_SIZE));
        pipe_->InitBuffer(indexQueue_, 1, indexBytes);
        pipe_->InitBuffer(locationQueue_, 1, indexBytes);
        pipe_->InitBuffer(gateQueue_, 1, gateBytes);
    }

    __aicore__ inline void InitOutputZero()
    {
        if (initLen_ > 0) {
            GlobalTensor<TByte> outputShard;
            outputShard.SetGlobalBuffer(yGm_.GetPhyAddr(initStart_), initLen_);
            Fill(outputShard, static_cast<uint64_t>(initLen_), static_cast<TByte>(0));
        }
    }

    __aicore__ inline void SyncBetweenStages()
    {
        // All launched blocks execute this barrier, including blocks with an empty init or route shard.
        SyncAll();
    }

    __aicore__ inline void LoadRouteMeta(int64_t base, int64_t count)
    {
        indexLocal_ = indexQueue_.template AllocTensor<int32_t>();
        locationLocal_ = locationQueue_.template AllocTensor<int32_t>();
        gateLocal_ = gateQueue_.template AllocTensor<TByte>();

        DataCopyExtParams indexParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0};
        DataCopyExtParams gateParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(TByte))), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPad{false, 0, 0, 0};
        DataCopyPadExtParams<TByte> gatePad{false, 0, 0, 0};

        DataCopyPad(indexLocal_, indicesGm_[base], indexParams, indexPad);
        DataCopyPad(locationLocal_, locationsGm_[base], indexParams, indexPad);
        DataCopyPad(gateLocal_, gatesGm_[base], gateParams, gatePad);

        indexQueue_.EnQue(indexLocal_);
        locationQueue_.EnQue(locationLocal_);
        gateQueue_.EnQue(gateLocal_);
        indexLocal_ = indexQueue_.template DeQue<int32_t>();
        locationLocal_ = locationQueue_.template DeQue<int32_t>();
        gateLocal_ = gateQueue_.template DeQue<TByte>();
    }

    __aicore__ inline void FreeRouteMeta()
    {
        indexQueue_.FreeTensor(indexLocal_);
        locationQueue_.FreeTensor(locationLocal_);
        gateQueue_.FreeTensor(gateLocal_);
    }

    __aicore__ inline bool RouteValid(int64_t localIndex, int64_t &dstRow) const
    {
        const int64_t location = static_cast<int64_t>(locationLocal_.GetValue(localIndex));
        if (location < 0 || location >= capacity_) {
            return false;
        }
        const int64_t expert = static_cast<int64_t>(indexLocal_.GetValue(localIndex));
        if (expert < 0 || expert >= expertE_) {
            return false;
        }
        if ((gateLocal_.GetValue(localIndex) & GateAbsMask<TByte>()) == static_cast<TByte>(0)) {
            return false;
        }
        dstRow = expert * capacity_ + location;
        return true;
    }

    __aicore__ inline static void PartitionRange(int64_t total, int64_t parts, int64_t index, int64_t &start,
                                                 int64_t &length)
    {
        const int64_t quotient = total / parts;
        const int64_t remainder = total % parts;
        start = index * quotient + MinInt64(index, remainder);
        length = quotient + (index < remainder ? 1 : 0);
    }

    TPipe *pipe_ = nullptr;
    const MoeTutelDispatchTilingData *__restrict tiling_ = nullptr;

    GlobalTensor<TByte> xGm_;
    GlobalTensor<TByte> gatesGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> locationsGm_;
    GlobalTensor<TByte> yGm_;

    TQue<QuePosition::VECIN, 1> indexQueue_;
    TQue<QuePosition::VECIN, 1> locationQueue_;
    TQue<QuePosition::VECIN, 1> gateQueue_;
    LocalTensor<int32_t> indexLocal_;
    LocalTensor<int32_t> locationLocal_;
    LocalTensor<TByte> gateLocal_;

    int64_t blockIdx_ = 0;
    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t expertE_ = 0;
    int64_t capacity_ = 0;
    int64_t routeNum_ = 0;
    int64_t yElemCnt_ = 0;
    int64_t usedCoreNum_ = 1;
    int64_t routeBatch_ = 1;
    int64_t rowBytes_ = 0;
    int64_t alignedH_ = 0;

    int64_t initStart_ = 0;
    int64_t initLen_ = 0;
    int64_t routeStart_ = 0;
    int64_t routeLen_ = 0;
    int64_t routeEnd_ = 0;
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_BASE_H_
