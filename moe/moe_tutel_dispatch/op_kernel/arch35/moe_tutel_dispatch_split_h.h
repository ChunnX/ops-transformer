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
 * \file moe_tutel_dispatch_split_h.h
 * \brief TilingKey 1：单行放不进 UB，按 hChunkElems 把 H 切块搬运
 */

#ifndef MOE_TUTEL_DISPATCH_SPLIT_H_H_
#define MOE_TUTEL_DISPATCH_SPLIT_H_H_

#include "moe_tutel_dispatch_base.h"

namespace MoeTutelDispatchNs {

template <typename TByte>
class MoeTutelDispatchSplitH : public MoeTutelDispatchBase<TByte> {
    using Base = MoeTutelDispatchBase<TByte>;

public:
    __aicore__ inline MoeTutelDispatchSplitH(TPipe *pipe, const MoeTutelDispatchTilingData *__restrict tiling)
        : Base(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y)
    {
        Base::InitCommon(x, gates, indices, locations, y);
        hChunkElems_ = this->tl_->hChunkElems;
        this->pipe_->InitBuffer(rowQue_, DOUBLE_BUFFER,
                                static_cast<uint32_t>(hChunkElems_ * static_cast<int64_t>(sizeof(TByte))));
    }

    __aicore__ inline void Process()
    {
        Base::InitOutputZero();
        Base::SyncBetweenStages();

        // tileRoutes 恒为 1：一次只解析一条路由，行内再按 hChunkElems 切块
        for (int64_t r = this->routeStart_; r < this->routeEnd_; ++r) {
            Base::LoadRouteMeta(r, 1);
            int64_t dstRow = 0;
            if (Base::RouteValid(0, dstRow)) {
                CopyRow((r % this->samples_) * this->hidden_, dstRow * this->hidden_);
            }
            Base::FreeRouteMeta();
        }
    }

private:
    __aicore__ inline void CopyRow(int64_t srcOffset, int64_t dstOffset)
    {
        DataCopyPadExtParams<TByte> rowPad;
        rowPad.isPad = 0;
        rowPad.leftPadding = 0;
        rowPad.rightPadding = 0;
        rowPad.paddingValue = 0;

        for (int64_t off = 0; off < this->hidden_; off += hChunkElems_) {
            const int64_t cur = MinInt64(hChunkElems_, this->hidden_ - off);
            DataCopyExtParams params{1, static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(TByte))), 0, 0, 0};

            LocalTensor<TByte> chunk = rowQue_.template AllocTensor<TByte>();
            DataCopyPad(chunk, this->xGm_[srcOffset + off], params, rowPad);
            rowQue_.EnQue(chunk);

            chunk = rowQue_.template DeQue<TByte>();
            DataCopyPad(this->yGm_[dstOffset + off], chunk, params);
            rowQue_.FreeTensor(chunk);
        }
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> rowQue_;
    int64_t hChunkElems_ = 0;
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_SPLIT_H_H_
