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
 * \file moe_tutel_dispatch_full_row.h
 * \brief TilingKey 0：一行完整驻留 UB，按 tileRoutes 条路由成批搬运
 */

#ifndef MOE_TUTEL_DISPATCH_FULL_ROW_H_
#define MOE_TUTEL_DISPATCH_FULL_ROW_H_

#include "moe_tutel_dispatch_base.h"

namespace MoeTutelDispatchNs {

template <typename TByte>
class MoeTutelDispatchFullRow : public MoeTutelDispatchBase<TByte> {
    using Base = MoeTutelDispatchBase<TByte>;

public:
    __aicore__ inline MoeTutelDispatchFullRow(TPipe *pipe, const MoeTutelDispatchTilingData *__restrict tiling)
        : Base(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y)
    {
        Base::InitCommon(x, gates, indices, locations, y);
    }

    __aicore__ inline void Process()
    {
        Base::InitOutputZero();
        Base::SyncBetweenStages();

        // Fill must finish before any InitBuffer. The payload queue holds routeBatch complete rows.
        Base::InitRouteBuffers();
        this->pipe_->InitBuffer(
            rowQue_, PAYLOAD_BUFFER_NUM,
            static_cast<uint32_t>(this->routeBatch_ * this->alignedH_ * static_cast<int64_t>(sizeof(TByte))));

        for (int64_t base = this->routeStart_; base < this->routeEnd_; base += this->routeBatch_) {
            const int64_t cnt = MinInt64(this->routeBatch_, this->routeEnd_ - base);
            ProcessTile(base, cnt);
        }
    }

private:
    __aicore__ inline void ProcessTile(int64_t base, int64_t cnt)
    {
        Base::LoadRouteMeta(base, cnt);

        LocalTensor<TByte> rowLocal = rowQue_.template AllocTensor<TByte>();
        CopyInputRows(rowLocal, base, cnt);
        rowQue_.EnQue(rowLocal);

        rowLocal = rowQue_.template DeQue<TByte>();
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(this->rowBytes_), 0, 0, 0};
        int64_t dstRow = 0;
        for (int64_t i = 0; i < cnt; ++i) {
            if (!Base::RouteValid(i, dstRow)) {
                continue;
            }
            DataCopyPad(this->yGm_[dstRow * this->hidden_], rowLocal[i * this->alignedH_], rowParams);
        }
        rowQue_.FreeTensor(rowLocal);

        Base::FreeRouteMeta();
    }

    __aicore__ inline void CopyInputRows(const LocalTensor<TByte> &rowLocal, int64_t base, int64_t cnt)
    {
        const uint8_t rightPadding = static_cast<uint8_t>(this->alignedH_ - this->hidden_);
        DataCopyPadExtParams<TByte> rowPad{false, 0, rightPadding, 0};
        const int64_t srcRow = base % this->samples_;
        const int64_t firstCount = MinInt64(cnt, this->samples_ - srcRow);
        DataCopyExtParams firstParams{static_cast<uint16_t>(firstCount), static_cast<uint32_t>(this->rowBytes_), 0, 0,
                                      0};
        DataCopyPad(rowLocal, this->xGm_[srcRow * this->hidden_], firstParams, rowPad);

        const int64_t secondCount = cnt - firstCount;
        if (secondCount > 0) {
            DataCopyExtParams secondParams{static_cast<uint16_t>(secondCount), static_cast<uint32_t>(this->rowBytes_),
                                           0, 0, 0};
            DataCopyPad(rowLocal[firstCount * this->alignedH_], this->xGm_, secondParams, rowPad);
        }
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, PAYLOAD_QUEUE_DEPTH> rowQue_;
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_FULL_ROW_H_
