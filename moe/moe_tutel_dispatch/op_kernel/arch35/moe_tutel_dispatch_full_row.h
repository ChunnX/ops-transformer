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
        // 一块 buffer 放 tileRoutes 行，每行按 32B 对齐；VECIN/VECOUT 绑定，省掉 UB→UB 拷贝
        this->pipe_->InitBuffer(
            rowQue_, DOUBLE_BUFFER,
            static_cast<uint32_t>(this->tileRoutes_ * this->alignedH_ * static_cast<int64_t>(sizeof(TByte))));
    }

    __aicore__ inline void Process()
    {
        Base::InitOutputZero();
        Base::SyncBetweenStages();

        for (int64_t base = this->routeStart_; base < this->routeEnd_; base += this->tileRoutes_) {
            const int64_t cnt = MinInt64(this->tileRoutes_, this->routeEnd_ - base);
            ProcessTile(base, cnt);
        }
    }

private:
    __aicore__ inline void ProcessTile(int64_t base, int64_t cnt)
    {
        Base::LoadRouteMeta(base, cnt);

        DataCopyExtParams rowParams{1, this->rowBytes_, 0, 0, 0};
        DataCopyPadExtParams<TByte> rowPad;
        rowPad.isPad = 0;
        rowPad.leftPadding = 0;
        rowPad.rightPadding = 0;
        rowPad.paddingValue = 0;

        LocalTensor<TByte> rowLocal = rowQue_.template AllocTensor<TByte>();
        int64_t slot = 0;
        int64_t dstRow = 0;
        for (int64_t i = 0; i < cnt; ++i) {
            if (!Base::RouteValid(i, dstRow)) {
                continue;
            }
            const int64_t srcRow = (base + i) % this->samples_;
            DataCopyPad(rowLocal[slot * this->alignedH_], this->xGm_[srcRow * this->hidden_], rowParams, rowPad);
            ++slot;
        }
        rowQue_.EnQue(rowLocal);

        rowLocal = rowQue_.template DeQue<TByte>();
        slot = 0;
        for (int64_t i = 0; i < cnt; ++i) {
            if (!Base::RouteValid(i, dstRow)) {
                continue;
            }
            DataCopyPad(this->yGm_[dstRow * this->hidden_], rowLocal[slot * this->alignedH_], rowParams);
            ++slot;
        }
        rowQue_.FreeTensor(rowLocal);

        Base::FreeRouteMeta();
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> rowQue_;
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_FULL_ROW_H_
