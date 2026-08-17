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
 * \file moe_tutel_dispatch_base.h
 * \brief 两阶段调度的公共部分：GM 绑定、阶段 A 清零、路由元数据分批读入与准入判定
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
          tl_(tiling)
    {}

protected:
    __aicore__ inline void InitCommon(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y)
    {
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        samples_ = tl_->samples;
        hidden_ = tl_->hidden;
        expertNum_ = tl_->expertNum;
        capacity_ = tl_->capacity;
        routeNum_ = tl_->routeNum;
        yElemCnt_ = tl_->yElemCnt;
        tileRoutes_ = tl_->tileRoutes;
        rowBytes_ = static_cast<uint32_t>(hidden_ * static_cast<int64_t>(sizeof(TByte)));
        alignedH_ = CeilAlignInt64(hidden_ * static_cast<int64_t>(sizeof(TByte)), BLOCK_SIZE) /
                    static_cast<int64_t>(sizeof(TByte));

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(x), samples_ * hidden_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(y), yElemCnt_);
        gatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TByte *>(gates), routeNum_);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), routeNum_);
        locationsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations), routeNum_);

        // 阶段 A / 阶段 B 各自的核内区间；分不到活的核区间为空，但仍要走到 SyncAll。
        initStart_ = blockIdx_ * tl_->initPerCore;
        initEnd_ = MinInt64(initStart_ + tl_->initPerCore, yElemCnt_);
        routeStart_ = blockIdx_ * tl_->routePerCore;
        routeEnd_ = MinInt64(routeStart_ + tl_->routePerCore, routeNum_);

        pipe_->InitBuffer(idxQue_, 1,
                          static_cast<uint32_t>(CeilAlignInt64(tileRoutes_ * static_cast<int64_t>(sizeof(int32_t)),
                                                               BLOCK_SIZE)));
        pipe_->InitBuffer(locQue_, 1,
                          static_cast<uint32_t>(CeilAlignInt64(tileRoutes_ * static_cast<int64_t>(sizeof(int32_t)),
                                                               BLOCK_SIZE)));
        pipe_->InitBuffer(gateQue_, 1,
                          static_cast<uint32_t>(CeilAlignInt64(tileRoutes_ * static_cast<int64_t>(sizeof(TByte)),
                                                               BLOCK_SIZE)));
    }

    // 阶段 A：把本核负责的 y 区间写 0。未命中槽位的输出值由此定型为 0。
    __aicore__ inline void InitOutputZero()
    {
        if (initEnd_ > initStart_) {
            GlobalTensor<TByte> dst = yGm_[initStart_];
            InitGlobalMemory(dst, static_cast<uint64_t>(initEnd_ - initStart_), static_cast<TByte>(0));
        }
    }

    // 阶段 A 与阶段 B 之间的唯一屏障，所有被拉起的核都无条件到达。
    __aicore__ inline void SyncBetweenStages()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void LoadRouteMeta(int64_t base, int64_t cnt)
    {
        idxLocal_ = idxQue_.template AllocTensor<int32_t>();
        locLocal_ = locQue_.template AllocTensor<int32_t>();
        gateLocal_ = gateQue_.template AllocTensor<TByte>();

        DataCopyExtParams idxParams{1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0};
        DataCopyExtParams gateParams{1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(TByte))), 0, 0, 0};
        DataCopyPadExtParams<int32_t> idxPad;
        idxPad.isPad = 0;
        idxPad.leftPadding = 0;
        idxPad.rightPadding = 0;
        idxPad.paddingValue = 0;
        DataCopyPadExtParams<TByte> gatePad;
        gatePad.isPad = 0;
        gatePad.leftPadding = 0;
        gatePad.rightPadding = 0;
        gatePad.paddingValue = 0;

        DataCopyPad(idxLocal_, indicesGm_[base], idxParams, idxPad);
        DataCopyPad(locLocal_, locationsGm_[base], idxParams, idxPad);
        DataCopyPad(gateLocal_, gatesGm_[base], gateParams, gatePad);

        idxQue_.EnQue(idxLocal_);
        locQue_.EnQue(locLocal_);
        gateQue_.EnQue(gateLocal_);
        idxLocal_ = idxQue_.template DeQue<int32_t>();
        locLocal_ = locQue_.template DeQue<int32_t>();
        gateLocal_ = gateQue_.template DeQue<TByte>();
    }

    __aicore__ inline void FreeRouteMeta()
    {
        idxQue_.FreeTensor(idxLocal_);
        locQue_.FreeTensor(locLocal_);
        gateQue_.FreeTensor(gateLocal_);
    }

    /*
     * 准入判定，与 canndev A2 kernel 的 `locations_ < capacity_ && indices_ >= 0 && gates_ != 0`
     * 在 A2 有定义的输入上逐位等价；额外的 `loc >= 0` / `idx < expertNum_` 只在 A2 会越界踩内存
     * 的输入上生效（跳过而不是写到 y 之外）。
     */
    __aicore__ inline bool RouteValid(int64_t i, int64_t &dstRow) const
    {
        const int64_t loc = static_cast<int64_t>(locLocal_.GetValue(i));
        if (loc < 0 || loc >= capacity_) {
            return false;
        }
        const int64_t idx = static_cast<int64_t>(idxLocal_.GetValue(i));
        if (idx < 0 || idx >= expertNum_) {
            return false;
        }
        if ((gateLocal_.GetValue(i) & GateAbsMask<TByte>()) == static_cast<TByte>(0)) {
            return false;
        }
        dstRow = idx * capacity_ + loc;
        return true;
    }

    TPipe *pipe_ = nullptr;
    const MoeTutelDispatchTilingData *__restrict tl_ = nullptr;

    GlobalTensor<TByte> xGm_;
    GlobalTensor<TByte> yGm_;
    GlobalTensor<TByte> gatesGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> locationsGm_;

    TQue<QuePosition::VECIN, 1> idxQue_;
    TQue<QuePosition::VECIN, 1> locQue_;
    TQue<QuePosition::VECIN, 1> gateQue_;
    LocalTensor<int32_t> idxLocal_;
    LocalTensor<int32_t> locLocal_;
    LocalTensor<TByte> gateLocal_;

    int64_t blockIdx_ = 0;
    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t expertNum_ = 0;
    int64_t capacity_ = 0;
    int64_t routeNum_ = 0;
    int64_t yElemCnt_ = 0;
    int64_t tileRoutes_ = 1;
    int64_t alignedH_ = 0;
    uint32_t rowBytes_ = 0;

    int64_t initStart_ = 0;
    int64_t initEnd_ = 0;
    int64_t routeStart_ = 0;
    int64_t routeEnd_ = 0;
};
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_BASE_H_
