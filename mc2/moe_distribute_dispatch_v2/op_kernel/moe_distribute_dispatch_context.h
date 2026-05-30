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
 * \file moe_distribute_dispatch_context.h
 * \brief 统一 v2 / v3 EP 上下文：v2 走 HcclOpParam，v3 走 Mc2MoeContext。
 *        对外接口一致，由模板选择具体实现。TP 域已下线，仅保留 EP 接口。
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_CONTEXT_H
#define MOE_DISTRIBUTE_DISPATCH_CONTEXT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "check_winsize.h"
#include "moe_distribute_v2_constant.h"
#include "../common/op_kernel/mc2_moe_context.h"
#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#endif

namespace MoeDistributeDispatchV2Impl {

using namespace AscendC;

// v3: 由 Mc2MoeContext 驱动。Mc2MoeContext 不携带 epWorldSize，
// Set() 时由调用方从 tilingData 显式传入并暂存。
class Mc2MoeContextHolderV3 {
public:
    __aicore__ inline Mc2MoeContextHolderV3() {};

    __aicore__ inline void Set(GM_ADDR ctxPtr, uint32_t epWorldSize)
    {
        mc2Context_ = (__gm__ Mc2Aclnn::Mc2MoeContext *)ctxPtr;
        epWorldSize_ = epWorldSize;
    }

    __aicore__ inline void InitEpAndCheck(uint64_t /*totalWinSize*/, TPipe * /*pipe*/, GM_ADDR /*expandXOut*/)
    {
        // mc2 上下文场景不做 winsize check
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[mc2Context_->epRankId]);
    }

    __aicore__ inline uint32_t GetEpRankId()
    {
        return mc2Context_->epRankId;
    }

    __aicore__ inline uint32_t GetEpWorldSize()
    {
        return epWorldSize_;
    }

    // mc2 数据区基址在 epHcclBuffer 的 STATE_SIZE 之后，这里把 STATE_SIZE 加掉，
    // 保证调用方拿到与 v2 helper 同一“数据区基址”语义。curRankId 不使用。
    __aicore__ inline GM_ADDR GetWindAddrByRankId(int32_t rankId, int32_t /*curRankId*/)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + Mc2Kernel::STATE_SIZE;
    }

    // 状态区基址即 epHcclBuffer 起始地址。curRankId 不使用。
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int32_t rankId, int32_t /*curRankId*/)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
    }

private:
    __gm__ Mc2Aclnn::Mc2MoeContext *mc2Context_{nullptr};
    uint32_t epWorldSize_{0};
};

// v2: 由 hccl context 驱动。Set() 入参 epWorldSize 仅为对齐接口签名，本侧忽略，
// 实际 worldSize 走 hccl GetRankDim。
class HcclContextHolderV2 {
public:
    __aicore__ inline HcclContextHolderV2() {};

    __aicore__ inline void Set(GM_ADDR ctxPtr, uint32_t /*epWorldSize*/)
    {
        epContext_ = (__gm__ Mc2Kernel::HcclOpParam *)ctxPtr;
    }

    __aicore__ inline void InitEpAndCheck(uint64_t totalWinSize, TPipe *pipe, GM_ADDR expandXOut)
    {
        CheckWindowSize(totalWinSize, Mc2Kernel::GetWinSize(epContext_), pipe, expandXOut);
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return Mc2Kernel::GetStatusDataSpaceGm(epContext_);
    }

    __aicore__ inline uint32_t GetEpRankId()
    {
        return Mc2Kernel::GetRankId(epContext_);
    }

    __aicore__ inline uint32_t GetEpWorldSize()
    {
        return Mc2Kernel::GetRankDim(epContext_);
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(int32_t rankId, int32_t curRankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(epContext_, rankId, curRankId);
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(int32_t rankId, int32_t curRankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(epContext_, rankId, curRankId);
    }

private:
    __gm__ Mc2Kernel::HcclOpParam *epContext_{nullptr};
};

} // namespace MoeDistributeDispatchV2Impl

#endif // MOE_DISTRIBUTE_DISPATCH_CONTEXT_H
