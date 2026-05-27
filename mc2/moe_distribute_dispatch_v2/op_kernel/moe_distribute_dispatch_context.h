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
 * \brief 统一 v2 / v3 上下文：v2 走 HcclOpParam，v3 走 Mc2MoeContext。
 *        对外接口一致，由模板选择具体实现。
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

// v2: 由 hccl context 驱动，构造入参为 void* 强转为 Mc2Kernel::HcclOpParam*（EP）。
// TP context 在 InitTpAndCheck 内部通过 GetHcclContext<1>() 获取。
class V2HcclContextHolder {
public:
    __aicore__ inline V2HcclContextHolder() {};

    __aicore__ inline void Set(GM_ADDR ctxPtr)
    {
        epContext_ = (__gm__ Mc2Kernel::HcclOpParam *)ctxPtr;
    }

    __aicore__ inline void InitEpAndCheck(uint64_t totalWinSize, TPipe *pipe, GM_ADDR expandXOut)
    {
        CheckWindowSize(totalWinSize, Mc2Kernel::GetWinSize(epContext_), pipe, expandXOut);
    }

    __aicore__ inline void InitTpAndCheck(uint64_t totalWinSize, TPipe *pipe, GM_ADDR expandXOut)
    {
        tpContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<1>();
        CheckWindowSize(totalWinSize, Mc2Kernel::GetWinSize(tpContext_), pipe, expandXOut);
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return Mc2Kernel::GetStatusDataSpaceGm(epContext_);
    }

    __aicore__ inline uint32_t GetEpRankId(uint32_t /*fallback*/)
    {
        return Mc2Kernel::GetRankId(epContext_);
    }

    __aicore__ inline uint32_t GetEpWorldSize(uint32_t /*fallback*/)
    {
        return Mc2Kernel::GetRankDim(epContext_);
    }

    // 返回的是数据区基址（不含 winDataSizeOffset，由调用方追加）
    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, int32_t rankId, int32_t curRankId)
    {
        __gm__ Mc2Kernel::HcclOpParam *ctx = (ctxIdx == Mc2Kernel::COMM_EP_IDX) ? epContext_ : tpContext_;
        return Mc2Kernel::GetBaseWindAddrByRankId(ctx, rankId, curRankId);
    }

    // 返回的是状态区基址（不含 dataState*WIN_STATE_OFFSET，由调用方追加）
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, int32_t rankId, int32_t curRankId)
    {
        __gm__ Mc2Kernel::HcclOpParam *ctx = (ctxIdx == Mc2Kernel::COMM_EP_IDX) ? epContext_ : tpContext_;
        return Mc2Kernel::GetBaseWindStateAddrByRankId(ctx, rankId, curRankId);
    }

private:
    __gm__ Mc2Kernel::HcclOpParam *epContext_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam *tpContext_{nullptr};
};

// v3: 由 Mc2MoeContext 驱动，构造入参为 void* 强转为 Mc2Aclnn::Mc2MoeContext*。
// TP 通道不支持（v3.cpp 的 IsNeedAllgather 模板参数始终为 false）。
class V3Mc2MoeContextHolder {
public:
    __aicore__ inline V3Mc2MoeContextHolder() {};

    __aicore__ inline void Set(GM_ADDR ctxPtr)
    {
        mc2Context_ = (__gm__ Mc2Aclnn::Mc2MoeContext *)ctxPtr;
    }

    __aicore__ inline void InitEpAndCheck(uint64_t /*totalWinSize*/, TPipe * /*pipe*/, GM_ADDR /*expandXOut*/)
    {
        // mc2 上下文场景不做 winsize check
    }

    __aicore__ inline void InitTpAndCheck(uint64_t /*totalWinSize*/, TPipe * /*pipe*/, GM_ADDR /*expandXOut*/)
    {
        // 不支持 TP
    }

    __aicore__ inline GM_ADDR GetStatusDataSpaceGm()
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[mc2Context_->epRankId]);
    }

    __aicore__ inline uint32_t GetEpRankId(uint32_t /*fallback*/)
    {
        return mc2Context_->epRankId;
    }

    __aicore__ inline uint32_t GetEpWorldSize(uint32_t fallback)
    {
        return fallback;
    }

    // mc2 数据区基址在 epHcclBuffer 的 STATE_SIZE 之后，这里把 STATE_SIZE 加掉，
    // 保证调用方拿到与 v2 helper 同一“数据区基址”语义。ctxIdx/curRankId 不使用。
    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t /*ctxIdx*/, int32_t rankId, int32_t /*curRankId*/)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + Mc2Kernel::STATE_SIZE;
    }

    // 状态区基址即 epHcclBuffer 起始地址。ctxIdx/curRankId 不使用。
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t /*ctxIdx*/, int32_t rankId, int32_t /*curRankId*/)
    {
        return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
    }

private:
    __gm__ Mc2Aclnn::Mc2MoeContext *mc2Context_{nullptr};
};

} // namespace MoeDistributeDispatchV2Impl

#endif // MOE_DISTRIBUTE_DISPATCH_CONTEXT_H
