/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_worker_batching_tiling_arch35.cpp
 * \brief FfnWorkerBatching arch35 (Ascend950 / DAV_3510) Regbase tiling（1000 档 + IsRegbaseSocVersion 守卫）。
 *        UB 容量/核数运行时经 GetCoreMemSize/GetCoreNumAiv 取值，禁写死 arch 常量。
 *        切分阈值/workspace 公式沿用 A2（ffn_worker_batching_tiling.cpp），输入换成 arch35 运行时值自动重算。
 *        TilingData 采用 host/kernel 共用平铺 struct（GetTilingData<FfnWorkerBatchingArch35TilingData>() 直写）。
 */
#include "ffn_worker_batching_tiling.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "../op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h"

namespace optiling {
namespace {
constexpr uint32_t EXPERT_NUM_ATTR = 0;
constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;
constexpr uint32_t TOKEN_DTYPE_ATTR = 2;
constexpr uint32_t NEED_SCHEDULE_ATTR = 3;
constexpr uint32_t LAY_NUM_ATTR = 4;

constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;

constexpr int64_t BATCH_MODE = 1;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MAX_RESERVE_WK_NUM = 128;

constexpr int64_t TILING_KEY_NORM = 100;
constexpr int64_t TILING_KEY_RECV = 101;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t EXPERT_IDX_MAX = 8192;
constexpr int64_t MAX_SESSION_NUM = 1024;
constexpr int64_t MAX_K_NUM = 64;
constexpr int64_t TH_RECV_CORE_NUM = 32;
constexpr int64_t TH_RECV_MIN_ROWS_PER_CORE = 64;

// arch35 系统预留 UB：GetCoreMemSize(UB) 返回平台标称 UB（本机 248KB），但 vector core
// 实际可用比标称少 32KB 系统预留（穿刺2 同进程实测 kernel 侧 UBUF_PER_VECTOR_CORE=216KB）。
// sort 多核每 loop 4-buffer footprint 按标称 248KB 派生 sortLoopMaxElement 会超物理 UB 溢出，
// 故派生前扣减系统预留，与同仓 arch35 算子（mhc_pre_sinkhorn_backward）一致。非写死 UB 容量。
constexpr uint64_t UB_SYS_RESERVED_SIZE = 32 * 1024;
} // namespace

class FfnWorkerBatchingTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FfnWorkerBatchingTilingArch35(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
    }
    ~FfnWorkerBatchingTilingArch35() override = default;

protected:
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckInputParam();
    ge::graphStatus GetAttrsInfo();

    FfnWorkerBatchingArch35TilingData *tilingDataPtr_ = nullptr;
    int64_t A_ = 0;
    int64_t BS_ = 0;
    int64_t K_ = 0;
    int64_t Y_ = 0;
    int64_t H_ = 0;
    int64_t expertNum_ = 0;
    int64_t tokenDtype_ = 0;
    int64_t needSchedule_ = 0;
    int64_t layerNum_ = 0;
    int64_t aivNum_ = 0;
    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
};

ge::graphStatus FfnWorkerBatchingTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    aivNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(aivNum_ == 0, OP_LOGE(context_->GetNodeName(), "Get aivNum failed."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(
        ubSize_ <= UB_SYS_RESERVED_SIZE,
        OP_LOGE(context_->GetNodeName(), "Get ubSize failed: %lu <= sys reserved %lu.", ubSize_, UB_SYS_RESERVED_SIZE),
        return ge::GRAPH_FAILED);
    // 扣减系统预留，得 vector core 实际可用 UB（与 kernel 侧真实口径对齐，防多核 sort footprint 溢出）。
    ubSize_ -= UB_SYS_RESERVED_SIZE;

    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::CheckInputParam()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType xdtype = inputDesc->GetDataType();
    OP_CHECK_IF(xdtype != ge::DT_INT8,
                OP_LOGE(context_->GetNodeName(), "Input dtype:%s not int8", Ops::Base::ToString(xdtype).c_str()),
                return ge::GRAPH_FAILED);

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "x shape %s dim num not 1", Ops::Base::ToString(xShape).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr);
    OP_CHECK_IF(
        *expertNumPtr > EXPERT_IDX_MAX || *expertNumPtr <= 0,
        OP_LOGE(context_->GetNodeName(), "expert_num:%ld should be in range (0, %ld]", *expertNumPtr, EXPERT_IDX_MAX),
        return ge::GRAPH_FAILED);
    expertNum_ = *expertNumPtr;

    const gert::ContinuousVector *maxOutShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxOutShapePtr);
    OP_CHECK_IF(maxOutShapePtr->GetSize() != static_cast<size_t>(NUM_FOUR),
                OP_LOGE(context_->GetNodeName(), "The max_out_shape size:%lu not equal 4.", maxOutShapePtr->GetSize()),
                return ge::GRAPH_FAILED);
    const int64_t *maxOutShapeArray = reinterpret_cast<const int64_t *>(maxOutShapePtr->GetData());
    A_ = maxOutShapeArray[INDEX_ZERO];
    BS_ = maxOutShapeArray[INDEX_ONE];
    K_ = maxOutShapeArray[INDEX_TWO];
    H_ = maxOutShapeArray[INDEX_THREE];
    OP_CHECK_IF(
        (A_ > MAX_SESSION_NUM || A_ <= 0),
        OP_LOGE(context_->GetNodeName(), "max_out_shape[0]:%ld should be in range of (0, %ld]", A_, MAX_SESSION_NUM),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(BS_ <= 0, OP_LOGE(context_->GetNodeName(), "max_out_shape[1]:%ld should be greater than 0", BS_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((K_ > MAX_K_NUM || K_ <= 0),
                OP_LOGE(context_->GetNodeName(), "max_out_shape[2]:%ld should be in range of (0, %ld]", K_, MAX_K_NUM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H_ <= 0, OP_LOGE(context_->GetNodeName(), "max_out_shape[3]:%ld should be greater than 0", H_),
                return ge::GRAPH_FAILED);

    Y_ = A_ * BS_ * K_;

    const int64_t *tokenDtype = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    if (tokenDtype != nullptr) {
        OP_CHECK_IF((*tokenDtype < 0 || *tokenDtype > NUM_TWO),
                    OP_LOGE(context_->GetNodeName(), "token_dtype:%ld must be one of [0, 1, 2]", *tokenDtype),
                    return ge::GRAPH_FAILED);
        tokenDtype_ = *tokenDtype;
    }

    const int64_t *needSchedulePtr = attrs->GetAttrPointer<int64_t>(NEED_SCHEDULE_ATTR);
    if (needSchedulePtr != nullptr) {
        OP_CHECK_IF((*needSchedulePtr < 0 || *needSchedulePtr > 1),
                    OP_LOGE(context_->GetNodeName(), "need_schedule:%ld must be one of [0, 1]", *needSchedulePtr),
                    return ge::GRAPH_FAILED);
        needSchedule_ = *needSchedulePtr;
    }

    const int64_t *layNumPtr = attrs->GetAttrPointer<int64_t>(LAY_NUM_ATTR);
    if (layNumPtr != nullptr) {
        OP_CHECK_IF(
            (*layNumPtr < 0 || *layNumPtr > *expertNumPtr),
            OP_LOGE(context_->GetNodeName(), "layer_num:%ld must be in range of [0, %ld]", *layNumPtr, *expertNumPtr),
            return ge::GRAPH_FAILED);
        layerNum_ = *layNumPtr;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckInputParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckInputParam failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetAttrsInfo failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::DoOpTiling()
{
    tilingDataPtr_ = context_->GetTilingData<FfnWorkerBatchingArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataPtr_);

    coreNum_ = aivNum_;
    // RECV 限核：A2 用 flat-32（Y<=200000 全覆盖），在 A5（满核 64 + 4T 带宽）下会把带宽 bound 的 gather 腰斩。
    // A5 策略「只增不减」：Y 大到每核可分满 TH_RECV_MIN_ROWS_PER_CORE 行（gather 占比高、带宽 bound）时放开满核；
    // Y 较小时 gather 非瓶颈，保留 A2 的 32 核，避免多核 SyncAll 空耗、严格不劣于原实现。
    // NORM 主线 needSchedule_=0 不触发。A2 路径（monolithic tiling）逐字不动，零回归。
    if (needSchedule_ == 1 && Y_ < aivNum_ * TH_RECV_MIN_ROWS_PER_CORE) {
        coreNum_ = std::min(aivNum_, TH_RECV_CORE_NUM);
    }

    // UB 派生切分阈值：ubSize 为运行时值（arch35 自动增大），禁写死 arch 常量。
    int64_t sortLoopMaxElement = static_cast<int64_t>(ubSize_) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) /
                                 ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;

    tilingDataPtr_->Y = Y_;
    tilingDataPtr_->H = H_;
    tilingDataPtr_->tokenDtype = tokenDtype_;
    tilingDataPtr_->expertNum = expertNum_;
    tilingDataPtr_->coreNum = coreNum_;
    tilingDataPtr_->ubSize = static_cast<int64_t>(ubSize_);
    tilingDataPtr_->sortLoopMaxElement = sortLoopMaxElement;
    tilingDataPtr_->sortNumWorkSpace = Y_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FfnWorkerBatchingTilingArch35::GetTilingKey() const
{
    // TilingKey 只由 need_schedule 决定（token_dtype 正交，kernel 内处理）。
    return needSchedule_ == 0 ? static_cast<uint64_t>(TILING_KEY_NORM) : static_cast<uint64_t>(TILING_KEY_RECV);
}

ge::graphStatus FfnWorkerBatchingTilingArch35::GetWorkspaceSize()
{
    // 与 A2 同公式；sysWorkspaceSize 经 GetLibApiWorkSpaceSize() 取 arch35 平台值。
    workspaceSize_ = MAX_RESERVE_WK_NUM * sizeof(int32_t) + Y_ * sizeof(int32_t) +
                     Y_ * sizeof(int32_t) * NUM_TWO * NUM_FOUR + expertNum_ * sizeof(int32_t) + sysWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTilingArch35::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    context_->SetScheduleMode(BATCH_MODE);

    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);

    OP_LOGI(context_->GetNodeName(),
            "arch35 tiling: coreNum:%ld ubSize:%ld Y:%ld H:%ld tokenDtype:%ld expertNum:%ld "
            "sortLoopMaxElement:%ld sortNumWorkSpace:%ld tilingKey:%lu",
            tilingDataPtr_->coreNum, tilingDataPtr_->ubSize, tilingDataPtr_->Y, tilingDataPtr_->H,
            tilingDataPtr_->tokenDtype, tilingDataPtr_->expertNum, tilingDataPtr_->sortLoopMaxElement,
            tilingDataPtr_->sortNumWorkSpace, GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, FfnWorkerBatchingTilingArch35, 1000);
} // namespace optiling
