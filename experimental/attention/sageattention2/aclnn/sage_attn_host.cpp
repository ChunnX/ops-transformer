#include "sage_attn_proto.h"
// op_api 层 TU：允许引用 libnnopbase 平台查询符号（动态核数探测）
#define SAGE_ATTN_HAVE_PLATFORM_INFO 1
#include "sage_attn_host_tiling.h"

#include <memory>

#ifdef SAGE_HAS_CANN_ACLNN_BASE
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#endif

namespace sage_attn {

/**
 * 宿主机轻量执行器上下文定义
 * （真实 CANN 运行时下执行器由 CREATE_EXECUTOR 创建，本结构仅服务本地桩路径；
 *   tiling 位元在 CANN 下由 op_host TilingImpl 序列化下发，本地桩只需保留
 *   workspace 校验所需的字节数，故仅存 workspace_size）
 */
struct SageOpExecutorContext {
    uint64_t workspace_size{0};
};

#ifndef SAGE_HAS_CANN_ACLNN_BASE
/**
 * 本地环境下的具体 aclOpExecutor 实现类
 */
struct ConcreteSageOpExecutor : public ::aclOpExecutor {
    SageOpExecutorContext ctx;

    uint64_t GetWorkspaceSize() const override {
        return ctx.workspace_size;
    }
};
#endif

/**
 * 张量维度辅助提取器
 * 支持从 CANN 原生 aclTensor 或本地轻量 mock 中统一提取形状
 */
inline bool ExtractTensorDimensions(
    const aclTensor *tensor,
    int64_t &dim0,
    int64_t &dim1,
    int64_t &dim2,
    int64_t &dim3)
{
    if (tensor == nullptr) {
        return false;
    }
#ifdef SAGE_HAS_CANN_ACLNN_BASE
    const op::Shape &viewShape = tensor->GetViewShape();
    if (viewShape.GetDimNum() != 4) {
        return false;
    }
    dim0 = viewShape[0];
    dim1 = viewShape[1];
    dim2 = viewShape[2];
    dim3 = viewShape[3];
    return true;
#else
    if (tensor->GetDimNum() != 4) {
        return false;
    }
    dim0 = tensor->GetShape(0);
    dim1 = tensor->GetShape(1);
    dim2 = tensor->GetShape(2);
    dim3 = tensor->GetShape(3);
    return true;
#endif
}


} // namespace sage_attn

// =============================================================================
// ACLNN 官方两段式 C 接口实现
// =============================================================================

extern "C" {

__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *keyMeanOptional,
    const aclTensor *lseCorrectionOptional,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 1. 空指针入参校验
    if (query == nullptr || key == nullptr || value == nullptr || out == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 2. 提取张量形状
    // lse_correction 修正随算子内融合（smooth_k 路径），独立张量入口暂不参与
    // Host Tiling 计算，显式抑制未用告警以保持 C 接口签名与 proto.h 声明一致
    (void)lseCorrectionOptional;

    int64_t q_d0 = 0, q_d1 = 0, q_d2 = 0, q_d3 = 0;
    int64_t k_d0 = 0, k_d1 = 0, k_d2 = 0, k_d3 = 0;
    int64_t v_d0 = 0, v_d1 = 0, v_d2 = 0, v_d3 = 0;

    if (!sage_attn::ExtractTensorDimensions(query, q_d0, q_d1, q_d2, q_d3) ||
        !sage_attn::ExtractTensorDimensions(key, k_d0, k_d1, k_d2, k_d3) ||
        !sage_attn::ExtractTensorDimensions(value, v_d0, v_d1, v_d2, v_d3)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 根据布局解析 B, S, H, D
    uint32_t b = 0, sq = 0, sk = 0, hq = 0, hkv = 0, d_og = 0;
    if (tensorLayout == sage_attn::LAYOUT_HND) {
        // HND 布局: [B, H, S, D]
        b = static_cast<uint32_t>(q_d0);
        hq = static_cast<uint32_t>(q_d1);
        sq = static_cast<uint32_t>(q_d2);
        d_og = static_cast<uint32_t>(q_d3);

        hkv = static_cast<uint32_t>(k_d1);
        sk = static_cast<uint32_t>(k_d2);
    } else {
        // NHD 布局: [B, S, H, D]
        b = static_cast<uint32_t>(q_d0);
        sq = static_cast<uint32_t>(q_d1);
        hq = static_cast<uint32_t>(q_d2);
        d_og = static_cast<uint32_t>(q_d3);

        sk = static_cast<uint32_t>(k_d1);
        hkv = static_cast<uint32_t>(k_d2);
    }

    // 4. 执行微秒级闭式 Tiling 与 Workspace 显存计算
    sage_attn::SageAttention2TilingData tiling_data;
    uint64_t calculated_workspace = 0;

    bool has_km = (keyMeanOptional != nullptr);
    bool has_lse = (lse != nullptr);

    // 移植态 C-ABI 无 dtype 形参：本路径固定 fp16 位型（bf16 走 canonical
    // op_api/op_host 路径，由 aclTensor/gert dtype 分派）
    aclnnStatus status = sage_attn::ComputeHostTiling(
        b, sq, sk, hq, hkv, /*dtype_mode*/ 1, d_og,
        smScale, qkQuantGran, pvAccumDtype,
        isCausal, tensorLayout, smoothK, smoothV,
        has_km, has_lse,
        tiling_data,
        calculated_workspace);

    if (status != ACLNN_SUCCESS) {
        return status;
    }

    *workspaceSize = calculated_workspace;

    // 5. 构造并返回算子执行器上下文
#ifdef SAGE_HAS_CANN_ACLNN_BASE
    // 在 CANN 运行时下，TilingData 经 op_host/sage_attention2_tiling.cpp 注册的
    // TilingImpl (IMPL_OP(SageAttention2).Tiling) 序列化进 gert::RawTilingData，
    // 由框架在 CommonOpExecutorRun 下发阶段写入 kernel；本阶段以同一闭式
    // ComputeHostTiling 计算 Workspace 与一致性校验，保证两处位元一致
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    // Workspace 大小经 *workspaceSize 返回给调用方分配，
    // 执行器内部 workspace 由框架在 CommonOpExecutorRun 阶段按入参绑定
    uniqueExecutor.ReleaseTo(executor);
#else
    auto op_exec = std::make_unique<sage_attn::ConcreteSageOpExecutor>();
    if (op_exec == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    op_exec->ctx.workspace_size = calculated_workspace;

    *executor = static_cast<::aclOpExecutor*>(op_exec.release());
#endif

    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

#ifdef SAGE_HAS_CANN_ACLNN_BASE
    // 在 CANN 运行时下，调用公共执行器调度下发算子
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
#else
    // 宿主机环境或单测桩执行
    (void)workspace;
    (void)workspaceSize;
    (void)stream;

    auto *concrete = static_cast<sage_attn::ConcreteSageOpExecutor *>(executor);
    if (workspaceSize < concrete->ctx.workspace_size) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
#endif
}

} // extern "C"
