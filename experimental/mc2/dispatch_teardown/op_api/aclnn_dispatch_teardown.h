/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_dispatch_teardown.h —— DispatchTeardown 的对外 aclnn 两段式接口。
 *  入参对应 op 定义：status(int32 输入) / done(int32 输出)。
 */
#ifndef OP_API_INC_DISPATCH_TEARDOWN_H
#define OP_API_INC_DISPATCH_TEARDOWN_H

#include "aclnn/aclnn_base.h"
// ACLNN_API 在本 CANN 版本的公共 aclnn 头里未定义（只在 ops-transformer 内部 aclnn_util.h 有，
// 对外部调用方/驱动不可达）。为让本对外头在【注册构建】与【驱动 standalone 构建】两上下文都自洽，
// 未定义时给出标准导出可见性定义；上游已定义则沿用（#ifndef 守卫，不与 aclnn_util.h 冲突）。
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDispatchTeardown 第一段：计算 workspace 大小、构建 executor。
 * @param [in]  status        : 本卡状态区（CCU dispatch_setup 写 1 的同一块 GM），int32。
 * @param [out] done          : 轮询到就绪后写 1 的完成标志，int32。
 * @param [out] workspaceSize : 返回 device workspace 大小。
 * @param [out] executor      : 返回算子执行器。
 */
ACLNN_API aclnnStatus aclnnDispatchTeardownGetWorkspaceSize(const aclTensor *status, aclTensor *done,
                                                            uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnDispatchTeardown 第二段：在 stream 上执行（AIV 忙等 status==1 → 写 done=1）。
 */
ACLNN_API aclnnStatus aclnnDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_TEARDOWN_H
