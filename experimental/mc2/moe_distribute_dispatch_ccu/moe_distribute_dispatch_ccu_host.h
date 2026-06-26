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
 * \file moe_distribute_dispatch_ccu_host.h
 * \brief dispatch-ccu host 直调编排入口声明
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_HOST_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_HOST_H

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "dispatch_ccu_common.h"

/**
 * @brief dispatch-ccu 直调入口：建 CCU 通信资源 + 构造 remote_va_table + 注册并启动 CCU setup
 *        kernel + 双流编排 + 启动 AICore teardown 核函数。无 torch / 无 PTA。
 *
 * @param comm        已初始化的 HCCL 通信域（调用方经 HcclCommInitClusterInfoConfig 创建）
 * @param x           [in]  本卡 token，shape [tokenNum, hidden]
 * @param expertIds   [in]  本卡 token 的 topk expert 路由，shape [tokenNum, topk]
 * @param expandXOut  [out] 本卡收到并整理后的 token
 * @param param       规模 / 拓扑参数
 * @param stream1     AICore 主算路 stream（teardown 在此流；CCU setup 走内部 stream2）
 * @return HcclResult 执行结果状态码
 */
HcclResult MoeDistributeDispatchCcu(HcclComm comm, void* x, void* expertIds, void* expandXOut,
                                    const DispatchCcuParam& param, aclrtStream stream1);

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_HOST_H
