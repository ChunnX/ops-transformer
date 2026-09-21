/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FFN_WORKER_BATCHING_V2_H_
#define OP_API_INC_FFN_WORKER_BATCHING_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取FfnWorkerBatching的workspace大小及执行器。
 * syncFlag为bool类型：false等待全部session，true消费ready快照；仅Ascend 950接收模式生效。
 *
 * scheduleContext虽然以const aclTensor*传入，const限定的是描述对象，不承诺设备载荷只读。
 * RECV会推进其polling_index并清理所选描述符；调用方不能把同一调度上下文无序并发使用。
 * 其中保存的token/descriptor设备指针必须保持有效，且ready发布前payload写入须可见。
 * flag清零只表示描述符已被接收，并不表示gather已搬完token；槽位复用须遵守完整消费协议。
 *
 * maxOutShape=[A,BS,K,H]用于输出容量，实际有效行数由actualTokenNum返回；只消费有效前缀。
 * tokenDtype=3/4/5表示已有MXFP8/MXFP4数据，dynamicScale使用E8M0，并不请求本算子量化。
 * FP4的aclTensor形状以逻辑H计，而实际每行token载荷为H/2字节；不得用字节数替代逻辑维度。
 * syncFlag=true且needSchedule=1时要求layerNum>0、expertNum能被layerNum整除，以构造跨层专家号。
 *
 * 此V2符号在layerNum后显式追加bool属性，旧V1符号保留原签名并固定同步接收语义。
 * 两段调用之间须保持输入输出及workspace有效；GetWorkspaceSize不代表已消费ready数据。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2GetWorkspaceSize(
    const aclTensor *scheduleContext, int64_t expertNum, const aclIntArray *maxOutShape, int64_t tokenDtype,
    int64_t needSchedule, int64_t layerNum, bool syncFlag, const aclTensor *y, const aclTensor *groupList,
    const aclTensor *sessionIds, const aclTensor *microBatchIds, const aclTensor *tokenIds,
    const aclTensor *expertOffsets, const aclTensor *dynamicScale, const aclTensor *actualTokenNum,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/** @brief 执行第一段接口构造的任务。 */
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
