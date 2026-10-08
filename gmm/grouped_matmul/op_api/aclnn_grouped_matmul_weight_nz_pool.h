/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_POOL_H
#define OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_POOL_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulWeightNzPool的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 本接口在aclnnGroupedMatmulWeightNz的基础上，把weight/antiquantScale的语义从「每个专家一份权重」改为
 * 「所有专家共用的一块 pool」，并新增两个参数描述 pool：
 *   - slotTableOptional: [slotCount] int64，第 g 个 group 使用 pool 的第 slotTable[g] 行；
 *   - poolMetaOptional:  attr，[rowBytes, wSegOff, sSegOff, n]，描述 pool 的行布局与输出 N。
 * 未传 poolMetaOptional（或长度不足4）时退化为与aclnnGroupedMatmulWeightNz完全一致的直址寻址。
 * 此时 weight/antiquantScale 必须按 aclnnGroupedMatmulWeightNz 的语义（每专家一份、NZ格式）传入。
 *
 * @param [in] x: 表示公式中的输入x，数据类型支持FLOAT16、BFLOAT16、INT8、INT4、INT32、FLOAT8_E4M3FN、
 * FLOAT4_E2M1、FLOAT4_E1M2，数据格式支持ND。
 * 部分场景支持的最大长度为1024个（详见接口文档约束说明），其他场景支持的最大长度为128个。
 * INT32为打包输入，每个INT32承载8个INT4数据。
 * @param [in]
 * weight：表示公式中的weight，数据类型支持FLOAT16、BFLOAT16、INT8、INT4、FLOAT4_E2M1、INT32、FLOAT32、
 * FLOAT8_E4M3FN、FLOAT4_E1M2，数据格式支持NZ。
 * pool模式下 weight 为 [slotCount, rowBytes] 的整块 pool（物理上仍是 NZ 排布），
 * 此时输出N由 poolMetaOptional[3] 给出，而不是由 weight 的末维给出。
 * 部分场景支持的最大长度为1024个（详见接口文档约束说明），其他场景支持的最大长度为128个。
 * @param [in]
 * biasOptional：表示公式中的bias，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT32，数据格式支持ND。
 * 部分场景支持的最大长度为1024个（详见接口文档约束说明），其他场景支持的最大长度为128个。
 * @param [in]
 * scaleOptional：代表量化参数中的缩放因子，数据类型支持BFLOAT16、FLOAT32、INT64、UINT64、FLOAT8_E8M0，数据格式支持ND和FRACTAL_NZ，长度与weight相同。
 * @param [in] offsetOptional：代表量化参数中的偏移量，数据类型支持FLOAT32，数据格式支持ND，长度与weight相同。
 * @param [in]
 * antiquantScaleOptional：代表伪量化参数中的缩放因子，数据类型支持FLOAT16、BFLOAT16、FLOAT8_E8M0，数据格式支持ND，长度与weight相同。
 * pool模式下同样为 [slotCount, rowBytes] 的整块 pool，scale 段在行内的偏移由 poolMetaOptional[2] 给出。
 * @param [in]
 * antiquantOffsetOptional：代表伪量化参数中的偏移量，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，长度与weight相同。
 * @param [in]
 * perTokenScaleOptional：代表量化参数中的由输入x量化引入的缩放因子，数据类型支持FLOAT32、FLOAT8_E8M0，数据格式支持ND，长度与x相同
 * @param [in]
 * groupListOptional：代表输入和输出分组轴的matmul大小分布，数据类型支持INT64，数据格式支持ND，长度与weight相同。
 * @param [in] slotTableOptional：
 * pool模式下代表每个group在pool中的行号，数据类型支持INT64，数据格式支持ND，长度为group个数。
 * 不传时按直址寻址处理（与aclnnGroupedMatmulWeightNz一致）。本参数为普通入参，允许随每次迭代变化。
 * @param [in] tuningConfigOptional：
 * 调优参数。数组中第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值进行最优tiling。
 * @param [in] poolMetaOptional：
 * pool行内布局与输出N，格式为[rowBytes, wSegOff, sSegOff, n]，其中rowBytes为pool一行的字节数（即相邻slot的间距），
 * wSegOff/sSegOff分别为weight段与antiquantScale段在一行内的字节偏移，n为输出N。
 * 该参数为编译期常量，注册为算子attr，tiling与InferShape都依赖它；不传或长度不足4时走直址寻址。
 * @param [in] quantGroupSize: 整数型参数，代表分组量化（per-group）的分组大小，不涉及分组量化时，填0。
 * @param [out] out:
 * 表示公式中的out，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT32数据类型，数据格式支持ND。
 * 部分场景支持的最大长度为1024个（详见接口文档约束说明），其他场景支持的最大长度为128个。
 * @param [out] activationFeatureOutOptional: 激活函数的输入数据。
 * @param [out] dynQuantScaleOutOptional: 存在激活函数的时候，对激活后的输出进行动态量化的量化系数输出。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulWeightNzPoolGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional, const aclTensor *slotTableOptional,
    const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
    int64_t actType, aclIntArray *tuningConfigOptional, aclIntArray *poolMetaOptional, int64_t quantGroupSize,
    aclTensorList *out, aclTensorList *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulWeightNzPool的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulWeightNzPoolGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulWeightNzPool(void *workspace,
                                                                                  uint64_t workspaceSize,
                                                                                  aclOpExecutor *executor,
                                                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
