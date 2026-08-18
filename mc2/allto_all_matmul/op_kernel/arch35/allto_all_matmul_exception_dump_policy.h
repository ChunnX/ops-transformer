/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul_exception_dump_policy.h
 * \brief AlltoAllMatmul 算子的 ExceptionDump Policy。
 *        Policy 提供 AlltoAllMatmul 专属类型与编译期常量，注入 ExceptionDump 模板。
 *        通用方法（UpdateStage/Dump/DumpTilingData）由 ExceptionDump 实现。
 */
#ifndef ALLTO_ALL_MATMUL_EXCEPTION_DUMP_POLICY_H
#define ALLTO_ALL_MATMUL_EXCEPTION_DUMP_POLICY_H

#include "../../../common/op_kernel/mc2_exception_dump.h"
#include "allto_all_matmul_tiling_data.h"

namespace MC2AlltoAllAdump {
// AlltoAllMatmul 执行阶段枚举，与 kernel 的流水线流程对应。
// 用于定位算子卡在哪个阶段，StageEnumT::END 需与 ExceptionDump 的 BlockStage 数组大小一致。
enum class Stage : uint32_t {
    INIT = 0,        // 流水线初始化
    COMM_PREPARE = 1, // 通信准备
    COMM_PROCESS = 2, // 通信执行
    TRANSPOSE = 3,    // 转置
    MATMUL = 4,       // 矩阵乘计算
    END = 5           // 哨兵，表示阶段总数，供 ExceptionDump BlockStage 数组使用
};

// 非量化场景 Policy
struct AlltoAllMatmulExceptionDumpPolicy {
    using TilingDataT = AlltoAllMatmulTilingData;
    using StageEnumT = Stage;
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_ALLTO_ALL_MATMUL;
};

// 量化场景 Policy（KC量化/MX量化共用 AlltoAllQuantMatmulTilingData）
struct AlltoAllQuantMatmulExceptionDumpPolicy {
    using TilingDataT = AlltoAllQuantMatmulTilingData;
    using StageEnumT = Stage;
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_ALLTO_ALL_MATMUL;
};

// 引擎实例类型别名（供 kernel 直接使用）
using FpExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<AlltoAllMatmulExceptionDumpPolicy>;
using QuantExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<AlltoAllQuantMatmulExceptionDumpPolicy>;
} // namespace MC2AlltoAllAdump
#endif // ALLTO_ALL_MATMUL_EXCEPTION_DUMP_POLICY_H
