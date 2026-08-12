/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H
#define MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H

#include "../../../common/op_kernel/mc2_exception_dump.h"
#include "common/mega_moe_workspace.h"
#include "mega_moe_tiling.h"

namespace MC2MegaMoeA5Dump {

enum class Stage : uint32_t {
    INIT = 0,
    INPUT_PREPARE = 1,
    SHARED_EXPERT_GMM1 = 2,
    CROSS_RANK_SYNC_INPUT = 3,
    MOE_GMM1_PIPELINE = 4,
    MOE_GMM2_COMBINE = 5,
    SHARED_EXPERT_GMM2 = 6,
    CROSS_RANK_SYNC_OUTPUT = 7,
    UNPERMUTE = 8,
    END = 9
};

struct MegaMoeExceptionDumpPolicy {
    using TilingDataT = MegaMoeTilingData;
    using StageEnumT = Stage;
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_MEGA_MOE;
};

using ExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<MegaMoeExceptionDumpPolicy>;

static_assert(MC2ExceptionDump::DUMP_HEADER_REGION_SIZE + MC2ExceptionDump::DUMP_TILING_REGION_SIZE +
                      MC2ExceptionDump::DUMP_BLOCK_STAGE_REGION_SIZE +
                      MC2ExceptionDump::MAX_DUMP_ENTRIES * sizeof(MC2ExceptionDump::DumpParams) <=
                  static_cast<size_t>(MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE),
              "MegaMoe arch35 exception-dump metadata exceeds its reserved communication-window prefix");

} // namespace MC2MegaMoeA5Dump

#endif // MEGA_MOE_ARCH35_EXCEPTION_DUMP_POLICY_H
