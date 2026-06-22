#ifndef ENGRAM_FETCH_H
#define ENGRAM_FETCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"
#include "engram_fetch_tiling_data.h"

#if __has_include("mc2_moe_context.h")
#include "mc2_moe_context.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#endif
constexpr uint32_t  MAX_QP_SIZE = 1024;
struct EngramBufferHandle {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[MAX_QP_SIZE];
    uint64_t hcommHandle_[MAX_QP_SIZE];
};

namespace EngramFetchImpl {
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t TILE_BYTES = 32U * 1024U;
constexpr uint32_t HCOMM_INIT_SIZE = 512UL;

using namespace AscendC;
using namespace Mc2Aclnn;
} // namespace EngramFetchImpl

#endif