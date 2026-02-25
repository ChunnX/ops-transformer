/*!
 * \file rfft1d_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RFFT1D_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RFFT1D_H_

#pragma once
#include "register/op_impl_registry.h"
#include "common/log.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/shape.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"

namespace optiling {

struct Rfft1DCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling
#endif