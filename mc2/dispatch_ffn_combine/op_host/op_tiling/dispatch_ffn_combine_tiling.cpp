/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_tiling.cpp
 * \brief
 */

#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "cann_version.h"

#if CANN_VERSION_NUM >= 90000000
#include "mc2_exception_dump.h"
using namespace Mc2Exception;
#endif

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;


namespace optiling {
static ge::graphStatus DispatchFFNCombineTilingFunc(gert::TilingContext* context)
{
    // Input indices: context=0, x=1, expert_ids=2, expert_scales=3, weight1=4(dynamic), weight2=5(dynamic),
    //                scales=6(optional), x_active_mask=7(optional), weight_scales1=8(dynamic), weight_scales2=9(dynamic)
    // Attr indices: ep_world_size=0, ep_rank_id=1, moe_expert_num=2, ccl_buffer_size=3,
    //               max_recv_token_num=4, shared_expert_num=5, dispatch_quant_mode=6,
    //               dispatch_quant_out_type=7, combine_quant_mode=8, comm_alg=9, global_bs=10

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter DispatchFFNCombine tiling");

    // TODO: Implement tiling logic

    return ge::GRAPH_SUCCESS;
}

struct DispatchFFNCombineCompileInfo {};
static ge::graphStatus TilingParseForDispatchFFNCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombine)
    .Tiling(DispatchFFNCombineTilingFunc)
    .TilingParse<DispatchFFNCombineCompileInfo>(TilingParseForDispatchFFNCombine);

#if CANN_VERSION_NUM >= 90000000
inline void DispatchFFNCombineExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2ExceptionImpl(args, userdata, "DispatchFFNCombine");
}

IMPL_OP(DispatchFFNCombine)
    .ExceptionDumpParseFunc(DispatchFFNCombineExceptionImplWrapper);
#endif
} // namespace optiling
