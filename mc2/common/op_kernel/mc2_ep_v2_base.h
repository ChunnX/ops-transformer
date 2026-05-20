#ifndef MC2_EP_V2_BASE_H
#define MC2_EP_V2_BASE_H

#include "mc2_ep_kernel_deps.h"

#if __has_include("moe_distribute_v2_base.h")
#include "moe_distribute_v2_base.h"
#elif __has_include("../common/moe_distribute_v2_base.h")
#include "../common/moe_distribute_v2_base.h"
#else
#include "../../common/op_kernel/moe_distribute_v2_base.h"
#endif

#endif // MC2_EP_V2_BASE_H
