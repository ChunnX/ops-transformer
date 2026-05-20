#ifndef MC2_EP_KERNEL_DEPS_H
#define MC2_EP_KERNEL_DEPS_H

// Same-directory includes (build: src/common/, source: mc2/common/op_kernel/)
#if __has_include("comm_args.h") && __has_include("moe_distribute_base.h")
#include "comm_args.h"
#include "data_copy.h"
#include "moe_distribute_base.h"
#include "mc2_kernel_utils.h"
#include "mc2_moe_context.h"
#elif __has_include("../common/comm_args.h")
#include "../common/comm_args.h"
#include "../common/data_copy.h"
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/comm_args.h"
#include "../../common/op_kernel/data_copy.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

#endif // MC2_EP_KERNEL_DEPS_H
