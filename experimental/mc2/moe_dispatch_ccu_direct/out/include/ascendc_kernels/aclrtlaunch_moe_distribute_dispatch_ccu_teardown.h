#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_moe_distribute_dispatch_ccu_teardown(uint32_t numBlocks, aclrtStream stream, void* x, void* y, void* expertIds, void* commCmdInfo, void* expandXOut, void* dynamicScalesOut, void* assistInfoForCombineOut, void* expertTokenNumsOut, void* workspaceGM, void* tilingGM);
#endif
