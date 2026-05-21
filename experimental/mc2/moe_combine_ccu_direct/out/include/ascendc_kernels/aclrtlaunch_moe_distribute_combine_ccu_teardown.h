#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_moe_distribute_combine_ccu_teardown(uint32_t numBlocks, aclrtStream stream, void* expandX, void* quantExpandX, void* expertIds, void* expandIdx, void* expertScales, void* commCmdInfo, void* xActiveMask, void* sharedExpertX, void* XOut, void* workspaceGM, void* tilingGM);
#endif
