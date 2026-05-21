
#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_moe_distribute_dispatch_ccu_setup(uint32_t numBlocks, void* stream, void* x, void* expertIds, void* scales, void* xActiveMask, void* YOut, void* expandIdxOut, void* commCmdInfoOut, void* workspaceGM, void* tilingGM);

inline uint32_t moe_distribute_dispatch_ccu_setup(uint32_t numBlocks, void* hold, void* stream, void* x, void* expertIds, void* scales, void* xActiveMask, void* YOut, void* expandIdxOut, void* commCmdInfoOut, void* workspaceGM, void* tilingGM)
{
    (void)hold;
    return aclrtlaunch_moe_distribute_dispatch_ccu_setup(numBlocks, stream, x, expertIds, scales, xActiveMask, YOut, expandIdxOut, commCmdInfoOut, workspaceGM, tilingGM);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_moe_distribute_dispatch_ccu_teardown(uint32_t numBlocks, void* stream, void* x, void* y, void* expertIds, void* commCmdInfo, void* expandXOut, void* dynamicScalesOut, void* assistInfoForCombineOut, void* expertTokenNumsOut, void* workspaceGM, void* tilingGM);

inline uint32_t moe_distribute_dispatch_ccu_teardown(uint32_t numBlocks, void* hold, void* stream, void* x, void* y, void* expertIds, void* commCmdInfo, void* expandXOut, void* dynamicScalesOut, void* assistInfoForCombineOut, void* expertTokenNumsOut, void* workspaceGM, void* tilingGM)
{
    (void)hold;
    return aclrtlaunch_moe_distribute_dispatch_ccu_teardown(numBlocks, stream, x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, workspaceGM, tilingGM);
}

#endif
