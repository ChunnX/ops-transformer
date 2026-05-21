
#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_SETUP_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_SETUP_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_moe_distribute_combine_ccu_setup(uint32_t numBlocks, void* stream, void* expandX, void* expertIds, void* assistInfoForCombine, void* quantExpandX, void* commCmdInfoOut, void* workspaceGM, void* tilingGM);

inline uint32_t moe_distribute_combine_ccu_setup(uint32_t numBlocks, void* hold, void* stream, void* expandX, void* expertIds, void* assistInfoForCombine, void* quantExpandX, void* commCmdInfoOut, void* workspaceGM, void* tilingGM)
{
    (void)hold;
    return aclrtlaunch_moe_distribute_combine_ccu_setup(numBlocks, stream, expandX, expertIds, assistInfoForCombine, quantExpandX, commCmdInfoOut, workspaceGM, tilingGM);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_moe_distribute_combine_ccu_teardown(uint32_t numBlocks, void* stream, void* expandX, void* quantExpandX, void* expertIds, void* expandIdx, void* expertScales, void* commCmdInfo, void* xActiveMask, void* sharedExpertX, void* XOut, void* workspaceGM, void* tilingGM);

inline uint32_t moe_distribute_combine_ccu_teardown(uint32_t numBlocks, void* hold, void* stream, void* expandX, void* quantExpandX, void* expertIds, void* expandIdx, void* expertScales, void* commCmdInfo, void* xActiveMask, void* sharedExpertX, void* XOut, void* workspaceGM, void* tilingGM)
{
    (void)hold;
    return aclrtlaunch_moe_distribute_combine_ccu_teardown(numBlocks, stream, expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMask, sharedExpertX, XOut, workspaceGM, tilingGM);
}

#endif
