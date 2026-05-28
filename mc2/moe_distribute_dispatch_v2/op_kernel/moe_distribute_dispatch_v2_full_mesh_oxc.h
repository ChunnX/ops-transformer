#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_OXC_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_OXC_H

#include "moe_distribute_dispatch_v2_full_mesh.h"

template <TemplateMC2TypeFullmeshClass>
class MoeDistributeDispatchV2FullMeshOxc
    : public MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc> {
public:
    __aicore__ MoeDistributeDispatchV2FullMeshOxc() = default;

    __aicore__ void Init(GM_ADDR mc2Context, GM_ADDR x,
                         GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                         GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
                         GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                         GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                         GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM,
                         TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData) {
        this->Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
                   performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
                   expertTokenNumsOut, sendCountsOut, tpSendCountsOut,
                   workspaceGM, pipe, tilingData);
    }
};

#endif
