#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_OXC_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_OXC_H

#include "moe_distribute_dispatch_v2_full_mesh.h"

template <TemplateMC2TypeFullmeshClass>
class MoeDistributeDispatchV2FullMeshOxc
    : public MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc> {
public:
    __aicore__ MoeDistributeDispatchV2FullMeshOxc() = default;
};

#endif
