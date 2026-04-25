/**
 * Bench stub: provides Mc2Hcom::MC2HcomTopology static APIs for infer-shape without
 * loading libhccl / ge::HcomTopoInfo (single-process validation).
 */

#include "common/mc2_hcom_topo_info.h"
#include "common/mc2_log.h"

#include <hccl/hccl_types.h>

namespace Mc2Hcom {

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char *group, uint32_t *rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        OP_LOGE("", "Group or rank num is nullptr.");
        return HCCL_E_PTR;
    }
    *rankNum = 8U;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char *group, uint32_t *topoType)
{
    if (group == nullptr || topoType == nullptr) {
        OP_LOGE("", "Group or topo type is nullptr.");
        return HCCL_E_PTR;
    }
    *topoType = COMM_MESH;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm)
{
    if (group == nullptr || cclBufferSize == nullptr || hcclComm == nullptr) {
        OP_LOGE("", "Group or rank num or hcclComm is nullptr.");
        return HCCL_E_PTR;
    }
    *cclBufferSize = 0;
    *hcclComm = nullptr;
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize)
{
    if (group == nullptr || cclBufferSize == nullptr) {
        return HCCL_E_PTR;
    }
    *cclBufferSize = (1ULL << 20);
    return HCCL_SUCCESS;
}

}  // namespace Mc2Hcom
