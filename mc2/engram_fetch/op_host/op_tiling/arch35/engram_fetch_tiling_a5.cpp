#include "engram_fetch_tiling_a5.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace {

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal;
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};
} // namespace

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(EngramFetch, EngramFetchTilingA5, 0);

void EngramFetchTilingA5::SetHcommCfg()
{
    OP_LOGD(nodeName_, "EngramFetch group = %s", group_.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint8_t aivEngineValue = mc2tiling::AIV_ENGINE;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group_, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(aivEngineValue);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData_->mc2CcTiling)->protocal = 1;
}

bool EngramFetchTilingA5::IsCapable()
{
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_LOGD(nodeName_, "Do EngramFetchTilingA5 tiling.");
        return true;
    }
    return false;
}

ge::graphStatus EngramFetchTilingA5::DoOpTiling()
{
    return EngramFetchTilingFuncImpl();
}
} // namespace optiling