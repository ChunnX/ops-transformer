#ifndef ENGRAM_FETCH_TILING_BASE_H_
#define ENGRAM_FETCH_TILING_BASE_H_

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../op_kernel/engram_fetch_tiling_data.h"

namespace optiling {
class EngramFetchTilingBase : public MoeTilingBase {
public:
    explicit EngramFetchTilingBase(gert::TilingContext *context)
        : MoeTilingBase(context), nodeName_(context->GetNodeName()) {};

protected:
    const char *socTilingName_;
    std::string nodeName_;
    std::string group_;
    EngramFetchTilingData *tilingData_ = nullptr;

    uint64_t GetTilingKey() const;

    ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetAttrAndSetTilingData();
    ge::graphStatus CheckInputTensorShape();
    ge::graphStatus CheckInputTensorDataType();
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData();

    void SetTilingKey();
    void SetPlatformInfo();
    ge::graphStatus SetWorkspace();

    virtual void SetHcommCfg();
    ge::graphStatus EngramFetchTilingFuncImpl();
};
} // namespace optiling
#endif