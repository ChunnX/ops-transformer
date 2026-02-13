#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_WITH_ARGMAX_H

#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"
#include "max_pool3d_grad_tiling_constants.h"

namespace optiling {
using namespace OpTiling;
const uint32_t DHW_DIMS = 3;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;

BEGIN_TILING_DATA_DEF(MaxPool3DGradWithArgmaxTilingData)
TILING_DATA_FIELD_DEF(uint64_t, ncDim);
TILING_DATA_FIELD_DEF(uint64_t, diDim);
TILING_DATA_FIELD_DEF(uint64_t, hiDim);
TILING_DATA_FIELD_DEF(uint64_t, wiDim);
TILING_DATA_FIELD_DEF(uint64_t, doDim);
TILING_DATA_FIELD_DEF(uint64_t, hoDim);
TILING_DATA_FIELD_DEF(uint64_t, woDim);
TILING_DATA_FIELD_DEF(uint64_t, kd);
TILING_DATA_FIELD_DEF(uint64_t, kh);
TILING_DATA_FIELD_DEF(uint64_t, kw);
TILING_DATA_FIELD_DEF(uint64_t, sd);
TILING_DATA_FIELD_DEF(uint64_t, sh);
TILING_DATA_FIELD_DEF(uint64_t, sw);
TILING_DATA_FIELD_DEF(uint64_t, padDTop);
TILING_DATA_FIELD_DEF(uint64_t, padHTop);
TILING_DATA_FIELD_DEF(uint64_t, padWTop);
TILING_DATA_FIELD_DEF(uint64_t, padDBottom);
TILING_DATA_FIELD_DEF(uint64_t, padHBottom);
TILING_DATA_FIELD_DEF(uint64_t, padWBottom);
TILING_DATA_FIELD_DEF(uint64_t, baseNc);
TILING_DATA_FIELD_DEF(uint64_t, baseDo);
TILING_DATA_FIELD_DEF(uint64_t, baseHo);
TILING_DATA_FIELD_DEF(uint64_t, baseWo);
TILING_DATA_FIELD_DEF(uint64_t, ncTail);
TILING_DATA_FIELD_DEF(uint64_t, doTail);
TILING_DATA_FIELD_DEF(uint64_t, hoTail);
TILING_DATA_FIELD_DEF(uint64_t, woTail);
TILING_DATA_FIELD_DEF(uint64_t, ncCnt);
TILING_DATA_FIELD_DEF(uint64_t, doCnt);
TILING_DATA_FIELD_DEF(uint64_t, hoCnt);
TILING_DATA_FIELD_DEF(uint64_t, woCnt);
TILING_DATA_FIELD_DEF(uint64_t, totalCnt);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalUBSize);

TILING_DATA_FIELD_DEF(uint64_t, singleCoreNc);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);

TILING_DATA_FIELD_DEF(uint64_t, ncRound);
TILING_DATA_FIELD_DEF(uint64_t, ncRoundTail);
TILING_DATA_FIELD_DEF(uint64_t, totalRound);
TILING_DATA_FIELD_DEF(uint64_t, preCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxTilingData);

constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t NUM_TWO = 2;
constexpr size_t DHW_DIM_NUM = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

static const std::map<ge::DataType, uint64_t> DTYPE_MAP = {
    {ge::DT_FLOAT, 0},
    {ge::DT_FLOAT16, 1},
    {ge::DT_BF16, 2}
};

struct Tiling4MaxPool3DGradWithArgmaxCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct MaxPoolGradWithArgmaxTilingParams {
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    uint64_t xDtypeSize{0};
    uint64_t indexDtypeSize{0};
    uint64_t ncDim{0};
    uint64_t diDim{0};
    uint64_t hiDim{0};
    uint64_t wiDim{0};
    uint64_t doDim{0};
    uint64_t hoDim{0};
    uint64_t woDim{0};
    uint64_t kd{0};
    uint64_t kh{0};
    uint64_t kw{0};
    uint64_t sd{0};
    uint64_t sh{0};
    uint64_t sw{0};
    uint64_t vl{0};
    uint64_t pDTop{0};
    uint64_t pHTop{0};
    uint64_t pWTop{0};
    uint64_t pDBottom{0};
    uint64_t pHBottom{0};
    uint64_t pWBottom{0};
    uint64_t dilationD{0};
    uint64_t dilationH{0};
    uint64_t dilationW{0};
    uint64_t baseNc{0};
    uint64_t baseDo{0};
    uint64_t baseHo{0};
    uint64_t baseWo{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t ncCnt{0};
    uint64_t doCnt{0};
    uint64_t hoCnt{0};
    uint64_t woCnt{0};
    uint64_t totalCnt{0};
    uint64_t usedCoreNum{0};

    uint64_t singleCoreNc{0};
    uint64_t singleCoreDo{0};
    uint64_t singleCoreHo{0};
    uint64_t singleCoreWo{0};

    uint64_t ncRound{0};
    uint64_t ncRoundTail{0};
    uint64_t totalRound{0};
    uint64_t preCoreNum{0};

    uint64_t workspaceSize{0};
    uint64_t tilingType{0};
    uint32_t ubCutAxis{0};
    bool ceilMode{false};
    bool isOverLap{false};
    bool isInitOutput{false};
    bool isNeedWorkspace{false};
};

class MaxPool3DGradWithArgmaxTilingBase : public TilingBaseClass {
public:
    explicit MaxPool3DGradWithArgmaxTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~MaxPool3DGradWithArgmaxTilingBase() override
    {}

    const std::string nodeName = "MaxPool3DGradWithArgmax";
    MaxPool3DGradWithArgmaxTilingData tilingData;
    MaxPoolGradWithArgmaxTilingParams maxPoolGradParams;

    bool CheckInputShape();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckAttrShape();
    ge::graphStatus CheckInputValid();
    ge::graphStatus SetInputParams();
    ge::graphStatus SetAttrParams();
    void SetCntTailTilingParams();
    void SetOtherInputParams();
    void SetBaseTilingData();
    void PrintTilingData();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class MaxPool3DGradWithArgmaxScatterTiling : public MaxPool3DGradWithArgmaxTilingBase {
public:
    explicit MaxPool3DGradWithArgmaxScatterTiling(gert::TilingContext* context)
        : MaxPool3DGradWithArgmaxTilingBase(context)
    {}
    ~MaxPool3DGradWithArgmaxScatterTiling() override
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

private:
    bool SetScatterTilingParams();
    void SetOtherTilingParams();
    void SetScatterTilingData();
    void PrintScatterTilingData();
};
} // namespace optiling

#endif