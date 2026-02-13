#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BASE_H_

#include <array>
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"

using namespace std;

namespace optiling {
using namespace OpTiling;
const int DHW_DIMS = 3;
const int DHW_PAD_DIMS = 6;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2TilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, inputShapes);
TILING_DATA_FIELD_DEF_ARR(uint64_t, DHW_DIMS, outShapes);
TILING_DATA_FIELD_DEF(uint64_t, kD);
TILING_DATA_FIELD_DEF(uint64_t, kW);
TILING_DATA_FIELD_DEF(uint64_t, kH);
TILING_DATA_FIELD_DEF(uint64_t, sD);
TILING_DATA_FIELD_DEF(uint64_t, sW);
TILING_DATA_FIELD_DEF(uint64_t, sH);
TILING_DATA_FIELD_DEF(uint64_t, pD);
TILING_DATA_FIELD_DEF(uint64_t, pW);
TILING_DATA_FIELD_DEF(uint64_t, pH);
TILING_DATA_FIELD_DEF(uint64_t, dD);
TILING_DATA_FIELD_DEF(uint64_t, dW);
TILING_DATA_FIELD_DEF(uint64_t, dH);
TILING_DATA_FIELD_DEF(float, minFloat);
TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nc);
TILING_DATA_FIELD_DEF(uint64_t, dx);
TILING_DATA_FIELD_DEF(uint64_t, hx);
TILING_DATA_FIELD_DEF(uint64_t, wx);
TILING_DATA_FIELD_DEF(uint64_t, kd);
TILING_DATA_FIELD_DEF(uint64_t, kh);
TILING_DATA_FIELD_DEF(uint64_t, kw);
TILING_DATA_FIELD_DEF(uint64_t, sd);
TILING_DATA_FIELD_DEF(uint64_t, sh);
TILING_DATA_FIELD_DEF(uint64_t, sw);
TILING_DATA_FIELD_DEF(uint64_t, pf);
TILING_DATA_FIELD_DEF(uint64_t, pb);
TILING_DATA_FIELD_DEF(uint64_t, pl);
TILING_DATA_FIELD_DEF(uint64_t, pr);
TILING_DATA_FIELD_DEF(uint64_t, pt);
TILING_DATA_FIELD_DEF(uint64_t, pd);
TILING_DATA_FIELD_DEF(uint64_t, dy);
TILING_DATA_FIELD_DEF(uint64_t, hy);
TILING_DATA_FIELD_DEF(uint64_t, wy);
TILING_DATA_FIELD_DEF(uint64_t, ncFactor);
TILING_DATA_FIELD_DEF(uint64_t, ncTail);
TILING_DATA_FIELD_DEF(uint64_t, ncOuter);
TILING_DATA_FIELD_DEF(uint64_t, dyFactor);
TILING_DATA_FIELD_DEF(uint64_t, dyTail);
TILING_DATA_FIELD_DEF(uint64_t, dyOuter);
TILING_DATA_FIELD_DEF(uint64_t, hyFactor);
TILING_DATA_FIELD_DEF(uint64_t, hyTail);
TILING_DATA_FIELD_DEF(uint64_t, hyOuter);
TILING_DATA_FIELD_DEF(uint64_t, wyFactor);
TILING_DATA_FIELD_DEF(uint64_t, wyTail);
TILING_DATA_FIELD_DEF(uint64_t, wyOuter);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTail);
TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
TILING_DATA_FIELD_DEF(uint64_t, coreNums);
TILING_DATA_FIELD_DEF(uint64_t, inputBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, outputMaxBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, outputIndiceBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, indiceTempBufferSize);
TILING_DATA_FIELD_DEF(uint64_t, maskBufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2, MaxPool3DWithArgmaxV2TilingData);

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_300001, MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_300002, MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);

struct InputInfo {
    uint64_t batches;
    array<uint64_t, DHW_DIMS> inputShape;
    array<uint64_t, DHW_DIMS> outShape;
    array<uint64_t, DHW_DIMS> kernelSize;
    array<uint64_t, DHW_DIMS> stride;
    array<uint64_t, DHW_DIMS> pad;
    array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
};

struct SplitInfo {
    uint64_t ncFactor = 1;
    uint64_t ncTail = 1;
    uint64_t ncOuter = 1;
    uint64_t dyFactor = 1;
    uint64_t dyTail = 1;
    uint64_t dyOuter = 1;
    uint64_t wyFactor = 1;
    uint64_t wyTail = 1;
    uint64_t wyOuter = 1;
    uint64_t hyFactor = 1;
    uint64_t hyTail = 1;
    uint64_t hyOuter = 1;
    uint64_t kdFactor = 1;
    uint64_t kdTail = 1;
    uint64_t kdOuter = 1;
    uint64_t khFactor = 1;
    uint64_t khTail = 1;
    uint64_t khOuter = 1;
    uint64_t kwFactor = 1;
    uint64_t kwTail = 1;
    uint64_t kwOuter = 1;

    uint64_t blockFactor = 1;
    uint64_t blockTail = 1;
    uint64_t totalIdx = 1;
    uint64_t coreNums = 1;
    array<uint64_t, DHW_DIMS> outShape;
    array<uint64_t, DHW_PAD_DIMS> pad;
};

struct BufferInfo {
    uint64_t inputBufferSize = 1;
    uint64_t outputMaxBufferSize = 1;
    uint64_t outputIndiceBufferSize = 1;
    uint64_t indiceTempBufferSize = 1;
    uint64_t maskBufferSize = 1;

    uint64_t inputCastBytes;
    uint64_t blockDataNum;
    uint64_t ncMaxFactor;
    uint64_t ncFactorAlign;
    uint64_t lbDyFactor;
    uint64_t lbHyFactor;
    uint64_t lbWyFactor;
    uint64_t ubDyFactor;
    uint64_t ubHyFactor;
    uint64_t ubWyFactor;
    uint64_t tmpDyFactor;
    uint64_t tmpHyFactor;
    uint64_t tmpWyFactor;
    uint64_t tmpWyFactorAlign;
    uint64_t outputMaxPoolSize;
    uint64_t outputIndicePoolSize;
    uint64_t tmpDxFactor;
    uint64_t tmpHxFactor;
    uint64_t tmpWxFactorAlign;
    uint64_t inputPoolSize;
    uint64_t indiceTempPoolSize;
    uint64_t maskPoolSize;
    uint64_t tmpTotalSize;
};

struct MaxPool3DWithArgmaxV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class MaxPool3DWithArgmaxV2BaseTiling : public TilingBaseClass {
public:
    explicit MaxPool3DWithArgmaxV2BaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~MaxPool3DWithArgmaxV2BaseTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

public:
    InputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
};

class MaxPool3DWithArgmaxV2NoExpandIndicesTiling : public MaxPool3DWithArgmaxV2BaseTiling {
public:
    explicit MaxPool3DWithArgmaxV2NoExpandIndicesTiling(gert::TilingContext* context)
        : MaxPool3DWithArgmaxV2BaseTiling(context)
    {}

    ~MaxPool3DWithArgmaxV2NoExpandIndicesTiling() override
    {}

private:
    void DoDAdjustment();
    void DoHAdjustment();
    void DoWAdjustment();
    void DoBoundaryAdjustment();
    void DoOutputPadAdjustment();
    void DoUBTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPool3DWithArgmaxV2NoExpandIndicesTilingData tiling;
    uint64_t inputBytes;
    SplitInfo splitData;
    BufferInfo bufferData;
};

} // namespace optiling

#endif