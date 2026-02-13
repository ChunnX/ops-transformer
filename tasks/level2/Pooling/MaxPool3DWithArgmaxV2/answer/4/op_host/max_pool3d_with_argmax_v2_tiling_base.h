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
const int MAX_CORE_NUM = 50;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t MAX_DIV = 2;
const uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const uint32_t MIN_TRANSPOSE_ROWS = 16;
const uint32_t INT64_FP32 = 2;
const uint32_t BINARY_SEARCH_COEFF = 2;
const uint32_t BLOCK_LEN_FP32 = 8;
const uint32_t BLOCK_LEN_FP16 = 16;

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

BEGIN_TILING_DATA_DEF(MaxPool3DWithArgmaxV2SplitWTilingData)
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
TILING_DATA_FIELD_DEF(uint64_t, batchesPerCore);
TILING_DATA_FIELD_DEF(uint64_t, leftOverBatches);
TILING_DATA_FIELD_DEF(uint64_t, partsPerCore);
TILING_DATA_FIELD_DEF(uint64_t, leftOverParts);
TILING_DATA_FIELD_DEF(uint64_t, partD);
TILING_DATA_FIELD_DEF(uint64_t, partH);
TILING_DATA_FIELD_DEF(uint64_t, partW);
TILING_DATA_FIELD_DEF(float, minFloat);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDIn);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHIn);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointWIn);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointDOut);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointHOut);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, splitPointWOut);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, sizeIn);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_CORE_NUM, batchStart);
TILING_DATA_FIELD_DEF(uint64_t, ceilD);
TILING_DATA_FIELD_DEF(uint64_t, ceilH);
TILING_DATA_FIELD_DEF(uint64_t, ceilW);
TILING_DATA_FIELD_DEF(uint64_t, sizeUb1);
TILING_DATA_FIELD_DEF(uint64_t, sizeUb2);
TILING_DATA_FIELD_DEF(uint64_t, sizeValues);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2, MaxPool3DWithArgmaxV2TilingData);

REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111100, MaxPool3DWithArgmaxV2SplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111101, MaxPool3DWithArgmaxV2SplitWTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3DWithArgmaxV2_111102, MaxPool3DWithArgmaxV2SplitWTilingData);

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

struct PadInputInfo {
    array<uint64_t, DHW_DIMS> padInputShape;
    array<uint64_t, DHW_DIMS> ceil;
};

struct UBBufferSize {
    uint64_t sizeUb1;
    uint64_t sizeUb2;
    uint64_t valSize;
};

struct SplitDataW {
    uint64_t batchesPerCore;
    uint64_t leftOverBatches;
    uint64_t partD;
    uint64_t partOutD;
    uint64_t partH;
    uint64_t partOutH;
    uint64_t partW;
    uint64_t partOutW;
    uint64_t partsPerCore = 0;
    uint64_t leftOverParts = 0;
    uint64_t splitPointDIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointHIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointWIn[MAX_CORE_NUM] = {0};
    uint64_t splitPointDOut[MAX_CORE_NUM] = {0};
    uint64_t splitPointHOut[MAX_CORE_NUM] = {0};
    uint64_t splitPointWOut[MAX_CORE_NUM] = {0};
    uint64_t sizeIn[MAX_CORE_NUM] = {0};
    uint64_t batchStart[MAX_CORE_NUM] = {0};
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

class MaxPool3DWithArgmaxV2BaseSplitTiling : public MaxPool3DWithArgmaxV2BaseTiling {
public:
    explicit MaxPool3DWithArgmaxV2BaseSplitTiling(gert::TilingContext* context)
        : MaxPool3DWithArgmaxV2BaseTiling(context)
    {}

    ~MaxPool3DWithArgmaxV2BaseSplitTiling() override
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t CalcBufferSizes(
        const array<uint64_t, DHW_DIMS> part, const array<uint64_t, DHW_DIMS> partOut, const uint64_t partHwInp,
        UBBufferSize& ubBufSizes);
    ge::graphStatus FindSplitParts(
        array<uint64_t, DHW_DIMS>& inParts, array<uint64_t, DHW_DIMS>& outParts, UBBufferSize& ubBufSizes, uint64_t dim);
    uint64_t RoundUpBlock(const uint64_t& src, const uint64_t& blockLen);
    uint64_t RoundDownBlock(const uint64_t& src, const uint64_t& blockLen);

private:
    ge::graphStatus PadCalc();
    uint64_t InputCalc(uint64_t dim);
    ge::graphStatus InputPadCalc(const array<int64_t, DHW_DIMS> inpDiff);

public:
    PadInputInfo padInputData;
    UBBufferSize bufSizes;
    uint64_t blockLength;
    uint64_t blockLengthS;
    uint64_t ubSizeNew;
};

class MaxPool3DWithArgmaxV2SplitWTiling : public MaxPool3DWithArgmaxV2BaseSplitTiling {
public:
    explicit MaxPool3DWithArgmaxV2SplitWTiling(gert::TilingContext* context)
        : MaxPool3DWithArgmaxV2BaseSplitTiling(context)
    {}

    ~MaxPool3DWithArgmaxV2SplitWTiling() override
    {}

private:
    void DoUBTiling();
    void SetTilingData();
    ge::graphStatus SmallBatchesTiling(void);
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPool3DWithArgmaxV2SplitWTilingData tiling;
    SplitDataW splitData;
};

} // namespace optiling

#endif