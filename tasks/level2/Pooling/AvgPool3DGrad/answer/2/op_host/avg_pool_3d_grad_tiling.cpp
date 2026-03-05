#include <string>
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"
#include "avg_pool_3d_grad_tiling.h"

namespace {
constexpr int32_t kAvgPool3DGradDimLimit = 6;
constexpr int32_t kAvgPool3DGradDedyInputIdx = 1;
} // namespace

namespace optiling {
constexpr int64_t GRAD_SHAPE = 5;
constexpr int64_t ATTR_SIZE = 3;
constexpr int64_t SCALR_UB_SIZE = 20480;
constexpr int64_t BF16_BYTE_SIZE = 2;
constexpr int64_t BF16_UB_PART = 3;
constexpr int64_t BF16_UB_PART_NO_OVERLAP = 4;
constexpr int64_t FP32_UB_PART = 2;
constexpr int64_t D_DIM = 0;
constexpr int64_t H_DIM = 1;
constexpr int64_t W_DIM = 2;
constexpr int64_t W_DIM_OFFSET = 1;
constexpr int64_t H_DIM_OFFSET = 2;
constexpr int64_t D_DIM_OFFSET = 3;
constexpr int64_t N_DIM_OFFSET = 5;
constexpr int64_t COUNT_IDX = 4;
constexpr int64_t DIVISOR_IDX = 5;
constexpr int64_t FORMAT_IDX = 6;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t NDHWC_N_DIM = 0;
constexpr int64_t NDHWC_D_DIM = 1;
constexpr int64_t NCDHW_D_DIM = 2;
constexpr size_t ORIG_INPUT_SHAPE_INDEX = 0;
constexpr size_t COMPATIABLE_PAD_DIM = 6;
constexpr size_t DIM0 = 0;
constexpr size_t DIM2 = 2;
constexpr size_t DIM4 = 4;
constexpr uint64_t DTYPE_LEN_B16 = 2;
constexpr uint64_t DTYPE_LEN_B32 = 4;
constexpr uint64_t FRACTOR_TWO = 2;
constexpr int64_t BATCH_MODE = 1;

struct DHWParam {
    int64_t n = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
};

template <typename T>
auto max(T a, T b) -> T
{
    return a >= b ? a : b;
}

class AvgPool3dGradTiling {
public:
    explicit AvgPool3dGradTiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline ge::graphStatus InitDHW();
    inline void Tiling4CParam(
        const uint64_t ubSizePlatform, const int64_t cShape, const int64_t ndhwShape, const ge::DataType& dtype);
    inline void SetTilingKey(const ge::DataType& dtype);
    inline int64_t AlignUp(int64_t num, int64_t rnd) const;
    inline int64_t AlignDown(int64_t num, int64_t rnd);

    ge::graphStatus InitBaseParams(gert::Shape& gradShape, ge::DataType& dtype);
    void CalcShapeVars(const gert::Shape& gradShape, int64_t& ncShape, int64_t& ndhwShape, int64_t& cShape);
    void SetWorkspace(ge::DataType dtype, int64_t ncShape);
    void CalcCoreNumAndDetermineFlag(uint32_t coreNum, int64_t ncShape, int64_t ndhwShape, int64_t gradDim0);
    void DispatchTilingStrategy(uint64_t ubSizePlatform, ge::DataType dtype, 
                                                 int64_t cShape, int64_t ndhwShape);
private:
    gert::TilingContext* tilingContext_ = nullptr;
    AvgPool3dGradTilingParam tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t isDetermine_ = 0;
    uint64_t countIncludePad_ = 0;
    int64_t divisorOverride_ = 0;
    uint64_t isOverlap_ = 0;
    uint64_t N_ = 0;
    uint64_t C_ = 0;
    std::string dataFormat_;
    DHWParam outDHW_;
    DHWParam inDHW_;
    DHWParam kDHW_;
    DHWParam dDHW_;
    DHWParam padDHW_;

    uint64_t normalCoreCNum_ = 0;
    uint64_t lastCoreCNum_ = 0;
    uint64_t cAlign_ = 0;
    uint64_t cTotal_ = 0;
    uint64_t cNum_ = 0;
    uint64_t cCount_ = 0;
    uint64_t cTail_ = 0;
    uint64_t cLine_ = 0;
};

int64_t AvgPool3dGradTiling::AlignDown(int64_t num, int64_t rnd)
{
    return ((rnd) == 0 ? 0 : ((num / rnd) * rnd));
}

int64_t AvgPool3dGradTiling::AlignUp(int64_t num, int64_t rnd) const
{
    return ((rnd) == 0 ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

ge::graphStatus AvgPool3dGradTiling::InitDHW()
{
    auto inputShape1 = tilingContext_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape1);
    auto gradShape = inputShape1->GetStorageShape();
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    auto kSize = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    if (kSize->GetSize() != ATTR_SIZE || strides->GetSize() != ATTR_SIZE ||
        (pads->GetSize() != ATTR_SIZE && pads->GetSize() != COMPATIABLE_PAD_DIM)) {
        OP_LOGE(
            tilingContext_->GetNodeName(),
            "kSize shape %lu"
            "strides shape %lu, pads shape %lu is not same with 3, or pads shape is not same with 6",
            kSize->GetSize(), strides->GetSize(), pads->GetSize());
        return ge::GRAPH_FAILED;
    }
    auto kSizeData = static_cast<const int64_t*>(kSize->GetData());
    auto stridesData = static_cast<const int64_t*>(strides->GetData());
    auto padsData = static_cast<const int64_t*>(pads->GetData());

    auto inputShape0 = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    const gert::Tensor* shapeTensor = tilingContext_->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapeTensor);
    const uint32_t* shapeValue = shapeTensor->GetData<uint32_t>();
    if (shapeValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t gradShapeDimD = NDHWC_D_DIM;
    inDHW_.n = shapeValue[shapeDim - N_DIM_OFFSET];
    inDHW_.d = shapeValue[shapeDim - D_DIM_OFFSET];
    inDHW_.h = shapeValue[shapeDim - H_DIM_OFFSET];
    inDHW_.w = shapeValue[shapeDim - W_DIM_OFFSET];
    outDHW_.n = gradShape.GetDim(NDHWC_N_DIM);
    outDHW_.d = gradShape.GetDim(gradShapeDimD);
    outDHW_.h = gradShape.GetDim(gradShapeDimD + H_DIM);
    outDHW_.w = gradShape.GetDim(gradShapeDimD + W_DIM);
    kDHW_.d = kSizeData[D_DIM];
    kDHW_.h = kSizeData[H_DIM];
    kDHW_.w = kSizeData[W_DIM];
    dDHW_.d = stridesData[D_DIM];
    dDHW_.h = stridesData[H_DIM];
    dDHW_.w = stridesData[W_DIM];

    if (pads->GetSize() != COMPATIABLE_PAD_DIM) {
        padDHW_.d = padsData[D_DIM];
        padDHW_.h = padsData[H_DIM];
        padDHW_.w = padsData[W_DIM];
    } else {
        padDHW_.d = padsData[DIM0];
        padDHW_.h = padsData[DIM2];
        padDHW_.w = padsData[DIM4];
    }
    isOverlap_ = static_cast<uint64_t>(dDHW_.d < kDHW_.d || dDHW_.h < kDHW_.h || dDHW_.w < kDHW_.w);
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::Tiling4CParam(
    const uint64_t ubSizePlatform, const int64_t cShape, const int64_t ndhwShape, const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for c param start");
    auto ubSizeLeft = ubSizePlatform - static_cast<uint64_t>(SCALR_UB_SIZE);
    auto ubPart = FP32_UB_PART;
    auto byteSize = GetSizeByDataType(dtype);
    auto ubSize4C = ubSizeLeft / ubPart;
    auto ubSize4CAlign = AlignDown(ubSize4C, ALIGN_BYTES);
    int64_t ubMaxCNum = ubSize4CAlign / byteSize;
    cTotal_ = static_cast<uint64_t>(cShape);
    if (cShape > ubMaxCNum) {
        cAlign_ = static_cast<uint64_t>(ubMaxCNum);
        cCount_ = static_cast<uint64_t>(ubMaxCNum);
        cNum_ = (static_cast<uint64_t>(cShape) + cCount_ - 1UL) / cCount_;
        cTail_ = static_cast<uint64_t>(cShape) - (cNum_ - 1UL) * cCount_;
        cLine_ = 1UL;
    } else {
        auto alignBlocks = ALIGN_BYTES / byteSize;
        cAlign_ = AlignUp(cShape, alignBlocks);
        if (cAlign_ == 0UL) {
            return;
        }
        cCount_ = static_cast<uint64_t>(cShape);
        cNum_ = 1UL;
        cTail_ = cCount_;
        cLine_ = static_cast<uint64_t>(ubMaxCNum) / cAlign_;
    }
    auto dividShape = static_cast<uint64_t>(ndhwShape) * cNum_;
    normalCoreCNum_ = dividShape / coreNum_;
    lastCoreCNum_ = dividShape - normalCoreCNum_ * (coreNum_ - 1UL);
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for c param end");
}

void AvgPool3dGradTiling::SetTilingKey(const ge::DataType& dtype)
{
    tilingContext_->SetTilingKey(2000);
}

ge::graphStatus AvgPool3dGradTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Tiling initing");
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    gert::Shape gradShape;
    ge::DataType dtype;
    OP_CHECK_IF(InitBaseParams(gradShape, dtype) != ge::GRAPH_SUCCESS, 
                OP_LOGE(tilingContext_->GetNodeName(), "InitBaseParams failed"),
                return ge::GRAPH_FAILED);

    int64_t ndhwShape = 1;
    int64_t ncShape = 1;
    int64_t cShape = 1;
    CalcShapeVars(gradShape, ncShape, ndhwShape, cShape);

    auto ret = InitDHW();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SetWorkspace(dtype, ncShape);
    CalcCoreNumAndDetermineFlag(coreNum, ncShape, ndhwShape, gradShape.GetDim(0));

    OP_CHECK_IF(coreNum_ == 0UL, 
                OP_LOGE(tilingContext_->GetNodeName(), "CoreNum is zero, error."),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint64_t ubSizePlatform = static_cast<int64_t>(ubSizePlatForm);
    DispatchTilingStrategy(ubSizePlatform, dtype, cShape, ndhwShape);

    SetTilingKey(dtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3dGradTiling::InitBaseParams(gert::Shape& gradShape, ge::DataType& dtype)
{
    gradShape = tilingContext_->GetInputShape(1)->GetStorageShape();
    dtype = tilingContext_->GetInputDesc(1)->GetDataType();
    auto attrs = tilingContext_->GetAttrs();
    
    countIncludePad_ = static_cast<uint64_t>(*attrs->GetAttrPointer<bool>(COUNT_IDX));
    divisorOverride_ = static_cast<int64_t>(*attrs->GetAttrPointer<int>(DIVISOR_IDX));
    dataFormat_ = "NDHWC";

    if (gradShape.GetDimNum() != GRAD_SHAPE) {
        OP_LOGE(tilingContext_->GetNodeName(), "gradShape dim num is not 5");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::CalcShapeVars(const gert::Shape& gradShape, int64_t& ncShape, int64_t& ndhwShape, int64_t& cShape)
{
    cShape = gradShape.GetDim(GRAD_SHAPE - 1);
    ndhwShape = 1;
    for (int i = NDHWC_N_DIM; i < NDHWC_D_DIM + ATTR_SIZE; i++) {
        ndhwShape *= gradShape.GetDim(i);
    }
    ncShape = gradShape.GetDim(0) * cShape;
}

void AvgPool3dGradTiling::SetWorkspace(ge::DataType dtype, int64_t ncShape)
{
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t castWorkspaceSize = 0;
    
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        size_t totalElements = static_cast<size_t>(inDHW_.d) * static_cast<size_t>(inDHW_.h) *
                               static_cast<size_t>(inDHW_.w) * static_cast<size_t>(ncShape);
        castWorkspaceSize = totalElements * sizeof(float);
    }

    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize + castWorkspaceSize;
}

void AvgPool3dGradTiling::CalcCoreNumAndDetermineFlag(
    uint32_t coreNum, int64_t ncShape, int64_t ndhwShape, int64_t gradDim0)
{
    coreNum_ = std::min(coreNum, static_cast<uint32_t>(ndhwShape));

    bool isDeterministic = (tilingContext_->GetDeterministic() == 1);
    if (!isDeterministic || isOverlap_ != 1) {
        return;
    }

    coreNum_ = std::min(coreNum, static_cast<uint32_t>(gradDim0));
    isDetermine_ = 1UL;
}

void AvgPool3dGradTiling::DispatchTilingStrategy(uint64_t ubSizePlatform, ge::DataType dtype, 
                                                 int64_t cShape, int64_t ndhwShape)
{
    Tiling4CParam(ubSizePlatform, cShape, ndhwShape, dtype);
}


ge::graphStatus AvgPool3dGradTiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.attrParam.set_N(N_);
    tilingData_.attrParam.set_C(C_);
    tilingData_.attrParam.set_inN(inDHW_.n);
    tilingData_.attrParam.set_inD(inDHW_.d);
    tilingData_.attrParam.set_inH(inDHW_.h);
    tilingData_.attrParam.set_inW(inDHW_.w);
    tilingData_.attrParam.set_outN(outDHW_.n);
    tilingData_.attrParam.set_outD(outDHW_.d);
    tilingData_.attrParam.set_outH(outDHW_.h);
    tilingData_.attrParam.set_outW(outDHW_.w);
    tilingData_.attrParam.set_kD(kDHW_.d);
    tilingData_.attrParam.set_kH(kDHW_.h);
    tilingData_.attrParam.set_kW(kDHW_.w);
    tilingData_.attrParam.set_dD(dDHW_.d);
    tilingData_.attrParam.set_dH(dDHW_.h);
    tilingData_.attrParam.set_dW(dDHW_.w);
    tilingData_.attrParam.set_padD(padDHW_.d);
    tilingData_.attrParam.set_padH(padDHW_.h);
    tilingData_.attrParam.set_padW(padDHW_.w);
    tilingData_.attrParam.set_countIncludePad(countIncludePad_);
    tilingData_.attrParam.set_divisorOverride(divisorOverride_);
    tilingData_.attrParam.set_isOverLap(isOverlap_);
    tilingData_.attrParam.set_isDetermine(isDetermine_);

    tilingData_.cParam.set_normalCoreCNum(normalCoreCNum_);
    tilingData_.cParam.set_lastCoreCNum(lastCoreCNum_);
    tilingData_.cParam.set_cAlign(cAlign_);
    tilingData_.cParam.set_cTotal(cTotal_);
    tilingData_.cParam.set_cCount(cCount_);
    tilingData_.cParam.set_cNum(cNum_);
    tilingData_.cParam.set_cLine(cLine_);
    tilingData_.cParam.set_cTail(cTail_);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    tilingContext_->SetScheduleMode(BATCH_MODE);
    TilingDataPrint();
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << coreNum_ << std::endl;
    std::cout << "N = " << N_ << std::endl;
    std::cout << "C = " << C_ << std::endl;
    std::cout << "inN = " << inDHW_.n << std::endl;
    std::cout << "inD = " << inDHW_.d << std::endl;
    std::cout << "inH = " << inDHW_.h << std::endl;
    std::cout << "inW = " << inDHW_.w << std::endl;
    std::cout << "outN = " << outDHW_.n << std::endl;
    std::cout << "outD = " << outDHW_.d << std::endl;
    std::cout << "outH = " << outDHW_.h << std::endl;
    std::cout << "outW = " << outDHW_.w << std::endl;
    std::cout << "kD = " << kDHW_.d << std::endl;
    std::cout << "kH = " << kDHW_.h << std::endl;
    std::cout << "kW = " << kDHW_.w << std::endl;
    std::cout << "dD = " << dDHW_.d << std::endl;
    std::cout << "dH = " << dDHW_.h << std::endl;
    std::cout << "dW = " << dDHW_.w << std::endl;
    std::cout << "padD = " << padDHW_.d << std::endl;
    std::cout << "padH = " << padDHW_.h << std::endl;
    std::cout << "padW = " << padDHW_.w << std::endl;
    std::cout << "countIncludePad = " << countIncludePad_ << std::endl;
    std::cout << "isOverlap = " << isOverlap_ << std::endl;
    std::cout << "normalCoreCNum = " << normalCoreCNum_ << std::endl;
    std::cout << "lastCoreCNum = " << lastCoreCNum_ << std::endl;
    std::cout << "cTotal = " << cTotal_ << std::endl;
    std::cout << "cAlign = " << cAlign_ << std::endl;
    std::cout << "cCount = " << cCount_ << std::endl;
    std::cout << "cNum = " << cNum_ << std::endl;
    std::cout << "cLine = " << cLine_ << std::endl;
    std::cout << "cTail = " << cTail_ << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_->GetNodeName(), "coreNum:                    %lu", coreNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "outn:                       %ld", outDHW_.n);
    OP_LOGD(tilingContext_->GetNodeName(), "outd:                       %ld", outDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "outh:                       %ld", outDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "outw:                       %ld", outDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "inn:                        %ld", inDHW_.n);
    OP_LOGD(tilingContext_->GetNodeName(), "ind:                        %ld", inDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "inh:                        %ld", inDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "inw:                        %ld", inDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "kd:                         %ld", kDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "kh:                         %ld", kDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "kw:                         %ld", kDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "sd:                         %ld", dDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "sh:                         %ld", dDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "sw:                         %ld", dDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "padd:                       %ld", padDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "padh:                       %ld", padDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "padw:                       %ld", padDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "countIncludePad:            %lu", countIncludePad_);
    OP_LOGD(tilingContext_->GetNodeName(), "isOverlap:                  %lu", isOverlap_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreCNum:            %lu", normalCoreCNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "lastCoreCNum:              %lu", lastCoreCNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "cTotal:                    %lu", cTotal_);
    OP_LOGD(tilingContext_->GetNodeName(), "cAlign:                    %lu", cAlign_);
    OP_LOGD(tilingContext_->GetNodeName(), "cCount:                    %lu", cCount_);
    OP_LOGD(tilingContext_->GetNodeName(), "cNum:                      %lu", cNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "cLine:                     %lu", cLine_);
    OP_LOGD(tilingContext_->GetNodeName(), "cTail:                     %lu", cTail_);
}

ge::graphStatus TilingAvgPool3dGradVec(gert::TilingContext* context)
{
    AvgPool3dGradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

ge::graphStatus TilingPrepareForAvgPool3dGradVec(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For AvgPool3dGrad start");
    OP_LOGD(context->GetNodeName(), "Tiling prepare for AvgPool3dGrad end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAvgPool3DGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("AvgPool3DGrad", "context is null"), return ge::GRAPH_FAILED);
    return TilingAvgPool3dGradVec(context);
}

static ge::graphStatus TilingPrepareForAvgPool3DGrad(gert::TilingParseContext* context)
{
    return TilingPrepareForAvgPool3dGradVec(context);
}

IMPL_OP_OPTILING(AvgPool3DGrad)
    .InputsDataDependency({ORIG_INPUT_SHAPE_INDEX})
    .Tiling(TilingForAvgPool3DGrad)
    .TilingParse<AvgPool3dGradCompileInfo>(TilingPrepareForAvgPool3DGrad);
} // namespace optiling