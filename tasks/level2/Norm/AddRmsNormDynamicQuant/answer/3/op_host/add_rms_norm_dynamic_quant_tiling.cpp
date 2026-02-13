#include "add_rms_norm_dynamic_quant_tiling.h"
#include "common/tiling_util.h"

namespace optiling {

template <typename T>
static inline T CeilDiv(T a, T b) {
    return (a + b - 1) / b;
}

constexpr int X1_IDX = 0;
constexpr int X2_IDX = 1;
constexpr int GAMMA_IDX = 2;
constexpr int SMOOTH1_IDX = 3;
constexpr int SMOOTH2_IDX = 4;
constexpr int BETA_IDX = 5;

constexpr int Y1_IDX = 0;
constexpr int Y2_IDX = 1;
constexpr int X_IDX = 2;
constexpr int SCALE1_IDX = 3;
constexpr int SCALE2_IDX = 4;

constexpr int EPS_IDX = 0;
constexpr int OUT_QUANT_1_IDX = 1;
constexpr int OUT_QUANT_2_IDX = 2;
constexpr int DST_TYPE_IDX = 2;

constexpr uint64_t USR_WORKSPACE_SIZE_910B = 1;

constexpr uint32_t SIZEOF_B16 = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t INT4_ALIGN_SIZE = 64;

constexpr uint32_t UB_TILING_POLICY_SLICE_D = 3;

constexpr uint32_t SLICE_COL_LEN = 8864;
constexpr uint32_t SLICE_COL_LEN_INT4 = 8832;

constexpr int32_t INT_NEGATIVE_ONE = -1;
constexpr int32_t INT_ZERO = 0;
constexpr int32_t INT_ONE = 1;
constexpr int32_t INT_TWO = 2;

bool CheckOptionalShapeExisting(const gert::StorageShape* smoothShape)
{
    OP_CHECK_IF(nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty smoothShape"), return false);
    return true;
}

bool CheckOptionalBetaExisting(const gert::StorageShape* betaShape)
{
    OP_CHECK_IF(nullptr == betaShape, OP_LOGD("CheckOptionalBetaExisting", "Get nullptr betaShape"), return false);
    int64_t betaShapeSize = betaShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((betaShapeSize <= 0), OP_LOGD("CheckOptionalBetaExisting", "Get empty betaShape"), return false);
    return true;
}

size_t GetworkspaceRowsNum(int32_t outQuant1Flag, int32_t outQuant2Flag, uint32_t smoothNum1_, uint32_t smoothNum2_)
{
    size_t workspaceRowsNum = INT_ZERO;
    if ((outQuant1Flag == INT_NEGATIVE_ONE && outQuant2Flag == INT_NEGATIVE_ONE)) {
        workspaceRowsNum = (smoothNum1_ == INT_ZERO && smoothNum2_ == INT_ZERO) ? INT_ONE : INT_TWO;
    } else {
        workspaceRowsNum = (outQuant1Flag == INT_ONE || outQuant2Flag == INT_ONE) ? INT_TWO : INT_ONE;
    }
    return workspaceRowsNum;
}

void AddRmsNormDynamicQuantTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(AddRmsNormDynamicQuantTilingData* tiling)
{
    context_->SetBlockDim(this->useCore_);
    tiling->set_useCore(this->useCore_);
    tiling->set_numFirstDim(this->numFirstDim_);
    tiling->set_numLastDim(this->numLastDim_);
    tiling->set_numLastDimAligned(this->numLastDimAligned_);
    tiling->set_firstDimPerCore(this->firstDimPerCore_);
    tiling->set_firstDimPerCoreTail(this->firstDimPerCoreTail_);
    tiling->set_firstDimPerLoop(this->firstDimPerLoop_);
    tiling->set_lastDimSliceLen(this->lastDimSliceLen_);
    tiling->set_lastDimLoopNum(this->lastDimLoopNum_);
    tiling->set_lastDimSliceLenTail(this->lastDimSliceLenTail_);
    tiling->set_smoothNum1(this->smoothNum1_);
    tiling->set_smoothNum2(this->smoothNum2_);
    tiling->set_epsilon(this->eps_);
    tiling->set_outQuant1Flag(this->outQuant1Flag);
    tiling->set_outQuant2Flag(this->outQuant2Flag);
    tiling->set_avgFactor(this->avgFactor_);
    tiling->set_betaFlag(this->betaFlag_);
    uint32_t tilingKey = 0;
    size_t usrSize = USR_WORKSPACE_SIZE_910B;

    tilingKey += UB_TILING_POLICY_SLICE_D;
    size_t workspaceRowsNum =
        GetworkspaceRowsNum(this->outQuant1Flag, this->outQuant2Flag, this->smoothNum1_, this->smoothNum2_);
    usrSize = this->useCore_ * this->numLastDim_ * sizeof(float) * workspaceRowsNum;

    context_->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = this->sysWorkspaceSize_ + usrSize;

    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata useCore_: %lu, smoothNum1_: %u, smoothNum2_: %u",
        this->useCore_, this->smoothNum1_, this->smoothNum2_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata N: %lu, D:%lu, DAligned: %lu", numFirstDim_, numLastDim_,
        numLastDimAligned_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        firstDimPerCore_, firstDimPerCoreTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerLoop_: %lu", firstDimPerLoop_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace",
        "Tilingdata lastDimSliceLen_: %lu, lastDimLoopNum_: %lu, lastDimSliceLenTail_: %lu", lastDimSliceLen_,
        lastDimLoopNum_, lastDimSliceLenTail_);
    OP_LOGI("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata eps_: %f, avgFactor_: %f", eps_, avgFactor_);
    OP_LOGI(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey, usrSize);
}

bool AddRmsNormDynamicQuantTilingHelper::DoTiling()
{
    OP_TILING_CHECK(
        (nullptr == context_), OP_LOGE("AddRmsNormDynamicQuantTiling", "Helper context_ get nullptr, return failed."),
        return false);
    OP_TILING_CHECK(!GetBaseInfo(), OP_LOGE(context_->GetNodeName(), "GetBaseInfo falied, return false"), return false);
    OP_TILING_CHECK(
        !GetShapeInfo(), OP_LOGE(context_->GetNodeName(), "GetShapeInfo falied, return false"), return false);
    OP_TILING_CHECK(
        !DoBlockTiling(), OP_LOGE(context_->GetNodeName(), "DoBlockTiling falied, return false"), return false);
    OP_TILING_CHECK(!DoUbTiling(), OP_LOGE(context_->GetNodeName(), "DoUbTiling falied, return false"), return false);
    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
    this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
    this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
    OP_LOGI(
        "DoBlockTiling", "BlockTiling Factor: useCore_: %lu, firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        this->useCore_, this->firstDimPerCore_, this->firstDimPerCoreTail_);
    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::InitializePlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->socCoreNums_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::GetBaseInfo()
{
    if (!InitializePlatformInfo()) {
        return false;
    }

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(
        nullptr == attrs, OP_LOGE(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    const float* epsPtr = attrs->GetFloat(EPS_IDX);
    if (epsPtr != nullptr) {
        this->eps_ = *epsPtr;
    }

    const gert::ContinuousVector* outputMaskAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUT_QUANT_1_IDX);
    if (outputMaskAttr != nullptr && outputMaskAttr->GetSize() == INT_TWO) {
        const bool* scalesArray = static_cast<const bool*>(outputMaskAttr->GetData());
        this->outQuant1Flag = (scalesArray[0] == true) ? 1 : 0;
        this->outQuant2Flag = (scalesArray[1] == true) ? 1 : 0;
    } else {
        this->outQuant1Flag = -1;
        this->outQuant2Flag = -1;
    }
    OP_LOGI("outputMask", "outQuant1Flag: %u, outQuant2Flag: %u", this->outQuant1Flag, this->outQuant2Flag);
    if (!ValidateBaseParameters()) {
        return false;
    }
    OP_LOGI(
        "GetBaseInfo", "socCoreNum: %lu, ubSize: %lu, sysWorkspaceSize: %lu, epsilon: %f", this->socCoreNums_,
        this->ubSize_, this->sysWorkspaceSize_, this->eps_);

    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::ValidateBaseParameters()
{
    OP_TILING_CHECK(
        this->eps_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Epsilon less or equal than precision threshold, please check."),
        return false);
    OP_TILING_CHECK(
        (this->ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize less or equal than zero, please check."),
        return false);
    OP_TILING_CHECK(
        (this->socCoreNums_ <= 0),
        OP_LOGE(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    return true;
}

ge::graphStatus CheckDtypeVaild(ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

bool AddRmsNormDynamicQuantTilingHelper::ValidateInputOutput()
{
    OP_TILING_CHECK(
        CheckInputOutputShape() == false, OP_LOGE(context_->GetNodeName(), "Check tensor shape failed."), return false);

    auto y1DataType = context_->GetOutputDesc(Y1_IDX)->GetDataType();
    auto y2DataType = context_->GetOutputDesc(Y2_IDX)->GetDataType();
    std::vector<ge::DataType> supportedYDtypes = {ge::DataType::DT_INT8, ge::DataType::DT_INT4};
    if ((ge::GRAPH_SUCCESS != CheckDtypeVaild(y1DataType, supportedYDtypes)) ||
        (ge::GRAPH_SUCCESS != CheckDtypeVaild(y2DataType, supportedYDtypes)) || (y1DataType != y2DataType)) {
        OP_LOGE(context_->GetNodeName(), "Output dtype should be int8 int4 hifp8 and y1DataType y2DataType need same.");
        return false;
    }

    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::CalculateShapeParameters()
{
    this->dtSize_ = SIZEOF_B16;

    auto xShape = context_->GetInputShape(X1_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }

    this->numFirstDim_ = numRow;
    this->numLastDim_ = numCol;
    auto y1DataType = context_->GetOutputDesc(Y1_IDX)->GetDataType();
    uint32_t alignSize = y1DataType == ge::DT_INT4 ? INT4_ALIGN_SIZE : BLOCK_SIZE;
    this->dstType_ = static_cast<uint32_t>(y1DataType);
    this->numLastDimAligned_ =
        CeilDiv(numCol, static_cast<uint64_t>(alignSize)) * static_cast<uint64_t>(alignSize);

    this->avgFactor_ = 1.0 / (static_cast<float>(this->numLastDim_));

    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::SetFlagsAndCheckConsistency()
{
    const gert::StorageShape* smooth1Shape = this->context_->GetOptionalInputShape(SMOOTH1_IDX);
    const gert::StorageShape* smooth2Shape = this->context_->GetOptionalInputShape(SMOOTH2_IDX);
    const gert::StorageShape* betaShape = this->context_->GetOptionalInputShape(BETA_IDX);
    bool smooth1Exist = CheckOptionalShapeExisting(smooth1Shape);
    bool smooth2Exist = CheckOptionalShapeExisting(smooth2Shape);
    bool betaExist = CheckOptionalBetaExisting(betaShape);

    this->smoothNum1_ = (smooth1Exist) ? 1 : 0;
    this->smoothNum2_ = (smooth2Exist) ? 1 : 0;
    this->betaFlag_ = (betaExist) ? 1 : 0;

    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    OP_TILING_CHECK(
        (smooth1Exist && smooth1Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth1Shape."), return false);
    OP_TILING_CHECK(
        (smooth2Exist && smooth2Shape->GetStorageShape() != gammaShape),
        OP_LOGE(context_->GetNodeName(), "GammaShape is not same to smooth2Shape."), return false);

    if (this->outQuant1Flag == INT_NEGATIVE_ONE && this->outQuant2Flag == INT_NEGATIVE_ONE) {
        OP_TILING_CHECK(
            (!smooth1Exist) && (smooth2Exist),
            OP_LOGE(context_->GetNodeName(), "Smooth2 exist but smooth1 not exist, bad input."), return false);
    }

    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::GetShapeInfo()
{
    if (!ValidateInputOutput()) {
        return false;
    }

    if (!CalculateShapeParameters()) {
        return false;
    }

    if (!SetFlagsAndCheckConsistency()) {
        return false;
    }

    OP_LOGI("GetShapeInfo", "[N, D] = [%lu, %lu]", this->numFirstDim_, this->numLastDim_);
    OP_LOGI("GetShapeInfo", "dtSize_=%lu, avgFactor_=%f", this->dtSize_, this->avgFactor_);
    return true;
}

bool AddRmsNormDynamicQuantTilingHelper::DoUbTiling()
{
    OP_TILING_CHECK(CheckUbSliceDTiling(), OP_LOGI(context_->GetNodeName(), "Ub Tiling: SliceD."), return true);
    return false;
}

bool AddRmsNormDynamicQuantTilingHelper::CheckUbSliceDTiling()
{
    OP_LOGI(this->context_->GetNodeName(), "CheckUbSliceDTiling success. Compute tiling by yourself.");
    this->ubTilingPolicy_ = UB_TILING_POLICY::SLICE_D;
    this->firstDimPerLoop_ = 1;
    if (this->dstType_ == 29) {
        this->lastDimSliceLen_ = SLICE_COL_LEN_INT4;
    } else {
        this->lastDimSliceLen_ = SLICE_COL_LEN;
    }
    this->lastDimSliceLenTail_ = (this->numLastDim_ % this->lastDimSliceLen_ == 0) ?
                                     this->lastDimSliceLen_ :
                                     this->numLastDim_ % this->lastDimSliceLen_;
    this->lastDimLoopNum_ = (this->numLastDim_ - this->lastDimSliceLenTail_) / this->lastDimSliceLen_;
    return true;
}

ge::graphStatus Tiling4AddRmsNormDynamicQuant(gert::TilingContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Tiling4AddRmsNormDynamicQuant");
    auto colShape = context->GetInputShape(GAMMA_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, colShape);
    auto colStorageShape = optiling::EnsureNotScalar(colShape->GetStorageShape());
    uint32_t col_val = colStorageShape.GetDim(0);
    bool isEmptyTensor = (col_val == 0);
    if (isEmptyTensor) {
        context->SetBlockDim(1);
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    AddRmsNormDynamicQuantTilingData tiling;
    AddRmsNormDynamicQuantTilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OP_TILING_CHECK(
        !status, OP_LOGE(context->GetNodeName(), "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);
    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4AddRmsNormDynamicQuant(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4AddRmsNormDynamicQuant.");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "PlatformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<AddRmsNormDynamicQuantCompileInfo>();
    if (compileInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
        compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantTilingHelper::CheckInputOutputShape()
{
    const gert::StorageShape* x1Shape = this->context_->GetInputShape(X1_IDX);
    const gert::StorageShape* x2Shape = this->context_->GetInputShape(X2_IDX);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);

    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    const gert::StorageShape* y2Shape = this->context_->GetOutputShape(Y2_IDX);
    const gert::StorageShape* xShape = this->context_->GetOutputShape(X_IDX);
    const gert::StorageShape* scale1Shape = this->context_->GetOutputShape(SCALE1_IDX);
    const gert::StorageShape* scale2Shape = this->context_->GetOutputShape(SCALE2_IDX);

    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale2Shape);

    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t y1DimNum = y1Shape->GetStorageShape().GetDimNum();
    size_t y2DimNum = y2Shape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t scale1DimNum = scale1Shape->GetStorageShape().GetDimNum();
    size_t scale2DimNum = scale2Shape->GetStorageShape().GetDimNum();

    OP_LOGI(
        this->context_->GetNodeName(),
        "ShapeDim info: x1.dim=%zu, x2.dim=%zu, gamma.dim=%zu, y1.dim=%zu, y2.dim=%zu, x.dim=%zu, scale1.dim=%zu, "
        "scale2.dim=%zu",
        x1DimNum, x2DimNum, gammaDimNum, y1DimNum, y2DimNum, xDimNum, scale1DimNum, scale2DimNum);

    bool hasZeroDimTensor = x1DimNum <= 0 || x2DimNum <= 0 || gammaDimNum <= 0;
    OP_TILING_CHECK(
        (hasZeroDimTensor),
        OP_LOGE(
            this->context_->GetNodeName(),
            "Input x1/x2/y1//x/scale1DimNum shape invaild, dim num should not be smaller or equal to zero."),
        return false);
    OP_TILING_CHECK(
        ((x1DimNum != x2DimNum)),
        OP_LOGE(this->context_->GetNodeName(), "Input x1/x2 shape dims not equal. Tiling failed. "), return false);
    OP_TILING_CHECK(
        ((gammaDimNum != 1)), OP_LOGE(this->context_->GetNodeName(), "gamma shape dims not equal to 1. Tiling failed."),
        return false);
    gert::Shape shapeOfX = xShape->GetStorageShape();
    gert::Shape shapeOfGamma = gammaShape->GetStorageShape();
    OP_TILING_CHECK(
        (shapeOfX[xDimNum - 1] != shapeOfGamma[gammaDimNum - 1]),
        OP_LOGE(context_->GetNodeName(), "gammaShape isn't consistent with the last dimension of x1."), return false);
    return true;
}

IMPL_OP_OPTILING(AddRmsNormDynamicQuant)
    .Tiling(Tiling4AddRmsNormDynamicQuant)
    .TilingParse<AddRmsNormDynamicQuantCompileInfo>(TilingPrepare4AddRmsNormDynamicQuant);

} // namespace optiling