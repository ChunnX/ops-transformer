#include <vector>
#include "common/tiling_util.h"
#include "add_layer_norm_quant_tiling.h"

namespace optiling {

static constexpr int64_t UB_RESERVED_BYTE = 256;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t FLOAT_BLOCK_ELEM = 8;
static constexpr int64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
static constexpr int64_t ROW_FACTOR = 16;

static constexpr uint64_t TILING_DATATYPE_FP16 = 1000;
static constexpr uint64_t TILING_DATATYPE_FP32 = 2000;
static constexpr uint64_t TILING_DATATYPE_BF16 = 3000;

static constexpr uint64_t TILING_X3_ELEWISE = 1;
static constexpr uint64_t TILING_X3_BROADCAST = 2;
static constexpr uint64_t TILING_DYNAMIC_QUANT = 100;

static constexpr uint32_t MODE_DUAL_WITHOUT_OFFSETS = 200;
static constexpr uint32_t MODE_DUAL_WITH_OFFSETS = 211;
static constexpr uint32_t MODE_SOLE_WITHOUT_OFFSETS = 100;
static constexpr uint32_t MODE_SOLE_WITH_OFFSETS = 110;

static constexpr uint32_t MODE_CODE_INPUT_SCALES = 100;
static constexpr uint32_t MODE_CODE_INPUT_ZEROPOINTS1 = 10;
static constexpr uint32_t MODE_CODE_INPUT_ZEROPOINTS2 = 1;
static constexpr int32_t ELEM_PER_REP_FP32 = 64;
static constexpr int32_t ONE_BLK_SIZE = 32;

const std::string OP_NAME = "AddLayerNormQuant";

inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

enum class TILING_TYPE : uint32_t {
    NORMAL,
    SINGLE_ROW,
    SLICE_EXT,
    SMALL_D
};

template <typename T>
auto GetOptionalAttr(const gert::RuntimeAttrs* attrs, const size_t idx, const T& defaultValue) -> T
{
    const T* attrPtr = attrs->GetAttrPointer<T>(idx);
    if (nullptr == attrPtr) {
        OP_LOGD("GetOptionalAttr", "attr[%zu] get unsuccess, use default value", idx);
    }
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

inline TILING_TYPE AddLayerNormQuantTilingImpl(
    uint64_t maxUbSize, int64_t& dtSize, int32_t bufferNum, int32_t numCol, int32_t firstdimPerCore, bool enableXOut,
    enum BIAS_TYPE biasType, uint32_t& rowPerTime, uint32_t& colPerTime)
{
    int64_t additionalOutputNum = enableXOut ? 1 : 0;
    int64_t biasNum = biasType == BIAS_TYPE::BROADCAST_BIAS ? 1 : 0;
    auto numColAligned = (numCol + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    float avaUb = static_cast<float>(static_cast<float>(maxUbSize) - 1024);
    float liveTensorFactor = static_cast<float>(14 * numColAligned + 6 * numColAligned + 8 + 32 + 256 + 4);
    float tmpRow = avaUb / liveTensorFactor;
    rowPerTime = floor(tmpRow);
    colPerTime = numCol;
    return TILING_TYPE::NORMAL;
}

inline bool CheckScaleOffset(bool isDynamicQuant, uint32_t mode)
{
    bool ret = true;
    ret = (mode % MODE_CODE_INPUT_SCALES) == 0;
    return ret;
}

static inline uint32_t ComputeOptCode(gert::TilingContext* context)
{
    uint32_t optionalScaleOffsetMode = 0;
    auto scale1Shape = context->GetOptionalInputShape(SCALE1_IDX);
    auto scale2Shape = context->GetOptionalInputShape(SCALE2_IDX);
    auto zeroPoint1Shape = context->GetOptionalInputShape(ZERO_POINT1_IDX);
    auto zeroPoint2Shape = context->GetOptionalInputShape(ZERO_POINT2_IDX);
    optionalScaleOffsetMode += (nullptr != scale1Shape) ? MODE_CODE_INPUT_SCALES : 0;
    optionalScaleOffsetMode += (nullptr != scale2Shape) ? MODE_CODE_INPUT_SCALES : 0;
    optionalScaleOffsetMode += (nullptr != zeroPoint1Shape) ? MODE_CODE_INPUT_ZEROPOINTS1 : 0;
    optionalScaleOffsetMode += (nullptr != zeroPoint2Shape) ? MODE_CODE_INPUT_ZEROPOINTS2 : 0;
    OP_LOGD(context, "optionalScaleOffsetMode=%d", optionalScaleOffsetMode);
    return optionalScaleOffsetMode;
}

static inline bool GetAttrs(
    gert::TilingContext* context, AddLayerNormQuantTilingData* tiling, bool& isDynamicQuant,
    bool& enableAdditionalOutput)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    const char* qms = attrs->GetAttrPointer<char>(QUANT_MODE_IDX);
    OP_CHECK_IF(nullptr == qms, OP_LOGE("GetAttrs", "Get required attr quantMode failed, tiling failed"), return false);
    std::string quantModeStr = qms;
    isDynamicQuant = (quantModeStr == "dynamic");
    OP_LOGD("GetAttrs", "isDynamicQuant=%d", isDynamicQuant);

    float eps = GetOptionalAttr<float>(attrs, EPS_IDX, (float)1e-5);
    enableAdditionalOutput = GetOptionalAttr<bool>(attrs, X_OUT_IDX, false);
    OP_LOGD("GetAttrs", "Attrs: quantMode=%s, eps=%f, additionalOutput=%d", qms, eps, enableAdditionalOutput);
    tiling->set_eps(eps);
    if (enableAdditionalOutput) {
        tiling->set_isXOut(1);
    } else {
        tiling->set_isXOut(0);
    }
    return true;
}

static inline uint64_t ComputeTilingKey(
    ge::DataType dataType, bool isDynamicQuant, TILING_TYPE tilingType, BIAS_TYPE biasType)
{
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_DATATYPE_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_DATATYPE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_DATATYPE_BF16;
    }

    tilingKey += TILING_DYNAMIC_QUANT;

    if (biasType == BIAS_TYPE::BROADCAST_BIAS) {
        tilingKey += TILING_X3_BROADCAST;
    } else if (biasType == BIAS_TYPE::ELEWISE_BIAS) {
        tilingKey += TILING_X3_ELEWISE;
    }
    OP_LOGI("AddLayerNormQuant", "tilingKey = %lu", tilingKey);
    return tilingKey;
}

static inline BIAS_TYPE FindBiasType(gert::TilingContext* context, int64_t numRow, int64_t numCol)
{
    BIAS_TYPE biasType = BIAS_TYPE::NO_BIAS;
    auto biasShape = context->GetOptionalInputShape(BIAS_IDX);
    if (nullptr == biasShape) {
        OP_LOGD("AddLayerNormQuant", "Bias is null");
        biasType = BIAS_TYPE::NO_BIAS;
    } else {
        OP_LOGD("AddLayerNormQuant", "Bias is not null");
        int64_t x1Size = numRow * numCol;
        int64_t biasSize = 1;
        for (size_t i = 0; i < biasShape->GetStorageShape().GetDimNum(); i++) {
            biasSize *= biasShape->GetStorageShape().GetDim(i);
        }
        biasType = x1Size == biasSize ? (BIAS_TYPE::ELEWISE_BIAS) : (BIAS_TYPE::BROADCAST_BIAS);
    }
    OP_LOGI("AddLayerNormQuant", "Bias Type is %u", static_cast<uint32_t>(biasType));
    return biasType;
}

static inline void GetSocVersion(gert::TilingContext* context, uint64_t& maxUbSize, uint32_t& maxCoreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGI(context, "Get Platform Info: maxUbSize = %lu, maxCoreNum = %u", maxUbSize, maxCoreNum);
}

static inline void ComputeFusedAxis(
    gert::TilingContext* context, AddLayerNormQuantTilingData* tiling, int32_t& numRow, int32_t& numCol, int64_t& dtSize)
{
    numRow = 1;
    for (size_t i = 0; i < context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1; i++) {
        numRow *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    numCol = context->GetInputShape(0)->GetStorageShape().GetDim(
        context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);
    float tempAve = (numCol == 0) ? 0 : float(1.0 / numCol);
    int32_t numColAligned = numCol;
    if (dtSize > 0) {
        numColAligned = ((numCol * dtSize + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE) / dtSize;
    }
    int32_t numLastDimRoundUp32 = (numCol + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;

    uint32_t mulLoopFp32 = numColAligned / ELEM_PER_REP_FP32;
    uint32_t mulTailFp32 = numColAligned - mulLoopFp32 * ELEM_PER_REP_FP32;
    uint8_t dstRepStrideFp32 = numColAligned / 8;

    tiling->set_numFirstDim(numRow);
    tiling->set_numLastDim(numCol);
    tiling->set_aveFactor(tempAve);
    tiling->set_numLastDimAlign(numColAligned);
    tiling->set_numLastDimAlign32(numLastDimRoundUp32);
    tiling->set_mulLoopFp32(mulLoopFp32);
    tiling->set_mulTailFp32(mulTailFp32);
    tiling->set_dstRepStrideFp32(dstRepStrideFp32);
}

static inline void DoBlockTiling(
    gert::TilingContext* context, AddLayerNormQuantTilingData* tiling, uint32_t maxCoreNum, int32_t numRow,
    uint32_t& firstdimPerCore, uint32_t& numCore, int32_t& nlFirstdimPerCoreNum)
{
    (void)nlFirstdimPerCoreNum;
    firstdimPerCore = CEIL_DIV(tiling->get_numFirstDim(), maxCoreNum);
    numCore = CEIL_DIV(tiling->get_numFirstDim(), firstdimPerCore);
    tiling->set_numCore(numCore);
    context->SetBlockDim(numCore);
    firstdimPerCore = CEIL_DIV(numRow, numCore);
    uint32_t firstDimPerCoreTail = numRow - firstdimPerCore * (numCore - 1);
    uint32_t gmOffset = firstdimPerCore * tiling->get_numLastDim();

    tiling->set_firstDimPerCore(firstdimPerCore);
    tiling->set_firstDimPerCoreTail(firstDimPerCoreTail);
    tiling->set_gmOffset(gmOffset);
    OP_LOGI("AddLayerNormQuant", "numCore = %u", numCore);
}

static inline void SetWorkSapce4AddLayerNormQuant(
    gert::TilingContext* context, AddLayerNormQuantTilingData* tiling, TILING_TYPE tilingType, int32_t numRow,
    int32_t numCol)
{
    size_t workspaceSize = 1;
    tiling->set_workspaceSize(workspaceSize);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
    currentWorkspace[0] = workspaceSize + sysWorkspaceSize;
    OP_LOGI("AddLayerNormQuant", "workspaceSize = %zu", workspaceSize);
}

static inline void PostUbTiling(
    AddLayerNormQuantTilingData* tiling, int32_t firstdimPerCore, int32_t numCol, uint32_t colPerTime,
    uint32_t& rowPerTime)
{
    uint32_t firstDimPerCoreTail = tiling->get_firstDimPerCoreTail();

    rowPerTime =
        (rowPerTime > static_cast<uint32_t>(firstdimPerCore)) ? static_cast<uint32_t>(firstdimPerCore) : rowPerTime;
    uint32_t firstDimPerTimeTail = rowPerTime < firstDimPerCoreTail ? rowPerTime : firstDimPerCoreTail;
    uint32_t tt = 0;
    if (rowPerTime > 0) {
        tt = firstdimPerCore % rowPerTime;
    }
    uint32_t rowTailPerBlock = tt == 0 ? rowPerTime : tt;
    uint32_t rowTailLastBlock = (firstDimPerCoreTail % firstDimPerTimeTail == 0) ? firstDimPerTimeTail : (firstDimPerCoreTail % firstDimPerTimeTail);
    int32_t colMoveCnt = CEIL_DIV(numCol, colPerTime);
    int32_t colTail = (numCol % colPerTime == 0) ? colPerTime : (numCol % colPerTime);

    tiling->set_firstDimPerTime(rowPerTime);
    tiling->set_firstDimPerTimeTail(firstDimPerTimeTail);
    tiling->set_rowTailPerBlock(rowTailPerBlock);
    tiling->set_rowTailLastBlock(rowTailLastBlock);
    tiling->set_lastDimPerTime(colPerTime);
    tiling->set_colMoveCnt(colMoveCnt);
    tiling->set_colTail(colTail);

    OP_LOGI(
        "AddLayerNormQuant", "rowPerTime = %u, colPerTime = %u, colMoveCnt = %d, colTail = %d", rowPerTime, colPerTime,
        colMoveCnt, colTail);
}

static ge::graphStatus Tiling4AddLayerNormQuantMembase(gert::TilingContext* context)
{
    AddLayerNormQuantTilingData tiling;
    OP_LOGI(context, "Begin to do Tiling4AddLayerNormQuant");
    bool isDynamicQuant = false;
    bool enableAdditionalOutput = false;
    OP_CHECK_IF(
        !GetAttrs(context, &tiling, isDynamicQuant, enableAdditionalOutput),
        OP_LOGE(context, "GetAttrs failed"), return ge::GRAPH_FAILED);

    auto scale1Shape = context->GetOptionalInputShape(SCALE1_IDX);
    uint32_t optionalScaleOffsetMode = ComputeOptCode(context);

    OP_CHECK_IF(
        !CheckScaleOffset(isDynamicQuant, optionalScaleOffsetMode),
        OP_LOGE(
            context, "Optional input Scales and Offsets are invalid, get mode = %u, tiling failed",
            optionalScaleOffsetMode),
        return ge::GRAPH_FAILED);

    tiling.set_scaleOffsetMode(optionalScaleOffsetMode);
    tiling.set_isPerTensor(0);
    if ((nullptr != scale1Shape)) {
        OP_LOGD(context, "scale1Shape size : %ld", scale1Shape->GetStorageShape().GetShapeSize());
    }

    auto dataType = context->GetInputDesc(0)->GetDataType();
    int64_t dtSize = ge::GetSizeByDataType(dataType);
    int64_t bufferNum = 1;

    uint64_t maxUbSize = 0;
    uint32_t maxCoreNum = 0;
    GetSocVersion(context, maxUbSize, maxCoreNum);

    int32_t numRow = 1;
    int32_t numCol = 1;
    uint32_t firstdimPerCore = 1;
    uint32_t numCore = 1;
    int32_t nlFirstdimPerCoreNum = 1;
    ComputeFusedAxis(context, &tiling, numRow, numCol, dtSize);
    DoBlockTiling(context, &tiling, maxCoreNum, numRow, firstdimPerCore, numCore, nlFirstdimPerCoreNum);

    uint32_t rowPerTime;
    uint32_t colPerTime;
    BIAS_TYPE biasType = FindBiasType(context, numRow, numCol);

    auto tilingType = AddLayerNormQuantTilingImpl(
        maxUbSize, dtSize, bufferNum, numCol, firstdimPerCore, enableAdditionalOutput, biasType, rowPerTime, colPerTime);
    PostUbTiling(&tiling, firstdimPerCore, numCol, colPerTime, rowPerTime);

    SetWorkSapce4AddLayerNormQuant(context, &tiling, tilingType, numRow, numCol);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "numCore = " << tiling.get_numCore() << std::endl;
    std::cout << "numFirstDim = " << tiling.get_numFirstDim() << std::endl;
    std::cout << "numLastDim = " << tiling.get_numLastDim() << std::endl;
    std::cout << "numLastDimAlign = " << tiling.get_numLastDimAlign() << std::endl;
    std::cout << "numLastDimAlign32 = " << tiling.get_numLastDimAlign32() << std::endl;
    std::cout << "firstDimPerCore = " << tiling.get_firstDimPerCore() << std::endl;
    std::cout << "firstDimPerCoreTail = " << tiling.get_firstDimPerCoreTail() << std::endl;
    std::cout << "firstDimPerTime = " << tiling.get_firstDimPerTime() << std::endl;
    std::cout << "firstDimPerTimeTail = " << tiling.get_firstDimPerTimeTail() << std::endl;
    std::cout << "rowTailPerBlock = " << tiling.get_rowTailPerBlock() << std::endl;
    std::cout << "rowTailLastBlock = " << tiling.get_rowTailLastBlock() << std::endl;
    std::cout << "lastDimPerTime = " << tiling.get_lastDimPerTime() << std::endl;
    std::cout << "colMoveCnt = " << tiling.get_colMoveCnt() << std::endl;
    std::cout << "colTail = " << tiling.get_colTail() << std::endl;
    std::cout << "gmOffset = " << tiling.get_gmOffset() << std::endl;
    std::cout << "aveFactor = " << tiling.get_aveFactor() << std::endl;
    std::cout << "eps = " << tiling.get_eps() << std::endl;
    std::cout << "isXOut = " << tiling.get_isXOut() << std::endl;
    std::cout << "scaleOffsetMode = " << tiling.get_scaleOffsetMode() << std::endl;
    std::cout << "isPerTensor = " << tiling.get_isPerTensor() << std::endl;
    std::cout << "workspaceSize = " << tiling.get_workspaceSize() << std::endl;
    std::cout << "mulLoopFp32 = " << tiling.get_mulLoopFp32() << std::endl;
    std::cout << "mulTailFp32 = " << tiling.get_mulTailFp32() << std::endl;
    std::cout << "dstRepStrideFp32 = " << tiling.get_dstRepStrideFp32() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint64_t tilingKey = ComputeTilingKey(dataType, isDynamicQuant, tilingType, biasType);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddLayerNormQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4AddLayerNormQuant start");
    auto compileInfo = context->GetCompiledInfo<AddLayerNormQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    compileInfo->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfo->isRegbase =
        (npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_5102);
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSize_ = ubSizePlatform;
    compileInfo->vecRegSize_ = 64;
    compileInfo->blockSize_ = 32;
    OP_LOGD(
        context,
        "aivCoreNum %u, ubSize %lu, blockSize %u, vecRegSize %u, sysWorkspaceSize %u, isRegbase %d",
        compileInfo->aivCoreNum_, compileInfo->ubSize_, compileInfo->blockSize_, compileInfo->vecRegSize_,
        compileInfo->sysWorkspaceSize_, compileInfo->isRegbase);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AddLayerNormQuant(gert::TilingContext* context)
{
    OP_CHECK_IF(
        nullptr == context, OP_LOGE("Tiling4AddLayerNormQuant", "TilingContext is NULL, failed"),
        return ge::GRAPH_FAILED);
    return Tiling4AddLayerNormQuantMembase(context);
}

inline ge::graphStatus GenSimplifiedKey4AddLayerNormQuant(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_LOGD("AddLayerNormQuant", "Enter AddLayerNormQuant genSimplifiedKey.");

    constexpr size_t DEST_MAX = 100;
    constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;

    OP_CHECK_IF(nullptr == context, OP_LOGE("AddLayerNormQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        nullptr == simplifiedKey, OP_LOGE(context, "SimplifiedKey is null"), return ge::GRAPH_FAILED);

    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_IDX));
    int32_t x1Dtype = static_cast<int32_t>(context->GetInputDesc(X1_IDX)->GetDataType());

    int32_t scale1Dtype = -1;
    int32_t scale2Dtype = -1;
    int32_t offset1Dtype = -1;
    int32_t offset2Dtype = -1;
    if (context->GetOptionalInputDesc(SCALE1_IDX) != nullptr) {
        scale1Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(SCALE1_IDX)->GetDataType());
    }
    if (context->GetOptionalInputDesc(SCALE2_IDX) != nullptr) {
        scale2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(SCALE2_IDX)->GetDataType());
    }
    if (context->GetOptionalInputDesc(ZERO_POINT1_IDX) != nullptr) {
        offset1Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINT1_IDX)->GetDataType());
    }
    if (context->GetOptionalInputDesc(ZERO_POINT2_IDX) != nullptr) {
        offset2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINT2_IDX)->GetDataType());
    }

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp
        .append(std::to_string(x1Dtype))
        .append("/")
        .append(std::to_string(x1Dtype))
        .append("/")
        .append(std::to_string(x1Dtype))
        .append("/")
        .append(std::to_string(x1Dtype))
        .append("/")
        .append(std::to_string(x1Dtype))
        .append("/")
        .append(std::to_string(scale1Dtype))
        .append("/")
        .append(std::to_string(scale2Dtype))
        .append("/")
        .append(std::to_string(offset1Dtype))
        .append("/")
        .append(std::to_string(offset2Dtype));
    OP_LOGD(context, "SimpleKeyTemp: %s", simpleKeyTemp.c_str());
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    OP_CHECK_IF(
        (err != 0), OP_LOGE(context, "Error: strcat_s failed with error code %d.", err),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
        OP_LOGE(context, "len of simplifiedKey exceeds max length."), return ge::GRAPH_FAILED);
    OP_LOGD("AddLayerNormQuant", "Finish AddLayerNormQuant genSimplifiedKey.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddLayerNormQuant)
    .Tiling(Tiling4AddLayerNormQuant)
    .TilingParse<AddLayerNormQuantCompileInfo>(TilingPrepare4AddLayerNormQuant)
    .GenSimplifiedKey(GenSimplifiedKey4AddLayerNormQuant);
} // namespace optiling