#include "foreach_pow_list_tiling.h"
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "common_dtype.h"
#include "common/log.h"

namespace optiling {
constexpr uint64_t WORK_SPACE_SIZE = 32; // foreach(vector) not need workspace

constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
constexpr uint32_t UB_DIVIDER_FOR_TEMP_CASTING = 10;
constexpr uint8_t FOREACH_POW_TENSOR_OP_CODE = 8;
constexpr uint8_t FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE = 22;

constexpr uint8_t BYTE_PER_BLOCK = 32;
constexpr int32_t POW_TENSOR_TENSOR_CALC_PROC[9] = {12, 3, 5, 3, 12, 12, 12, 12, 12};

constexpr int32_t MAX_SUPPORT_DIMS_NUMS = 8;

enum class ForeachInputType : uint8_t
{
    TYPE_TENSORS = 0,        // only tensor type
    TYPE_SCALARS_TENSOR = 1, // include scalarlist type
    TYPE_SCALAR = 2          // include scalar type
};

enum class ForeachReturnType : uint8_t
{
    TYPE_BASE = 0,    // inplace or return
    TYPE_INPLACE = 1, // only inplace
};

class ForeachCommonTiling
{
public:
    explicit ForeachCommonTiling(gert::TilingContext* context) : tilingContext(context) {};
    
    ge::graphStatus Init(
        uint8_t theCode = 0, ForeachInputType inputType = ForeachInputType::TYPE_TENSORS,
        ForeachReturnType returnType = ForeachReturnType::TYPE_BASE)
    {
        opCode = theCode;
        opInputType = inputType;
        opReturnType = returnType;
        int dynamicIdx = 0; 
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(dynamicIdx, i);
            if (srcTensor == nullptr) {
                break;
            }

            auto temp = tilingContext->GetInputDesc(0);
            if (temp == nullptr) {
                return ge::GRAPH_FAILED;
            }

            auto srcDtype = temp->GetDataType();

            // Determine whether all data types are consistent.
            if (dataType == ge::DT_UNDEFINED) {
                dataType = srcDtype;
                dataTypeSize = GetDataTypeSize(dataType);
                if (dataTypeSize == 0) {
                    dataTypeSize = BYTE_LEN_4;
                }
                elementsPerBlock = static_cast<uint8_t>(BYTE_BLOCK / dataTypeSize);
            } else if (srcDtype != dataType) {
                return ge::GRAPH_FAILED;
            }
            gert::Shape tempShape = srcTensor->GetStorageShape();
            // Make a 32-byte alignment for each Tensor
            tensorDataCountList[i] = tempShape.GetShapeSize();
            totalDataCount += tensorDataCountList[i];
            totalTensorCount++;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RunBigKernelTiling()
    {
        // checke op info
        OP_CHECK_IF(
            CheckOpInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckOpInfo failed."),
            return ge::GRAPH_FAILED);

        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));

        uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());
        needCoreNum = needCoreNum > MAX_CORE_CONT ? MAX_CORE_CONT : needCoreNum;
        AssignDataToEachCore(static_cast<int64_t>(needCoreNum));
        DivideUbMemory(ubSizePlatForm);
        FillTilingData();
        tilingContext->SetBlockDim(needCoreNum);
        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = WORK_SPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

private:
    ge::graphStatus CheckOutputShapeAndDtype()
    {
        uint64_t inputIndexZero = 0;
        
        for (uint32_t i = 0; i < totalTensorCount; i++) {
            auto tempDesc = tilingContext->GetOutputDesc(i);
            OP_CHECK_IF(
                tempDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u desc is null.", i),
                return ge::GRAPH_FAILED);
            auto dstDtype = tempDesc->GetDataType();
            OP_CHECK_IF(
                dstDtype != dataType,
                OP_LOGE(tilingContext->GetNodeName(), "The tensor %u of output datatype should be same with input.", i),
                return ge::GRAPH_FAILED);
            auto srcShape = tilingContext->GetDynamicInputShape(inputIndexZero, i);
            OP_CHECK_IF(
                srcShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The input %u shape is null.", i),
                return ge::GRAPH_FAILED);
            auto dstShape = tilingContext->GetOutputShape(i);
            OP_CHECK_IF(
                dstShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u shape is null.", i),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                srcShape->GetStorageShape() != dstShape->GetStorageShape() &&
                    srcShape->GetStorageShape().GetShapeSize() > dstShape->GetStorageShape().GetShapeSize(),
                OP_LOGE(
                    tilingContext->GetNodeName(),
                    "The tensor %u of output shape should be same with input."
                    "shape: [%s]", i),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckInputTensorlistShape(size_t inputTensorsNum)
    {
        uint64_t inputIndexZero = 0;
        
        for (size_t tensorIndex = 0; tensorIndex < totalTensorCount; ++tensorIndex) {
            auto x1Shape = tilingContext->GetDynamicInputShape(inputIndexZero, tensorIndex);
            OP_CHECK_IF(
                x1Shape == nullptr,
                OP_LOGE(tilingContext->GetNodeName(), "The input %lu shape is null.", inputIndexZero),
                return ge::GRAPH_FAILED);

            // check max dim
            OP_CHECK_IF(
                x1Shape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
                OP_LOGE(
                    tilingContext->GetNodeName(),
                    "The input %lu shape is invalid, and it cannot be larger than %zu dimensions.", inputIndexZero,
                    static_cast<size_t>(MAX_SUPPORT_DIMS_NUMS)),
                return ge::GRAPH_FAILED);

            // checke tensorlist input shape consistent
            for (size_t listId = static_cast<size_t>(inputIndexZero) + 1U;
                 listId < static_cast<size_t>(inputTensorsNum); ++listId) {
                OP_CHECK_IF(
                    x1Shape->GetStorageShape() !=
                        tilingContext->GetDynamicInputShape(listId, tensorIndex)->GetStorageShape(),
                    OP_LOGE(
                        tilingContext->GetNodeName(), "The input %lu shape should be same with input %lu.", listId,
                        inputIndexZero),
                    return ge::GRAPH_FAILED);
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckInputTensorlistSize(size_t inputTensorsNum)
    {
        uint64_t startIndex = 0;
        
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);

        auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(startIndex);
        OP_CHECK_IF(
            anchorInstanceInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."),
            return ge::GRAPH_FAILED);
        size_t xSize = anchorInstanceInfo->GetInstanceNum();

        for (uint64_t i = startIndex + 1U; i < static_cast<uint64_t>(inputTensorsNum); i++) {
            auto tempAnchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(i);
            OP_CHECK_IF(
                tempAnchorInstanceInfo == nullptr,
                OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."), return ge::GRAPH_FAILED);
            size_t otherSize = tempAnchorInstanceInfo->GetInstanceNum();
            OP_CHECK_IF(
                otherSize != xSize,
                OP_LOGE(
                    tilingContext->GetNodeName(),
                    "The number of input tensors [%lu] should be same with input tensors [0], expect: %lu, actual: %lu",
                    i, xSize, otherSize),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckOpInfo()
    {
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);

        auto inputTensorsNum = computeNodeInfo->GetIrInputsNum();
        
        // check tensors num
        OP_CHECK_IF(
            CheckInputTensorlistSize(static_cast<size_t>(inputTensorsNum)),
            OP_LOGE(tilingContext->GetNodeName(), "CheckInputTensorlistSize failed."), return ge::GRAPH_FAILED);

        // check shape
        OP_CHECK_IF(
            CheckInputTensorlistShape(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckInputTensorlistShape failed."), return ge::GRAPH_FAILED);
        
        OP_CHECK_IF(
            CheckOutputShapeAndDtype() != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckOutputShapeAndDtype failed."), return ge::GRAPH_FAILED);

        // check tensor list dtype
        OP_CHECK_IF(
            CheckTensorListDtype(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckTensorListDtype() failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckTensorListDtype(size_t inputTensorsNum)
    {
        uint64_t startIndex = 0;
        
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);
        for (uint64_t i = startIndex; i < inputTensorsNum; i++) {
            auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(i);
            OP_CHECK_IF(
                anchorInstanceInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."),
                return ge::GRAPH_FAILED);
            for (uint32_t j = 0; j < totalTensorCount; j++) {
                auto tempDesc = tilingContext->GetDynamicInputDesc(i, j);
                OP_CHECK_IF(
                    tempDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The input %u desc is null.", j),
                    return ge::GRAPH_FAILED);
                auto checkDtype = tempDesc->GetDataType();
                // Determine whether all data types are consistent.
                if (dataType == ge::DT_UNDEFINED) {
                    return ge::GRAPH_FAILED;
                } else if (dataType != checkDtype) {
                    OP_LOGE(
                        tilingContext->GetNodeName(),
                        "DataType of all input should be same. The tensor %u of input %lu datatype is not same with "
                        "other input.",
                        j, i);
                    return ge::GRAPH_FAILED;
                }
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform)
    {
        uint32_t tempCoreNum = static_cast<uint32_t>(CeilA2B(totalDataCount, elementsPerBlock));
        if (tempCoreNum == 0U) {
            tempCoreNum = static_cast<uint32_t>(1);
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    void AssignDataToEachCore(int64_t needCoreNum)
    {
        // Kernel the input data according to 32 byte alignment.
        int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
        // Divisible, representing the amount of data each core needs to process.
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
        int64_t remainderCount = blockCount % needCoreNum; // remainder.
        uint16_t coreIndex = 0;
        int64_t dataCount = 0;
        int64_t curCmpCount = 0;
        int64_t cursorPosition = 0;
        tensorStartList[coreIndex] = static_cast<uint16_t>(0);
        tensorStartOffsetList[coreIndex] = 0;
        for (uint16_t i = 0; i < totalTensorCount; i++) {
            // When the remainder is not 0, each kernel index with less than the remainder processes one more block of
            // data.
            if (remainderCount && static_cast<int64_t>(coreIndex) < remainderCount) {
                curCmpCount = tempPerCoreCount + static_cast<int64_t>(elementsPerBlock);
            } else {
                curCmpCount = tempPerCoreCount;
            }
            int64_t tempCount = tensorDataCountList[i] - cursorPosition;

            if (dataCount + tempCount < curCmpCount) {
                dataCount += tempCount;
                cursorPosition = 0;
                continue;
            }
            // dataCount >= curCmpCount, Calculate the offset
            tensorEndList[coreIndex] = i;
            cursorPosition = cursorPosition + curCmpCount - dataCount;
            tensorEndOffsetList[coreIndex] = cursorPosition - 1;
            dataCount = 0;
            coreIndex++;
            if (cursorPosition < tensorDataCountList[i]) {
                tensorStartList[coreIndex] = i;
                tensorStartOffsetList[coreIndex] = cursorPosition;
                --i; // The next loop continues to allocate the current tensor
            } else if (static_cast<int64_t>(coreIndex) != needCoreNum) {
                tensorStartList[coreIndex] = static_cast<uint16_t>(i + 1);
                tensorStartOffsetList[coreIndex] = 0;
                cursorPosition = 0;
            }
        }
        /* The temporary count variable is not 0, which means that the last tensor is truncated,
            and you need to manually set the offset of the last core. */
        if (dataCount) {
            tensorEndList[coreIndex] = static_cast<uint16_t>(totalTensorCount - 1);
            tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
        }
    }

    void DivideUbMemory(uint64_t ubSizePlatForm)
    {
        DivideUbMemory2(ubSizePlatForm);
    }

    void DivideUbMemory2(uint64_t ubSizePlatForm)
    {
        // foreach_pow_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        uint32_t canUseUbSize;
        if (dataType == ge::DT_BF16) {
            canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER * UB_DIVIDER_FOR_TEMP_CASTING +
                                        POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType) - 1]);
        } else {
            canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER +
                                        POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType) - 1]);
        }
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }

    void FillTilingData()
    {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        tilingData.SaveToBuffer(
            tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

private:
    ForeachCommonTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
    uint8_t opCode = 0;
    ForeachInputType opInputType = ForeachInputType::TYPE_TENSORS;
    ForeachReturnType opReturnType = ForeachReturnType::TYPE_BASE;
};

static ge::graphStatus Tiling4ForeachPowListTiling(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_POW_TENSOR_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachPowList)
    .Tiling(Tiling4ForeachPowListTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);
} // namespace optiling