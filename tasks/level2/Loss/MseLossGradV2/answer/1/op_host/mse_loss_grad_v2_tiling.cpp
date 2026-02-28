#include "mse_loss_grad_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint64_t INPUT_BUFFER_NUM = 4;
constexpr uint64_t FP32_SIZE = 4;
constexpr uint64_t USED_DOUBLE_BUFFER_SIZE = 256 * 256;

static constexpr uint32_t FLOAT_KEY = 1;
static constexpr uint32_t FLOAT16_KEY = 2;

#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

void GetTilingKey(const uint32_t dtypeKey, uint32_t& tilingKey)
{
    tilingKey = dtypeKey;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MseLossGradTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    tiling.set_totalLength(totalLength);
    auto dyDataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = 0;
    uint32_t sizeofDataType = 0;
    uint32_t tmpBufferNum = 4;
    if (ge::DT_FLOAT == dyDataType) {
        sizeofDataType = 4;
        dtypeKey = FLOAT_KEY;
        tmpBufferNum = 0;
    } else if (ge::DT_FLOAT16 == dyDataType) {
        sizeofDataType = 2;
        dtypeKey = FLOAT16_KEY;
    }

    // get attr info
    const char* reduction = context->GetAttrs()->GetStr(0);
    const char* model = "mean";
    float reduceElts = 1.0;
    float cof = 1.0;
    auto shape_reduction = context->GetInputTensor(0)->GetOriginShape();
    int32_t dimNum = shape_reduction.GetDimNum();
    if (strcmp(reduction, model) == 0) {
        for (int32_t i = 0; i < dimNum; i++) {
            reduceElts = reduceElts * shape_reduction.GetDim(i);
        }
        cof = 1.0f / reduceElts * 2.0f;
        tiling.set_cof(cof);
    } else {
        tiling.set_cof(2.0);
    }
    uint32_t tilingKey = 0;
    GetTilingKey(dtypeKey, tilingKey);
    context->SetTilingKey(tilingKey);

    // set numBlocks
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    // ub size
    uint64_t ub_size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // tmp buffertype is fp16 or bf16 need cast fp32
    uint64_t buffersize = INPUT_BUFFER_NUM * sizeofDataType + tmpBufferNum * FP32_SIZE;
    // tile length
    if (buffersize == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t maxTileLength = (ub_size + buffersize - 1) / buffersize;
    // allLoopNumAllign
    if (maxTileLength == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t allLoopNumAllign = (totalLength + maxTileLength - 1) / maxTileLength;
    uint64_t usedCoreNum = coreNum;

    if (allLoopNumAllign <= coreNum) {
        usedCoreNum = allLoopNumAllign;
    }
    uint64_t blockLength = 0;
    uint64_t padLength = totalLength % usedCoreNum;
    context->SetBlockDim(usedCoreNum);
    if (padLength == 0) {
        blockLength = totalLength / usedCoreNum;
    } else {
        usedCoreNum -= 1;
        padLength = totalLength % usedCoreNum;
        blockLength = totalLength / usedCoreNum;
        if (padLength == 0) {
            context->SetBlockDim(usedCoreNum);
        }
    }
    uint64_t usedDB = 0;
    uint64_t TILE_NUM = (allLoopNumAllign + usedCoreNum - 1) / usedCoreNum;
    if (totalLength > USED_DOUBLE_BUFFER_SIZE) {
        usedDB = 1;
    }

    tiling.set_usedDb(usedDB);
    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_padLength(padLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "totalLength = " << tiling.get_totalLength() << std::endl;
    std::cout << "cof = " << tiling.get_cof() << std::endl;
    std::cout << "usedDb = " << tiling.get_usedDb() << std::endl;
    std::cout << "blockLength = " << tiling.get_blockLength() << std::endl;
    std::cout << "tileNum = " << tiling.get_tileNum() << std::endl;
    std::cout << "padLength = " << tiling.get_padLength() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}

struct MseLossGradCompileInfo {
};

static ge::graphStatus TilingParse4MseLossGrad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE(context, "TilingParse4MseLossGrad got context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MseLossGradV2).Tiling(TilingFunc).TilingParse<MseLossGradCompileInfo>(TilingParse4MseLossGrad);
} // namespace optiling