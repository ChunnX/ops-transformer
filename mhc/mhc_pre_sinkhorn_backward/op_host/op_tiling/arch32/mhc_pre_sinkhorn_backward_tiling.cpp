#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_sinkhorn_backward_tiling.h"

#define CHECK_NULLPTR(ptr)       \
    if (ptr == nullptr) {        \
        return ge::GRAPH_FAILED; \
    }

#define ADD_TILING_DATA(context, tiling)                                                                     \
    CHECK_NULLPTR(context->GetRawTilingData())                                                               \
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity()); \
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

namespace {
constexpr uint8_t INPUT_X_IDX = 0;

constexpr uint8_t BATCH_SIZE_DIM_IDX = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX = 1;
constexpr uint8_t N_DIM_IDX = 2;
constexpr uint8_t C_DIM_IDX = 3;
constexpr uint8_t ITER_COUNT_IDX = 0;
}

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {

ge::graphStatus TilingMhcPreSinkhornBackward(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();

    ubSize = ubSize - 40 * 1024;
    context->SetDynUBufSize(ubSize);

    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    const auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    CHECK_NULLPTR(xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    auto iterCountPtr = attrsPtr->GetAttrPointer<int32_t>(ITER_COUNT_IDX);
    CHECK_NULLPTR(iterCountPtr);

    int64_t batchSize = xShape.GetDim(BATCH_SIZE_DIM_IDX);
    int64_t seqLength = xShape.GetDim(SEQ_LENGTH_DIM_IDX);
    int64_t n = xShape.GetDim(N_DIM_IDX);
    int64_t c = xShape.GetDim(C_DIM_IDX);
    int64_t c0 = 256;    // must be VL
    int64_t c1 = c / c0;
    int64_t cTail = c % c0;
    int64_t cTailunaglin = cTail % 64 + 64 * cTail / (64 + 128);
    // int64_t tileGradY = 40;
    // int64_t tileHHat2 = 32;
    int64_t tile = 32;
    int64_t skIterCount = *iterCountPtr;

    // int64_t mm1K = n * n + 2 * n;
    // int64_t mm1M = tileHHat2 * 2;
    // int64_t mm1N = n * c;

    // int64_t mm2K = tileHHat2 * 2;
    // int64_t mm2M = n * n + 2 * n;
    // int64_t mm2N = n * c;

    int64_t mm1K = n * n + 2 * n;
    int64_t mm1M = tile * 2;
    int64_t mm1N = n * c;

    int64_t mm2K = tile * 2;
    int64_t mm2M = n * n + 2 * n;
    int64_t mm2N = n * c;

    MhcPreSinkhornBackwardTilingData tilingData;
    auto featureDataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData.mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetOrgShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    if (mm2Tiling.GetTiling(tilingData.mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    tilingData.set_n(n);
    tilingData.set_batchSize(batchSize);
    tilingData.set_seqLength(seqLength);
    tilingData.set_c(c);
    tilingData.set_cTail(cTail);
    tilingData.set_n(n);
    tilingData.set_c0(c0);
    tilingData.set_c1(c1);
    tilingData.set_aivNum(aivNum);
    // tilingData.set_tileGradY(tileGradY);
    // tilingData.set_tileHHat2(tileHHat2);
    tilingData.set_skIterCount(skIterCount);

    tilingData.set_tileSize(tile);

    ADD_TILING_DATA(context, tilingData);

    size_t gradPreWorkspace = batchSize * seqLength * n * sizeof(float);
    size_t gradHat2Workspace = batchSize * seqLength * (n * n + 2 * n) * sizeof(float);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = gradPreWorkspace + gradHat2Workspace;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMhcPreSinkhornBackward(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

struct MhcPreSinkhornBackwardCompileInfo {};

IMPL_OP_OPTILING(MhcPreSinkhornBackward)
    .Tiling(TilingMhcPreSinkhornBackward)
    .TilingParse<MhcPreSinkhornBackwardCompileInfo>(TilingParseForMhcPreSinkhornBackward); // 向框架注册入口函数
} // namespace optiling