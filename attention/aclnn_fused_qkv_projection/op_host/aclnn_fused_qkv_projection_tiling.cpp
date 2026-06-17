#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "lib/tiling_api.h"
#include "../op_kernel/aclnn_fused_qkv_projection_tiling_data.h"
#include "../op_kernel/aclnn_fused_qkv_projection_tiling_key.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace matmul_tiling;

constexpr uint32_t MAX_BLOCK_DIM = 8;

inline int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

struct CompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* ctx, uint64_t& ubSize, int64_t& coreNum) {
    auto* pi = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, pi);
    auto plat = platform_ascendc::PlatformAscendC(pi);
    coreNum = plat.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(ctx, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(ctx, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct TilingInput {
    int64_t M, N, K;
    int64_t qDim, kDim, vDim;
    bool hasBias;
    int32_t dtype; // ge::DT_FLOAT=0, ge::DT_FLOAT16=1
};

static ge::graphStatus ParseTilingInputs(gert::TilingContext* context, TilingInput* in) {
    auto hsShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    int64_t batch = hsShape.GetDim(0);
    int64_t seqLen = hsShape.GetDim(1);
    int64_t hiddenSize = hsShape.GetDim(2);

    auto wtShape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
    int64_t fusedDim = wtShape.GetDim(1);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    in->qDim = *attrs->GetAttrPointer<int64_t>(0);
    in->kDim = *attrs->GetAttrPointer<int64_t>(1);
    in->vDim = *attrs->GetAttrPointer<int64_t>(2);
    in->hasBias = (context->GetInputShape(2) != nullptr);
    in->M = batch * seqLen;
    in->K = hiddenSize;
    in->N = fusedDim;

    in->dtype = static_cast<int32_t>(context->GetInputTensor(0)->GetDataType());
    return ge::GRAPH_SUCCESS;
}

static void ConfigCubeTiling(
    matmul_tiling::MultiCoreMatmulTiling* ct, const TilingInput& in)
{
    auto dt = (in.dtype == static_cast<int32_t>(ge::DT_FLOAT16)) ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_FLOAT;
    
    ct->SetAType(TPosition::GM, CubeFormat::ND, dt);
    ct->SetBType(TPosition::GM, CubeFormat::ND, dt);
    ct->SetCType(TPosition::LCM, CubeFormat::ND, dt);
    ct->SetBiasType(TPosition::GM, CubeFormat::ND, dt);

    int64_t alignedM = CeilDiv(in.M, 16) * 16;
    int64_t alignedN = CeilDiv(in.N, 16) * 16;

    if (dt == matmul_tiling::DataType::DT_FLOAT16 && alignedN == 32) {
        alignedN = 48;
    }

    if (dt == matmul_tiling::DataType::DT_FLOAT16) {
        ct->SetMadType(matmul_tiling::MadType::USE_FP32_MAD);
    }
    
    ct->SetShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(in.K));
    ct->SetOrgShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(in.K));
    ct->SetBias(in.hasBias);
    ct->SetBufferSpace(-1, -1, -1);
    ct->SetDim(1);
}

static ge::graphStatus DoCubeTiling(
    gert::TilingContext* context, matmul_tiling::MultiCoreMatmulTiling* ct,
    AclnnFusedQkvProjectionTilingData* td, int32_t N)
{
    memset_s(td, sizeof(AclnnFusedQkvProjectionTilingData), 0, sizeof(AclnnFusedQkvProjectionTilingData));
    if (ct->GetTiling(td->cubeTiling) != 0) {
        OP_LOGE(context, "GetTiling failed");
        return ge::GRAPH_FAILED;
    }

    if (td->cubeTiling.baseN < N) {
        ct->SetFixSplit(static_cast<int32_t>(1), static_cast<int32_t>(1));
        if (ct->GetTiling(td->cubeTiling) != 0) {
            OP_LOGE(context, "GetTiling(fixsplit) failed");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static void FillTilingOutputs(
    gert::TilingContext* context, AclnnFusedQkvProjectionTilingData* td,
    const TilingInput& in, platform_ascendc::PlatformAscendC& plat)
{
    constexpr uint32_t BLOCK_DIM = 1;
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = plat.GetLibApiWorkSpaceSize();

    td->M = static_cast<int32_t>(in.M);
    td->N = static_cast<int32_t>(in.N);
    td->K = static_cast<int32_t>(in.K);
    td->singleCoreM = td->cubeTiling.singleCoreM;
    td->singleCoreN = td->cubeTiling.singleCoreN;
    td->baseM = td->cubeTiling.baseM;
    td->baseN = td->cubeTiling.baseN;
    td->baseK = td->cubeTiling.baseK;
    td->qDim = static_cast<int32_t>(in.qDim);
    td->kDim = static_cast<int32_t>(in.kDim);
    td->vDim = static_cast<int32_t>(in.vDim);
    td->hasBias = in.hasBias;
    td->dtype = in.dtype;
    td->blockDim = BLOCK_DIM;

    context->SetBlockDim(BLOCK_DIM);
    uint64_t dtypeVal = (in.dtype == static_cast<int32_t>(ge::DT_FLOAT16)) ? TPL_DTYPE_FLOAT16 : TPL_DTYPE_FLOAT;
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0, dtypeVal));
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    TilingInput in;
    OP_CHECK_IF(ParseTilingInputs(context, &in) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ParseTilingInputs error"), return ge::GRAPH_FAILED);

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(plat);
    ConfigCubeTiling(&cubeTiling, in);

    auto* td = context->GetTilingData<AclnnFusedQkvProjectionTilingData>();
    OP_CHECK_IF(DoCubeTiling(context, &cubeTiling, td, static_cast<int32_t>(in.N)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "DoCubeTiling error"), return ge::GRAPH_FAILED);

    FillTilingOutputs(context, td, in, plat);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(AclnnFusedQkvProjection)
    .Tiling(TilingFunc)
    .TilingParse<CompileInfo>(TilingParse);
} // namespace optiling
