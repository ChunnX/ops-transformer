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

static ge::graphStatus GetPlatformInfo(gert::TilingContext* ctx, uint64_t& ubSize, int64_t& coreNum)
{
    auto* pi = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, pi);
    auto plat = platform_ascendc::PlatformAscendC(pi);
    coreNum = plat.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(ctx, "coreNum is 0"), return ge::GRAPH_FAILED);
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(ctx, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize; int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto hsShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
    int64_t batch = hsShape.GetDim(0), seqLen = hsShape.GetDim(1), hiddenSize = hsShape.GetDim(2);

    auto wtShape = EnsureNotScalar(context->GetInputShape(1)->GetStorageShape());
    int64_t fusedDim = wtShape.GetDim(1);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t qDim = *attrs->GetAttrPointer<int64_t>(0);
    int64_t kDim = *attrs->GetAttrPointer<int64_t>(1);
    int64_t vDim = *attrs->GetAttrPointer<int64_t>(2);
    bool hasBias = (context->GetInputShape(2) != nullptr);

    int64_t M = batch * seqLen;
    int64_t K = hiddenSize;
    int64_t N = fusedDim;

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(plat); 
    
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::LCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);  
    
    int64_t alignedM = CeilDiv(M, 16) * 16;
    int64_t alignedN = CeilDiv(N, 16) * 16;

    cubeTiling.SetShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(K));
    cubeTiling.SetOrgShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(K));
    cubeTiling.SetBias(hasBias);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    cubeTiling.SetShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(K));
    cubeTiling.SetOrgShape(static_cast<int32_t>(alignedM), static_cast<int32_t>(alignedN), static_cast<int32_t>(K));
    cubeTiling.SetBias(hasBias);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    cubeTiling.SetDim(1);

    auto* td = context->GetTilingData<AclnnFusedQkvProjectionTilingData>();
    memset_s(td, sizeof(AclnnFusedQkvProjectionTilingData), 0, sizeof(AclnnFusedQkvProjectionTilingData));
    if (cubeTiling.GetTiling(td->cubeTiling) != 0) {
        OP_LOGE(context, "GetTiling failed");
        return ge::GRAPH_FAILED;
    }

    // 如果 baseN < N，GetTensorC fullC=true 会截断列。尝试用 SetFixSplit 强制单 N-tile
    if (td->cubeTiling.baseN < static_cast<int32_t>(N)) {
        cubeTiling.SetFixSplit(static_cast<int32_t>(1), static_cast<int32_t>(1));
        if (cubeTiling.GetTiling(td->cubeTiling) != 0) {
            OP_LOGE(context, "GetTiling(fixsplit) failed");
            return ge::GRAPH_FAILED;
        }
    }

    constexpr uint32_t BLOCK_DIM = 1;

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = plat.GetLibApiWorkSpaceSize();

    td->M = static_cast<int32_t>(M);
    td->N = static_cast<int32_t>(N);
    td->K = static_cast<int32_t>(K);
    td->singleCoreM = td->cubeTiling.singleCoreM;
    td->singleCoreN = td->cubeTiling.singleCoreN; 
    td->baseM = td->cubeTiling.baseM;
    td->baseN = td->cubeTiling.baseN;
    td->baseK = td->cubeTiling.baseK;
    td->qDim = static_cast<int32_t>(qDim);
    td->kDim = static_cast<int32_t>(kDim);
    td->vDim = static_cast<int32_t>(vDim);
    td->hasBias = hasBias;
    td->blockDim = BLOCK_DIM;
    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

// ======= 【修复 3：沿用原本规范的 IMPL 注册宏】 =======
IMPL_OP_OPTILING(AclnnFusedQkvProjection)
    .Tiling(TilingFunc)
    .TilingParse<CompileInfo>(TilingParse);

} // namespace optiling