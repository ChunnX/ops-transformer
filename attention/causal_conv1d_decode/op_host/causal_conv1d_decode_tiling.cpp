#include "causal_conv1d_decode_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 1;
constexpr int64_t CONV_STATES_INDEX = 2;
constexpr int64_t QUERY_START_LOC_INDEX = 3;
constexpr int64_t CACHE_INDICES_INDEX = 4;
constexpr int64_t BIAS_INDEX = 5;
constexpr int64_t NUM_ACCEPTED_TOKENS_INDEX = 6;
constexpr int64_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int64_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int64_t ATTR_CONV_MODE_INDEX = 2;
constexpr int64_t MAX_WIDTH = 4;
constexpr int64_t MIN_SEQ_LEN = 1;
constexpr int64_t MAX_SEQ_LEN = 4;
constexpr int64_t MIN_DIM = 64;
constexpr int64_t MAX_DIM = 16384;
constexpr int64_t DIM_ALIGN = 16;
constexpr int64_t DIM_FINE = 128;
constexpr int64_t DIM_COARSE = 1024;
constexpr int64_t MAX_BATCH = 256;
constexpr int64_t DTYPE_SIZE = 2;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t RESERVED_UB = 8 * 1024;
constexpr uint64_t TILING_KEY_BH_BF16 = 20000;

struct CausalConv1dDecodeCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct TilingWork {
    int64_t totalCoreNum = 1;
    int64_t ubSize = 192 * 1024;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t dim = 0;
    int64_t kernelWidth = 0;
    int64_t stateLen = 0;
    int64_t xStride = 0;
    int64_t cacheStride0 = 0;
    int64_t cacheStride1 = 0;
    int64_t padSlotId = -1;
    int64_t xInputMode = 0;
    bool hasBias = false;
    bool hasQueryStartLoc = false;
    bool hasCacheIndices = false;
    bool hasAccepted = false;
    CausalConv1dDecodeCutBHTilingData td{};
};

static int64_t AlignUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

static int64_t GetDim(const gert::Shape &shape, int32_t idx)
{
    return shape.GetDim(idx);
}

static bool HasInput(gert::TilingContext *ctx, int64_t idx)
{
    auto shapePtr = ctx->GetOptionalInputShape(idx);
    if (shapePtr == nullptr) {
        return false;
    }
    const auto &shape = shapePtr->GetOriginShape();
    if (shape.GetDimNum() == 0) {
        return false;
    }
    return !(shape.GetDimNum() == 1 && GetDim(shape, 0) <= 0);
}

static ge::graphStatus GetPlatform(gert::TilingContext *ctx, TilingWork &w)
{
    auto platformInfo = ctx->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfo = reinterpret_cast<const CausalConv1dDecodeCompileInfo *>(ctx->GetCompileInfo());
        if (compileInfo != nullptr && compileInfo->coreNum > 0 && compileInfo->ubSize > 0) {
            w.totalCoreNum = static_cast<int64_t>(compileInfo->coreNum);
            w.ubSize = static_cast<int64_t>(compileInfo->ubSize);
        }
        return ge::GRAPH_SUCCESS;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    w.totalCoreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    w.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(w.totalCoreNum <= 0 || w.ubSize <= 0, OP_LOGE(ctx->GetNodeName(), "invalid platform core/ub"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReadShapeAndAttrs(gert::TilingContext *ctx, TilingWork &w)
{
    auto xShapePtr = ctx->GetInputShape(X_INDEX);
    auto weightShapePtr = ctx->GetInputShape(WEIGHT_INDEX);
    auto stateShapePtr = ctx->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_IF(xShapePtr == nullptr || weightShapePtr == nullptr || stateShapePtr == nullptr,
                OP_LOGE(ctx->GetNodeName(), "required input shape is null"), return ge::GRAPH_FAILED);
    const auto &x = xShapePtr->GetOriginShape();
    const auto &weight = weightShapePtr->GetOriginShape();
    const auto &state = stateShapePtr->GetOriginShape();
    OP_CHECK_IF((x.GetDimNum() != 2 && x.GetDimNum() != 3) || weight.GetDimNum() != 2 || state.GetDimNum() != 3,
                OP_LOGE(ctx->GetNodeName(), "rank mismatch"), return ge::GRAPH_FAILED);
    w.hasQueryStartLoc = HasInput(ctx, QUERY_START_LOC_INDEX);
    if (x.GetDimNum() == 3) {
        w.xInputMode = 0;
        w.batch = GetDim(x, 0);
        w.seqLen = GetDim(x, 1);
        w.dim = GetDim(x, 2);
    } else {
        w.dim = GetDim(x, 1);
        w.seqLen = MAX_SEQ_LEN;
        if (w.hasQueryStartLoc) {
            w.xInputMode = 1;
            auto qShapePtr = ctx->GetOptionalInputShape(QUERY_START_LOC_INDEX);
            OP_CHECK_IF(qShapePtr == nullptr || qShapePtr->GetOriginShape().GetDimNum() != 1 ||
                            GetDim(qShapePtr->GetOriginShape(), 0) < 2,
                        OP_LOGE(ctx->GetNodeName(), "query_start_loc must be [batch + 1]"), return ge::GRAPH_FAILED);
            w.batch = GetDim(qShapePtr->GetOriginShape(), 0) - 1;
        } else {
            w.xInputMode = 2;
            w.batch = GetDim(x, 0);
            w.seqLen = 1;
        }
    }
    w.kernelWidth = GetDim(weight, 0);
    w.stateLen = GetDim(state, 1);
    w.hasAccepted = HasInput(ctx, NUM_ACCEPTED_TOKENS_INDEX);
    OP_CHECK_IF(w.batch < 1 || w.batch > MAX_BATCH, OP_LOGE(ctx->GetNodeName(), "batch out of range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(w.seqLen < MIN_SEQ_LEN || w.seqLen > MAX_SEQ_LEN, OP_LOGE(ctx->GetNodeName(), "seqLen must be 1..4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(w.dim < MIN_DIM || w.dim > MAX_DIM || w.dim % DIM_ALIGN != 0,
                OP_LOGE(ctx->GetNodeName(), "dim range/alignment mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(w.kernelWidth != MAX_WIDTH || GetDim(weight, 1) != w.dim,
                OP_LOGE(ctx->GetNodeName(), "weight must be [4, dim]"), return ge::GRAPH_FAILED);
    int64_t minStateLen = MAX_WIDTH - 1;
    if (w.hasAccepted) {
        minStateLen = std::max<int64_t>(minStateLen, w.seqLen + 2);
    }
    OP_CHECK_IF(GetDim(state, 2) != w.dim || w.stateLen < minStateLen,
                OP_LOGE(ctx->GetNodeName(), "state shape mismatch"), return ge::GRAPH_FAILED);
    w.hasCacheIndices = HasInput(ctx, CACHE_INDICES_INDEX);
    if (w.hasCacheIndices) {
        auto cacheShapePtr = ctx->GetOptionalInputShape(CACHE_INDICES_INDEX);
        OP_CHECK_IF(cacheShapePtr == nullptr || cacheShapePtr->GetOriginShape().GetDimNum() != 1 ||
                        GetDim(cacheShapePtr->GetOriginShape(), 0) != w.batch,
                    OP_LOGE(ctx->GetNodeName(), "cache_indices must be [batch]"), return ge::GRAPH_FAILED);
    }
    auto xDesc = ctx->GetInputDesc(X_INDEX);
    auto weightDesc = ctx->GetInputDesc(WEIGHT_INDEX);
    auto stateDesc = ctx->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_IF(xDesc->GetDataType() != ge::DT_BF16 || weightDesc->GetDataType() != ge::DT_BF16 ||
                    stateDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE(ctx->GetNodeName(), "v1 supports BF16 only"), return ge::GRAPH_FAILED);

    w.xStride = w.dim;
    w.cacheStride0 = w.stateLen * w.dim;
    w.cacheStride1 = w.dim;
    w.hasBias = HasInput(ctx, BIAS_INDEX);
    if (w.hasBias) {
        auto biasShapePtr = ctx->GetOptionalInputShape(BIAS_INDEX);
        OP_CHECK_IF(biasShapePtr == nullptr || biasShapePtr->GetOriginShape().GetDimNum() != 1 ||
                        GetDim(biasShapePtr->GetOriginShape(), 0) != w.dim,
                    OP_LOGE(ctx->GetNodeName(), "bias must be [dim]"), return ge::GRAPH_FAILED);
    }
    if (w.hasAccepted) {
        auto accShapePtr = ctx->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
        OP_CHECK_IF(accShapePtr == nullptr || accShapePtr->GetOriginShape().GetDimNum() != 1 ||
                        GetDim(accShapePtr->GetOriginShape(), 0) != w.batch,
                    OP_LOGE(ctx->GetNodeName(), "num_accepted_tokens must be [batch]"), return ge::GRAPH_FAILED);
    }
    auto attrs = ctx->GetAttrs();
    if (attrs != nullptr && attrs->GetInt(ATTR_ACTIVATION_MODE_INDEX) != nullptr) {
        OP_CHECK_IF(*attrs->GetInt(ATTR_ACTIVATION_MODE_INDEX) != 0,
                    OP_LOGE(ctx->GetNodeName(), "v1 supports activation_mode=0 only"), return ge::GRAPH_FAILED);
    }
    if (attrs != nullptr && attrs->GetInt(ATTR_PAD_SLOT_ID_INDEX) != nullptr) {
        w.padSlotId = *attrs->GetInt(ATTR_PAD_SLOT_ID_INDEX);
    }
    if (attrs != nullptr && attrs->GetInt(ATTR_CONV_MODE_INDEX) != nullptr) {
        OP_CHECK_IF(*attrs->GetInt(ATTR_CONV_MODE_INDEX) != 0,
                    OP_LOGE(ctx->GetNodeName(), "v1 supports conv_mode=0 only"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static void SplitInterCore(TilingWork &w)
{
    int64_t grain = (w.dim >= DIM_COARSE) ? DIM_COARSE : DIM_FINE;
    int64_t blocks = std::max<int64_t>(1, w.dim / grain);
    w.td.dimRemainderElems = w.dim % grain;
    int64_t limited = std::max<int64_t>(1, std::min<int64_t>(w.totalCoreNum, w.batch * blocks));
    int64_t bestDim = 1;
    int64_t bestBatch = 1;
    int64_t bestUsed = 1;
    for (int64_t dc = std::min(blocks, limited); dc >= 1; --dc) {
        int64_t bc = std::min<int64_t>(w.batch, limited / dc);
        if (bc <= 0) {
            continue;
        }
        int64_t used = dc * bc;
        if (used > bestUsed || (used == bestUsed && dc > bestDim)) {
            bestDim = dc;
            bestBatch = bc;
            bestUsed = used;
        }
        if (bestUsed == limited) {
            break;
        }
    }
    w.td.dimCoreCnt = bestDim;
    int64_t base = (w.dim / grain) / bestDim;
    int64_t rem = (w.dim / grain) % bestDim;
    w.td.dimMainCoreCnt = (rem == 0) ? bestDim : rem;
    w.td.dimTailCoreCnt = (rem == 0) ? 0 : bestDim - rem;
    w.td.mainCoreDimLen = ((rem == 0) ? base : base + 1) * grain;
    w.td.tailCoreDimLen = (base == 0) ? w.td.mainCoreDimLen : base * grain;
    w.td.batchCoreCnt = bestBatch;
    int64_t bsBase = w.batch / bestBatch;
    int64_t bsRem = w.batch % bestBatch;
    w.td.batchMainCoreCnt = (bsRem == 0) ? bestBatch : bsRem;
    w.td.batchTailCoreCnt = (bsRem == 0) ? 0 : bestBatch - bsRem;
    w.td.mainCoreBatchNum = (bsRem == 0) ? bsBase : bsBase + 1;
    w.td.tailCoreBatchNum = bsBase;
    w.td.usedCoreNum = bestDim * bestBatch;
}

static void ComputeUbFor(TilingWork &w, int64_t coreDim, int64_t coreBS, int64_t &ubDim, int64_t &ubBS,
                         int64_t &loopDim, int64_t &loopBS, int64_t &tailDim, int64_t &tailBS)
{
    int64_t fixed = AlignUp((w.batch + 1) * sizeof(int32_t), ALIGN_BYTES) * 3;
    int64_t avail = std::max<int64_t>(DIM_ALIGN * DTYPE_SIZE, w.ubSize - RESERVED_UB - fixed);
    int64_t grain = (w.dim >= DIM_COARSE) ? DIM_COARSE : DIM_FINE;
    int64_t historyRows = MAX_WIDTH - 1;
    int64_t perDim = (MAX_WIDTH + 1 + historyRows * 2 + coreBS * w.seqLen * 2) * DTYPE_SIZE;
    int64_t maxDim = std::max<int64_t>(DIM_ALIGN, (avail / std::max<int64_t>(1, perDim) / DIM_ALIGN) * DIM_ALIGN);
    ubDim = std::min<int64_t>(coreDim, std::min<int64_t>(maxDim, grain));
    int64_t perBatch = w.seqLen * ubDim * DTYPE_SIZE * 2;
    int64_t fixedDim = (MAX_WIDTH + 1 + historyRows * 2) * ubDim * DTYPE_SIZE;
    ubBS = std::min<int64_t>(coreBS, std::max<int64_t>(1, (avail - fixedDim) / std::max<int64_t>(1, perBatch)));
    loopDim = (coreDim + ubDim - 1) / ubDim;
    tailDim = (loopDim == 1) ? ubDim : coreDim - (loopDim - 1) * ubDim;
    loopBS = (coreBS + ubBS - 1) / ubBS;
    tailBS = (loopBS == 1) ? ubBS : coreBS - (loopBS - 1) * ubBS;
}

static void FillUb(TilingWork &w)
{
    ComputeUbFor(w, w.td.mainCoreDimLen, w.td.mainCoreBatchNum, w.td.ubMainFactorDim, w.td.ubMainFactorBS,
                 w.td.loopNumDim, w.td.loopNumBS, w.td.ubTailFactorDim, w.td.ubTailFactorBS);
    int64_t tailDim = (w.td.dimTailCoreCnt > 0) ? w.td.tailCoreDimLen : w.td.mainCoreDimLen;
    int64_t tailBS = (w.td.batchTailCoreCnt > 0) ? w.td.tailCoreBatchNum : w.td.mainCoreBatchNum;
    ComputeUbFor(w, tailDim, tailBS, w.td.tailBlockUbFactorDim, w.td.tailBlockUbFactorBS,
                 w.td.tailBlockLoopNumDim, w.td.tailBlockLoopNumBS, w.td.tailBlockUbTailFactorDim,
                 w.td.tailBlockUbTailFactorBS);
    int64_t lastDim = tailDim + w.td.dimRemainderElems;
    int64_t tmpBS = 0;
    int64_t tmpLoopBS = 0;
    int64_t tmpTailBS = 0;
    ComputeUbFor(w, lastDim, w.td.mainCoreBatchNum, w.td.lastCoreUbMainFactorDim, tmpBS,
                 w.td.lastCoreLoopNumDim, tmpLoopBS, w.td.lastCoreUbTailFactorDim, tmpTailBS);
}

static ge::graphStatus TilingCausalConv1dDecode(gert::TilingContext *ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE("CausalConv1dDecode", "context is null"), return ge::GRAPH_FAILED);
    TilingWork w;
    OP_CHECK_IF(GetPlatform(ctx, w) != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "platform failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadShapeAndAttrs(ctx, w) != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "shape/attr failed"),
                return ge::GRAPH_FAILED);
    SplitInterCore(w);
    FillUb(w);
    w.td.batchSize = w.batch;
    w.td.seqLen = w.seqLen;
    w.td.dim = w.dim;
    w.td.kernelWidth = w.kernelWidth;
    w.td.stateLen = w.stateLen;
    w.td.xStride = w.xStride;
    w.td.cacheStride0 = w.cacheStride0;
    w.td.cacheStride1 = w.cacheStride1;
    w.td.padSlotId = w.padSlotId;
    w.td.xInputMode = w.xInputMode;
    w.td.hasBias = w.hasBias ? 1 : 0;
    w.td.hasQueryStartLoc = w.hasQueryStartLoc ? 1 : 0;
    w.td.hasCacheIndices = w.hasCacheIndices ? 1 : 0;
    w.td.hasNumAcceptedTokens = w.hasAccepted ? 1 : 0;
    auto raw = ctx->GetRawTilingData();
    OP_CHECK_IF(raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(w.td),
                OP_LOGE(ctx->GetNodeName(), "raw tiling invalid"), return ge::GRAPH_FAILED);
    std::memcpy(raw->GetData(), &w.td, sizeof(w.td));
    raw->SetDataSize(sizeof(w.td));
    ctx->SetBlockDim(w.td.usedCoreNum);
    ctx->SetTilingKey(TILING_KEY_BH_BF16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareCausalConv1dDecode(gert::TilingParseContext *ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE("CausalConv1dDecode", "context is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1dDecode)
    .Tiling(TilingCausalConv1dDecode)
    .TilingParse<CausalConv1dDecodeCompileInfo>(TilingPrepareCausalConv1dDecode);
} // namespace
} // namespace optiling
