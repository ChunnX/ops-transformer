/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <string>
#include <securec.h>
#include "log/log.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "experimental/attention/sliding_tile_attention/op_kernel/sliding_tile_attention_tiling_data.h"
#include "experimental/attention/sliding_tile_attention/op_kernel/sliding_tile_attention_tiling_key.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q = 0U;
constexpr size_t ATTR_WINDOW_SIZE = 0U;
constexpr size_t ATTR_TEXT_LENGTH = 1U;
constexpr size_t ATTR_HAS_TEXT = 2U;
constexpr size_t ATTR_SEQ_SHAPE = 3U;
constexpr size_t BNSD_RANK = 4U;
constexpr int64_t DEFAULT_TILE_T = 6;
constexpr int64_t DEFAULT_TILE_H = 8;
constexpr int64_t DEFAULT_TILE_W = 8;
constexpr int64_t VECTOR_HEAD_DIM_LIMIT = 128;
constexpr int64_t SUPPORTED_HEAD_DIM = 64;
constexpr int64_t STA_BUFFER_NUM = 2;
constexpr size_t GM_WORKSPACE_ALIGN_BYTES = 512U;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct SlidingTileAttentionCompileInfo {
};

static bool ParseSeqShape(const char *shapeStr, int64_t &seqT, int64_t &seqH, int64_t &seqW)
{
    std::string value = shapeStr == nullptr ? std::string("30x48x80") : std::string(shapeStr);
    std::replace(value.begin(), value.end(), 'x', ' ');
    std::replace(value.begin(), value.end(), 'X', ' ');
    std::stringstream ss(value);
    ss >> seqT >> seqH >> seqW;
    return !ss.fail() && seqT > 0 && seqH > 0 && seqW > 0;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, int64_t &aivCoreNum, int64_t &aicCoreNum,
                                       uint32_t &blockDim)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    aicCoreNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aivCoreNum <= 0 || aicCoreNum <= 0,
                OP_LOGE(context, "GetCoreNum failed, aivCoreNum=%ld, aicCoreNum=%ld.", aivCoreNum, aicCoreNum),
                return ge::GRAPH_FAILED);
    blockDim = ascendcPlatform.CalcTschBlockDim(static_cast<uint32_t>(aivCoreNum),
                                                static_cast<uint32_t>(aicCoreNum),
                                                static_cast<uint32_t>(aivCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillWindow(const gert::RuntimeAttrs *attrs, gert::TilingContext *context,
                                  SlidingTileAttentionTilingData *tiling, int64_t headNum)
{
    auto windowAttr = attrs->GetListInt(ATTR_WINDOW_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, windowAttr);
    const int64_t windowSize = windowAttr->GetSize();
    const int64_t *windowData = windowAttr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, windowData);
    OP_CHECK_IF(windowSize != 3 && windowSize != headNum * 3,
                OP_LOGE(context,
                        "window_size should be [t,h,w] or flattened per-head triples with size head_num*3, got %ld.",
                        windowSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headNum > SLIDING_TILE_ATTENTION_MAX_HEAD_NUM,
                OP_LOGE(context, "head_num %ld exceeds max supported %ld.", headNum,
                        SLIDING_TILE_ATTENTION_MAX_HEAD_NUM),
                return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < SLIDING_TILE_ATTENTION_MAX_HEAD_NUM; ++i) {
        const int64_t base = (windowSize == 3) ? 0 : i * 3;
        const int64_t winT = (i < headNum) ? windowData[base] : 1;
        const int64_t winH = (i < headNum) ? windowData[base + 1] : 1;
        const int64_t winW = (i < headNum) ? windowData[base + 2] : 1;
        OP_CHECK_IF(winT <= 0 || winH <= 0 || winW <= 0,
                    OP_LOGE(context, "window_size values should be positive, got [%ld,%ld,%ld] for head %ld.", winT,
                            winH, winW, i),
                    return ge::GRAPH_FAILED);
        tiling->windowT[i] = winT;
        tiling->windowH[i] = winH;
        tiling->windowW[i] = winW;
    }
    return ge::GRAPH_SUCCESS;
}

static size_t AlignUpSize(size_t value, size_t align)
{
    return ((value + align - 1U) / align) * align;
}
} // namespace

static ge::graphStatus SlidingTileAttentionTilingFunc(gert::TilingContext *context)
{
    int64_t aivCoreNum = 0;
    int64_t aicCoreNum = 0;
    uint32_t taskBlockDim = 0;
    OP_CHECK_IF(GetPlatformInfo(context, aivCoreNum, aicCoreNum, taskBlockDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    auto qShapeHolder = context->GetInputShape(INPUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShapeHolder);
    auto qShape = qShapeHolder->GetStorageShape();
    OP_CHECK_IF(qShape.GetDimNum() != BNSD_RANK,
                OP_LOGE(context, "SlidingTileAttention only supports BNSD rank-4 input, got rank %zu.",
                        qShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t batch = qShape.GetDim(0);
    const int64_t headNum = qShape.GetDim(1);
    const int64_t seqLen = qShape.GetDim(2);
    const int64_t headDim = qShape.GetDim(3);
    OP_CHECK_IF(batch <= 0 || headNum <= 0 || seqLen <= 0 || headDim <= 0,
                OP_LOGE(context, "Static positive BNSD shape is required, got [%ld,%ld,%ld,%ld].", batch, headNum,
                        seqLen, headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headDim > VECTOR_HEAD_DIM_LIMIT,
                OP_LOGE(context, "SlidingTileAttention vector path only supports head_dim <= %ld, got %ld.",
                        VECTOR_HEAD_DIM_LIMIT, headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headDim != SUPPORTED_HEAD_DIM,
                OP_LOGE(context, "This SlidingTileAttention kernel is optimized only for head_dim=%ld, got %ld.",
                        SUPPORTED_HEAD_DIM, headDim),
                return ge::GRAPH_FAILED);

    int64_t seqT = 0;
    int64_t seqH = 0;
    int64_t seqW = 0;
    const char *seqShapeAttr = attrs->GetAttrPointer<char>(ATTR_SEQ_SHAPE);
    OP_CHECK_IF(!ParseSeqShape(seqShapeAttr, seqT, seqH, seqW),
                OP_LOGE(context, "seq_shape should be formatted like 30x48x80."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqT % DEFAULT_TILE_T != 0 || seqH % DEFAULT_TILE_H != 0 || seqW % DEFAULT_TILE_W != 0,
                OP_LOGE(context, "seq_shape must be divisible by tile shape 6x8x8, got %ldx%ldx%ld.", seqT, seqH,
                        seqW),
                return ge::GRAPH_FAILED);

    const bool *hasTextPtr = attrs->GetAttrPointer<bool>(ATTR_HAS_TEXT);
    const int64_t *textLengthPtr = attrs->GetAttrPointer<int64_t>(ATTR_TEXT_LENGTH);
    const bool hasText = hasTextPtr == nullptr ? true : *hasTextPtr;
    const int64_t textLength = hasText ? (textLengthPtr == nullptr ? 0 : *textLengthPtr) : 0;
    const int64_t imageSeqLen = seqT * seqH * seqW;
    OP_CHECK_IF(textLength < 0, OP_LOGE(context, "text_length should be non-negative, got %ld.", textLength),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(imageSeqLen + textLength > seqLen,
                OP_LOGE(context, "image tokens plus text tokens exceed seq_len: %ld + %ld > %ld.", imageSeqLen,
                        textLength, seqLen),
                return ge::GRAPH_FAILED);

    SlidingTileAttentionTilingData *tiling = context->GetTilingData<SlidingTileAttentionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SlidingTileAttentionTilingData), 0, sizeof(SlidingTileAttentionTilingData)) !=
                    EOK,
                OP_LOGE(context, "memset tiling data failed."), return ge::GRAPH_FAILED);

    tiling->batchSize = batch;
    tiling->headNum = headNum;
    tiling->seqLen = seqLen;
    tiling->headDim = headDim;
    tiling->scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    tiling->headDimAlign = ((headDim + 15) / 16) * 16;
    tiling->aicCoreNum = aicCoreNum;
    tiling->aivCoreNum = aivCoreNum;
    tiling->imageSeqLen = imageSeqLen;
    tiling->textLength = textLength;
    tiling->hasText = hasText ? 1 : 0;
    tiling->seqT = seqT;
    tiling->seqH = seqH;
    tiling->seqW = seqW;
    tiling->tileT = DEFAULT_TILE_T;
    tiling->tileH = DEFAULT_TILE_H;
    tiling->tileW = DEFAULT_TILE_W;
    tiling->tileTokenNum = DEFAULT_TILE_T * DEFAULT_TILE_H * DEFAULT_TILE_W;
    tiling->tileNumT = seqT / DEFAULT_TILE_T;
    tiling->tileNumH = seqH / DEFAULT_TILE_H;
    tiling->tileNumW = seqW / DEFAULT_TILE_W;
    tiling->tileNumHW = tiling->tileNumH * tiling->tileNumW;
    tiling->imageTileNum = (imageSeqLen + tiling->tileTokenNum - 1) / tiling->tileTokenNum;
    tiling->validSeqLen = imageSeqLen + textLength;
    const int64_t textTaskNum = textLength > 0 ? 1 : 0;
    const int64_t paddingTaskNum = tiling->validSeqLen < seqLen ? 1 : 0;
    tiling->queryTaskNum = tiling->imageTileNum + textTaskNum + paddingTaskNum;
    tiling->cubeTaskNum = batch * headNum * tiling->queryTaskNum;
    tiling->maxQueryRows = std::max(tiling->tileTokenNum, textLength);
    tiling->maxVisibleKvTokens = std::max(tiling->tileTokenNum, textLength);
    tiling->scoreWorkspaceElems = tiling->maxQueryRows * tiling->maxVisibleKvTokens;
    tiling->probWorkspaceElems = tiling->maxQueryRows * tiling->maxVisibleKvTokens;
    tiling->outWorkspaceElems = tiling->maxQueryRows * tiling->headDimAlign;
    tiling->scoreWorkspaceBytes = static_cast<int64_t>(
        AlignUpSize(static_cast<size_t>(STA_BUFFER_NUM * tiling->scoreWorkspaceElems * sizeof(float)),
                    GM_WORKSPACE_ALIGN_BYTES));
    tiling->probWorkspaceBytes = static_cast<int64_t>(
        AlignUpSize(static_cast<size_t>(STA_BUFFER_NUM * tiling->probWorkspaceElems * sizeof(uint16_t)),
                    GM_WORKSPACE_ALIGN_BYTES));
    tiling->outWorkspaceBytes = static_cast<int64_t>(
        AlignUpSize(static_cast<size_t>(STA_BUFFER_NUM * tiling->outWorkspaceElems * sizeof(float)),
                    GM_WORKSPACE_ALIGN_BYTES));
    tiling->cubeWorkspacePerCore =
        tiling->scoreWorkspaceBytes + tiling->probWorkspaceBytes + tiling->outWorkspaceBytes;
    tiling->totalQueryNum = batch * headNum * seqLen;

    OP_CHECK_IF(FillWindow(attrs, context, tiling, headNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FillWindow failed."), return ge::GRAPH_FAILED);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WS_SYS_SIZE + static_cast<size_t>(aicCoreNum) *
                                     static_cast<size_t>(tiling->cubeWorkspacePerCore);

    context->SetBlockDim(taskBlockDim);

    auto inputDesc = context->GetInputDesc(INPUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(SLIDING_TILE_ATTENTION_FP16_TILING_KEY);
    } else if (dtype == ge::DT_BF16) {
        context->SetTilingKey(SLIDING_TILE_ATTENTION_BF16_TILING_KEY);
    } else {
        OP_LOGE(context, "Unsupported dtype for SlidingTileAttention.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SlidingTileAttentionTilingParse([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SlidingTileAttention)
    .Tiling(SlidingTileAttentionTilingFunc)
    .TilingParse<SlidingTileAttentionCompileInfo>(SlidingTileAttentionTilingParse);
} // namespace optiling
