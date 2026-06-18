#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {

static constexpr size_t EPILOGUE_INPUT_CONTEXT_INDEX = 0;
static constexpr size_t EPILOGUE_INPUT_DST_SLOT_IDX_INDEX = 1;
static constexpr size_t EPILOGUE_INPUT_NUM_RECV_PER_RANK_INDEX = 2;
static constexpr size_t EPILOGUE_INPUT_NUM_RECV_PER_EXPERT_INDEX = 3;
static constexpr size_t EPILOGUE_INPUT_CACHED_RECV_SRC_METADATA_INDEX = 4;

static constexpr size_t EPILOGUE_OUTPUT_RECV_X_INDEX = 0;
static constexpr size_t EPILOGUE_OUTPUT_RECV_TOPK_IDX_INDEX = 1;
static constexpr size_t EPILOGUE_OUTPUT_RECV_TOPK_WEIGHTS_INDEX = 2;
static constexpr size_t EPILOGUE_OUTPUT_RECV_SRC_METADATA_INDEX = 3;
static constexpr size_t EPILOGUE_OUTPUT_RECV_SCALES_INDEX = 4;

static constexpr size_t EPILOGUE_ATTR_EP_WORLD_SIZE_INDEX = 0;
static constexpr size_t EPILOGUE_ATTR_EP_RANK_ID_INDEX = 1;
static constexpr size_t EPILOGUE_ATTR_NUM_EXPERTS_INDEX = 2;
static constexpr size_t EPILOGUE_ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 3;
static constexpr size_t EPILOGUE_ATTR_CCL_BUFFER_SIZE_INDEX = 4;
static constexpr size_t EPILOGUE_ATTR_EXPERT_ALIGNMENT_INDEX = 5;

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;

static ge::graphStatus InferShapeMoeEpDispatchEpilogue(gert::InferShapeContext *context) {
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape for MoeEpDispatchEpilogue.");

    const gert::Shape *dstSlotIdxShape = context->GetInputShape(EPILOGUE_INPUT_DST_SLOT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dstSlotIdxShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(EPILOGUE_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    int64_t topK = dstSlotIdxShape->GetDim(1);
    int64_t aAlloc = *epWorldSize * *attrs->GetAttrPointer<int64_t>(EPILOGUE_ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX);

    gert::Shape *recvX = context->GetOutputShape(EPILOGUE_OUTPUT_RECV_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvX);
    recvX->SetDimNum(DIM_TWO);
    recvX->SetDim(0U, aAlloc);
    recvX->SetDim(1U, -1);

    gert::Shape *recvTopkIdx = context->GetOutputShape(EPILOGUE_OUTPUT_RECV_TOPK_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvTopkIdx);
    recvTopkIdx->SetDimNum(DIM_TWO);
    recvTopkIdx->SetDim(0U, aAlloc);
    recvTopkIdx->SetDim(1U, topK);

    gert::Shape *recvTopkWeights = context->GetOutputShape(EPILOGUE_OUTPUT_RECV_TOPK_WEIGHTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvTopkWeights);
    recvTopkWeights->SetDimNum(DIM_ONE);
    recvTopkWeights->SetDim(0U, aAlloc);

    gert::Shape *recvSrcMetadata = context->GetOutputShape(EPILOGUE_OUTPUT_RECV_SRC_METADATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetadata);
    recvSrcMetadata->SetDimNum(DIM_TWO);
    recvSrcMetadata->SetDim(0U, aAlloc);
    recvSrcMetadata->SetDim(1U, 3);

    OP_LOGD(context->GetNodeName(), "InferShape for MoeEpDispatchEpilogue done.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeEpDispatchEpilogue(gert::InferDataTypeContext *context) {
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    context->SetOutputDataType(EPILOGUE_OUTPUT_RECV_X_INDEX, ge::DT_BF16);
    context->SetOutputDataType(EPILOGUE_OUTPUT_RECV_TOPK_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(EPILOGUE_OUTPUT_RECV_TOPK_WEIGHTS_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(EPILOGUE_OUTPUT_RECV_SRC_METADATA_INDEX, ge::DT_INT32);
    context->SetOutputDataType(EPILOGUE_OUTPUT_RECV_SCALES_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeEpDispatchEpilogue)
    .InferShape(InferShapeMoeEpDispatchEpilogue)
    .InferDataType(InferDataTypeMoeEpDispatchEpilogue);

}  // namespace ops
