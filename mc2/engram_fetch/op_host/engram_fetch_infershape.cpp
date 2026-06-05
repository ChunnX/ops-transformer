#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {

static constexpr size_t ENGRAM_FETCH_INPUT_BUFFERHANDLE_INDEX = 0;
static constexpr size_t ENGRAM_FETCH_INPUT_INDICES_INDEX = 1;
static constexpr size_t ENGRAM_FETCH_OUTPUT_FETCHED_INDEX = 0;

static constexpr size_t ENGRAM_FETCH_ATTR_NUM_ENTRIES_PER_RANK_INDEX = 0;
static constexpr size_t ENGRAM_FETCH_ATTR_HIDDEN_SIZE_INDEX = 1;
static constexpr size_t ENGRAM_FETCH_ATTR_OUTPUT_DTYPE_INDEX = 3;

static ge::graphStatus InferShapeEngramFetch(gert::InferShapeContext *context)
{
    const gert::Shape *indicesShape = context->GetInputShape(ENGRAM_FETCH_INPUT_INDICES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *hiddenSize = attrs->GetInt(ENGRAM_FETCH_ATTR_HIDDEN_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, hiddenSize);

    gert::Shape *fetchedShape = context->GetOutputShape(ENGRAM_FETCH_OUTPUT_FETCHED_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, fetchedShape);

    int64_t numTokens = indicesShape->GetDim(0);
    fetchedShape->SetDimNum(2);
    fetchedShape->SetDim(0, numTokens);
    fetchedShape->SetDim(1, *hiddenSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramFetch(gert::InferDataTypeContext *context)
{
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *outputDtype = attrs->GetInt(ENGRAM_FETCH_ATTR_OUTPUT_DTYPE_INDEX);
    OP_CHECK_IF(outputDtype == nullptr, OP_LOGE("EngramFetch", "output_dtype attr is null"), return ge::GRAPH_FAILED);

    ge::DataType dtype = static_cast<ge::DataType>(*outputDtype);
    OP_CHECK_IF(dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
                OP_LOGE("EngramFetch", "output_dtype must be BF16(1)/FP16(2)/FP32(3), got %ld", *outputDtype),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(ENGRAM_FETCH_OUTPUT_FETCHED_INDEX, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFetch).InferShape(InferShapeEngramFetch).InferDataType(InferDataTypeEngramFetch);

} // namespace ops