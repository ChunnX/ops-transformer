/*!
 * \file hans_decode_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
// #include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;
static constexpr int PDF_NUMEL_LENGTH = 256;
static constexpr int ENCODE_BLOCK_LENGTH = 64;
static constexpr int INT32_BYTES = 4;
static constexpr int FP32_BYTES = 4;
static constexpr int FP16_BYTES = 2;

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static ge::graphStatus InferShape4HansDecode(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4HansDecode");

    // get input shapes
    const gert::Shape* mantissaShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, mantissaShape);
    // check output

    OP_LOGD(context->GetNodeName(), "End to do InferShape4HansDecode");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HansDecode).InferShape(InferShape4HansDecode);
} // namespace ops