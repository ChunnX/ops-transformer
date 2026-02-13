/*!
 * \file rms_norm_quant_infershape.cpp
 * \brief
 */

#include <cstddef>
#include <cstdint>
#include <exception>
#include "common/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

namespace {
class NullPtrException : public std::exception {
public:
    explicit NullPtrException(const char* message = "Null pointer exception occurred") : msg_(message)
    {}
    const char* what() const noexcept override
    {
        return msg_;
    }

private:
    const char* msg_;
};

size_t GetInputDimNum(size_t inputIdx, const gert::InferShapeContext* context)
{
    const auto* shape = context->GetInputShape(inputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    return shape->GetDimNum();
}

size_t GetInputDim(size_t inputIdx, size_t dimIdx, const gert::InferShapeContext* context)
{
    const auto* shape = context->GetInputShape(inputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    return shape->GetDim(dimIdx);
}

void SetOutputDimNum(size_t outputIdx, size_t dimNum, gert::InferShapeContext* context)
{
    auto* shape = context->GetOutputShape(outputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    shape->SetDimNum(dimNum);
}

void SetOutputDim(size_t outputIdx, size_t dimIdx, int64_t value, gert::InferShapeContext* context)
{
    auto* shape = context->GetOutputShape(outputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    shape->SetDim(dimIdx, value);
}

void CopyInputDimsToOutput(size_t inputIdx, size_t outputIdx, gert::InferShapeContext* context)
{
    size_t dimNum = GetInputDimNum(inputIdx, context);
    SetOutputDimNum(outputIdx, dimNum, context);
    for (size_t dimIdx = 0; dimIdx < dimNum; dimIdx++) {
        SetOutputDim(outputIdx, dimIdx, GetInputDim(inputIdx, dimIdx, context), context);
    }
}

} // namespace

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_GAMMA = 1;
constexpr size_t INPUT_BETA = 2;

constexpr size_t OUTPUT_Y = 0;
constexpr size_t ATTR_DST_TYPE = 3;

static ge::graphStatus InferShape4RmsNormQuant(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4RmsNormQuant", "Context is nullptr, check failed.");
        return ge::GRAPH_FAILED;
    }

    std::string OP_NAME = "RmsNorm";
    OP_NAME = OP_NAME + "Quant";

    OP_LOGD(OP_NAME, "Begin to do NormInferShape");

    if (GetInputDimNum(INPUT_BETA, context) != 0) {
        if (GetInputDimNum(INPUT_GAMMA, context) != GetInputDimNum(INPUT_BETA, context)) {
            OP_LOGE(OP_NAME, "gamma_shape are not equal to beta_shape");
            return ge::GRAPH_FAILED;
        }

        for (size_t i = 0; i < GetInputDimNum(INPUT_GAMMA, context); i++) {
            if (GetInputDim(INPUT_GAMMA, i, context) != GetInputDim(INPUT_BETA, i, context)) {
                OP_LOGE(OP_NAME, "gamma_shape dim are not equal to beta_shape dim.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (GetInputDim(INPUT_GAMMA, GetInputDimNum(INPUT_GAMMA, context) - 1, context) !=
        GetInputDim(INPUT_X, GetInputDimNum(INPUT_X, context) - 1, context)) {
        OP_LOGE(OP_NAME, "gamma_shape last dim are not equal to x_shape last dim.");
        return ge::GRAPH_FAILED;
    }

    CopyInputDimsToOutput(INPUT_X, OUTPUT_Y, context);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormQuant(gert::InferDataTypeContext* context)
{
    ge::DataType y_dtype = DT_INT8;
    auto dstTypePtr = context->GetAttrs()->GetInt(ATTR_DST_TYPE);
    if (dstTypePtr != nullptr) {
        y_dtype = static_cast<ge::DataType>(*dstTypePtr);
    }
    context->SetOutputDataType(OUTPUT_Y, y_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormQuant).InferShape(InferShape4RmsNormQuant).InferDataType(InferDataType4RmsNormQuant);
} // namespace ops