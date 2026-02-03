/*!
 * \file histogram_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "common/log.h"

using namespace ge;
namespace ops {
static constexpr int64_t BINS_IDX = 0;
static constexpr int64_t OUTPUT_IDX = 0;

static ge::graphStatus HistogramV2InferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramV2InferShapeFunc");
    // 获取bins
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    int64_t bins = *(attr->GetAttrPointer<int64_t>(BINS_IDX));
    OP_LOGD(context, "bins = %ld", bins);
    if (bins <= 0) {
        OP_LOGE(context, "bins has to be positive, but get %ld", bins);
        return ge::GRAPH_FAILED;
    }

    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, bins);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HistogramV2).InferShape(HistogramV2InferShapeFunc);
} // namespace ops