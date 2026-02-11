#include "repeat_interleave_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                      \
        std::printf("nullptr error!");           \
        return ge::GRAPH_FAILED;                 \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

namespace optiling {

constexpr uint32_t DTYPE_KEY_FP16 = 0;
constexpr uint32_t DTYPE_KEY_BF16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_INT32 = 0;
constexpr uint32_t DTYPE_KEY_INT64 = 10;

static void RepeatInterleaveGradPrintParam(const gert::TilingContext* context, RepeatInterleaveGradTilingData& tiling)
{
    // output param
    OP_LOGD(context, "batch_dim_num = %ld", tiling.get_batch_dim_num());
    OP_LOGD(context, "repeats_i_grad_dim_num = %ld", tiling.get_repeats_i_grad_dim_num());
    OP_LOGD(context, "repeats_o_grad_dim_num = %ld", tiling.get_repeats_o_grad_dim_num());
    OP_LOGD(context, "data_dim_num = %ld", tiling.get_data_dim_num());

    OP_LOGD(context, "core_num = %ld", tiling.get_core_num());
    OP_LOGD(context, "batch_dim_num_each_core = %ld", tiling.get_batch_dim_num_each_core());
    OP_LOGD(context, "batch_dim_num_last_core = %ld", tiling.get_batch_dim_num_last_core());
    OP_LOGD(context, "core_num_each_batch = %ld", tiling.get_core_num_each_batch());
    OP_LOGD(context, "element_num_each_core = %ld", tiling.get_element_num_each_core());
    OP_LOGD(context, "element_num_last_core = %ld", tiling.get_element_num_last_core());

    OP_LOGD(context, "element_num_each_loop = %ld", tiling.get_element_num_each_loop());
}

static void RepeatInterleaveGradInitShapeInfo(
    const gert::TilingContext* context, RepeatInterleaveGradTilingData& tiling)
{
    // output param
    auto attrs = context->GetAttrs();
    auto inputshape0 = context->GetInputShape(0);
    auto inputshape1 = context->GetInputShape(1);
    auto outputshape0 = context->GetOutputShape(0);
    OP_CHECK_IF(
        attrs == nullptr || inputshape0 == nullptr || inputshape1 == nullptr || outputshape0 == nullptr,
        OP_LOGE(context, "attrs or shape is null."), return);
    const gert::Shape input_grad_shape = inputshape0->GetStorageShape();
    const gert::Shape repeats_shape = inputshape1->GetStorageShape();
    const gert::Shape output_grad_shape = outputshape0->GetStorageShape();
    const int64_t* axis = attrs->GetInt(0);
    int64_t axis_attr = *axis;
    int64_t input_dim_num = input_grad_shape.GetDimNum();
    if (axis_attr < 0) {
        axis_attr = axis_attr + input_dim_num;
    }
    int64_t batch_dim_num = 1;
    int64_t repeats_i_grad_dim_num = 1;
    int64_t repeats_o_grad_dim_num = 1;
    int64_t data_dim_num = 1;
    for (int64_t shape_index = 0; shape_index < input_dim_num; shape_index++) {
        if (shape_index < axis_attr) {
            batch_dim_num = batch_dim_num * input_grad_shape.GetDim(shape_index);
        } else if (shape_index == axis_attr) {
            repeats_o_grad_dim_num = repeats_shape.GetDim(0);
            if (repeats_o_grad_dim_num == 1) {
                repeats_i_grad_dim_num = input_grad_shape.GetDim(shape_index) / output_grad_shape.GetDim(shape_index);
                batch_dim_num = batch_dim_num * output_grad_shape.GetDim(shape_index);
            } else {
                repeats_i_grad_dim_num = input_grad_shape.GetDim(shape_index);
            }
        } else {
            data_dim_num = data_dim_num * input_grad_shape.GetDim(shape_index);
        }
    }

    tiling.set_batch_dim_num(batch_dim_num);
    tiling.set_repeats_i_grad_dim_num(repeats_i_grad_dim_num);
    tiling.set_repeats_o_grad_dim_num(repeats_o_grad_dim_num);
    tiling.set_data_dim_num(data_dim_num);
}

static void RepeatInterleaveGradInitSplitInfo(
    const gert::TilingContext* context, RepeatInterleaveGradTilingData& tiling)
{
    // output param
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto max_core_num = ascendcPlatform.GetCoreNum();
    int64_t batch_dim_num = tiling.get_batch_dim_num();
    int64_t data_dim_num = tiling.get_data_dim_num();
    int64_t core_num;
    int64_t batch_dim_num_each_core;
    int64_t batch_dim_num_last_core;
    int64_t core_num_each_batch;
    int64_t element_num_each_core;
    int64_t element_num_last_core;
    int64_t align_data_num = 64;
    if (batch_dim_num < max_core_num) {
        batch_dim_num_each_core = 1;
        batch_dim_num_last_core = 1;

        core_num_each_batch = max_core_num / batch_dim_num;
        element_num_each_core = (data_dim_num + core_num_each_batch - 1) / core_num_each_batch;
        element_num_each_core = (element_num_each_core + align_data_num - 1) / align_data_num * align_data_num;
        core_num_each_batch = (data_dim_num + element_num_each_core - 1) / element_num_each_core;
        element_num_last_core = data_dim_num - (core_num_each_batch - 1) * element_num_each_core;
        core_num = core_num_each_batch * batch_dim_num;
    } else {
        batch_dim_num_each_core = (batch_dim_num + max_core_num - 1) / max_core_num;
        core_num = (batch_dim_num + batch_dim_num_each_core - 1) / batch_dim_num_each_core;
        batch_dim_num_last_core = batch_dim_num - (core_num - 1) * batch_dim_num_each_core;
        core_num_each_batch = 1;
        element_num_each_core = data_dim_num;
        element_num_last_core = data_dim_num;
    }

    tiling.set_core_num(core_num);
    tiling.set_batch_dim_num_each_core(batch_dim_num_each_core);
    tiling.set_batch_dim_num_last_core(batch_dim_num_last_core);
    tiling.set_core_num_each_batch(core_num_each_batch);
    tiling.set_element_num_each_core(element_num_each_core);
    tiling.set_element_num_last_core(element_num_last_core);
}

ge::graphStatus Tiling4RepeatInterleaveGrad(gert::TilingContext* context)
{
    OP_LOGD(context, "RepeatInterleaveGradTiling tiling start");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    
    RepeatInterleaveGradTilingData tiling;
    RepeatInterleaveGradInitShapeInfo(context, tiling);
    OP_LOGD(context, "RepeatInterleaveGradTiling RepeatInterleaveGradInitShapeInfo");
    RepeatInterleaveGradInitSplitInfo(context, tiling);
    OP_LOGD(context, "RepeatInterleaveGradTiling RepeatInterleaveGradInitSplitInfo");
    
    // data_size * tensor_num * buffer_num
    uint64_t max_ub_size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);
    int64_t element_size = 4 * 4 * 2;
    int64_t align_data_num = 64;
    int64_t element_num_each_loop = max_ub_size / element_size / align_data_num * align_data_num;
    tiling.set_element_num_each_loop(element_num_each_loop);

    RepeatInterleaveGradPrintParam(context, tiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(tiling.get_core_num());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto grad_dtype = context->GetInputDesc(0)->GetDataType();
    auto index_dtype = context->GetInputDesc(1)->GetDataType();
    uint32_t grad_dtype_key = 0;
    uint32_t index_dtype_key = 0;
    if (grad_dtype == ge::DT_FLOAT16) {
        grad_dtype_key = DTYPE_KEY_FP16;
    } else if (grad_dtype == ge::DT_BF16) {
        grad_dtype_key = DTYPE_KEY_BF16;
    } else if (grad_dtype == ge::DT_FLOAT) {
        grad_dtype_key = DTYPE_KEY_FP32;
    }
    if (index_dtype == ge::DT_INT32) {
        index_dtype_key = DTYPE_KEY_INT32;
    } else if (index_dtype == ge::DT_INT64) {
        index_dtype_key = DTYPE_KEY_INT64;
    }
    uint32_t tiling_key = grad_dtype_key + index_dtype_key;
    context->SetTilingKey(tiling_key);
    OP_LOGD(context, "RepeatInterleaveGradTiling tiling end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RepeatInterleaveGrad)
    .Tiling(Tiling4RepeatInterleaveGrad);

} // namespace optiling