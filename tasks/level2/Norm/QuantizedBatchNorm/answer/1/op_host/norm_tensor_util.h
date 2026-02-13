#ifndef NORM_COMMON_NORM_TENSOR_UTIL
#define NORM_COMMON_NORM_TENSOR_UTIL
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"

namespace op {
const aclIntArray* getAllDims(const aclTensor* self, aclOpExecutor* executor);

const aclTensor* ResizeFrom1D(
    const aclTensor* cdim, const aclTensor* input, bool isSupportNcdhw, aclOpExecutor* executor);

const aclTensor* ResizeTo1D(
    const aclTensor* result, const aclTensor* output, bool isSupportNcdhw, aclOpExecutor* executor);

const aclTensor* ResizeFromND(const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeToND(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeFrom5D(const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeTo5D(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor);

aclTensor* FillScalar(int64_t dim, int value, aclOpExecutor* executor);

aclnnStatus ProcessEmptyTensorWithValue(aclTensor* src, float initValue, aclOpExecutor* executor);

op::DataType CombineCategories(op::DataType higher, op::DataType lower);
} // namespace op

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus BatchNorm(const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* runningMean,
                      aclTensor* runningVar, bool training, float momentum, float eps, aclTensor** output,
                      aclTensor* saveMean, aclTensor* saveInvstd, aclOpExecutor* executor);

#ifdef __cplusplus
}
#endif
#endif // NORM_COMMON_NORM_TENSOR_UTIL