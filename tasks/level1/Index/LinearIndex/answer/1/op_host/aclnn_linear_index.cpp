#include "aclnn_linear_index.h"
#include "linear_index.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                             op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static bool CheckNotNull(const aclTensor *indices, const aclTensor *shape, const aclTensor *out) {
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(shape, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *indices, const aclTensor *shape, const aclTensor *out) {
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(shape, OUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *indices, const aclTensor *shape, const aclTensor *out) {
  CHECK_RET(CheckNotNull(indices, shape, out), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(indices, shape, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinearIndexGetWorkspaceSize(const aclTensor *indices, const aclTensor *shape, int64_t axits, bool combine,
                                                 const aclTensor * out,uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnLinearIndex, DFX_IN(indices, shape, axits, combine), DFX_OUT(out));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(indices, shape, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (indices->IsEmpty() || shape->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto shapeContiguous = l0op::Contiguous(shape, uniqueExecutor.get());
  CHECK_RET(shapeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto linearIndexRes = l0op::LinearIndex(indicesContiguous, shapeContiguous, axits, combine, uniqueExecutor.get());
  CHECK_RET(linearIndexRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(linearIndexRes, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinearIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLinearIndex);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif