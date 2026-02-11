

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
  // 检查indices的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
  // 检查shape的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(shape, OUT_DTYPE_SUPPORT_LIST, return false);
  // 检查out的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *indices, const aclTensor *shape, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(indices, shape, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(indices, shape, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinearIndexGetWorkspaceSize(const aclTensor *indices, const aclTensor *shape, int64_t axits, bool combine,
                                                 const aclTensor * out,uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnLinearIndex, DFX_IN(indices, shape, axits, combine), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(indices, shape, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (indices->IsEmpty() || shape->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入indices转换成连续的tensor
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入shape转换成连续的tensor
  auto shapeContiguous = l0op::Contiguous(shape, uniqueExecutor.get());
  CHECK_RET(shapeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 执行L0算子
  auto linearIndexRes = l0op::LinearIndex(indicesContiguous, shapeContiguous, axits, combine, uniqueExecutor.get());
  CHECK_RET(linearIndexRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出data上
  auto viewCopyResult = l0op::ViewCopy(linearIndexRes, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinearIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLinearIndex);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
