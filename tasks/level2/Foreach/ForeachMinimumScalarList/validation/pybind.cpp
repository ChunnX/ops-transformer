#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
 * - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */

template <typename T>
void create_tensor_to_scalar(std::vector<at::Scalar> &scalarsVec, std::vector<at::Tensor> &scalars)
{
    for (int64_t i = 0; i < scalars.size(); i++) {
        scalarsVec.push_back(at::Scalar(scalars[i].index({0}).item<T>()));
    }
}

std::vector<at::Tensor> foreach_minimum_scalar_list(
    std::vector<at::Tensor> &x, std::vector<at::Tensor> &scalars)
{
    // alloc output memory
    if (x.size() <= 0) {
        throw std::runtime_error("x size should be greater then 0");
    }
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }
    if (scalars.size() <= 0) {
        throw std::runtime_error("scalars size should be greater then 0");
    }
    std::vector<at::Scalar> scalarsVec;
    scalarsVec.reserve(scalars.size());
    if (scalars[0].scalar_type() == at::ScalarType::Float) {
        create_tensor_to_scalar<float>(scalarsVec, scalars);
    } else if (scalars[0].scalar_type() == at::ScalarType::Half) {
        create_tensor_to_scalar<at::Half>(scalarsVec, scalars);
    } else if (scalars[0].scalar_type() == at::ScalarType::Long) {
        create_tensor_to_scalar<int64_t>(scalarsVec, scalars);
    } else {
        throw std::runtime_error("Unsupported scalar type for foreach_minimum_scalar_list");
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList resultList = at::TensorList(result);
    at::TensorList xList = at::TensorList(x);
    at::ArrayRef<at::Scalar> scalarsList(scalarsVec);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnForeachMinimumScalarList, xList, scalarsList, resultList);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("foreach_minimum_scalar_list", &foreach_minimum_scalar_list, "");
}