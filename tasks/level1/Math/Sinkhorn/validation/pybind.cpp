/**
* @file extension_sinkhorn.cpp
*/
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
at::Tensor sinkhorn(const at::Tensor cost, c10::optional<float> tol)
{
    // 获取输入 shape
    auto shape = cost.sizes();
    c10::SmallVector<int64_t, 2> output_size;
    output_size.emplace_back(shape[0]);
    output_size.emplace_back(shape[1]);

    c10::TensorOptions options = cost.options().dtype(cost.scalar_type());
    at::Tensor result = at::empty(output_size, options);

    float tolf = 0.0001;
    if (tol != c10::nullopt) {
        tolf = tol.value();
    }

    // 调用 ACLNN 算子：aclnnSinkhorn(cost, tol_val, result)
    EXEC_NPU_CMD(aclnnSinkhorn, cost, tolf, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `sinkhorn`
    m.def("sinkhorn", &sinkhorn, "");
}