/**
* @file pybind.cpp
* Python bindings for HansEncode operator
*/
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
* 使用说明：
* 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnHansEncode
* 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 hans_encode
*
* 替换示例：
* - aclnnCustomOp -> aclnnHansEncode
* - custom_pybind_api -> hans_encode
*
* 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
*/
std::vector<at::Tensor> hans_encode(
    const at::Tensor& input_tensor,
    const at::Tensor& pdf_ref,
    bool statistic = true,
    bool reshuff = false
) {
    // 1. 输入校验
    TORCH_CHECK(input_tensor.defined(), "input_tensor is not defined");
    TORCH_CHECK(pdf_ref.defined(), "pdf_ref is not defined");
    TORCH_CHECK(pdf_ref.size(0) == 1 && pdf_ref.size(1) == 256, "pdf_ref must be [1, 256]");
    TORCH_CHECK(input_tensor.numel() % 64 == 0, "input_tensor numel must be multiple of 64");
    TORCH_CHECK(input_tensor.numel() >= 32768, "input_tensor numel must >= 32768");

    // 2. 确定输出 shape
    auto output_shape = input_tensor.sizes();
    int64_t numel = input_tensor.numel();
    c10::TensorOptions options = input_tensor.options();
    // at::Tensor pdf_out = at::empty({1, 256}, torch::kInt32);  // [1, 256], int32
    at::ScalarType dtype = input_tensor.scalar_type();
    int dTypeBytes = dtype == at::ScalarType::Float? 4 : 2;
    int64_t mantissa_size = numel * (dTypeBytes - 1) / dTypeBytes;
    int64_t rest_size =  numel / dTypeBytes;
    at::Tensor mantissa_out = at::empty({1, mantissa_size}, options);
    at::Tensor fixed_out = at::empty({1, rest_size}, options);
    at::Tensor var_out = at::empty({1, rest_size}, options);

    EXEC_NPU_CMD(
        aclnnHansEncode,
        input_tensor,
        pdf_ref,
        statistic,
        reshuff,
        mantissa_out,
        fixed_out,
        var_out
    );

    // 返回输出（4 个张量）
    return {pdf_ref, mantissa_out, fixed_out, var_out};
}

// 注册 Python 接口
PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `hans_encode`
    m.def("hans_encode", &hans_encode, "");
}