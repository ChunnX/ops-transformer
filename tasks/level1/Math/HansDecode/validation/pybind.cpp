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
* 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 hans_decode
*
* 替换示例：
* - aclnnCustomOp -> aclnnHansEncode
* - custom_pybind_api -> hans_decode
*
* 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
*/
std::vector<at::Tensor> hans_decode(
    const at::Tensor& mantissa_out,
    const at::Tensor& fixed_out,
    const at::Tensor& var_out,
    const at::Tensor& pdf_out,
    bool reshuff = false
) {

    c10::TensorOptions options = mantissa_out.options();
    at::ScalarType dtype = mantissa_out.scalar_type();
    int dTypeBytes = dtype == at::ScalarType::Float? 4 : 2;
    int64_t numel_recover_out = mantissa_out.numel() * dTypeBytes / (dTypeBytes - 1);

    printf("numel_mantissa: %ld", mantissa_out.numel());
    printf("numel_fixed_out: %ld", fixed_out.numel());
    printf("numel_pdf_out: %ld", pdf_out.numel());
    
    printf("numel_recover_out: %ld", numel_recover_out);
    at::Tensor recover_out = at::empty({1, numel_recover_out}, options);

    EXEC_NPU_CMD(
        aclnnHansDecode,
        mantissa_out,
        fixed_out,
        var_out,
        pdf_out,
        reshuff,
        recover_out
    );

    //  返回输出
    return {recover_out};
}


// 注册 Python 接口
PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `hans_decode`
    m.def("hans_decode", &hans_decode, "");

}
