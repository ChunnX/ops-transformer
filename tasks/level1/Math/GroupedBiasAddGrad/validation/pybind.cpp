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
#include "acl/acl.h"
#include "acl/acl_rt.h"
#define CHECK_ACL(expr) \
    do { \
        aclError ret = (expr); \
        if (ret != ACL_SUCCESS) { \
            printf("CHECK_ACL failed: %d\n", ret); \
            return {}; \
        } \
    } while (0)

std::vector<at::Tensor> grouped_bias_add_grad(const at::Tensor& grad_y, c10::optional<at::Tensor> group_idx) 
{
    auto shape = grad_y.sizes();

    c10::SmallVector<int64_t, 2> output_size;
    
    if (grad_y.dim() == 2) {
        output_size.emplace_back(group_idx.value().sizes()[0]);
        output_size.emplace_back(shape[1]);
    } else if (grad_y.dim() == 3) {
        output_size.emplace_back(shape[0]);
        output_size.emplace_back(shape[2]);
    } else {
        TORCH_CHECK(false, "grad_y must be 2D or 3D");
    }
    
    // 3. 创建输出张量
    c10::TensorOptions options = grad_y.options().dtype(grad_y.scalar_type());

    at::Tensor out = at::empty(output_size, options);

    int64_t groupIdxType = 0;
    if (group_idx.has_value()){
        EXEC_NPU_CMD(aclnnGroupedBiasAddGrad, grad_y, group_idx, groupIdxType, out);
    } else {
        at::Tensor empty_tensor = at::Tensor();
        EXEC_NPU_CMD(aclnnGroupedBiasAddGrad, grad_y, empty_tensor, groupIdxType, out);
    }
    
    return {out};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `grouped_bias_add_grad`
    m.def("grouped_bias_add_grad", &grouped_bias_add_grad, "");
}