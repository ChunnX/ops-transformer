/**
 * @file extension_quantized_batch_norm.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

// Use fixed version of pytorch_npu_helper with corrected ConvertType
// The fix: Use tensor shape as storage_dims instead of storage size
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnQuantizedBatchNorm
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 quantized_batch_norm
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnQuantizedBatchNorm)
 * - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如quantized_batch_norm)
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
at::Tensor quantized_batch_norm(
    at::Tensor x,
    at::Tensor mean,
    at::Tensor var,
    at::Tensor input_scale,
    at::Tensor input_zero_point,
    at::Tensor output_scale,
    at::Tensor output_zero_point,
    c10::optional<at::Tensor> weightOptional,
    c10::optional<at::Tensor> biasOptional,
    double epsilon)
{
    // Ensure all input tensors are contiguous and have correct format
    x = x.contiguous();
    mean = mean.contiguous();
    var = var.contiguous();
    input_scale = input_scale.contiguous();
    input_zero_point = input_zero_point.contiguous();
    output_scale = output_scale.contiguous();
    output_zero_point = output_zero_point.contiguous();
    
    // Output tensor - ensure same shape and dtype as input
    at::Tensor y = torch::empty_like(x);

    // Prepare optional weight and bias tensors for NPU_CMD
    at::Tensor weight_npu = weightOptional.has_value() ? weightOptional.value().contiguous() : at::Tensor();
    at::Tensor bias_npu = biasOptional.has_value() ? biasOptional.value().contiguous() : at::Tensor();

    // Convert scalar tensors (shape=[1]) to aclScalar* for NPU operation
    // The NPU operation expects aclScalar* for quantization parameters, not aclTensor*
    // Extract values to local variables first to ensure lifetime safety
    // Then create aclScalar* using aclCreateScalar with correct data types
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    TORCH_CHECK(aclCreateScalar != nullptr, "aclCreateScalar function not found");
    
    // Extract float values for scale parameters
    float input_scale_value = input_scale.item<float>();
    float output_scale_value = output_scale.item<float>();
    
    // Extract int32 values for zero_point parameters  
    int32_t input_zero_point_value = input_zero_point.item<int32_t>();
    int32_t output_zero_point_value = output_zero_point.item<int32_t>();
    
    // Get correct aclDataType from conversion table
    aclDataType acl_float_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::Float)];
    aclDataType acl_int32_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::Int)];
    
    // Create aclScalar* with correct data types
    aclScalar* input_scale_scalar = aclCreateScalar(&input_scale_value, acl_float_type);
    aclScalar* input_zero_point_scalar = aclCreateScalar(&input_zero_point_value, acl_int32_type);
    aclScalar* output_scale_scalar = aclCreateScalar(&output_scale_value, acl_float_type);
    aclScalar* output_zero_point_scalar = aclCreateScalar(&output_zero_point_value, acl_int32_type);

    EXEC_NPU_CMD(aclnnQuantizedBatchNorm,
                 x, mean, var, input_scale_scalar, input_zero_point_scalar, output_scale_scalar, output_zero_point_scalar,
                 weight_npu, bias_npu, epsilon,
                 y);

    return y;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `quantized_batch_norm`
    m.def("quantized_batch_norm", &quantized_batch_norm, "");
}
