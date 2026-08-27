#ifndef GROUPED_MATMUL_QUANT_TORCH_ADPT_H
#define GROUPED_MATMUL_QUANT_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor grouped_matmul_quant(const at::Tensor &x, const at::Tensor &quantized_weight,
                       const at::Tensor &weight_scale, const at::Tensor &weight_offset,
                       const c10::optional<at::Tensor> &group_list, int64_t scale_group_size) {
    constexpr int64_t FRACTAL_FLOAT16 = 16;
    int64_t m = x.size(0);
    int64_t fracN = quantized_weight.size(2);
    int64_t n = fracN * FRACTAL_FLOAT16;
    at::Tensor output = at::empty({m, n}, x.options());
    EXEC_NPU_CMD(aclnnGroupedMatmulQuant, x, quantized_weight, weight_scale, weight_offset,
                 group_list, scale_group_size, output);
    return output;
}

}  // namespace vllm_ascend
#endif