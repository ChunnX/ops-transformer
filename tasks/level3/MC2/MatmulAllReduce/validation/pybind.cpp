#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <string>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * MatmulAllReduce：与 build_out/autogen/aclnn_matmul_all_reduce.h 中
 * aclnnMatmulAllReduceGetWorkspaceSize 参数顺序一致。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x1,
    const at::Tensor &x2,
    c10::optional<at::Tensor> bias,
    const std::string &group,
    const std::string &reduce_op,
    int64_t comm_turn)
{
    at::ScalarType dtype = x1.scalar_type();
    int64_t m = x1.size(0);
    int64_t n = x2.size(1);

    at::Tensor out = at::empty(
        {m, n},
        at::TensorOptions().dtype(dtype).device(x1.device()));

    char *group_ptr = const_cast<char *>(group.c_str());
    char *reduce_ptr = reduce_op.empty() ? nullptr : const_cast<char *>(reduce_op.c_str());

    const bool is_trans_a = false;
    const bool is_trans_b = false;
    const int64_t antiquant_group_size = 0;
    const int64_t group_size = 0;
    // Match op_def default: ge::DT_UNDEFINED.
    const int64_t y_dtype = 0;
    const int64_t comm_quant_mode = 0;

    EXEC_NPU_CMD(
        aclnnCustomOp,
        x1,
        x2,
        bias,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        group_ptr,
        reduce_ptr,
        is_trans_a,
        is_trans_b,
        comm_turn,
        antiquant_group_size,
        group_size,
        y_dtype,
        comm_quant_mode,
        out);

    return {out};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
