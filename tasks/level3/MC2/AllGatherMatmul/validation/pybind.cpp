#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * AllGatherMatmul: MC2 通信计算融合算子
 * 参数与 aclnnAllGatherMatmulGetWorkspaceSize 一致：
 * x1, x2, bias, group, gatherIndex, commTurn, streamMode, output, gatherOut
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x1,
    const at::Tensor &x2,
    c10::optional<at::Tensor> bias,
    const std::string &group,
    int64_t gatherIndex,
    int64_t commTurn,
    int64_t streamMode)
{
    (void)streamMode;
    at::ScalarType dtype = x1.scalar_type();
    int64_t m = x1.size(0);
    int64_t k = x1.size(1);
    int64_t n = x2.size(1);
    int64_t rank_size = 1;
    if (const char *world_size_env = std::getenv("WORLD_SIZE"); world_size_env != nullptr) {
        const int64_t parsed = std::strtoll(world_size_env, nullptr, 10);
        rank_size = parsed > 0 ? parsed : 1;
    }
    bool is_trans_a = false;
    bool is_trans_b = false;
    bool is_gather_out = true;

    at::Tensor output = at::empty(
        {m * rank_size, n},
        at::TensorOptions().dtype(dtype).device(x1.device()));

    at::Tensor gatherOut = at::empty(
        {m * rank_size, k},
        at::TensorOptions().dtype(dtype).device(x1.device()));

    // autogen 头文件参数为 char*，这里保证可变指针类型对齐 ABI。
    char *group_ptr = const_cast<char *>(group.c_str());
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x1,
        x2,
        bias,
        group_ptr,
        is_trans_a,
        is_trans_b,
        gatherIndex,
        commTurn,
        rank_size,
        is_gather_out,
        output,
        gatherOut);

    return {output, gatherOut};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}