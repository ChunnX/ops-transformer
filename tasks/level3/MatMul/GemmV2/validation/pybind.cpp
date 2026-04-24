#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>
#include "pytorch_npu_helper.hpp"

namespace {
std::vector<std::string> ResolveOppBases(const std::string &base)
{
    std::vector<std::string> bases{base};
    if (!base.empty() && base[0] != '/') {
        const std::string source_file(__FILE__);
        const std::string marker = "/validation/pybind.cpp";
        const auto marker_pos = source_file.rfind(marker);
        const auto src_pos = base.find("/src/");
        if (marker_pos != std::string::npos && src_pos != std::string::npos) {
            const std::string src_dir = source_file.substr(0, marker_pos);
            bases.push_back(src_dir + "/" + base.substr(src_pos + 5));
        }
    }
    return bases;
}

void LoadCustomTiling()
{
    static bool loaded = false;
    if (loaded) {
        return;
    }

    const char *opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    TORCH_CHECK(opp_path != nullptr, "ASCEND_CUSTOM_OPP_PATH is not set");

    std::string last_error;
    for (const auto &base : ResolveOppBases(opp_path)) {
        const std::string candidates[] = {
            base + "/op_impl/ai_core/tbe/op_tiling/liboptiling.so",
            base + "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
            base + "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so",
        };

        for (const auto &path : candidates) {
            void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (handle != nullptr) {
                loaded = true;
                return;
            }
            const char *err = dlerror();
            if (err != nullptr) {
                last_error = err;
            }
        }
    }
    TORCH_CHECK(false, "failed to dlopen custom GemmV2 tiling library: ", last_error);
}
} // namespace

at::Tensor gemm_v2(at::Tensor &a, at::Tensor &b, at::Tensor &alpha, at::Tensor &beta, at::Tensor &c,
                   bool transpose_a, bool transpose_b)
{
    LoadCustomTiling();
    at::Tensor result = c.clone();
    EXEC_NPU_CMD(aclnnGemmV2, a, b, alpha, beta, result, transpose_a, transpose_b);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("gemm_v2", &gemm_v2, "");
}
