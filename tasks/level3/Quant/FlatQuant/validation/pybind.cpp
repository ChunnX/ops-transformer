#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <acl/acl_base.h>

// FlatQuant outputs int4 (packed) + float32 quant_scale.
// PyTorch has no int4 dtype, so we manually create aclTensors with ACL_INT4
// and unpack the result to int8 for comparison.
//
// Autogen aclnn signature:
//   aclnnFlatQuantGetWorkspaceSize(x, p1, p2, clipRatio:double, dstDtype:int64_t,
//                                   outOut, quantScaleOut, workspaceSize:uint64_t*, executor)

static aclTensor *CreateAclInt4Tensor(void *data, int64_t K, int64_t M, int64_t N)
{
    static const auto fn = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(fn != nullptr, "aclCreateTensor not found");

    int64_t shape[3] = {K, M, N};
    int64_t strides[3] = {M * N, N, 1};
    int64_t total_elements = K * M * N;

    return fn(shape, 3, ACL_INT4, strides, 0, ACL_FORMAT_ND,
              &total_elements, 1, data);
}

static void UnpackInt4ToInt8(const uint8_t *packed, int8_t *unpacked, int64_t total_elements)
{
    int64_t num_pairs = total_elements / 2;
    for (int64_t i = 0; i < num_pairs; i++) {
        uint8_t byte_val = packed[i];
        int8_t hi = static_cast<int8_t>(byte_val) >> 4;        // sign extend high nibble
        int8_t lo = static_cast<int8_t>(byte_val << 4) >> 4;  // sign extend low nibble
        unpacked[i * 2] = hi;      // high nibble first (AscendC convention)
        unpacked[i * 2 + 1] = lo;
    }
    if (total_elements % 2 == 1) {
        uint8_t byte_val = packed[num_pairs];
        int8_t hi = static_cast<int8_t>(byte_val) >> 4;
        unpacked[total_elements - 1] = hi;
    }
}

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    at::Tensor &x,
    at::Tensor &kronecker_p1,
    at::Tensor &kronecker_p2,
    double clip_ratio,
    int64_t dst_dtype)
{
    auto x_sizes = x.sizes();
    int64_t K = x_sizes[0], M = x_sizes[1], N = x_sizes[2];

    // Allocate output buffers on NPU
    int64_t out_elements = K * M * N;
    int64_t out_bytes = (out_elements + 1) / 2;
    at::Tensor out_buf = torch::empty({out_bytes}, x.options().dtype(at::kByte));
    at::Tensor quant_scale = torch::empty({K}, x.options().dtype(at::kFloat));

    // Create aclTensors for inputs
    aclTensor *x_acl = ConvertType(x);
    aclTensor *p1_acl = ConvertType(kronecker_p1);
    aclTensor *p2_acl = ConvertType(kronecker_p2);

    // Create int4 output tensor manually
    aclTensor *out_acl = CreateAclInt4Tensor(
        const_cast<void *>(out_buf.storage().data()), K, M, N);
    aclTensor *qs_acl = ConvertType(quant_scale);

    // Get aclnn function pointers
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnCustomOpGetWorkspaceSize");
    static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnCustomOp");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnCustomOp API not found");

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    typedef int32_t (*GetWorkspaceSizeFunc)(const aclTensor *, const aclTensor *, const aclTensor *,
                                            double, int64_t,
                                            const aclTensor *, const aclTensor *,
                                            uint64_t *, aclOpExecutor **);
    auto ws_status = reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr)(
        x_acl, p1_acl, p2_acl, clip_ratio, dst_dtype,
        out_acl, qs_acl, &workspace_size, &executor);
    TORCH_CHECK(ws_status == 0, "aclnnCustomOpGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

    void *workspace_addr = nullptr;
    at::Tensor workspace;
    if (workspace_size > 0) {
        workspace = torch::empty({(int64_t)workspace_size},
            at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace.storage().data());
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    typedef int32_t (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    auto op_status = reinterpret_cast<OpApiFunc>(opApiFuncAddr)(
        workspace_addr, workspace_size, executor, acl_stream);
    TORCH_CHECK(op_status == 0, "aclnnCustomOp failed, detail:", aclGetRecentErrMsg());

    // Release acl resources
    Release(x_acl);
    Release(p1_acl);
    Release(p2_acl);
    Release(out_acl);
    Release(qs_acl);

    // Unpack int4 output to int8 (must copy to CPU first since NPU memory is not CPU-accessible)
    at::Tensor out_buf_cpu = out_buf.to(at::kCPU);
    at::Tensor out_int8 = torch::empty({K, M, N}, at::kChar);
    UnpackInt4ToInt8(out_buf_cpu.data_ptr<uint8_t>(), out_int8.data_ptr<int8_t>(), out_elements);

    return std::make_tuple(out_int8, quant_scale);
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "kernel_gen_ops custom operator");
}
