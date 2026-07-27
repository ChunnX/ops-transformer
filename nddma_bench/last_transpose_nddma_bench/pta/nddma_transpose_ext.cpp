// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// PTA pybind 扩展：把 LastTranspose NDDMA kernel 暴露为 PyTorch 可调用算子。
// 仅依赖 torch 张量 API + 启动器 C 接口，不直接调用 torch_npu C++ 符号。

#include <torch/extension.h>
#include <cstdint>

extern "C" int LaunchNddmaTranspose(void* xDev, void* yDev,
                                    uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                    uint32_t dtype, uint32_t isMte3, void* extStream);

extern "C" int LaunchNddmaTransposeHost(const void* xHost, void* yHost,
                                        uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                        uint32_t dtype, uint32_t isMte3, int32_t deviceId);

// 输入 x: (dim0,dim1,dim2,dim3)，返回转置后 (dim0,dim1,dim3,dim2) 的连续张量。
// dtype: 0=fp16, 1=fp32, 2=int8; isMte3=1 正常输出（=0 仅测读带宽，输出无意义）。
at::Tensor nddma_transpose(const at::Tensor& x,
                           int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                           int64_t dtype, int64_t isMte3)
{
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.numel() == dim0 * dim1 * dim2 * dim3, "numel mismatch with dims");

    // 输出与输入同元素数、同 dtype、同 device；shape 记为转置后
    auto y = at::empty({dim0, dim1, dim3, dim2}, x.options());

    // extStream=nullptr：启动器内部自建流并同步（调用前 python 侧已 synchronize）
    int ret = LaunchNddmaTranspose(
        x.data_ptr(), y.data_ptr(),
        static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1),
        static_cast<uint32_t>(dim2), static_cast<uint32_t>(dim3),
        static_cast<uint32_t>(dtype), static_cast<uint32_t>(isMte3),
        nullptr);
    TORCH_CHECK(ret == 0, "LaunchNddmaTranspose failed, acl ret=", ret);
    return y;
}

// 自包含 host 路径：输入为 CPU 连续张量，启动器内部管理 device 内存与 ACL。
// 用于 torch_npu 不支持当前 SoC 时，仍以 PyTorch(CPU) 张量驱动真实 NPU kernel。
at::Tensor nddma_transpose_host(const at::Tensor& x,
                                int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                                int64_t dtype, int64_t isMte3, int64_t deviceId)
{
    TORCH_CHECK(x.device().is_cpu(), "host path expects a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.numel() == dim0 * dim1 * dim2 * dim3, "numel mismatch with dims");

    auto y = at::empty({dim0, dim1, dim3, dim2}, x.options());

    int ret = LaunchNddmaTransposeHost(
        x.data_ptr(), y.data_ptr(),
        static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1),
        static_cast<uint32_t>(dim2), static_cast<uint32_t>(dim3),
        static_cast<uint32_t>(dtype), static_cast<uint32_t>(isMte3),
        static_cast<int32_t>(deviceId));
    TORCH_CHECK(ret == 0, "LaunchNddmaTransposeHost failed, acl ret=", ret);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nddma_transpose", &nddma_transpose,
          "LastTranspose [0,1,3,2] via NDDMA kernel (PTA, device ptr)",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"), py::arg("dim2"),
          py::arg("dim3"), py::arg("dtype"), py::arg("isMte3"));
    m.def("nddma_transpose_host", &nddma_transpose_host,
          "LastTranspose [0,1,3,2] via NDDMA kernel (PTA, self-contained host path)",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"), py::arg("dim2"),
          py::arg("dim3"), py::arg("dtype"), py::arg("isMte3"), py::arg("deviceId"));
}
