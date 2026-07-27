// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// PTA pybind 扩展：把 Broadcast NDDMA kernel 暴露为 PyTorch 可调用算子——3D NDDMA 方案（DESIGN §9，权威）。
// 仅依赖 torch 张量 API + 启动器 C 接口，不直接调用 torch_npu C++ 符号。
//
// 5D 四场景（AB/BA/BAB/ABA）：输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。
// 单条 3D NDDMA：dim1 唯一核内 for；dim2/dim3/dim4 为 loopInfo 三轴；tile=dim2*dim3*dim4。

#include <torch/extension.h>
#include <cstdint>

extern "C" int LaunchNddmaBroadcast5d(void* xDev, void* yDev,
                                      uint32_t dim0, uint32_t dim1, uint32_t dim2,
                                      uint32_t dim3, uint32_t dim4, uint32_t scene,
                                      uint32_t dtype, uint32_t isMte3,
                                      uint32_t blockNum, void* extStream);

// 3D NDDMA 广播场景（DESIGN §9）：4 场景（AB/BA/BAB/ABA）→ 输出恒 (dim0,dim1,dim2,dim3,dim4)。
//   输入 numel 随 scene（广播轴长度=1，单条 3D 整块，DESIGN §9.2）：
//     AB     : dim0*dim1*dim2*dim3  （源块 (dim2,dim3,1)，dim4 广播）
//     BA/ABA : dim0*dim1*dim2*dim4  （源块 (dim2,1,dim4)，dim3 广播；BA≡ABA §9.1）
//     BAB    : dim0*dim1*dim3       （源块 (1,dim3,1)，dim2+dim4 广播）
//   dtype: 0=fp16 1=fp32 2=int8；isMte3=1 正常输出（=0 仅测读带宽）。
//   block_num: 核数扫描发射核数（0 → 缺省 min(dim0,56)）。
at::Tensor nddma_broadcast_5d(const at::Tensor& x,
                              int64_t dim0, int64_t dim1, int64_t dim2,
                              int64_t dim3, int64_t dim4, int64_t scene,
                              int64_t dtype, int64_t isMte3,
                              int64_t block_num = 0)
{
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    int64_t srcNumel;
    if (scene == 0) {                         // AB : src(d2,d3,1) 整块 d2*d3
        srcNumel = dim0 * dim1 * dim2 * dim3;
    } else if (scene == 2) {                  // BAB: src(1,d3,1) 整块 d3
        srcNumel = dim0 * dim1 * dim3;
    } else {                                  // BA / ABA: src(d2,1,d4) 整块 d2*d4（BA≡ABA §9.1）
        srcNumel = dim0 * dim1 * dim2 * dim4;
    }
    TORCH_CHECK(x.numel() == srcNumel,
                "5D src numel mismatch: expect ", srcNumel, " for scene ", scene,
                " got ", x.numel());

    // 输出恒满 5D shape (dim0,dim1,dim2,dim3,dim4)
    at::Tensor y = at::empty({dim0, dim1, dim2, dim3, dim4}, x.options());

    int ret = LaunchNddmaBroadcast5d(
        x.data_ptr(), y.data_ptr(),
        static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1),
        static_cast<uint32_t>(dim2), static_cast<uint32_t>(dim3),
        static_cast<uint32_t>(dim4), static_cast<uint32_t>(scene),
        static_cast<uint32_t>(dtype), static_cast<uint32_t>(isMte3),
        static_cast<uint32_t>(block_num), nullptr);
    TORCH_CHECK(ret == 0, "LaunchNddmaBroadcast5d failed, acl ret=", ret);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nddma_broadcast_5d", &nddma_broadcast_5d,
          "5D broadcast (AB/BA/BAB/ABA) via single 3D NDDMA, dim1-only for (PTA, device ptr)",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"), py::arg("dim2"),
          py::arg("dim3"), py::arg("dim4"), py::arg("scene"),
          py::arg("dtype"), py::arg("isMte3"), py::arg("block_num") = 0);
}
