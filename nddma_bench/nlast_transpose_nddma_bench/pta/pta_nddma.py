# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""PTA 接口封装：JIT 构建 pybind 扩展并暴露 nlast_transpose。

依赖：pta/build/libnlast_transpose_launch.so（由 ASC 编译，需先在本 pta/ 目录
cmake -DBUILD_PTA_LAUNCH=ON && make nlast_transpose_launch，详见 pta/README.md）。
registry-invoke 迁移后 compute 逻辑在 ../op_kernel/arch35/*.h，launch .so 独立建在 pta/build。

torch_npu>=2.7.1.post6 已支持 Ascend950PR 9579，走原生 PTA device 路径：
Python 侧用 `.npu()` device 张量，pybind 直接取 device data_ptr 驱动 kernel（零拷贝）。
启动器 extStream 传 nullptr（内部自建流并同步）；Python 侧前后 synchronize 保证跨流可见。
host 路径（nlast_transpose_host）保留作 CPU-张量 fallback。
"""

import os
import torch
import torch_npu  # noqa: F401  注册 NPU 后端
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# registry-invoke 迁移：launch .so 独立建在本 pta/ 目录的 build 子目录（pta/build），
# 而非算子根 build/（后者是主线 registry-invoke 工程的产物，不含 launch lib）。
_BUILD = os.path.join(_HERE, "build")
_CANN = (os.environ.get("ASCEND_HOME")
         or os.environ.get("ASCEND_HOME_PATH")
         or os.path.expanduser("~/Ascend/cann-9.0.0-beta.2"))

DTYPE_CODE = {torch.float16: 0, torch.float32: 1, torch.int8: 2}

_ext = load(
    name="nlast_transpose_ext",
    sources=[os.path.join(_HERE, "nlast_transpose_ext.cpp")],
    extra_include_paths=[os.path.join(_CANN, "include")],
    extra_ldflags=[
        f"-L{_BUILD}", "-lnlast_transpose_launch",
        f"-Wl,-rpath,{_BUILD}",
        f"-L{os.path.join(_CANN, 'lib64')}", "-lascendcl",
        f"-Wl,-rpath,{os.path.join(_CANN, 'lib64')}",
    ],
    verbose=True,
)


def nlast_transpose(x: torch.Tensor, is_mte3: int = 1) -> torch.Tensor:
    """原生 PTA device 路径：对 NPU 上 5D 张量做 [0,1,3,2,4] 转置，返回 (dim0,dim1,dim3,dim2,dim4)。"""
    assert x.dim() == 5, "expect 5D tensor (dim0,dim1,dim2,dim3,dim4)"
    assert x.dtype in DTYPE_CODE, f"unsupported dtype {x.dtype}"
    x = x.contiguous()
    if not x.is_npu:
        x = x.npu()
    d0, d1, d2, d3, d4 = x.shape
    torch.npu.synchronize()
    y = _ext.nlast_transpose(x, d0, d1, d2, d3, d4, DTYPE_CODE[x.dtype], is_mte3)
    torch.npu.synchronize()
    return y


def nlast_transpose_host(x: torch.Tensor, is_mte3: int = 1, device_id: int = 0) -> torch.Tensor:
    """自包含 host 路径（CPU 张量 fallback）：启动器内部 malloc/H2D/发射/D2H。"""
    assert x.dim() == 5, "expect 5D tensor (dim0,dim1,dim2,dim3,dim4)"
    assert x.dtype in DTYPE_CODE, f"unsupported dtype {x.dtype}"
    x = x.contiguous().cpu()
    d0, d1, d2, d3, d4 = x.shape
    return _ext.nlast_transpose_host(x, d0, d1, d2, d3, d4, DTYPE_CODE[x.dtype], is_mte3, device_id)


if __name__ == "__main__":
    # PTA 直调最小 demo：构造一个 5D NPU 张量 → nlast_transpose → 与 torch [0,1,3,2,4] 转置 bitwise 比对。
    # 需真实 NPU + torch_npu，且已按 pta/README.md 构建 pta/build/libnlast_transpose_launch.so。
    torch.manual_seed(0)
    x = torch.arange(8 * 4 * 64 * 64 * 16, dtype=torch.float16).reshape(8, 4, 64, 64, 16).npu()
    y = nlast_transpose(x, is_mte3=1)
    golden = x.transpose(2, 3).contiguous()
    ok = torch.equal(y.cpu(), golden.cpu())
    print(f"[demo] nlast_transpose x{tuple(x.shape)} -> y{tuple(y.shape)} bitwise_match={ok}")
