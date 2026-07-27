# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""PTA 接口封装：JIT 构建 pybind 扩展并暴露 nddma_broadcast_5d——3D NDDMA 方案（DESIGN §9，权威）。

依赖：pta/build/libnddma_broadcast_launch.so（由 ASC 编译，需先在本 pta/ 目录
cmake -DBUILD_PTA_LAUNCH=ON && make nddma_broadcast_launch，详见 pta/README.md）。
registry-invoke 迁移后 compute 逻辑在 ../op_kernel/arch35/*.h，launch .so 独立建在 pta/build。

torch_npu>=2.7.1.post6 已支持 Ascend950PR 9579，走原生 PTA device 路径：
Python 侧用 `.npu()` device 张量，pybind 直接取 device data_ptr 驱动 kernel（零拷贝）。
启动器 extStream 传 nullptr（内部自建流并同步）；Python 侧前后 synchronize 保证跨流可见。

5D 四场景（AB/BA/BAB/ABA）：单条 3D NDDMA（dim1 唯一核内 for，dim2/dim3/dim4 为 loopInfo 三轴，
tile=dim2*dim3*dim4）；输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。
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
    name="nddma_broadcast_ext",
    sources=[os.path.join(_HERE, "nddma_broadcast_ext.cpp")],
    extra_include_paths=[os.path.join(_CANN, "include")],
    extra_ldflags=[
        f"-L{_BUILD}", "-lnddma_broadcast_launch",
        f"-Wl,-rpath,{_BUILD}",
        f"-L{os.path.join(_CANN, 'lib64')}", "-lascendcl",
        f"-Wl,-rpath,{os.path.join(_CANN, 'lib64')}",
    ],
    verbose=True,
)


# ---- 5D 广播场景（DESIGN §9，权威）：AB/BA/BAB/ABA 四场景（单条 3D NDDMA）----
SCENE_AB = 0    # (dim2/dim3 保留, dim4 广播)       输入 (d0,d1,dim2,dim3,1)
SCENE_BA = 1    # (dim2/dim4 保留, dim3 广播)       输入 (d0,d1,dim2,1,dim4)
SCENE_BAB = 2   # (dim3 保留, dim2+dim4 广播)       输入 (d0,d1,1,dim3,1)
SCENE_ABA = 3   # (dim2/dim4 保留, dim3 广播)       输入 (d0,d1,dim2,1,dim4)（≡ BA，DESIGN §9.1）


def nddma_broadcast_5d(x: torch.Tensor, dim0: int, dim1: int, dim2: int,
                       dim3: int, dim4: int, scene: int, is_mte3: int = 1,
                       block_num: int = 0) -> torch.Tensor:
    """5D 广播场景直调（DESIGN §9）：4 场景统一为单条 3D NDDMA（dim1 唯一核内 for）。
      x 为 NPU 连续张量，按 scene 的输入 shape（广播轴长度=1）构造。
      返回满 5D 输出 (dim0,dim1,dim2,dim3,dim4)（device 张量）。
      block_num>0 → 核数扫描按指定核数发射（§9.6）；0 → 缺省 min(dim0,56)。
    """
    assert x.dtype in DTYPE_CODE, f"unsupported dtype {x.dtype}"
    x = x.contiguous()
    if not x.is_npu:
        x = x.npu()
    torch.npu.synchronize()
    y = _ext.nddma_broadcast_5d(x, dim0, dim1, dim2, dim3, dim4, scene,
                                DTYPE_CODE[x.dtype], is_mte3, block_num)
    torch.npu.synchronize()
    return y


def _make_src_5d(dim0, dim1, dim2, dim3, dim4, scene, dtype):
    """按 scene 构造 5D 输入张量（广播轴长度=1，NPU device；单条 3D 整块，DESIGN §9.2）。
      AB : (d0,d1,dim2,dim3,1)   BA/ABA : (d0,d1,dim2,1,dim4)   BAB : (d0,d1,1,dim3,1)
      contiguous 布局保证相邻 g（dim0 分核 × dim1 for）的源块在 GM 内连续排列，
      与 kernel SrcOff 整块基址（AB=d2*d3 / BA·ABA=d2*d4 / BAB=d3）逐 scene 对齐（§9.2 注）。
    """
    if scene == SCENE_AB:
        shp = (dim0, dim1, dim2, dim3, 1)      # src(d2,d3,1) 整块 d2*d3
    elif scene == SCENE_BAB:
        shp = (dim0, dim1, 1, dim3, 1)         # src(1,d3,1) 整块 d3
    else:  # BA / ABA：src(d2,1,d4) 整块 d2*d4（BA≡ABA §9.1）
        shp = (dim0, dim1, dim2, 1, dim4)
    total = 1
    for s in shp:
        total *= s
    if dtype == torch.int8:
        return torch.randint(-128, 128, shp, dtype=torch.int8)
    flat = (torch.arange(1, total + 1, dtype=torch.float32) % 2048)
    return flat.reshape(shp).to(dtype)


def bcast_5d(dim0, dim1, dim2, dim3, dim4, scene, dtype=torch.float32,
             is_mte3: int = 1, block_num: int = 0):
    """便捷封装：按 scene 构造输入 + 调 kernel + 返回 (y, golden)。
      golden = x.broadcast_to((d0,d1,d2,d3,d4)).contiguous()（§9.5 bitwise golden）。"""
    x = _make_src_5d(dim0, dim1, dim2, dim3, dim4, scene, dtype).npu()
    y = nddma_broadcast_5d(x, dim0, dim1, dim2, dim3, dim4, scene,
                           is_mte3=is_mte3, block_num=block_num)
    golden = x.broadcast_to((dim0, dim1, dim2, dim3, dim4)).contiguous()
    return y, golden
