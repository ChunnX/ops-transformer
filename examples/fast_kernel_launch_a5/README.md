# Fast Kernel Launch — A5 MoE Dispatch & Combine

A5（Ascend 950PR/950DT, dav-3510）MTE 通信模式下，MoE Distribute Dispatch V2 / Combine V2 的 `<<<>>>` 直调工程。方案同 A3 版 `fast_kernel_launch_example`，通过 `updateContext` 预构建 `Mc2MoeContext`（含真实 HCCL buffer 地址），kernel 绕开 `GetHcclContext<>()` 直取 buffer。

## 快速开始

```bash
cd examples/fast_kernel_launch_a5
rm -rf build && python3 -m build --wheel -n
pip install dist/*.whl --force-reinstall --no-deps
```

### 验证

```bash
# 1. 静态检查（CPU/NPU 通用）：确认 .so 包含注册符号
nm ascend_ops/_C.abi3.so | grep "TORCH_LIBRARY.*static_init" | grep -oP 'ascend_ops[^E]+' | sort -u
# 预期: Update_Context, MoeDistributeCombineV2A5, MoeDistributeDispatchV2A5

# 2. 全局初始化函数
nm ascend_ops/_C.abi3.so | grep "_GLOBAL__sub_I"
readelf -x .init_array ascend_ops/_C.abi3.so | tail -2

# 3. NPU 机器：ctypes 加载 + 触发懒加载（PyTorch dir() 需先访问才显示）
python3 -c "
import torch, ctypes
ctypes.cdll.LoadLibrary('ascend_ops/_C.abi3.so')
torch.ops.ascend_ops.updateContext
torch.ops.ascend_ops.npu_moe_distribute_dispatch_v2_a5
torch.ops.ascend_ops.npu_moe_distribute_combine_v2_a5
print(dir(torch.ops.ascend_ops))
"
# 包含 updateContext, npu_moe_distribute_dispatch_v2_a5, npu_moe_distribute_combine_v2_a5 即可
```

### 多卡测试（仅 NPU）

```bash
torchrun --nproc_per_node=2 tests/test_a5_mte.py
```

## 调用示例

```python
import torch, torch_npu, torch.distributed as dist, ascend_ops

dist.init_process_group(backend="hccl")
rank, world = dist.get_rank(), dist.get_world_size()

x = torch.randn(4, 128, dtype=torch.float16).npu()
eid = torch.randint(0, 4, (4, 2), dtype=torch.int32).npu()

# 1) 创建 mc2Context
mc2_ctx, buf_size = torch.ops.ascend_ops.updateContext("group_ep", world)

# 2) Dispatch
out = torch.ops.ascend_ops.npu_moe_distribute_dispatch_v2_a5(
    x, eid, "group_ep", world, rank, 4, mc2_ctx, buf_size)
expand_x, dyn_scales, assist, token_nums, ep_cnts, _ = out

# 3) Combine
x_out = torch.ops.ascend_ops.npu_moe_distribute_combine_v2_a5(
    expand_x, eid, assist, ep_cnts, dyn_scales,
    "group_ep", world, rank, 4, mc2_ctx, buf_size)
```

## 目录结构

```
├── CMakeLists.txt          # ASC + Torch + bisheng 编译
├── setup.py                # python3 -m build --wheel 构建
├── cmake/                  # ascend.cmake, func.cmake, torch.cmake, torch_npu.cmake
├── csrc/
│   ├── op_kernel/          # 共享 kernel 头文件
│   ├── extension.cpp       # PYBIND11_MODULE + TORCH_LIBRARY
│   ├── torch/              # g++ 编译（torch binding）
│   │   ├── dispatch_torch.cpp
│   │   ├── combine_torch.cpp
│   │   └── update_context_torch.cpp
│   ├── moe_distribute_dispatch_v2/ascend950/  # bisheng -xasc 编译（entry.cpp）
│   └── moe_distribute_combine_v2/ascend950/   # 同上
└── tests/test_a5_mte.py    # torchrun 端到端测试
```

## 架构适配

所有路径基于环境变量 `ASCEND_HOME_PATH` 和 `uname -m` 动态推导，不写死 `aarch64-linux`。`NPU_ARCH` 默认 `ascend950`，可通过 cmake 参数覆盖。
