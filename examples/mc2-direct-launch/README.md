# MC2 MoE <<<>>> Direct Launch

MoE Distribute Dispatch/Combine V2 算子的 `<<<>>>` kernel 直调示例。

使用 bisheng 编译器 (`-xasc --npu-arch=dav-3510`) 从源码编译 kernel，通过 `<<<>>>` 语法直接启动，绕过 aclnn 框架。

## 验证结果

| 测试 | 结果 |
|------|------|
| dispatch 独立 | PASSED |
| combine 独立 | PASSED |
| dispatch→combine 级联 | PASSED |
| dispatch vs aclnn golden | bit-identical (max_diff=0) |
| combine vs aclnn golden | bit-identical (max_diff=0) |
| 级联 vs aclnn golden | bit-identical (max_diff=0) |
| combine int8 量化 | PASSED |

## 目录结构

```
mc2-direct-launch/
├── README.md
├── build.sh                          # 一键编译
├── CMakeLists.txt                    # 顶层 cmake
├── cmake/                            # cmake 基础设施（bisheng 编译器发现等）
├── csrc/
│   ├── dispatch_entry.cpp            # dispatch <<<>>> 入口（bisheng -xasc 编译）
│   ├── combine_entry.cpp             # combine <<<>>> 入口（bisheng -xasc 编译）
│   ├── mc2_torch.cpp                 # pybind11 Python 接口
│   ├── tiling_host.h                 # host 兼容 tiling 结构体（无 SDK 依赖）
│   ├── CMakeLists.txt                # kernel + torch 编译配置
│   └── op_kernel/                    # kernel 源码（从 mc2/ 零修改复制）
│       ├── moe_distribute_dispatch_v2.h
│       ├── moe_distribute_combine_v2.h
│       └── ... (共 21 个头文件)
└── tests/
    ├── test_direct_launch.py         # dispatch 独立测试
    ├── test_combine_standalone.py    # combine 独立测试
    └── test_dispatch_compare.py      # dispatch vs aclnn golden 对比
```

## 编译

```bash
source /home/l00946875/Ascend/cann/set_env.sh
export PATH="/root/miniconda3/envs/zzx_py310/bin:$PATH"
bash build.sh
# 输出: build/csrc/mc2_direct_launch.so
```

## 运行测试

```bash
export LD_LIBRARY_PATH=/home/l00946875/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export RANK_TABLE_FILE=/mnt/hccl/new_hccl_rootinfo.json

# dispatch 独立
python3 tests/test_direct_launch.py

# combine 独立
python3 tests/test_combine_standalone.py

# dispatch vs golden
python3 tests/test_dispatch_compare.py
```

## 技术要点

### <<<>>> 调用链

```
Python → pybind11 (mc2_torch.cpp)
  → mc2_dispatch_launch() / mc2_combine_launch()
    → mc2_dispatch_kernel<<<blockDim, nullptr, stream>>>(...)  ← 真正的 <<<>>>
      → MoeDistributeDispatchV2<bf16,bf16,0,false,false>::Init/Process
```

### kernel 零修改

`csrc/op_kernel/` 下的头文件从 `mc2/` 目录原样复制，仅修改了 `#include` 相对路径为平级引用。kernel 类实现、算法逻辑、通信模式完全不变。

### host/device 隔离

- `tiling_host.h`: 只含 POD 结构体定义，不依赖 SDK 类型（`GM_ADDR`/`Mc2InitTiling` 等）
- `dispatch_entry.cpp` / `combine_entry.cpp`: bisheng -xasc 编译，可见 SDK 类型
- `mc2_torch.cpp`: 普通 g++ 编译，只引用 `tiling_host.h`

### 级联同步

dispatch <<<>>> 的 DPU 远程写入会占用 HCCL 通道资源，导致 `dist.barrier()` 挂起。
级联测试使用文件同步替代 HCCL barrier。

### tilingKey 编码

**dispatch**:
- 0: fp16, no_fullmesh
- 1: bf16, no_fullmesh
- 10: fp16, fullmesh
- 11: bf16, fullmesh

**combine**:
- 0: fp16, no_quant
- 1: bf16, no_quant
- 20: fp16, int8_quant
- 21: bf16, int8_quant

## 依赖

- CANN 9.1.0+ (含 bisheng 编译器)
- PyTorch 2.6+ / torch_npu
- Ascend 950PR 硬件 (NPU_ARCH=3510)
- ops-transformer 自定义算子包（已安装）
- CommContextManager（来自 torch_extension）
