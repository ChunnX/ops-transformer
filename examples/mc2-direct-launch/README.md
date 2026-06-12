# MC2 MoE `<<<>>>` Direct Launch

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

## 环境依赖

| 依赖 | 说明 |
|------|------|
| CANN 9.1.0+ | 含 bisheng 编译器 (`-xasc --npu-arch=dav-3510`) |
| PyTorch 2.6+ / torch_npu | NPU 运行时 |
| Ascend 950PR | NPU_ARCH=3510 |
| ops-transformer | 自定义算子包（含 CommContextManager） |
| pta-moe-test | golden 对比库（见下方步步骤 3） |

## 目录结构

```
mc2-direct-launch/
├── README.md
├── build.sh                          # 一键编译脚本
├── CMakeLists.txt                    # 顶层 cmake
├── cmake/                            # cmake 基础设施
├── csrc/
│   ├── dispatch_entry.cpp            # dispatch <<<>>> 入口（bisheng 编译）
│   ├── combine_entry.cpp             # combine <<<>>> 入口（bisheng 编译）
│   ├── mc2_torch.cpp                 # pybind11 Python 接口
│   ├── tiling_host.h                 # host 兼容 tiling 结构体
│   ├── CMakeLists.txt                # kernel + torch 编译配置
│   └── op_kernel/                    # kernel 源码（21 个头文件）
└── tests/
    ├── test_golden.py                # 主测试入口（dispatch/combine/cascade）
    ├── test_direct_launch.py         # dispatch 独立测试 + golden 对比 + CSV
    ├── test_combine_standalone.py    # combine 独立测试
    ├── test_dispatch_compare.py      # dispatch vs aclnn bit-identical 验证
    ├── gen_cpu_golden.py             # 生成 CPU golden 数据
    ├── run_kernel_launch.py          # 三步分离流程的 kernel 运行步骤
    └── compare_results.py            # 三步分离流程的结果对比步骤
```

## 快速开始

所有命令在 `examples/mc2-direct-launch/` 目录下执行。

### Step 1: 环境准备

```bash
# 初始化 CANN 环境（设置 ASCEND_HOME_PATH 等）
source $ASCEND_HOME_PATH/set_env.sh

# 激活 conda 环境
conda activate <your_env>

# 设置 RANK_TABLE
export RANK_TABLE_FILE=<your_rank_table_file>
```

### Step 2: 编译

```bash
bash build.sh
# 产出: build/csrc/mc2_direct_launch.so
```

### Step 3: 克隆对比依赖

部分测试需要 `pta-moe-test` 中的 `compare_output` 进行 golden 数据对比。

```bash
git clone https://gitcode.com/zhong-zixin/pta-moe-test.git
```

### Step 4: 运行测试

#### 4a. 主测试入口（推荐）

`test_golden.py` 支持三种模式，每种都与 CPU golden 对比：

```bash
python3 tests/test_golden.py --dispatch              # 仅 dispatch
python3 tests/test_golden.py --combine               # 仅 combine
python3 tests/test_golden.py --cascade               # 级联（dispatch→combine）
```

#### 4b. 独立测试

```bash
# dispatch 独立（含 golden 对比 + CSV 输出）
python3 tests/test_direct_launch.py

# combine 独立（全本地路由）
python3 tests/test_combine_standalone.py
```

#### 4c. bit-identical 验证

验证 `<<<>>>` dispatch 与 aclnn dispatch 的逐 bit 一致性：

```bash
python3 tests/test_dispatch_compare.py
```

#### 4d. 三步分离流程（用于调试）

```bash
# Step 1: 生成 CPU golden
python3 tests/gen_cpu_golden.py           # 输出: tests/golden.pt

# Step 2: 运行 kernel 产出 NPU 数据
python3 tests/run_kernel_launch.py        # 输入: tests/golden.pt，输出: tests/npu_output.pt

# Step 3: 对比 golden 与 NPU
python3 tests/compare_results.py          # 输入: tests/golden.pt + tests/npu_output.pt
```

## 测试设计说明

### 测试用例

| 文件 | 用途 | 依赖 pta-moe-test |
|------|------|:---:|
| `test_golden.py` | 主入口，dispatch/combine/cascade 三种模式 + CPU golden 对比 | 是 |
| `test_direct_launch.py` | dispatch 独立快测，含 golden 对比和 CSV 输出 | 是 |
| `test_combine_standalone.py` | combine 独立，全本地路由，不依赖 dispatch | 否 |
| `test_dispatch_compare.py` | `<<<>>>` vs aclnn bit-identical 验证 | 否 |
| `gen_cpu_golden.py` | 生成 CPU golden | 否 |
| `run_kernel_launch.py` | 运行 kernel 产出 NPU 数据 | 否 |
| `compare_results.py` | 对比 golden.pt vs npu_output.pt | 是 |

### 测试参数

| 参数 | 值 | 说明 |
|------|-----|------|
| BS (batch_size) | 55 | 每卡 token 数 |
| H (hidden_dim) | 1038 | 隐藏层维度 |
| K (top_k) | 1 | 每个 token 分发的专家数 |
| EP (ep_world_size) | 2 | 专家并行度 |
| MOE (num_experts) | 2 | 专家总数 |
| dtype | bf16 | 计算精度 |
| block_dim | 32 | 核函数 grid 大小 |

### 级联同步机制

dispatch `<<<>>>` 的 DPU 远程写入会占用 HCCL 通道资源，无法使用 `dist.barrier()`（会触发 HCCL socket 超时）。级联测试使用文件同步替代：

```
dispatch 完成后写 /tmp/mc2_gc_{rank} → combine 开始时等待所有 /tmp/mc2_gc_* 就绪
```

## 技术要点

### `<<<>>>` 调用链

```
Python → pybind11 (mc2_torch.cpp)
  → mc2_dispatch_launch() / mc2_combine_launch()
    → mc2_dispatch_kernel<<<blockDim, nullptr, stream>>>(...)
      → MoeDistributeDispatchV2<bf16,bf16,0,false,false>::Init/Process
```

### kernel 零修改

`csrc/op_kernel/` 下的头文件从 `mc2/` 目录原样复制，仅修改了 `#include` 相对路径为平级引用。kernel 类实现、算法逻辑、通信模式完全不变。

### host/device 隔离

- `tiling_host.h`: 只含 POD 结构体定义，不依赖 SDK 类型（`GM_ADDR`/`Mc2InitTiling` 等）
- `dispatch_entry.cpp` / `combine_entry.cpp`: bisheng -xasc 编译，可见 SDK 类型
- `mc2_torch.cpp`: 普通 g++ 编译，只引用 `tiling_host.h`

### tilingKey 编码

**dispatch**:
| key | 含义 |
|-----|------|
| 0 | fp16, no_fullmesh |
| 1 | bf16, no_fullmesh |
| 10 | fp16, fullmesh |
| 11 | bf16, fullmesh |

**combine**:
| key | 含义 |
|-----|------|
| 0 | fp16, no_quant |
| 1 | bf16, no_quant |
| 20 | fp16, int8_quant |
| 21 | bf16, int8_quant |
