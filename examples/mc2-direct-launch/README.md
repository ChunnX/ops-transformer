# MC2 Direct Launch (`<<<>>>` kernel 直调)

**moe_distribute_dispatch_v2** + **moe_distribute_combine_v2** 大EP DPU通信算子的 `<<<>>>` kernel 直调版本。

## 交付类型

**kernel + tiling + pybind11 direct-launch glue 闭环**（不依赖自定义算子 aclnn 包）

## 环境准备

```bash
# 1. Source CANN
source /home/l00946875/Ascend/cann/set_env.sh

# 2. 激活 conda 环境
conda activate mc2_py310_lbh

# 3. 设置 rank table（多卡测试需要）
export RANK_TABLE_FILE=/mnt/hccl/new_hccl_rootinfo.json

# 4. 创建 CANN 日志目录（避免 stdout 污染）
mkdir -p ~/ascend/log
```

## 编译

```bash
cd /home/l00946875/ops-transformer/examples/mc2-direct-launch
bash build.sh
```

编译成功输出：`build/csrc/mc2_direct_launch.so`

### 编译说明

- **Kernel 入口文件**（`dispatch_entry.cpp`, `combine_entry.cpp`）使用 BiSheng `-xasc` 双 pass 编译
- **Python 接口**（`mc2_torch.cpp`）使用 BiSheng 普通编译，链接 torch/torch_npu
- Kernel 实现头文件整文件从 `mc2/` 搬运，算法代码零修改
- 不支持 A2/A3/CCU 架构，仅支持 ascend950 (dav-3510)

## 测试

### 前提

测试需要 `comm_context.cpp`（JIT 编译创建 mc2Context），该文件位于：
```
../../torch_extension/cann_ops_transformer/ops/csrc/comm_context.cpp
```

### 测试方法

参照原始 `moe_operator_test_v3v4.py`：**CPU 提前计算 golden → NPU 运行 → 逐卡调用 `toolss.data_compare_np` 对比**。

| 模式 | golden 来源 | NPU 侧 |
|------|-----------|--------|
| `--dispatch` | `moe_dispatch_golden_cpu()` 在 CPU 模拟 all-to-all scatter | 各 rank 调用 `mc2_dispatch`，主进程收集 `expand_x` |
| `--combine` | `combine_golden_cpu()` 逐 token 计算 scale×expand_x 累加和 | 全本地路由，各 rank 调用 `mc2_combine`，主进程收集 `x_out` |
| `--cascade` | x 原值（dispatch→combine 回环恒等式） | 先 dispatch → 文件屏障 → combine，x_out 应等于原输入 |

### 标准测试命令

```bash
python3 tests/test_golden.py --dispatch   # 多 rank 通信，expand_x vs CPU golden
python3 tests/test_golden.py --combine    # 本地计算，x_out vs CPU golden
python3 tests/test_golden.py --cascade    # 回环，x_out vs 原输入（最权威）
```

### 测试参数

- BS=55, H=1038, K=1, EP=2, MOE=2
- dtype=bfloat16, blockDim=32
- workspace=16MB
- 所有 rank 使用相同随机种子（42+rank）

### 环境限制

- `test_dispatch_compare.py`（与 aclnn 逐 bit 对比）：需要自定义算子 aclnn 包 `libcust_opapi.so`，**未部署时跳过**。dispatch 正确性由 `--cascade` 回环保证。
- 多卡测试需要 2 张 Ascend950PR NPU 卡
- 每次跑完一轮测试后需要清理 NPU 进程：`fuser -k /dev/davinci*`

### 同步机制

- MC2 DPU 直驱通信占用 HCCL 通道，**不能使用 `dist.barrier()`**（socket 超时）
- 独立测试使用 `Lock+Value` 共享内存屏障
- Cascade 使用文件屏障（`/tmp/mc2_gc_*`）

## 目录结构

```
mc2-direct-launch/
├── CMakeLists.txt              # 顶层 cmake（bisheng 发现 + torch/torch_npu/python 查找）
├── build.sh                    # 一键编译
├── cmake/                      # cmake 模块
│   ├── ascend.cmake
│   ├── torch.cmake
│   ├── torch_npu.cmake
│   └── python.cmake
├── csrc/
│   ├── dispatch_entry.cpp      # dispatch kernel 入口（-xasc 编译）
│   ├── combine_entry.cpp       # combine kernel 入口（-xasc 编译）
│   ├── mc2_torch.cpp           # pybind11 Python 接口
│   ├── tiling_host.h           # Host/Kernel 共用 tiling POD struct
│   └── op_kernel/              # Kernel 实现头文件（零修改搬运）
│       ├── arch35/             # Ascend950 专用头文件
│       ├── moe_distribute_dispatch_v2*.h  # dispatch kernel
│       ├── moe_distribute_combine_v2*.h   # combine kernel
│       ├── moe_distribute_v2_*.h          # 共享基类
│       ├── hccl_context_holder.h          # v2 HcclOpParam 上下文
│       ├── mc2_moe_context.h              # Mc2MoeContext 结构
│       └── ... (共 26 个头文件)
├── tests/
│   └── test_golden.py          # 统一测试入口
├── pta-moe-test/               # golden 对比工具（软链）
└── README.md
```

## tilingKey 编码

| key | 算子 | dtype | 附加 |
|-----|------|-------|------|
| 0 | dispatch | fp16 | - |
| 1 | dispatch | bf16 | - |
| 10 | dispatch | fp16 | fullmesh |
| 11 | dispatch | bf16 | fullmesh |
| 0 | combine | fp16 | - |
| 1 | combine | bf16 | - |
| 20 | combine | fp16 | int8 quant |
| 21 | combine | bf16 | int8 quant |

## 性能验证

### 采集基线（原始算子）
```bash
cd /home/l00946875/ops-transformer/mc2/moe_distribute_dispatch_v2
python3 -m build --wheel -n && python3 -m pip install dist/*.whl --force-reinstall --no-deps
msprof op --warm-up=10 --output=./perf_baseline ./build_out/demo
```

### 采集直调性能
```bash
# 运行 cascade 测试
msprof op --warm-up=10 --output=./perf_after python3 tests/test_golden.py --cascade
```

### 判定标准
- 劣化率 < 3%：✅ 通过
- 3%-10%：⚠️ 需诊断
- > 10%：❌ 不可交付

## 常见问题

1. **`import torch_npu` 报 `[LOG_WARNING]` 污染输出**：创建 `~/ascend/log` 目录
2. **编译报 `cannot find -lcust_opapi`**：a3xx 类不需要，跳过 `test_dispatch_compare.py`
3. **HCCL channel failed ret: 9**：`fuser -k /dev/davinci*` 清理上一轮进程
4. **dist.barrier() 超时**：MC2 DPU 运行时禁用 HCCL barrier，改用 Lock+Value/文件屏障
5. **build/ 目录 root 属主**：`sudo rm -rf build` 后重新编译

## 代码迁移原则

- ✅ Kernel 实现代码零修改（整文件搬运）
- ✅ Tiling 计算逻辑零修改
- ✅ 只改注册框架胶水（入口改造 + TilingData POD化）
- ✅ `<<<>>>` 直调：tiling struct by-value 入参，去掉 `GET_TILING_DATA`/`TILING_KEY_IS`
- ❌ 不修改 kernel 类成员函数、不合并分支、不优化公式
