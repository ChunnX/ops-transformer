# QLI V2 Metadata CPU Test

在 CPU 环境下运行 QLI V2 Metadata 算子的负载均衡分核逻辑，验证分核结果。

## 项目结构

```
cpu_test/
├── CMakeLists.txt
├── main.cpp                                # 主程序
├── example_config.txt                      # 完整配置示例（所有字段）
├── example_tnd.txt                         # TND 布局示例（部分 tensor 自动生成）
├── example_910b.txt                        # 910B 芯片示例
├── op_kernel/
│   └── quant_lightning_indexer_v2_metadata.h          # 原始头文件（未修改）
├── op_kernel_aicpu/
│   ├── quant_lightning_indexer_v2_metadata_aicpu.cpp  # 原始实现（未修改）
│   └── quant_lightning_indexer_v2_metadata_aicpu.h    # 原始头文件（未修改）
└── stubs/                                  # CANN SDK mock 头文件
    ├── status.h / log.h / cpu_types.h
    ├── cpu_tensor_shape.h / cpu_tensor.h
    ├── cpu_attr_value.h / cpu_context.h / cpu_kernel.h
```

## 编译

```bash
cd cpu_test
mkdir -p build
cmake -S . -B build
cmake --build build
```

## 运行

### 方式一：配置文件（推荐）

```bash
./build/qli_v2_metadata_test <config_file>
```

配置文件为纯文本 `key = value` 格式，`#` 开头为注释。支持指定全部 15 个属性和 5 个 tensor。

### 方式二：默认参数

```bash
./build/qli_v2_metadata_test
```

不传参数时使用内置默认值（batch_size=2, max_seqlen_q=4, max_seqlen_k=128, ...）。

## 配置文件格式

### 属性（标量）

```
# --- 必填 ---
soc_version   = Ascend950          # 芯片版本：Ascend950 / Ascend910B
num_heads_q   = 8                  # Query 头数
num_heads_k   = 1                  # Key 头数
head_dim      = 128                # 头维度
topk          = 8                  # TopK 数量

# --- 可选（有默认值）---
batch_size    = 2                  # Batch 大小
max_seqlen_q  = 4                  # Query 最大序列长度
max_seqlen_k  = 128                # Key 最大序列长度
layout_q      = BSND               # Query 布局：BSND / TND
layout_k      = BSND               # Key 布局：BSND / TND
mask_mode     = 3                  # 0=DEFAULT, 1=ALL, 2=LEFT_UP_CAUSAL, 3=RIGHT_DOWN_CAUSAL, 4=BAND
cmp_ratio     = 1                  # 压缩比
quant_mode    = 0                  # 量化模式
aic_core_num  = 36                 # AIC 核数
aiv_core_num  = 72                 # AIV 核数
```

### Tensor（数组）

```
# 每个 batch 的实际序列长度, shape [batch_size]
seqused_q = [4, 8]
seqused_k = [128, 256]

# 累积序列长度（前缀和）, shape [batch_size + 1]
# 省略时自动从 seqused 计算
cu_seqlens_q = [0, 4, 12]
cu_seqlens_k = [0, 128, 384]

# 压缩残差, shape [batch_size]
# 省略时默认为全 0
cmp_residual_k = [0, 0]
```

### Tensor 自动填充规则

| Tensor | 省略时的自动生成规则 |
|--------|---------------------|
| `seqused_q` | `[max_seqlen_q×1, max_seqlen_q×2, ...]` |
| `seqused_k` | `[max_seqlen_k×1, max_seqlen_k×2, ...]` |
| `cu_seqlens_q` | `seqused_q` 的前缀和，首元素为 0 |
| `cu_seqlens_k` | `seqused_k` 的前缀和，首元素为 0 |
| `cmp_residual_k` | 全 0，长度 = batch_size |

输出中每个 tensor 后面会标注 `(from config)` 或 `(auto: ...)` 表示数据来源。

## 示例

```bash
# 完整配置（所有 tensor 手动指定）
./build/qli_v2_metadata_test example_config.txt

# TND 布局（cu_seqlens 和 cmp_residual 自动生成）
./build/qli_v2_metadata_test example_tnd.txt

# 910B 芯片
./build/qli_v2_metadata_test example_910b.txt

# 不传参数，使用内置默认值
./build/qli_v2_metadata_test
```

## 输出说明

程序输出三部分：

1. **Input Parameters**：所有属性值 + 5 个 tensor 的值及来源标注
2. **AIC (LI) Metadata**：每个 AIC 核的 BN2/M/S2 起止范围（FA 分核切分）
3. **AIV (LD) Metadata**：每个 AIV 核的 FD 归约任务分配（存在跨核行时输出）

## TTK 集成（npu_preprocess 仿真模式）

TTK 资产（`tests/assets/impl/npu_preprocess.py`）默认调用真正的 NPU metadata
算子生成 metadata；设置环境变量 `IS_NPU_NOW=0` 可切换为本 cpu_test 二进制
生成（与 pytest 仿真模式 `IS_NPU_NOW=False` 语义一致）：

```bash
export IS_NPU_NOW=0            # 0=cpu_test 仿真生成 metadata；不设或 1=NPU metadata 算子
export SOC_VERSION=Ascend950   # 芯片版本，须与上板芯片一致（默认 Ascend950）
export AIC_CORE_NUM=64         # 实际分核数，须与上板芯片一致（默认 64）
export AIV_CORE_NUM=64         # 实际分核数，须与上板芯片一致（默认 64）
```

说明：

- 直接 API（`torch.ops.cann_ops_transformer.lightning_indexer`）与 aclnn C API
  （`aclnnLightningIndexerV2`）两种用例规格（`LightningIndexerV2Spec` /
  `AclnnLightningIndexerV2Spec`）均已接入，均经
  `lightning_indexer_v2_golden.generate_metadata_from_cpu` 桥接。
- 运行环境需有 cmake 与 C++ 编译器（二进制缺失时自动编译），或预先执行
  `cmake -S . -B build && cmake --build build`。
- metadata 由二进制生成后仍会回写 TTK manual-data 零占位（sidecar 协议不变），
  回放场景不受运行模式影响。
- `AIC_CORE_NUM`/`AIV_CORE_NUM` 与上板芯片不一致时，分核计划与真实硬件不匹配，
  主算子负载均衡行为会失真，务必正确设置。
