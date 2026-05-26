# block_sparse_attention CPU 验证

本目录提供 `block_sparse_attention` 算子的 CPU 参考实现，用于与 NPU 侧 aclnn 测试结果做数值对比。NPU 测试入口在 `../examples/test_aclnn_block_sparse_attention.cpp`；本目录只负责读取 C++ 落盘的输入/输出 tensor，在 CPU 上复算并校验。

## 目录结构

```
py_verify/
├── README.md              # 本文档
├── verify_bsa_cpu.py      # CPU 参考实现 + 对比脚本
└── bsa_output/            # NPU 测试落盘目录（运行时生成，仓库内保持为空）
    ├── .gitignore
    ├── metadata.json
    ├── query.bin
    ├── key.bin
    ├── value.bin
    ├── block_sparse_mask.bin
    └── attention_out.bin
```

`bsa_output/` 不在仓库中预置真实结果。若本地出现上述文件，均为 C++ 测试运行产物，可安全删除后重新跑测试。

## 验证流程

```
examples/test_aclnn_block_sparse_attention.cpp
        │  构造 Q/K/V/mask，NPU 执行 aclnnBlockSparseAttention
        ▼
py_verify/bsa_output/
        │  metadata.json + query/key/value/mask/out 二进制文件
        ▼
verify_bsa_cpu.py
        │  直接读入落盘 tensor，CPU 块稀疏注意力（Online Softmax）
        ▼
PASS / FAIL
```

1. 在 Ascend 环境编译并运行 C++ 测试（工作目录建议在 `examples/`）。
2. 测试将 C++ 侧构造的 Q/K/V/mask 与 NPU 输出一并写入 `py_verify/bsa_output/`。
3. 在任意有 Python 3 与 NumPy 的机器上运行 `verify_bsa_cpu.py`，读取落盘 tensor 并对比。

## 依赖

- Python 3.8+
- NumPy

```bash
pip install numpy
```

## 使用方式

### 1. NPU 侧跑测试并落盘

在 `examples/` 目录下执行（具体编译命令以工程构建说明为准）：

```bash
./test_aclnn_block_sparse_attention
```

默认将结果写入 `attention/block_sparse_attention/py_verify/bsa_output/`（`build.sh --run_example` 从 `build/` 目录运行时自动解析到该路径）。每次成功运行会**覆盖**该目录下全部落盘文件。也可通过环境变量或命令行指定：

```bash
export BSA_VERIFY_OUTPUT_DIR=/path/to/custom_output
./test_aclnn_block_sparse_attention

# 或
./test_aclnn_block_sparse_attention /path/to/custom_output
```

### 2. CPU 侧对比

`verify_bsa_cpu.py` **只读取**落盘文件，不会写入或修改 `bsa_output/` 内容。

在 `py_verify/` 目录下：

```bash
python verify_bsa_cpu.py
```

指定落盘目录：

```bash
python verify_bsa_cpu.py --output-dir ./bsa_output
python verify_bsa_cpu.py --output-dir /path/to/custom_output
```

成功时输出示例：

```
Running CPU block_sparse_attention: batch=1, q=1024, kv=1024, ...
max_abs_diff=..., mismatch=0/131072, tol(atol=0.05, rtol=0.05)
PASS: CPU reference matches NPU output.
```

失败时会打印前若干 mismatch 的 flat index 及 CPU/NPU 数值。

## 落盘文件格式

### metadata.json

记录测试参数与各二进制文件的 shape/文件名。Python **不**根据填充模式或 seed 重建输入，而是直接读取 `query.bin` / `key.bin` / `value.bin` / `block_sparse_mask.bin`。

| 字段 | 说明 |
|------|------|
| `batch`, `q_seqlen`, `kv_seqlen` | batch 与序列长度 |
| `num_heads`, `num_kv_heads`, `head_dim` | head 维度 |
| `block_shape_x`, `block_shape_y` | 块形状 |
| `q_block_num`, `kv_block_num` | Q/KV 块数量 |
| `sparsity_mode` | 稀疏模式（0~3，仅作记录） |
| `scale_value`, `inner_precise` | 缩放系数与精度模式 |
| `input_dtype` | Q/K/V 落盘 dtype，固定 `float32` |
| `query_file`, `query_shape` | Query tensor 文件与 shape |
| `key_file`, `key_shape` | Key tensor 文件与 shape |
| `value_file`, `value_shape` | Value tensor 文件与 shape |
| `block_sparse_mask_file`, `block_sparse_mask_shape`, `block_sparse_mask_dtype` | 块稀疏 mask（`int8`） |
| `output_dtype`, `output_file`, `output_shape` | NPU 输出 |

### 二进制 tensor 文件

| 文件 | 类型 | 布局 |
|------|------|------|
| `query.bin` | `float32` | TND `[totalQTokens, numHeads, headDim]` |
| `key.bin` | `float32` | TND `[totalKvTokens, numKvHeads, headDim]` |
| `value.bin` | `float32` | TND `[totalKvTokens, numKvHeads, headDim]` |
| `block_sparse_mask.bin` | `int8` | `[batch, numHeads, qBlockNum, kvBlockNum]` |
| `attention_out.bin` | `float32` | TND `[totalQTokens, numHeads, headDim]` |

均为小端序、无文件头、按 shape 行优先展平。Q/K/V/out 在 C++ 侧由 fp16 host 数据转为 `float32` 落盘，与控制台打印精度一致；Python 读入后再转为 `float16` 参与 CPU 参考计算。

## 与 C++ 测试的对应关系

C++ 测试中的关键开关（见 `test_aclnn_block_sparse_attention.cpp`）：

| C++ 项 | 默认值（验证模式） | 说明 |
|--------|-------------------|------|
| `kCpuVerifyMode` | `true` | `true` 时 `q/kv_seqlen=1024`，便于 CPU 对比；压测改为 `false` 则 seq=1048576 |
| `sparsityMode` | `SPARSITY_NONE` (0) | 0 全稠密，1 高稀疏，2 下三角 causal，3 低稀疏 |
| `inputFillMode` | `INPUT_FILL_ALL_ONE` (0) | 0：Q/K/V 全 1.0；1：Q/K/V 随机 0/1 |
| `innerPrecise` | `4` | 混合精度（Softmax fp16 + RescaleO fp32） |
| 布局 | TND | `[totalTokens, numHeads, headDim]` |

无论 C++ 使用何种输入填充方式，Python 均以落盘 tensor 为准。

## CPU 参考实现说明

`verify_bsa_cpu.py` 读入落盘的 Q/K/V/mask 后，按块稀疏 mask 遍历有效 Q-KV 块对，对每个 Q token 使用 **Online Softmax** 跨 KV 块累加，语义与算子文档中的 MM1 → Softmax → MM2 流水线一致：

- MM1：`S = Q × K^T × scale`
- Softmax：行方向 Online Softmax（支持 `inner_precise` 0/1/4）
- MM2：`O = P × V`，输出经 RescaleO 归一化

## 对比容差

按 `inner_precise` 自动选择：

| inner_precise | 含义 | atol / rtol |
|---------------|------|-------------|
| 0 | 全 fp32 | 1e-3 |
| 1 | 全 fp16 | 2e-2 |
| 4 | 混合精度 | 5e-2 |

## 当前限制

- `kCpuVerifyMode=false` 且全稠密 mask 时，序列过长会导致 CPU 参考无法在合理时间内完成，此类场景仅适合 NPU 压测，不宜做 CPU 全量对比。

## 相关路径

- NPU 测试：`../examples/test_aclnn_block_sparse_attention.cpp`
- 算子语义说明：`AI_learning/document/main/Ascend-ops/A5算子调优手段/BSA详解/block_sparse_attention_详解.md`
