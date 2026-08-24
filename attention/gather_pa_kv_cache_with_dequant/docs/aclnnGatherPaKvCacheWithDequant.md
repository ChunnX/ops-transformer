# aclnnGatherPaKvCacheWithDequant

[📄 查看源码](../op_host/gather_pa_kv_cache_with_dequant_def.cpp)

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 功能说明

融合 **GatherPaKvCache + Dequant** 算子。根据 `blockTables` 中的 blockId 和 `seqLens` 中的序列长度，从 `keyCache`/`valueCache` 中将内存不连续的 paged token 搬运、拼接成连续序列，并在 kernel 内完成反量化：

```
Key 路径:   INT8 cache     --(per-token-per-head  动态 scale)-->  BF16
Value 路径: FP8 e4m3 cache --(per-head-per-channel 静态 scale)--> BF16
```

等价于把以下小算子链融合为单一 kernel：
`GatherPaKvCache → Cast(int8→float) → Mul(×scale) → Cast(float→bf16)`（Key）
`GatherPaKvCache → AscendAntiQuantV2(fp8→bf16)`（Value）

- **isSeqLensCumsum 为 true**：`seqLens` 为累加和，`seqLens[-1]` 即输出第一维大小。
- **isSeqLensCumsum 为 false**：`seqLens` 各值之和为输出第一维大小。
- **cache_mode**：控制 KV cache 的量化方式与物理布局（输出始终为 ND BF16，不受影响）。合法取值见下表，未识别的值会在 tiling 阶段报错。

  | cache_mode | key scale 量化方式 | cache 物理布局 | 说明 |
  |---|---|---|---|
  | `"Norm"`（默认） | per-token-per-head 动态 | ND 连续 | key 走 `GatherDequantKey`（fp32 Brcb 广播 scale） |
  | `"C8"` | per-channel 静态 `[numHeadsK, headDimK]` | ND 连续 | key 走 `GatherDequantValue`（int8→half→Mul→BF16） |
  | `"Norm_NZ"` | per-token-per-head 动态 | FRACTAL_NZ 5D | NZ 物理由生产 `npu_scatter` 写入，`LoadSrcNzToUb` 读成 UB 内 ND |
  | `"C8_NZ"` | per-channel 静态 | FRACTAL_NZ 5D | C8 量化 + NZ 物理的组合 |

  > `C8`（量化方式）与 `NZ`（物理布局）两个维度正交：`Norm_NZ` 是 paged per-token scale + NZ 物理，**不**隐含 per-channel。误设 `C8` 会使 Key 错走 per-channel 路径，把 paged scale 当静态 scale 读，导致 `dim1+` 维度错 scale。

## 函数原型

每个算子分为[两段式接口]，必须先调用 `aclnnGatherPaKvCacheWithDequantGetWorkspaceSize` 获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnGatherPaKvCacheWithDequant` 执行计算。

```Cpp
aclnnStatus aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(
  const aclTensor *keyCache,
  const aclTensor *valueCache,
  const aclTensor *keyScaleCache,
  const aclTensor *valueScale,
  const aclTensor *blockTables,
  const aclTensor *seqLens,
  aclTensor       *keyRef,
  aclTensor       *valueRef,
  const aclTensor *seqOffsetOptional,
  bool             isSeqLensCumsum,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherPaKvCacheWithDequant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGatherPaKvCacheWithDequantGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| keyCache | 输入 | INT8 paged key cache，shape `[num_blocks, block_size, num_heads_k, head_dim_k]`，ND 或 FRACTAL_NZ。 |
| valueCache | 输入 | FP8 e4m3（DT_FLOAT8_E4M3FN）paged value cache，shape `[num_blocks, block_size, num_heads_v, head_dim_v]`，同 keyCache 排布。 |
| keyScaleCache | 输入 | FLOAT，key 反量化 scale，per-token-per-head 动态量化，shape `[num_blocks, block_size, num_heads_k]`，与 keyCache 同构（少最后一维 head_dim_k）。 |
| valueScale | 输入 | FLOAT，value 反量化 scale，per-head-per-channel 静态量化，shape `[num_heads_v, head_dim_v]`，单独连续存储。 |
| blockTables | 输入 | INT32，每个 batch 中 KV Cache 逻辑块到物理块的映射，shape `[batch, block_indices]`。 |
| seqLens | 输入 | INT32，每个 batch 对应的序列长度。非累加和时 shape `[batch]`；累加和时 shape `[batch + 1]`。 |
| keyRef | 输入/输出 | BF16，反量化后的 key。输入时其 shape 用于推导输出，输出时 in-place 写入结果。shape `[num_tokens, num_heads_k, head_dim_k]`。 |
| valueRef | 输入/输出 | BF16，反量化后的 value。同 keyRef，shape `[num_tokens, num_heads_v, head_dim_v]`。 |
| seqOffsetOptional | 输入 | INT32，可选，blockTables 获取 blockId 时的首偏移，shape `[batch]`。不使用时传 nullptr。 |
| isSeqLensCumsum | 输入 | Bool，表示 seqLens 是否为累加和。 |
| workspaceSize | 输出 | 该算子执行所需的 workspace 大小。 |
| executor | 输出 | 包含算子计算流程的执行器。 |

### 返回值说明

返回 `aclnnStatus`：
- `ACLNN_SUCCESS`：成功。
- 其它：失败，具体含义参考 `aclnn_base.h` 错误码。

## aclnnGatherPaKvCacheWithDequant

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| workspace | 输入 | 在 Device 上申请的 workspace 内存地址。 |
| workspaceSize | 输入 | workspace 大小，由第一段接口计算得出。 |
| executor | 输入 | 第一段接口返回的执行器。 |
| stream | 输入 | 指定执行操作的 Stream。 |

### 返回值说明

返回 `aclnnStatus`：
- `ACLNN_SUCCESS`：成功。
- 其它：失败。

## 约束说明

- **确定性计算**：本算子默认确定性实现。
- **dtype 约束**：
  - `key_cache`：INT8（per-token-per-head int8 量化）。
  - `value_cache`：FP8 e4m3（`DT_FLOAT8_E4M3FN`，对齐非融合算子 `AscendAntiQuantV2` 的 `fp8→bf16` 路径）。
  - `key_scale_cache` / `value_scale`：FLOAT。
  - `block_tables` / `seq_lens` / `seq_offset`：INT32。
  - `key` / `value`（ref 与输出）：BF16。
- **排布约束**：
  - `Norm` / `C8` 模式：`key_cache`/`value_cache` 为 ND 排布。
  - `Norm_NZ` / `C8_NZ` 模式：`key_cache`/`value_cache` 为 FRACTAL_NZ 物理排布（NZ 5D，由生产 `npu_scatter` 写入）；输出 key/value 始终为 ND。
  - NZ 模式要求 `head_dim` 32 对齐（`npu_scatter` 按 32B 块写入，非 32 倍数会在 tiling 阶段报错）。
- **shape 约束（Norm 模式）**：

  | 张量 | shape |
  |---|---|
  | key_cache | `[num_blocks, block_size, num_heads_k, head_dim_k]` |
  | value_cache | `[num_blocks, block_size, num_heads_v, head_dim_v]` |
  | key_scale_cache | `[num_blocks, block_size, num_heads_k]` |
  | value_scale | `[num_heads_v, head_dim_v]` |
  | block_tables | `[batch, block_indices]` |
  | seq_lens | `[batch]`（非累加和）或 `[batch + 1]`（累加和） |
  | key / value | `[num_tokens, num_heads, head_dim]` |
  | seq_offset | `[batch]`（可选） |

- **量化方式差异**：
  - `key_scale_cache`：per-token-per-head **动态**量化，存储在 KV cache 中（每个 token 每个 head 一个 scale）。
  - `value_scale`：per-head-per-channel **静态**量化，单独连续存储（每个 head 每个 channel 一个 scale）。
- **属性**（通过 op attr 设置，aclnn 接口使用默认值）：
  - `cache_mode`：String，默认 `"Norm"`。合法值 `Norm` / `C8` / `Norm_NZ` / `C8_NZ`，未识别值在 tiling 阶段报错。
  - `is_seq_lens_cumsum`：Bool，默认 `true`。
  - `key_dst_type`：Int，默认 `DT_BF16`。**保留兼容，实际恒为 BF16** —— InferDataType 硬编码返回 BF16，传入其它值不影响输出 dtype。
  - `value_dst_type`：Int，默认 `DT_BF16`。同 `key_dst_type`，**保留兼容，实际恒为 BF16**。

## 调用示例

示例代码见 [`examples/test_aclnn_gather_pa_kv_cache_with_dequant.cpp`](../examples/test_aclnn_gather_pa_kv_cache_with_dequant.cpp)，编译与执行过程参考 CANN 样例文档。

```bash
aarch64-linux-gnu-g++ examples/test_aclnn_gather_pa_kv_cache_with_dequant.cpp \
    -I$ASCEND_HOME_PATH/vendors/customize/op_api/include \
    -L$ASCEND_HOME_PATH/vendors/customize/op_api/lib \
    -laclnn_gather_pa_kv_cache_with_dequant -laclnn_base -lascendcl
```
