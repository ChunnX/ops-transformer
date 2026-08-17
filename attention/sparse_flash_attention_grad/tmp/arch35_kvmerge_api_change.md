# SparseFlashAttentionGrad arch35 kvMerge 实现说明

## 1. 结论

`SparseFlashAttentionGrad` 的公开 API 由 arch22 和 arch35 共用。

arch22 在提交 `46a1f1592` 中已经完成共享接口契约修改：

```text
value: REQUIRED -> OPTIONAL
d_value: REQUIRED -> OPTIONAL
```

并定义 kvMerge 语义：

```text
value == nullptr
dValueOut == nullptr
dKeyOut = dK + dV
```

因此 arch35 不需要再次修改 OpDef、ACLNN 签名或 torch_npu schema。本次工作只让 arch35 Host、tiling 和 Kernel 实现 arch22 已经定义好的 kvMerge 契约。

## 2. 修改范围

本次保留的修改范围如下：

| 模块 | 修改内容 |
| --- | --- |
| arch35 Host | 识别 optional `value`，生成 kvMerge tiling key |
| arch35 Pre | kvMerge 时跳过无用的 dV workspace 清零 |
| arch35 Scatter | 将缩放后的 dK 与 dV 相加后写入 dK workspace |
| arch35 Post | kvMerge 时只搬运 dQ、合并后的 dK，不写 dV |
| arch35 Kernel dispatch | 增加 `KvMerge` 模板参数并复用 Key 作为 Value |
| UT | 覆盖普通场景、kvMerge、kvMerge+RoPE |

以下共享或非 kvMerge 修改不属于本次范围，已回退：

- 不修改共享 `SparseFlashAttentionGrad` OpDef。
- 不修改公开 ACLNN 函数签名和参数顺序。
- 不修改共享 InferShape 的 RoPE 校验规则。
- 不扩展共享 `OutputIndex` 枚举。
- 不引入 arch35 专属 torch_npu/op-plugin schema。
- 不包含单 RoPE 输入、错误 dKeyRope shape 等独立 RoPE 修复。

## 3. arch22 已有共享 API

### 3.1 OpDef

arch22 已经将内部输入顺序调整为：

| 索引 | 输入 | 属性 |
| --- | --- | --- |
| 0 | `query` | REQUIRED |
| 1 | `key` | REQUIRED |
| 2 | `sparse_indices` | REQUIRED |
| 3 | `d_out` | REQUIRED |
| 4 | `out` | REQUIRED |
| 5 | `softmax_max` | REQUIRED |
| 6 | `softmax_sum` | REQUIRED |
| 7 | `value` | OPTIONAL |
| 8 | `actual_seq_lengths_query` | OPTIONAL |
| 9 | `actual_seq_lengths_kv` | OPTIONAL |
| 10 | `query_rope` | OPTIONAL |
| 11 | `key_rope` | OPTIONAL |

共享输出槽位为：

| 索引 | 输出 | 属性 |
| --- | --- | --- |
| 0 | `d_query` | REQUIRED |
| 1 | `d_key` | REQUIRED |
| 2 | `d_value` | OPTIONAL |
| 3 | `d_query_rope` | OPTIONAL |
| 4 | `d_key_rope` | OPTIONAL |

arch35 直接复用该定义，不再修改。

### 3.2 ACLNN

公开 ACLNN 参数顺序保持兼容：

```cpp
aclnnSparseFlashAttentionGradGetWorkspaceSize(
    query,
    key,
    value,
    sparseIndices,
    dOut,
    out,
    softmaxMax,
    softmaxSum,
    // optional inputs and attrs
    dQueryOut,
    dKeyOut,
    dValueOut,
    dQueryRopeOut,
    dKeyRopeOut,
    workspaceSize,
    executor);
```

kvMerge 调用由 arch22 已有 wrapper 支持：

```cpp
aclnnSparseFlashAttentionGradGetWorkspaceSize(
    query,
    key,
    nullptr,       // value
    sparseIndices,
    dOut,
    out,
    softmaxMax,
    softmaxSum,
    // optional inputs and attrs
    dQueryOut,
    dKeyOut,
    nullptr,       // dValueOut
    dQueryRopeOut,
    dKeyRopeOut,
    workspaceSize,
    executor);
```

## 4. arch35 Host/Tiling 实现

### 4.1 判断 kvMerge

arch35 原实现要求 `value` 必须存在，并直接读取 Value shape。修改后使用共享 optional 输入：

```cpp
const gert::StorageShape *valueStorageShape =
    context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));

tmpData.kvMerge = (valueStorageShape == nullptr);

const gert::Shape &valueShape =
    tmpData.kvMerge ? keyShape : valueStorageShape->GetStorageShape();
```

kvMerge 下使用 `keyShape` 作为逻辑上的 `valueShape`，满足 K、V 共享输入的语义。

### 4.2 dValue 输出处理

普通场景仍然要求 `dValueOut` 存在并校验其 shape：

```text
value != nullptr -> dValueOut 必须存在
```

kvMerge 场景跳过 `dValueOut` 解引用和 shape 校验：

```text
value == nullptr -> dValueOut == nullptr
```

### 4.3 tiling key

arch35 template tiling key 增加 `KvMerge` 模板位：

```text
KvMerge = 0：普通 K/V 分离场景
KvMerge = 1：K/V 共享输入，梯度合并场景
```

这样 Kernel 可以在编译期裁剪不需要的 dV workspace 和输出路径。

## 5. arch35 Kernel 实现

### 5.1 Value 输入复用

kvMerge 模板分支中，Kernel 使用 Key 地址作为 Value 输入：

```cpp
__gm__ uint8_t *valueIn = KV_MERGE ? key : value;
```

不会访问空的 Value 指针。

### 5.2 梯度合并公式

kvMerge 的数学语义不是拼接 dK 和 dV，而是共享输入梯度求和：

```text
dKV = dK + dV
```

结合 SFAG 的缩放顺序，实际计算为：

```text
dKV = scaleValue * dK_mm4 + dV_mm5
```

必须先对 dK 分支执行 `scaleValue`，再与 dV 相加，不能计算成：

```text
scaleValue * (dK_mm4 + dV_mm5)
```

### 5.3 Pre 阶段

普通场景初始化：

```text
dQ workspace
dK workspace
dV workspace
```

kvMerge 场景只初始化：

```text
dQ workspace
dK workspace
```

跳过无用的 dV workspace 清零，减少 GM 带宽。

### 5.4 Scatter 阶段

普通场景分别写入：

```text
dK -> dK workspace
dV -> dV workspace
```

kvMerge 场景执行：

```text
scaled dK + dV -> dK workspace
```

确定性和非确定性 Scatter 路径都使用相同的 kvMerge 语义。

### 5.5 Post 阶段

普通场景搬运三路结果：

```text
dQ workspace -> dQueryOut
dK workspace -> dKeyOut
dV workspace -> dValueOut
```

kvMerge 场景只搬运：

```text
dQ workspace -> dQueryOut
dK workspace -> dKeyOut，内容为 dK + dV
```

不会访问 `dValueOut`。

## 6. 输出语义

### 6.1 普通场景

```text
输入：value != nullptr
输出：
  dQueryOut = dQ
  dKeyOut = dK
  dValueOut = dV
```

### 6.2 kvMerge 场景

```text
输入：value == nullptr
输出：
  dQueryOut = dQ
  dKeyOut = dK + dV
  dValueOut = nullptr
```

RoPE 输入输出继续使用共享 API 原有规则，本次不修改其公共校验行为和输出索引定义。

## 7. 测试范围

arch35 OpAPI UT 保留：

```text
A1_bsnd_bf16_kvmerge
A2_bsnd_bf16_kvmerge_rope
```

arch35 tiling UT 保留：

```text
A1_bsnd_bf16_value_present
A2_bsnd_bf16_kvmerge
A3_bsnd_bf16_kvmerge_rope
```

其中普通场景用于确认新增 `KvMerge` 模板位不会改变原路径，两个 kvMerge 场景用于确认无 Value、无 dValue 输出时能够生成正确 tiling key。

单边 RoPE 和错误 dKeyRope shape 用例属于独立 RoPE 参数校验，不纳入本次 kvMerge 修改。

## 8. torch_npu 说明

torch_npu/op-plugin API 同样是 arch22 和 arch35 共用，不存在 arch35 专属 schema。

测试前只需要检查当前安装版本是否已经包含 arch22 的共享 optional value 契约：

```python
import torch
import torch_npu

print(torch.ops.npu.npu_sparse_flash_attention_grad.default._schema)
```

预期 schema 中包含：

```text
Tensor? value
```

如果目标 op-plugin 分支已经是 `Tensor? value`，本次 arch35 修改不应再改 YAML。只有目标发布版本尚未同步 arch22 共享契约时，才应将 arch22 对应的 op-plugin 提交合入，而不是增加一套 arch35 专属 API。

## 9. 最终范围总结

```text
共享 API：复用 arch22 已有实现，不修改
共享 InferShape：不修改
共享 OutputIndex：不修改
arch35 Host/Tiling：增加 kvMerge 分支
arch35 Kernel：增加 K/V 复用和 dK+dV 合并
arch35 UT：覆盖普通、kvMerge、kvMerge+RoPE
torch_npu：复用共享 schema，不增加 arch35 专属改动
```
