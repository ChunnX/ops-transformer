# SFAG arch35 TND 尾 padding 支持修改计划

## 1. 修改范围

本次只考虑 `arch35/regbase` 分支，不修改 README、接口文档，也不改 aclnn 对外函数签名。

主要关注文件：

- `op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp`
- `op_kernel/arch35/sparse_flash_attention_grad_kernel.h`
- `op_kernel/arch35/sparse_flash_attention_grad_kernel_base.h`
- `tests/ut/op_host` 下 arch35 或可复用的 tiling 单测

不在本次计划内：

- `arch22/basic` 分支
- README 和 `docs/aclnnSparseFlashAttentionGrad.md`
- 输出 padding 区域强制清零

## 2. 目标语义

TND 场景下，输入物理 shape 仍然是：

```text
query: [T1, N1, D]
key:   [T2, N2, D]
value: [T2, N2, D1]
```

`actual_seq_lengths_query` 和 `actual_seq_lengths_kv` 表示每个 batch 的累计有效长度。

原约束是：

```text
actual_seq_lengths_query[B - 1] == T1
actual_seq_lengths_kv[B - 1] == T2
```

新语义放开为：

```text
actual_seq_lengths_query[B - 1] <= T1
actual_seq_lengths_kv[B - 1] <= T2
```

其中：

```text
t1Index < actual_seq_lengths_query[B - 1]  表示有效 query token
t1Index >= actual_seq_lengths_query[B - 1] 表示尾 padding token
```

尾 padding token 不参与计算，kernel 侧应直接跳过。

## 3. Host 侧 tiling 修改计划

### 3.1 放开 TND shape 校验

在 `op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp` 中检查 TND 相关校验逻辑。

需要确认并调整：

- TND 下 `actual_seq_lengths_query` 和 `actual_seq_lengths_kv` 仍然必须存在。
- 不再要求 `actual_seq_lengths_query` 最后一个元素必须等于 `T1`。
- 不再要求 `actual_seq_lengths_kv` 最后一个元素必须等于 `T2`。
- 如果 host 侧能拿到 actual seq tensor 的常量值，则改为校验最后一个元素不超过物理 shape。

建议校验规则：

```text
0 <= actual_seq_lengths_query[i] <= T1
0 <= actual_seq_lengths_kv[i] <= T2
actual_seq_lengths_query 单调非降
actual_seq_lengths_kv 单调非降
actual_seq_lengths_query[B - 1] <= T1
actual_seq_lengths_kv[B - 1] <= T2
```

如果 host tiling 阶段拿不到 runtime tensor 的实际内容，则不要在 shape 校验中强制 `last == T1/T2`，只保留 shape、dtype、非空等静态校验。

### 3.2 暂不收缩 workspace

本次建议 workspace 仍按物理 shape 计算，不按有效 token 数收缩。

继续按物理 `T1/T2` 计算：

- `dqWorkspaceLen`
- `dkWorkspaceLen`
- `dvWorkspaceLen`
- post/cast 相关 workspace
- block dim
- 输出 shape

原因：

- 输出 tensor 的 shape 仍然是物理 shape。
- host tiling 未必能稳定读取 actual seq runtime 值。
- 先保证功能正确，避免 workspace 偏移和 post 阶段引入额外风险。

### 3.3 暂不修改多核切分粒度

`DoBlockTiling()` 中 TND 场景目前按 `tmpData.t1` 切分任务：

```cpp
int32_t nNums = tmpData.layout == static_cast<uint32_t>(InputLayout::TND) ? tmpData.t1 : tmpData.b * tmpData.s1;
```

本次建议保持不变。

也就是说，host 侧仍然可能把尾 padding 的 `t1Index` 分配给某些核，但 kernel 会在真正计算前判断并跳过。

## 4. Kernel 侧修改计划

### 4.1 增加 TND 有效 t1 判断

在 `op_kernel/arch35/sparse_flash_attention_grad_kernel_base.h` 中增加一个小的内联 helper。

建议语义：

```cpp
__aicore__ inline bool IsValidTndT1Index(int64_t t1Index)
{
    if constexpr (!IS_TND) {
        return true;
    }
    int64_t validT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[batchNum - 1];
    return t1Index < validT1;
}
```

实现时需要使用当前 base 类里已有的 batch 数字段。若字段名不是 `batchNum`，则以现有 `constInfo` 或 tiling data 中保存的 batch 维度为准。

### 4.2 在 `GetTndSeqLen()` 前跳过 padding

在 `op_kernel/arch35/sparse_flash_attention_grad_kernel.h` 中，重点检查两处主循环：

```cpp
this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
this->GetTndSeqLen(this->t1Index, this->bIndex);
```

需要改为：

```cpp
this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
if (!this->IsValidTndT1Index(this->t1Index)) {
    break;
}
this->GetTndSeqLen(this->t1Index, this->bIndex);
```

关键点：

- 判断必须放在 `GetTndSeqLen()` 前。
- `t1Index >= actual_seq_lengths_query[B - 1]` 时必须跳出循环。
- AIC 和 AIV 两侧都要使用完全一致的判断条件。

### 4.3 为什么必须在 `GetTndSeqLen()` 前判断

当前 `GetTndSeqLen()` 会根据 `t1Idx` 查找所在 batch，大致逻辑是：

```cpp
curT1 = actualSeqQlen[bIndex];
while (curT1 <= t1Idx) {
    curT1 = actualSeqQlen[++bIndex];
}
```

当存在尾 padding 时，如果 `t1Idx >= actualSeqQlen[B - 1]`，这里可能继续读 `actualSeqQlen[B]`，造成越界。

所以本次改动的核心不是只为了少算，而是为了避免 TND 尾 padding 下的 batch 查找越界。

### 4.4 检查同步一致性

arch35 是 AIC/AIV 协同执行，跳过逻辑必须两边一致。

检查项：

- cube 侧循环跳过 padding token。
- vector 侧循环跳过同一个 padding token。
- 不出现一侧等待 flag、另一侧已经 break 的情况。
- 对于完全没有有效任务的 core，退出路径仍然能参与必要的全局同步。

### 4.5 检查其他 `GetTndSeqLen()` 调用点

用 `rg` 搜索 arch35 下所有 `GetTndSeqLen` 调用。

凡是调用前使用了 `t1Index`，都要确认：

```text
TND 场景下 t1Index < validT1
```

如果有 scatter、post 或辅助流程会按 `T1` 遍历并读取 per-token 输入，也需要加同样保护。

## 5. 输出 padding 区域处理

本次建议不额外清零尾 padding 输出。

原因：

- 用户需求是“无效数据不需要计算，需要跳过”。
- 输出物理 shape 仍然包含 padding 区域。
- 强制清零会引入额外 post/cast 或 init 开销，改动面更大。

预期行为：

```text
[0, validT1)       有效区域正常计算
[validT1, T1)      padding 区域不保证数值
```

如果后续要求 padding 区域必须为 0，再单独设计清零流程。

## 6. 单测计划

### 6.1 Host tiling 单测

新增 arch35 TND case：

1. 无 padding，保持原行为：

```text
T1 = 64
actual_seq_lengths_query = [64]
期望 tiling 成功
```

2. 单 batch 尾 padding：

```text
T1 = 80
actual_seq_lengths_query = [64]
期望 tiling 成功
```

3. 多 batch 尾 padding：

```text
T1 = 160
actual_seq_lengths_query = [32, 96, 128]
期望 tiling 成功
```

4. 非法超长：

```text
T1 = 64
actual_seq_lengths_query = [80]
如果 host 能读取 seq 值，则期望失败
```

5. TND 缺少 seq tensor：

```text
layout = TND
actual_seq_lengths_query = null
期望失败
```

### 6.2 Kernel 或 golden 验证

建议至少覆盖：

- arch35 TND 无 padding，与原结果一致。
- arch35 TND 尾 padding，有效区域 `[0, validT1)` 与无 padding reference 对齐。
- 数值比较时忽略 padding 区域 `[validT1, T1)`。

## 7. 实施顺序

建议按以下顺序修改：

1. 在 arch35 kernel base 中添加 `IsValidTndT1Index()`。
2. 在 arch35 kernel 主循环中，在 `GetTndSeqLen()` 前加 padding 跳过判断。
3. 搜索 arch35 其他 `GetTndSeqLen()` 调用点，补齐保护。
4. 放开 arch35 host tiling 中 TND `last == T1/T2` 的校验。
5. 增加或调整 arch35 TND tail padding 单测。
6. 跑现有 SFAG 相关 UT，确认无 padding case 不回退。

## 8. Code Review 检查清单

- 不修改 README 和 docs。
- 不修改 aclnn 对外接口。
- 只影响 arch35/regbase 分支。
- 所有 arch35 `GetTndSeqLen()` 调用前都保证 `t1Index < validT1`。
- AIC/AIV 跳过条件完全一致。
- workspace 仍按物理 shape 分配。
- 无 padding TND 行为保持不变。
- padding 输出区域不参与数值校验。
