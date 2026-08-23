# QuantBlockSparseAttn MXFP8 极端输入的 CUDA、Golden 与 Kernel 语义

## 1. 文档范围

本文说明 QuantBlockSparseAttn 的 MXFP8 路径在空行、全 `-Inf` score、`NaN`、`+Inf/-Inf`、混合无穷以及单 tile/多 tile 场景下应产生什么输出，并解释 CPU golden 与 NPU kernel 如何对齐 CUDA SDPA/cuDNN 的行为。

实现基线为 `fix_nan` 分支，主要对应以下提交：

```text
872ccf7ae  fix(qbsa): align all-negative-infinity softmax rows
b6fdea181  fix(qbsa): align mxfp8 overflow rows and log ttk diffs
188a28675  fix(qbsa): handle zero-sum MX multi-tile rows
316757f6f  fix(qbsa): align MX empty rows with CUDA
4bc6e239a  fix(qbsa): stabilize MXFP8 nonfinite STC cases
1ce5230fb  fix(qbsa): avoid mixed-inf MX dot products in STC
f2ae1309d  test(qbsa): add resource-bounded MXFP8 generalized cases
```

本文当前在 `fix_nan` 分支继续补充有限 FP32 极值与联合空行判断；其他分支只有合入对应修改后才具备相同行为。

本文中的 `LSE=-Inf` 指真正的 IEEE FP32 `-Inf`，不是旧实现使用的 `-3.4028235e+38`（`-FLT_MAX`）占位值。

## 2. 一行 Attention 的状态分类

对每个 Q×Head 行，需要区分三个概念：

```text
存在合法 KV 位置       any_position_valid
softmax 数值状态已启动  any_softmax_active
最终可写回行           row_active

row_active = any_position_valid && any_softmax_active
```

“存在合法 KV”只表示 KV block、实际序列长度和 mask 允许该位置参与计算，并不保证 score 能建立有效 softmax 状态。

### 2.1 空行

没有任何合法 KV 位置，例如 `seqused_kv=0`、稀疏索引没有选中 KV，或该 Q 行的 KV 全被 mask：

```text
attention_out = 全 0
LSE           = -Inf
```

### 2.2 有合法 KV，但所有 score 都是 `-Inf`

MX kernel 的行最大值从有限哨兵 `-FLT_MAX` 开始。`-Inf` 不会更新它，因此该行不会启动 softmax，概率和 softmax sum 保持 0：

```text
attention_out = 全 0
LSE           = -Inf
```

这类行在输出语义上与空行相同，但形成原因不同：前者没有合法位置，后者有合法位置但没有有效数值状态。

### 2.3 score 中包含真实 `NaN` 或 `+Inf`

`NaN` 不会被当作空行过滤。`+Inf` 在 fused softmax 中会遇到 `+Inf - +Inf`，CUDA SDPA/cuDNN 与当前 MX 路径均传播 NaN：

```text
attention_out = NaN
LSE           = NaN
```

这里的 NaN 是真实输入或真实 QK 运算产生的 NaN，不包括 CPU 提前反量化制造的伪 NaN。

## 3. CUDA/Golden 期望总表

以下表格按“单个 Q×Head 行”描述。`attention_out` 中的 NaN/Inf 可以只影响部分 D 通道；LSE 每行只有一个值。

| 场景 | CUDA SDPA/cuDNN | CPU golden | `attention_out` | LSE |
|---|---|---|---|---|
| 没有合法 KV | 空 softmax 行 | 不写回预初始化结果 | 全 0 | `-Inf` |
| 有合法 KV，但 score 全为 `-Inf` | 空 softmax 行 | `any_softmax_active=False` | 全 0 | `-Inf` |
| score 有有限值和 `-Inf`，无 NaN/+Inf | 仅有限 score 形成概率 | 正常 online softmax | 有限加权和 | 有限值 |
| Q 或 K payload 含 NaN | QK score 传播 NaN | 不修复真实 payload NaN | NaN | NaN |
| Q/K 产生 `+Inf` score | fused softmax 传播 NaN | 保留 `+Inf-+Inf` 结果 | NaN | NaN |
| Q/K 只产生 `-Inf` score | softmax 不启动 | 按全 `-Inf` 行处理 | 全 0 | `-Inf` |
| V 含 NaN，QK/softmax 正常 | PV 传播 NaN | PV 传播 NaN | 对应通道 NaN | 仍由 QK 决定，通常有限 |
| V 仅含 `+Inf`，权重非零 | PV 传播 `+Inf` | PV 传播 `+Inf` | 对应通道 `+Inf` | 不受 V 影响 |
| V 仅含 `-Inf`，权重非零 | PV 传播 `-Inf` | PV 传播 `-Inf` | 对应通道 `-Inf` | 不受 V 影响 |
| 同一输出通道同时累加 `+Inf/-Inf` | 未定义无穷和 | 保留 IEEE 运算 | NaN | 不受 V 影响 |
| CPU 提前反量化出现 `Inf×0` 伪 NaN | Cube 中零 group 贡献仍为 0 | 按 Cube group 顺序修复 | 取修复后 score 的语义 | 取修复后 score 的语义 |
| 多 tile 最终 `sum=0 && max=-FLT_MAX` | 输出空行 | max/sum 联合判空 | 全 0 | `-Inf` |
| 最终 `sum=0`，但 max 不是 `-FLT_MAX` | 非空异常行 | 不按空行吞掉 | NaN | NaN |

### 3.1 不能只根据 Q/K/V 的 range 判断整张输出

最终结果还取决于：

- 非有限值实际落在哪个 token、head、D group；
- 对应 KV 是否被稀疏索引选中；
- KV 是否越过实际序列长度；
- mask 是否屏蔽该位置；
- 同一行是否还存在有限 score；
- V 的不同符号是否落在同一个输出通道并同时获得非零权重。

因此上表是逐行、逐通道规则，不表示只要 Tensor 中出现一个 NaN，整张输出就必然全是 NaN。

## 4. 当前非有限值 STC 的具体期望

STC 名称已统一为结构化长名称。下表用编号和场景简称描述；完整名称可以在 Pytest/TTK 的 `qbsa_mxfp8_stc.csv` 中查询。

### 4.1 标量 Q/K 非有限值

这些用例只有一个有效 KV，另一侧为正有限值：

| STC | 输入 | QK score | `attention_out` | LSE |
|---|---|---|---|---|
| 082 | Q=`+Inf` | `+Inf` | NaN | NaN |
| 083 | Q=`-Inf` | `-Inf` | 全 0 | `-Inf` |
| 084 | Q=`NaN` | NaN | NaN | NaN |
| 085 | K=`+Inf` | `+Inf` | NaN | NaN |
| 086 | K=`-Inf` | `-Inf` | 全 0 | `-Inf` |
| 087 | K=`NaN` | NaN | NaN | NaN |

### 4.2 标量 V 非有限值

Q/K 为有限正数且 softmax 正常，V 不参与 LSE：

| STC | 输入 | `attention_out` | LSE |
|---|---|---|---|
| 088 | V=`+Inf` | 全通道 `+Inf` | 有限，由 QK 决定 |
| 089 | V=`-Inf` | 全通道 `-Inf` | 有限，由 QK 决定 |
| 090 | V=`NaN` | 全通道 NaN | 有限，由 QK 决定 |

这三条 STC 只有一个有效 KV，因此 LSE 等于该有限 QK score。若反量化后的 Q/K 都是全 1，`D=128` 且 `softmax_scale=1/sqrt(128)`，则该 score 约为 `sqrt(128)=11.313708`。

### 4.3 `[-Inf,+Inf]` 场景

无界区间不是随机均匀分布。输入生成器把两个端点放在相邻 D 向量中，避免在同一次 MX QK 点积里人为制造顺序相关的 `-Inf + +Inf`。

| STC | 构造 | 逐行期望 |
|---|---|---|
| 091 | 相邻 Q 向量分别带 `-Inf`、`+Inf`，K 为正有限 | `-Inf` Q 行为 `0/-Inf`；`+Inf` Q 行为 `NaN/NaN` |
| 092 | 相邻 K token 分别带 `-Inf`、`+Inf`，Q 为正有限 | 同一 Q 行看到 `+Inf` score，OUT/LSE 为 NaN |
| 093 | 相邻 V token 分别带 `-Inf`、`+Inf`，并放在不同 D group/输出通道 | 对应通道分别传播 `-Inf`、`+Inf`；LSE 有限 |

### 4.4 原始极值问题与多 tile 回归

原始问题场景（现为 STC081）中：

```text
Q = 3.3895313892515355e38
K in [-0.5, 0]
V in [-0.5, 0]
```

按 MX Cube 的 group-dot 后应用 descale 的顺序，合法 QK score 最终为 `-Inf`，而不是 CPU 提前反量化产生的 NaN。因此：

```text
attention_out = 全 0
LSE           = -Inf
```

STC094 使用 `S2=513`，专门覆盖跨 512 边界的多 tile 全 `-Inf` 行：

```text
attention_out = 全 0
LSE           = -Inf
```

泛化集合还成对覆盖 `S2=511/513`，并对原始极值场景覆盖 `511/513/1024/1025`，用于捕获“单 tile 正常、进入第二个 tile 合并后失败”的问题。

STC097/098 继续叠加多 tile 与 GQA：

| STC | 场景 | `attention_out` | LSE |
|---|---|---|---|
| 097 | K 含 NaN，`S2=513`，GQA | 受影响行 NaN | 受影响行 NaN |
| 098 | Q/K 为 0，V 的 `-Inf/+Inf` 位于不同通道，`S2=513`，GQA | 对应通道分别为 `-Inf/+Inf` | 有限；QK 全 0 时由有效 KV 数量决定 |

### 4.5 zero-KV、ragged 与 mask 空行

STC095 包含 ragged batch 和 `seqused_kv=0` 的 batch。只有实际没有合法 KV 的 Q 行输出空行结果：

```text
attention_out = 全 0
LSE           = -Inf
```

同一个 batch 中有合法 KV 的其他行仍按正常 QK/V 计算，不能因为某个 batch 为空而把整张输出清零。

## 5. CPU Golden 为何需要特殊处理

### 5.1 位置合法与 softmax 状态分离

Golden 保留两个状态：

```python
any_position_valid = ...
any_softmax_active = ...
row_active = any_position_valid & any_softmax_active
```

每个 subloop 使用：

```python
subloop_active = has_valid_position & ~torch.isneginf(local_max)
```

其效果是：

- `local_max=-Inf`：不启动 softmax；
- `local_max=NaN`：不主动过滤，继续传播真实 NaN；
- `local_max=+Inf`：启动数值路径，后续 `+Inf-+Inf` 传播 NaN；
- 有限 `local_max`：正常更新 online max、sum 和 PV 累加。

Golden 的输出 Tensor 预初始化为：

```text
attention_out = 0
softmax_lse   = -Inf
```

只有 `row_active=True` 的行才覆盖预初始化值。

### 5.2 修复 CPU 提前反量化产生的伪 NaN

旧 CPU 路径先做：

```text
Q_fp8 × Q_descale -> FP32 Q
K_fp8 × K_descale -> FP32 K
FP32 Q @ FP32 K^T
```

极大 descale 可能让有限 payload 先变成 `Inf`。如果另一侧 payload 为 0，CPU 会得到：

```text
Inf × 0 = NaN
```

MX Cube 的顺序是：

```text
每 32 个 payload 先做有限点积 -> group_dot
再乘 Q/K group descale       -> scaled_group_dot
最后跨 group 累加             -> score
```

因此零 group 贡献应保持 0。Golden 只修复满足以下条件的 score：

1. 普通反量化 `torch.matmul` 得到 NaN；
2. Q/K FP8 payload 本身不含 NaN；
3. 重新按 32 元素 group 计算；
4. 使用 `where(group_dot == 0, 0, group_dot * group_scale)`；
5. 只替换原路径中的伪 NaN。

如果 payload 本身含 NaN，则绝不进入修复，真实 NaN 继续传播。

## 6. Kernel 为与 CUDA 保持一致所做的处理

### 6.1 单 tile：max 与 sum 联合判空

单 tile 的除法阶段仍可能先把 `expSum==0` 选择成 0，但写回前会使用最终 max 与 sum 重新分类：

```text
max == -FLT_MAX && sum == 0 -> attention_out = 0
max != -FLT_MAX && sum == 0 -> attention_out = NaN
其他                         -> 保留归一化结果
```

第一类是真正空行或全负无穷行；第二类典型来自正向 FP32 极值的 fused-softmax 状态，CUDA 会传播 NaN，不能被单独的 `sum==0` 吞成空行。

### 6.2 多 tile：最终写回使用同一联合条件

原先多 tile 最终路径直接执行：

```text
(history * rescale + current) / expSum
```

多 tile 的最终除法仍启用原有 zero-sum 防护；随后与单 tile 共用最终行状态修正：

```text
max == -FLT_MAX && expSum == 0 -> 0
max != -FLT_MAX && expSum == 0 -> NaN
```

因此 512/513 边界不会因为进入第二个 tile 而改变空行分类。

### 6.3 空行 LSE 写成真正的 `-Inf`

MXFP8 定义独立的空行值：

```cpp
MX_EMPTY_LSE_VALUE = -3e+99;  // 编译为 FP32 -Inf
```

两处都会输出 `-Inf`：

1. 整个 Q block 没有有效 task 时的预初始化；
2. `ComputeLseOutputVF<float, true>` 同时检测到 `max==-FLT_MAX` 且 `sum==0` 时。

正常活跃行仍计算：

```text
LSE = log(sum) + max
```

### 6.4 不吞掉真实 NaN

Kernel 没有把 Q/K 产生的 NaN 或 `+Inf-+Inf` 结果改成 0，也没有在输出阶段执行通用 `nan_to_num`。只有 `max==-FLT_MAX && sum==0` 的明确空行状态才写 `0/-Inf`。

因此：

```text
空行/全 -Inf -> 0 / -Inf
真实 NaN     -> NaN / NaN
```

两者不会混淆。

### 6.5 修改仅对 MXFP8 开启

共享 VF 使用模板参数控制行为：

```text
zeroWhenSumZero=false        默认共享/FP8 行为
negativeInfWhenEmpty=false   默认共享/FP8 行为

MXFP8 的 LSE 与最终 OUT 写回执行 max/sum 联合判断
```

因此这次 CUDA 语义对齐不会改变 FP8 的空行 LSE 表示或正常计算路径。

## 7. MXFP8 非有限输入如何编码

### 7.1 `+Inf/-Inf`

E4M3FN payload 使用有限端点 `+448/-448`，配合最大有限 E8M0 descale `2^127`：

```text
+448 × 2^127 -> FP32 +Inf
-448 × 2^127 -> FP32 -Inf
```

### 7.2 NaN

NaN 保留在 E4M3FN payload 中，descale 使用合法有限值；全 NaN group 使用安全 scale 1。不能用 E8M0 的 NaN 编码代替数据 NaN。

### 7.3 `[-Inf,+Inf]`

同一个 QK 点积如果同时包含 `-Inf/+Inf`，不同 group 的硬件累加顺序可能产生顺序相关结果。为了让 CUDA 反量化路径、CPU golden 和 MX Cube 测试同一个确定问题，生成器采用：

```text
相邻 D 向量分别放 -Inf 与 +Inf
V 的两个符号放在不同 D group/通道
```

这样既覆盖两个端点，又避免把不可移植的混合无穷点积当作固定精度预期。

## 8. 单 tile 与多 tile 必测矩阵

每种数值状态都应至少成对执行：

| 维度 | 单 tile | 多 tile/归并边界 |
|---|---|---|
| S2 | 511 或 512 | 513 |
| 更高边界 | 1024 | 1025 |
| Q/K `+Inf` | OUT/LSE NaN | 第二 tile 参与后仍传播 NaN |
| Q/K `-Inf` 且 score 全 `-Inf` | `0/-Inf` | 最终 merge 后仍为 `0/-Inf` |
| Q/K NaN | OUT/LSE NaN | 任一有效 tile 含 NaN 都传播 |
| V NaN/Inf | 只影响 OUT | 跨 tile PV 合并后保持相同 IEEE 语义 |
| zero-KV/mask 空行 | `0/-Inf` | 没有有效 tile 时仍为 `0/-Inf` |
| 有限正 score 越过 base-2 对齐临界值 | `NaN/NaN` | `NaN/NaN` |
| 有限负 score 越过 base-2 对齐临界值 | `0/-Inf` | `0/-Inf` |

### 8.1 有限 FP32 极值实测

本机环境为 RTX 3060 Laptop GPU、CUDA 13.2、cuDNN 9.23.1。Windows PyTorch 未编译 `USE_FLASH_ATTENTION`，因此这里使用可返回 LSE 的 CUDA cuDNN fused-attention 接口实测，不能把它表述为独立 FlashAttention 内核实测。

固定 `D=128`、Q/V 全 1，K 全 `+1` 或 `-1`。在 `S2=1/512/513/1024` 上结果一致：

| softmax_scale | score 符号 | CUDA OUT | CUDA LSE |
|---:|---:|---:|---:|
| `1.8427011218064372e36` | 正/负 | 全 1 | 有限 `±2.3586574359122396e38` |
| `1.8427012802627622e36` | 正 | NaN | NaN |
| `1.8427012802627622e36` | 负 | 全 0 | `-Inf` |
| `2.6584558331135067e36` | 正（`+FLT_MAX`） | NaN | NaN |
| `2.6584558331135067e36` | 负（`-FLT_MAX`） | 全 0 | `-Inf` |

对应 STC103~109 覆盖临界安全值、首个越界值、精确 `±FLT_MAX` 以及 `S2=513` 多 tile。

同时至少叠加一次：

- GQA（`N1/N2 > 1`）；
- ragged batch；
- `mask_mode=0/3`；
- LSE on/off；
- `p_scale=None` 与显式 p_scale；
- sparse block 64/128；
- dense/random sparse mode。

## 9. 精度比较注意事项

### 9.1 `-Inf` 的打印

正确结果应打印为：

```text
ExpectOut = -inf
RealOut   = -inf
```

旧结果 `-3.4028235e+38` 只是 `-FLT_MAX`，数值上不是 `-Inf`。两者不能因为都代表“空行”就在精度比较中视为相等。

### 9.2 NaN/Inf 比较

精度工具应遵循：

```text
NaN vs NaN       -> 相等
+Inf vs +Inf     -> 相等
-Inf vs -Inf     -> 相等
NaN vs finite    -> 不相等
+Inf vs -Inf     -> 不相等
Inf vs finite    -> 不相等
```

对相等的 `-Inf/-Inf`，日志中的普通 `AbsDiff` 可能显示 `Inf` 或 `NaN`，这是直接做无穷减法的展示副作用，不应改变最终 PASS 判定。

## 10. 最终对齐原则

### 10.1 验证口径

本文把以下行为作为已经核对的 CUDA/cuDNN 基准语义：

- 无合法 KV 行为 `OUT=0, LSE=-Inf`；
- 合法 score 全为 `-Inf` 时同样为 `OUT=0, LSE=-Inf`；
- Q/K 产生 NaN 或 `+Inf` score 时 OUT/LSE 传播 NaN；
- V 的 NaN/Inf 只影响 OUT，LSE 保持 QK/softmax 的结果；
- 上述行为在单 tile 和多 tile 最终归一化路径中必须一致。
- 有限正极值形成 `sum=0/max!=−FLT_MAX` 时传播 NaN；有限负极值越界时输出 `0/-Inf`。

CPU golden 通过对应 STC 和 Cube-order 重算路径核对；NPU kernel 通过空行、全 `-Inf`、`S2=513` 多 tile 以及非有限输入 STC 对齐。

以下情况不定义为固定精度期望：同一次 QK 点积的不同 MX group 同时产生 `-Inf` 和 `+Inf`。它存在累加顺序依赖，因此输入生成器会把两个符号放到相邻向量，而不是要求不同后端对该未定义组合逐位一致。

### 10.2 规则汇总

整套实现遵循以下四条规则：

1. **空行和全 `-Inf` 行不是 NaN 行**：输出固定为 `attention_out=0`、`LSE=-Inf`。
2. **真实 NaN 必须传播**：Q/K 的 NaN 或 `+Inf-+Inf` 使 OUT/LSE 为 NaN；V 的 NaN 只影响 OUT。
3. **Golden 必须模拟 MX Cube 的运算顺序**：只修复提前反量化造成、但 Cube 中不存在的 `Inf×0` 伪 NaN。
4. **单 tile 与多 tile 必须一致**：只有 `sum==0 && max==-FLT_MAX` 输出 `0/-Inf`；`sum==0` 但 max 有效时传播 NaN。

简化后的判定流程为：

```text
没有合法 KV？
  是 -> OUT=0, LSE=-Inf
  否 -> 计算 QK

所有合法 score 都是 -Inf？
  是 -> OUT=0, LSE=-Inf
  否 -> 启动 softmax

softmax 中出现真实 NaN/+Inf？
  是 -> OUT=NaN, LSE=NaN
  否 -> 正常 OUT/LSE

V 含非有限值？
  是 -> 只按 IEEE 规则影响 OUT，LSE 保持 QK 结果
```
