# LI v2 / QLI v2 A5 白盒测试报告

## 1. 目标与结论

本次在 Ascend 950、`shy39`、CANN 9.1.0 环境验证 LI v2 / QLI v2，重点覆盖 dtype、layout、S1/S2 边界、TopK 分段与多轮合并、PA、mask、实际序列长度、metadata 调度、`return_value`、eager 和 graph 链路。

收敛后的结果：

| 模式 | LI v2 | QLI v2 |
| --- | ---: | ---: |
| eager，原矩阵连续 5 轮 | 每轮 72/72 通过 | 每轮 109/109 通过 |
| eager，补强后完整矩阵 | 74/74 通过 | 113/113 通过 |
| graph，补强后完整矩阵 | 74/74 通过 | 113/113 通过 |

QLI v2 覆盖 FP8、INT8、MXFP8、HIFLOAT8、MXFP4 五种量化模式及 BSND/BSND、BSND/PA、TND/TND、TND/PA 四种合法布局组合。测试矩阵采用边界值与成对组合，不对彼此独立的参数做无意义的全笛卡尔积。

补强用例进一步覆盖：

- LI v2 / QLI v2 两类多轮 TopK merge 在 `return_value=0/1` 下的完整组合；
- QLI v2 的 FP8/INT8 在非 FD 场景连续处理 17 个 S2 基本块，验证第 16 个 block 后的 per-token scale cache 刷新；
- PA golden 对物理 block id 的随机置换，以及多 batch 对非顺序物理 block 的共同寻址。

## 2. 本次发现及修复

### 2.1 BSND metadata 的最大序列长度取值错误

**现象**

当 BSND Tensor 的物理 S 轴大于 `seqused` 指定的有效长度时，golden 构造的 metadata 使用了有效长度，不能准确描述输入 Tensor 的物理形状。

**错误代码**

```python
max_seqlen_q_meta = actual_seq_lengths_query.max().item()
max_seqlen_k_meta = actual_seq_lengths_key.max().item()
```

LI v2 还会在 BSND 下用 `seqused_k` 覆盖物理 `k_seq`：

```python
if seqused_k is not None:
    max_seqlen_k = max(seqused_k).item()
```

**根因**

BSND 的 `max_seqlen_q/k` 描述 Tensor 的物理 S 轴；`seqused_q/k` 描述其中参与计算的有效长度，两者含义不同。TND 和 PA 没有固定的 BSND 物理 S 轴，才应从各 batch 的实际长度取最大值。

**定位方式**

构造“物理长度大于有效长度”的 BSND 用例，对照输入 shape、`seqused` 与传入 metadata 的 `max_seqlen`，确认 metadata 被错误缩短。

**正确代码**

```python
max_seqlen_q_meta = (
    q_seq if layout_query == "BSND" else actual_seq_lengths_query.max().item()
)
max_seqlen_k_meta = (
    k_seq if layout_key == "BSND" else actual_seq_lengths_key.max().item()
)
```

LI v2 的 BSND key 同样固定使用物理长度：

```python
max_seqlen_k = k_seq
```

### 2.2 `max_seqlen_q=-1/0` 放在了不合适的 BSND 用例中

**现象**

原用例在 `q_seq=5` 的 BSND Tensor 上同时展开 `max_seqlen_q=-1/0/4/5`，其中 `-1/0` 无法描述该 BSND 物理 S1 轴，测试失败发生在参数语义检查阶段，未覆盖预期分支。

**错误代码**

```python
"q_seq": [5],
"seqused_q": [[4]],
"max_seqlen_q": [-1, 0, 4, 5],
```

**根因**

将 attr 的取值边界与 BSND 的物理 shape 约束混在一个用例中。

**定位方式**

分别检查 BSND 和 TND 的参数语义；确认 `-1/0` 可以在不依赖固定 S1 轴描述的 TND 用例中覆盖。

**正确代码**

```python
# BSND：只保留与物理 S1 相容的值。
"max_seqlen_q": [4, 5]

# TND：单独覆盖 -1/0。
"layout_q": ["TND"],
"layout_k": ["TND"],
"max_seqlen_q": [-1, 0]
```

### 2.3 随机零 scale 造成 TopK 并列值假失败

**现象**

量化用例的 scale 随机范围包含 0 时，可能生成大量相同 score。CPU 与 NPU 都选到合法 TopK，但边界并列元素的索引选择不同，精确索引集合比较出现随机失败。

**错误代码**

```python
"q_scale_datarange": [[0, 255]],
"k_scale_datarange": [[0, 65504]],
```

**根因**

TopK 对相同 score 的索引顺序没有唯一要求；随机白盒用例不应在并非专门测试 tie-breaking 时大量制造并列值。

**定位方式**

保存失败 PT，回查完整 CPU score，确认差异索引位于等值或近似等值的 TopK 边界；排除越界、重复索引和输出 shape 错误。

**正确代码**

```python
"q_scale_datarange": [[1, 255]],
"k_scale_datarange": [[1, 65504]],
```

### 2.4 INT8、FD、`return_value=0` 的比较器消歧能力不足

**现象**

`S2=641`、INT8、FD、`return_value=0` 的随机用例独立运行 20 轮，14 轮通过、6 轮失败；相同失败 PT 改为 `return_value=1` 后，索引比较和 value 比较均通过，value 满足率为 100%。

**问题写法**

```python
"return_value": [0, 1],
"_quant_profiles": ["FP8", "INT8", "MXFP8", "HIFLOAT8", "MXFP4"],
```

**根因**

INT8 的 CPU/NPU 数值路径会使 TopK 边界近邻元素发生交换。`return_value=1` 时，比较器可使用 NPU 返回的 value 验证这些索引是否等价；`return_value=0` 没有该信息，只能使用 CPU score 估算 NPU 边界，可能把有效结果判为失败。这是比较信息不足，不是已证实的算子错误。

**定位方式**

每轮独立进程并保存 PT；对同一个失败 PT 仅切换 `return_value` 后复跑，确认 NPU value 与 golden 满足现有精度标准。

**解决方式**

白盒矩阵使用成对覆盖：INT8 的 FD 组合固定验证 `return_value=1`；INT8 的 `return_value=0` 由其他场景覆盖；FD 的 `return_value=0` 由其余四种 dtype 覆盖。不修改算子，也不放宽公共精度标准。

```python
"META_70": {
    "return_value": [0, 1],
    "_quant_profiles": ["FP8", "MXFP8", "HIFLOAT8", "MXFP4"],
},
"META_70_INT8": {
    "return_value": [1],
    "_quant_profiles": ["INT8"],
},
```

### 2.5 mask3 + `cmp_ratio` 用例缺少明确 residual

**现象**

用于覆盖“前一 S1 行无有效 S2、后一行恢复有效”的 metadata 用例没有显式给出 `cmp_residual_k`，目标序列长度依赖默认值，测试意图不完整。

**错误代码**

```python
"sparse_mode": [3],
"cmp_ratio": [2],
```

**根因**

mask3 下原始 K 长度由压缩长度、`cmp_ratio` 和 residual 共同决定。缺少 residual 时，无法从用例本身唯一读出预期边界。

**定位方式**

按 `actual_s2_orig = seqused_k * cmp_ratio + residual` 反推相邻 S1 行的有效范围，确认本用例目标 residual 为 0。

**正确代码**

```python
"sparse_mode": [3],
"cmp_ratio": [2],
"cmp_residual_k": [[0]],
```

## 3. 白盒用例暴露的主线历史问题

尾部零 cost batch 场景会在 FD 最后一个非零任务通过按行、按块或强制分配后，将 `unassignedCost` 清零；旧 metadata 实现随即结束分配，导致后续零 cost batch 未初始化。该问题已由上游提交 `3d2df35c4`（PR 10761）修复。

本分支基于已包含该修复的最新 `upstream/master`，因此本提交只保留对应白盒看护，不重复携带算子功能修复。

## 4. 覆盖率说明

仓库自带 Host 覆盖率命令可以执行，但当前 A5 配置只实际运行 LI v2 的 4 个 infer-shape UT：该文件行/函数覆盖率为 100%，整个目标 Host 源码行覆盖率约 7.5%、函数覆盖率约 8.8%，tiling 为 0%；QLI v2 当前没有对应 Host UT。该数字不能代表 A5 kernel 覆盖率。

Kernel 源码覆盖率需要为 LI v2 / QLI v2 新增基于 `tikicpulib` 和 `ICPU_RUN_KF` 的 ICPU UT/CMake 测试入口。仓库目前没有这两个算子的 kernel UT，因此本次不伪造源码覆盖率数字，也不把临时覆盖率改动带入提交。

本次 A5 STC 是硬件功能覆盖：能验证真实 tiling、metadata、kernel 和 eager/graph 链路，但不能替代逐行源码覆盖率。建议后续单独建设 kernel ICPU UT，提交范围与功能 STC 解耦。

## 5. 最终提交范围

仅包含：

- LI v2 / QLI v2 的 STC 用例调整；
- BSND metadata 所需的 golden 修复；
- 本报告。

Host 覆盖率兼容改动、临时脚本、日志、PT、Excel 和本地构建产物均不提交。
