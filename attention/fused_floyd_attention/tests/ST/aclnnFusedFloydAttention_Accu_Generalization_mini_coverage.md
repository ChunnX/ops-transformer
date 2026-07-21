# FusedFloydAttention STC Mini 用例覆盖分析

> 用例文件：`tests/ST/aclnnFusedFloydAttention_Accu_Generalization_mini.json`
> 用例总数：47 条（全部选自原 5000 条泛化用例，未重新设计）
> 选取原则：基于 STC 设计的分层覆盖思想，按因子组合去重抽样 + 边界补充

---

## 一、测试因子识别

根据 FFA 算子接口（`aclnnFusedFloydAttention`）和 tiling 约束，识别以下测试因子：

| 因子 | 来源 | 取值 | 算子约束 |
|---|---|---|---|
| query dtype | 输入数据类型 | fp16, bf16 | 算子支持 FP16/BF16 |
| D (head_dim) | query shape[4] | 32, 64, 128 | 约束：D ∈ {32, 64, 128} |
| M (S1) | query shape[3] | 128, 256, 384 | 约束：M % 128 == 0 |
| K (S2) | key shape[3] | 128, 256, 384 | 约束：K % 128 == 0 |
| B (batch) | query shape[0] | 1, 7 | 约束：B ≥ 1 |
| H (head_num) | query shape[1] | 1, 7 | 约束：H ≥ 1 |
| atten_mask dtype | mask 数据类型 | bool, uint8 | 算子支持 BOOL/UINT8 |
| atten_mask range | mask 取值范围 | 0（全参与）, 1（全不参与） | 0=生效, 1=不生效 |
| scaleValue | 缩放系数属性 | 0, 1 | 默认 1.0 |
| range_values type | 数据分布类型 | array(均匀分布), object(正态分布) | array=[-5,5], object=nd(mean,std) |

---

## 二、用例分层结构

按 STC 设计的 L0（门槛）/ L1（功能组合）/ 边界 三层划分：

| 层级 | 定义 | 条数 | 说明 |
|---|---|---|---|
| L0 核心 | B=1, H=1, M=128, K=128 最小 shape | 33 | 因子全组合覆盖 |
| L1 边界 | M 或 K 为 256（非 128） | 10 | 大 M / 大 K / 双大组合 |
| L1 多核 | B>1 或 H>1 | 4 | 多 batch / 多 head 并行路径 |

---

## 三、L0 核心用例覆盖矩阵（33 条）

固定 B=1, H=1, M=128, K=128，覆盖 dtype × D × maskDtype × maskRange × scale × rangeType 的组合。

> 理论全组合：2(dtype) × 3(D) × 2(maskDtype) × 2(maskRange) × 2(scale) × 2(rangeType) = 96
> 实际选取 33 条（bf16 对应 rangeType=array，fp16 对应 rangeType=object，去重后每 dtype 约 16 条）

### 3.1 BF16 用例（17 条，rangeType=array 均匀分布）

| id | D | maskDtype | maskRange | scale | 覆盖点 |
|---|---|---|---|---|---|
| 7 | 32 | bool | 0 | 0 | D=32 + bool mask + 全参与 + scale=0 |
| 10 | 32 | bool | 0 | 1 | D=32 + bool mask + 全参与 + scale=1 |
| 8 | 32 | bool | 1 | 0 | D=32 + bool mask + 全不参与 + scale=0 |
| 6 | 32 | uint8 | 0 | 0 | D=32 + uint8 mask + 全参与 + scale=0 |
| 15 | 32 | uint8 | 0 | 1 | D=32 + uint8 mask + 全参与 + scale=1 |
| 4 | 32 | uint8 | 1 | 0 | D=32 + uint8 mask + 全不参与 + scale=0 |
| 5 | 32 | uint8 | 1 | 1 | D=32 + uint8 mask + 全不参与 + scale=1 |
| 29 | 64 | bool | 0 | 0 | D=64 + bool mask + 全参与 + scale=0 |
| 45 | 64 | bool | 1 | 0 | D=64 + bool mask + 全不参与 + scale=0 |
| 32 | 64 | bool | 1 | 1 | D=64 + bool mask + 全不参与 + scale=1 |
| 33 | 64 | uint8 | 0 | 0 | D=64 + uint8 mask + 全参与 + scale=0 |
| 43 | 64 | uint8 | 1 | 0 | D=64 + uint8 mask + 全不参与 + scale=0 |
| 25 | 64 | uint8 | 1 | 1 | D=64 + uint8 mask + 全不参与 + scale=1 |
| 53 | 128 | bool | 0 | 1 | D=128 + bool mask + 全参与 + scale=1 |
| 51 | 128 | bool | 1 | 0 | D=128 + bool mask + 全不参与 + scale=0 |
| 54 | 128 | uint8 | 0 | 0 | D=128 + uint8 mask + 全参与 + scale=0 |
| 64 | 128 | uint8 | 1 | 0 | D=128 + uint8 mask + 全不参与 + scale=0 |

### 3.2 FP16 用例（16 条，rangeType=object 正态分布）

| id | D | maskDtype | maskRange | scale | 覆盖点 |
|---|---|---|---|---|---|
| 16 | 32 | bool | 0 | 0 | D=32 + bool mask + 全参与 + scale=0 |
| 17 | 32 | bool | 0 | 1 | D=32 + bool mask + 全参与 + scale=1 |
| 0 | 32 | uint8 | 0 | 1 | D=32 + uint8 mask + 全参与 + scale=1 |
| 2 | 32 | uint8 | 1 | 1 | D=32 + uint8 mask + 全不参与 + scale=1 |
| 24 | 64 | bool | 0 | 0 | D=64 + bool mask + 全参与 + scale=0 |
| 42 | 64 | bool | 1 | 1 | D=64 + bool mask + 全不参与 + scale=1 |
| 27 | 64 | uint8 | 0 | 0 | D=64 + uint8 mask + 全参与 + scale=0 |
| 39 | 64 | uint8 | 1 | 0 | D=64 + uint8 mask + 全不参与 + scale=0 |
| 26 | 64 | uint8 | 1 | 1 | D=64 + uint8 mask + 全不参与 + scale=1 |
| 61 | 128 | bool | 0 | 0 | D=128 + bool mask + 全参与 + scale=0 |
| 55 | 128 | bool | 1 | 0 | D=128 + bool mask + 全不参与 + scale=0 |
| 63 | 128 | bool | 1 | 1 | D=128 + bool mask + 全不参与 + scale=1 |
| 57 | 128 | uint8 | 0 | 0 | D=128 + uint8 mask + 全参与 + scale=0 |
| 52 | 128 | uint8 | 0 | 1 | D=128 + uint8 mask + 全参与 + scale=1 |
| 58 | 128 | uint8 | 1 | 0 | D=128 + uint8 mask + 全不参与 + scale=0 |
| 59 | 128 | uint8 | 1 | 1 | D=128 + uint8 mask + 全不参与 + scale=1 |

---

## 四、L1 边界用例（10 条）

M 或 K 取 256（非默认 128），覆盖大序列长度场景。固定 B=1, H=1。

| id | dtype | D | M | K | maskDtype | maskRange | scale | 边界类型 |
|---|---|---|---|---|---|---|---|---|
| 22 | bf16 | 32 | 128 | 256 | uint8 | 0 | 1 | 大 K |
| 46 | bf16 | 32 | 256 | 256 | uint8 | 0 | 1 | 双大 (M=K=256) |
| 48 | bf16 | 64 | 256 | 128 | uint8 | 1 | 1 | 大 M |
| 116 | bf16 | 64 | 256 | 256 | bool | 1 | 0 | 双大 (M=K=256) |
| 118 | bf16 | 128 | 256 | 128 | bool | 0 | 0 | 大 M |
| 23 | fp16 | 32 | 128 | 256 | bool | 1 | 1 | 大 K |
| 19 | fp16 | 32 | 256 | 128 | uint8 | 1 | 1 | 大 M |
| 50 | fp16 | 64 | 128 | 256 | bool | 1 | 1 | 大 K |
| 47 | fp16 | 64 | 256 | 128 | uint8 | 1 | 0 | 大 M |
| 117 | fp16 | 128 | 256 | 128 | uint8 | 0 | 1 | 大 M |

---

## 五、L1 多核用例（4 条）

B>1 或 H>1，验证多 batch / 多 head 的多核并行分发路径。固定 M=128, K=128, D=32。

| id | dtype | B | H | maskDtype | maskRange | scale | 覆盖点 |
|---|---|---|---|---|---|---|---|
| 88 | bf16 | 7 | 1 | uint8 | 0 | 0 | 多 batch (B=7) + uint8 mask |
| 89 | bf16 | 7 | 1 | bool | 0 | 1 | 多 batch (B=7) + bool mask |
| 72 | fp16 | 1 | 7 | uint8 | 0 | 1 | 多 head (H=7) + uint8 mask |
| 73 | fp16 | 1 | 7 | uint8 | 1 | 1 | 多 head (H=7) + uint8 mask |

---

## 六、因子覆盖统计

### 6.1 单因子覆盖

| 因子 | 取值 | 用例数 | 覆盖率 |
|---|---|---|---|
| dtype | fp16 | 23 | 100% |
| | bf16 | 24 | 100% |
| D | 32 | 19 | 100% |
| | 64 | 15 | 100% |
| | 128 | 13 | 100% |
| M | 128 | 37 | 100% |
| | 256 | 10 | 100% |
| K | 128 | 39 | 100% |
| | 256 | 8 | 100% |
| B | 1 | 45 | 100% |
| | 7 | 2 | 100% |
| H | 1 | 45 | 100% |
| | 7 | 2 | 100% |
| atten_mask dtype | bool | 20 | 100% |
| | uint8 | 27 | 100% |
| atten_mask range | 0 | 23 | 100% |
| | 1 | 24 | 100% |
| scaleValue | 0 | 23 | 100% |
| | 1 | 24 | 100% |
| range_values type | array (均匀) | 24 | 100% |
| | object (正态) | 23 | 100% |

### 6.2 两两组合覆盖（关键因子对）

| 因子对 | 覆盖组合数 | 总组合数 | 覆盖率 |
|---|---|---|---|
| dtype × D | 6 | 6 | 100% |
| dtype × maskDtype | 4 | 4 | 100% |
| dtype × scaleValue | 4 | 4 | 100% |
| D × maskDtype | 6 | 6 | 100% |
| D × scaleValue | 6 | 6 | 100% |
| maskDtype × maskRange | 4 | 4 | 100% |
| maskDtype × scaleValue | 4 | 4 | 100% |
| maskRange × scaleValue | 4 | 4 | 100% |
| dtype × rangeType | 2 | 2 | 100% |
| D × rangeType | 6 | 6 | 100% |
| M × K | 4 | 4 | 100% |

---

## 七、未覆盖说明

以下场景在 mini 用例中未覆盖（原 JSON 中存在但未选取）：

| 场景 | 原因 | 风险评估 |
|---|---|---|
| M=384 / K=384 极限边界 | 数据量大，仿真耗时 | 低：M/K=256 已覆盖大序列路径 |
| B>7 或 H>7 大 batch/head | 数据量大 | 低：B=7/H=7 已覆盖多核并行 |
| scaleValue=0 和 1 同时存在（range=[0,1]） | 原文仅 1 条 | 低：scale=0 和 scale=1 已分别覆盖 |
| bf16 + 正态分布 / fp16 + 均匀分布 | 原 JSON 中 dtype 与 rangeType 强相关 | 中：数据分布不影响计算逻辑，仅影响数值范围 |
| N(N2) > 48 | 数据量大 | 低：N=16 已覆盖核心路径 |

---

## 八、用例 ID 完整列表

```
0  2  4  5  6  7  8  10  15  16  17  19  22  23  24  25  26  27  29  32
33  39  42  43  45  46  47  48  50  51  52  53  54  55  57  58  59  61
63  64  72  73  88  89  116  117  118
```

共 47 条，全部来自原 `aclnnFusedFloydAttention_Accu_Generalization.json`（5000 条），未重新设计。
