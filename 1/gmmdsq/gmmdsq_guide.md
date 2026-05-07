# gmmdsq 算子 Kernel 生成 Guide

## 基本信息 [必填]

- **算子名称**: gmmdsq
- **功能描述**: Grouped MatMul with Dynamic Scale Quantization（分组矩阵乘法 + 动态缩放量化）
  - 对输入 x [M,K] 和分组权重 weight [E, K, N] 执行分组矩阵乘法
  - 结果经动态缩放量化后输出 int8 量化结果和对应缩放因子
- **作者**:
- **日期**: 2026-05-07

---

## 输入/输出 [已从代码分析自动填充]

### 输入张量

| 名称 | Shape | Dtype | Layout | 说明 |
|------|-------|-------|--------|------|
| x | [M, K] | int8 (XMAT_T) | ND | 全部 token 的激活值 |
| weight | tensorList [E, K, N] | int8 (WMAT_T) | ND | 所有专家权重，每个专家 [K, N] |
| weightScale | tensorList [E, N] | float (SCALE_T) | ND | 所有专家的 per-channel scale |
| xScale | [M] | float (SCALE_T) | ND | 全部 token 的 per-token scale |
| groupList | [E] | int64 | ND | 累积 token 数（累积和 = M） |

### 输出张量

| 名称 | Shape | Dtype | Layout | 说明 |
|------|-------|-------|--------|------|
| quantOutput | [M, N/2] | int8 (OUT_T) | ND | 量化输出 |
| quantScaleOutput | [M] | float (SCALE_T) | ND | 输出缩放因子 |

### 标量参数（可选）

| 名称 | Dtype | 默认值 | 说明 |
|------|-------|--------|------|
| （无） | | | |

---

## TilingData 字段 [已从 tiling.h 自动填充]

| 字段名 | 类型 | 说明 |
|--------|------|------|
| matmulTiling | TCubeTiling | MatMul tiling 配置 |
| coreNum | uint32_t | AICore 核数 |
| totalLength | uint64_t | 总输出长度 = M * (N/2) |
| matM | int64_t | x 行数 |
| matK | int64_t | x 列数 / weight 行数 |
| matN | int64_t | weight 列数 |

### Matmul 配置（已从 tiling.cpp 确认）

- A (x): int8, ND format, **不转置** (isTransposeA=false)
- B (weight): int8, ND format, **转置** (isTransposeB=true)
- C (output): int32, ND format
- 无 bias

---

## 典型 Shape 配置 [必填 - 需要用户填写]

<!-- TODO: 填写各轴的典型维度大小，用于切分决策和资源估算 -->

| 场景 | M | K | N | E (专家数) | 核数 | 说明 |
|------|---|---|---|-------------|------|------|
| 默认 | 4096 | 5120 | 7192 | 8 | 最大核数 | groupList 元素累积和 = M |

**说明**：
- M: 总 token 数（x 的行数）
- K: hidden 维度（x 的列数 = weight 的行数）
- N: 输出维度（weight 的列数）
- E: 专家/分组数量（groupList 长度、weight tensorList 元素数）
- 核数: 目标 AICore 数量

---

## Python Golden [必填 - 需要用户填写]

<!-- TODO: 在下方填入完整的 Python Golden 代码 -->
<!-- 核心计算逻辑：分组矩阵乘法 → 动态缩放量化 -->

```python
import torch
import numpy as np

def GMM_Swiglu_quant(x, weight, perChannelScale, perTokenScale, m):
    """
    单组的核心计算逻辑。
    Args:
        x: int8 [m, K]           — 当前组的激活值
        weight: int8 [K, N]      — 当前组的权重
        perChannelScale: float32 [N] — 当前组的 per-channel scale
        perTokenScale: float32 [m]   — 当前组的 per-token scale
        m: int                   — 当前组的 token 数
    Returns:
        quantOutput: int8 [m, N//2]
        quantScaleOutput: float32 [m]
    """
    # Step 1: int32 精度矩阵乘法
    c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))  # [m, N] int32

    # Step 2: 转换为 float32
    c_temp1 = c_temp1.to(torch.float32)  # [m, N] float32

    # Step 3: 反量化 — perChannelScale 广播 + perTokenScale 广播
    c_temp2 = torch.mul(c_temp1, perChannelScale)                      # [m, N] * [N] → [m, N]
    c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))         # [m, N] * [m, 1] → [m, N]

    # Step 4: SwiGLU 激活 — 沿 N 维切半
    c_temp4, gate = c_temp3.chunk(2, dim=-1)  # 各 [m, N/2]
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # silu(left): [m, N/2]
    c_temp6 = c_temp5 * gate                     # [m, N/2]

    # Step 5: 动态量化到 int8
    abs_max = torch.max(torch.abs(c_temp6), -1).values  # [m]
    quantScaleOutput = 127.0 / abs_max                    # [m] 量化 scale
    quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # [m, N/2]
    quantScaleOutput = 1.0 / quantScaleOutput             # [m] 反量化 scale (= absmax/127)

    return quantOutput, quantScaleOutput


def process_groups(x, weight, perChannelScale, perTokenScale, groupList):
    """
    分组处理入口。
    Args:
        x: int8 [M, K]                — 全部 token 的激活值
        weight: int8 [E, K, N]         — 所有专家权重
        perChannelScale: float32 [E, N] — 所有专家的 per-channel scale
        perTokenScale: float32 [M]     — 全部 token 的 per-token scale
        groupList: int64 [E]           — 累积 token 数
    Returns:
        quantOutput: int8 [M, N//2]
        quantScaleOutput: float32 [M]
    """
    M, N = x.shape[0], weight.shape[2]
    quantOutput = torch.zeros(M, N // 2, dtype=torch.int8)
    quantScaleOutput = torch.zeros(M, dtype=torch.float32)

    start_idx = 0
    preV = 0
    for i, v in enumerate(groupList.tolist()):
        currV = v
        tempV = currV - preV  # 当前组的实际 token 数
        preV = currV
        if tempV > 0:
            quantOutput[start_idx:start_idx + tempV], \
            quantScaleOutput[start_idx:start_idx + tempV] = \
                GMM_Swiglu_quant(
                    x[start_idx:start_idx + tempV],
                    weight[i],
                    perChannelScale[i],
                    perTokenScale[start_idx:start_idx + tempV],
                    tempV
                )
        start_idx += tempV

    return quantOutput, quantScaleOutput
```

---

## CV 协作模式建模 [AI 分析 + 用户确认]

### 核启动模式分析

**初步分析**（基于代码结构）:
- 有 Matmul (Cube 操作): x [M,K] × weight^T [E, K, N] → int32 [M,N]
- 有量化后处理 (Vector 操作): 动态缩放量化 → int8 输出
- [ ] **1c2v** - 有矩阵乘法 + 逐元素操作

**AI 推荐模式**: 待 Golden 填写后分析
**推荐原因**: 检测到 matmul + 量化后处理，具体模式取决于量化逻辑复杂度

### CV 序列建模 [待 Golden 填写后分析]

**建模表达式**:
```
<!-- TODO: AI 根据 Golden 分析后填充 -->
<!-- 预期: C0->V0 (Cube matmul → Vector 量化) -->
```

**推导依据**:
```
<!-- TODO: AI 填充 -->
```

### Stage/Phase 配置 [待分析]

```yaml
# 待 Step 0 分析后填充
stages: []
```

---

## 切分策略配置 [Step 1-3, 待分析]

### Step 1: 多核切分

| 参数 | 值 | 说明 |
|------|-----|------|
| **切分轴** | 待分析 | M 轴（行维度）是候选 |
| **变量命名** | total_matM_, single_matM_ | 根据物理轴派生 |
| **核数** | coreNum (Tiling) | |

### Step 2: Vector 切分（仅 1c2v）

| 参数 | 值 | 说明 |
|------|-----|------|
| **切分类型** | 待分析 | |
| **切分轴** | 待分析 | |

### Step 3: 核内切分

| 参数 | 值 | 说明 |
|------|-----|------|
| **切分轴** | 待分析 | |
| **表达式** | 待分析 | |
| **innerChunkSize** | 常量默认值，Step 5 优化 | 常量定义，不是 Tiling 参数 |

---

## 参数参与模式 [Step 4, 待分析]

```yaml
tensor_patterns:
  x: "M,K"
  weight: "K,N"
  weightScale: "TODO"
  xScale: "M"
  groupList: "G"
  quantOutput: "M,N/2"
  quantScaleOutput: "M"
```

---

## 指令序列 [Step 5, 待分析]

<!-- AI 根据 Python Golden 自动生成 -->

---

## UB 资源分配 [Step 6, 待分析]

<!-- 基于指令序列进行生命周期分析和 Buffer 分配 -->

---

## 补充说明

### 特殊约束
- weight 为 tensorList（动态输入），需通过 groupList 索引不同组
- 输出 quantOutput shape 为 [M, N/2]，N 维度减半，可能与量化打包策略有关
- matmul 为 int8×int8→int32，需要 Cube（Matmul API）

### 性能优化建议
- 待 Golden 填写后分析

### 参考实现
- tiling.cpp 中已有 Matmul tiling 配置：int8×int8→int32, B transposed