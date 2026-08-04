# Flash Attention ND 路径（arch35）数学推导

## 一、符号定义

对 Q 的一个 S1G 行块（每行 M 个 Query token），K/V 按 S2 行切块。每个 K block 尺寸为 `s2BaseSize × D`，每轮循环处理一个 K block。

```
M     = actMSize (≤128)            —— Q 方向本轮处理行数
N     = actSingleLoopS2Size        —— K/V 方向本轮列数 (≤128 或 ≤256)
D     = dSize                      —— Q/K head dim
DV    = dSizeV                     —— V head dim
α     = scaleValue = 1 / √D        —— scale factor
Q ∈ ℝ^{M×D}                       —— Query 分块
K_j ∈ ℝ^{N×D}                     —— Key 分块 (第 j 块)
V_j ∈ ℝ^{N×DV}                    —— Value 分块 (第 j 块)
```

## 二、标准单遍 Online Softmax

**初始化：**

```
m = -∞       (M×1)    —— 历史行最大值
s = 0        (M×1)    —— 历史分母 (指数和)
O = 0        (M×DV)   —— 未归一化累加输出
```

**遍历每个 K block j，对每一行 0 ≤ r < M，执行四步：**

```
S_j = Q × K_j^T × α                                    [S]   Q×K^T → 乘 scale

(a) m_new = max(m_old, rowmax(S_j))                    [M]   更新全局行最大

(b) s_new = s_old × exp(m_old - m_new)                 [S1]  历史 s 乘以 rescale
          + rowsum(exp(S_j - m_new))                   [S2]  加上当前块的贡献

(c) O = O × exp(m_old - m_new)                         [O1]  历史 O 乘以 rescale
       + exp(S_j - m_new) × V_j                        [O2]  加上当前块的 PV

(d) m_old = m_new,  s_old = s_new                            状态继承
```

**关键点说明**：

- **rescaling 的原理**：在 (b)[S1] 和 (c)[O1] 中，`exp(m_old - m_new)` 就是 rescale 因子。因为指数函数满足 `exp(a) × exp(b) = exp(a+b)`，所以：

  ```
  exp(S_old - m_old) × exp(m_old - m_new) = exp(S_old - m_new)
  ```

  即把历史项的基准从旧的 `m_old` 对齐到新的 `m_new`，只需乘以一个因子。

- **O 更新的物理含义**：(c)[O1] 将历史输出缩放到新 max 基准，(c)[O2] 加上当前块在新基准下的 PV 贡献。O 始终保持的不变量为：

  ```
  处理完前 j 个 K block 后:  O = Σ_{k=1..j} exp(S_k - m_j) × V_k
                             s = Σ_{k=1..j} exp(S_k - m_j)
  ```

- **最终归一化**：遍历全部 K block 后，输出 `O = O / s_final`：

  ```
  O / s = Σ_j exp(S_j - m_final) × V_j / Σ_j exp(S_j - m_final)
        = softmax(QK^T/√d) × V   ✓
  ```

## 三、算子两步拆解（AIC/AIV 异构流水）

标准单遍中 (a)(b) 在向量单元完成，(c) 需要 BMM2 的矩阵乘结果。arch35 将四步对半拆到 AIC（Cube）和 AIV（Vector）两类核上，通过核间同步信号组成流水线。

每一步触发前需等待前一步的核间同步：

```
Step 1 [AIC] — BMM1
     S_j = Q × K_j^T                                    [S]  Q×K^T (Matmul, Cube)

Step 2 [AIV] — Vec1 (scale + mask + softmax)
     α·S_j + mask                                       [scale+mask]
     m_new = max(m_old, rowmax(α·S_j))                  [M]   更新行最大  ─┐
     rescale = exp(m_old - m_new)                       [R]   rescale 因子   │ 与参考四步
     s_new = s_old × rescale                            [S1]  历史 s 缩放    │ (a)(b)(d)
           + rowsum(exp(α·S_j - m_new))                [S2]  当前块贡献     │ 一一对应
     P_j = exp(α·S_j - m_new)                           [P]   softmax 输出 ─┘
     m_old = m_new,  s_old = s_new                      (d)   状态继承

Step 3 [AIC] — BMM2
     O_j = P_j × V_j                                    [P*V] P×V (Matmul, Cube)

Step 4 [AIV] — Vec2 (flash update)
     O = O × rescale                                    [O1]  历史 O 乘以 rescale
       + O_j                                            [O2]  加上当前块

     若为最后一个 K block:
     O = O / s_new                                      最终归一化
```

**Step 2 / Step 4 在 AIV 核上连续执行**（`ProcessVec1` → `ProcessVec2`），中间插入 Step 1 / Step 3 在 AIC 核上的 Matmul。4 步通过双缓冲 + 预取 2 个 task 实现流水线重叠。

## 四、Vec1（Step 2）详述

**Vec1 = 顺序执行 `ProcessVec1Nd` → `Vec1PostProcess`**，对应代码 `flash_attn_block_vec_nd.h:381,506`。

对第一个 K block (`isFirstS2Loop=true`)：m_old = -∞，rescale 因子不生效，等价于标准初始化：

```
m_new = rowmax(α·S_j)                                 [M]
s_new = rowsum(exp(α·S_j - m_new))                    [S2]  (S1 项因 s_old=0 而消除)
P_j   = exp(α·S_j - m_new)                             [P]
rescale 在 Vec1PostProcess 中不计算
```

对后续 K block (`isFirstS2Loop=false`)：执行完整的 (a)(b)(d) + rescale 计算：

```
m_new = max(m_old, rowmax(α·S_j))                     [M]
```

**Vec1PostProcess** 完成 rescale 与 s 更新：

```
rescale = exp(m_old - m_new)                           [R]
s_new   = s_old × rescale + rowsum(exp(α·S_j - m_new)) [S1+S2]
```

函数 `UpdateExpSumAndExpMax` 实现 `(m_new, rescale, s_new)` 的计算，并将 rescale 写入 `expUb` 供后续 Step 4 使用。

## 五、Vec2（Step 4）详述

**Vec2 = `ProcessVec2`**，对应代码 `flash_attn_block_vec_nd.h:649`。根据当前 K block 位置分三种路径：

**首个 block** (`isFirstS2Loop=true`)：直接初始化 O，无 rescale：

```
O = O_j                                                (DataCopy mm2Res → O_acc)
```

**中间 block** (`isLastS2Loop=false`)：与标准 (c) 一致：

```
O = O × rescale + O_j                                  [O1+O2]
```

函数 `FlashUpdateNew` 实现 (`vf_flashupdate_new.h:182`)：`dst = pre × expMax + cur`。

**最后一个 block** (`isLastS2Loop=true`)：rescaling + 最终除以 s。内部分支：

> 若 T > 1（有多于一个 block），进入 `FlashUpdateLastNew` (`vf_flashupdate_new.h:379`)：
> ```
> O = (O × rescale + O_j) / s_new
> ```
>
> 若 T = 1（只有一个 block 且就是最后一个），进入 `LastDivNew` (`vf_flashupdate_new.h:439`)：
> ```
> O = O_j / s_new
> ```
> 因为没有历史 O，只需除以 s。

最终归一化后再做 RowInvalid（无效行置零）、Cast（fp32→fp16/bf16）、copy 到 GM。

## 六、代码步骤对照

| 数学步骤 | 硬件 | 代码位置 | 核心函数 |
|----------|------|----------|----------|
| [S] Q×K^T | AIC | `flash_attn_block_cube_nd.h:326` | `IterateBmm1NdL0Split` |
| [M][S1][S2][P][R][d] | AIV | `flash_attn_block_vec_nd.h:381` | `ProcessVec1Nd` |
| [S1+S2 合并][R] | AIV | `flash_attn_block_vec_nd.h:506` | `Vec1PostProcess` |
| [P*V] P×V | AIC | `flash_attn_block_cube_nd.h:376` | `IterateBmm2l0Split` |
| [O1+O2] flash update | AIV | `flash_attn_block_vec_nd.h:649` | `ProcessVec2` |
| O1+O2 (中间) | AIV | `vf_flashupdate_new.h:182` | `FlashUpdateNew` → `dst = pre × expMax + cur` |
| (O1+O2)/s (最后) | AIV | `vf_flashupdate_new.h:379` | `FlashUpdateLastNew` → `dst = (pre × expMax + cur) / expSum` |
| O/s (T=1) | AIV | `vf_flashupdate_new.h:439` | `LastDivNew` → `dst = cur / expSum` |
