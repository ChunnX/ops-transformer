# CGDR arch35 重构后数据流与改动点

> 本文档描述重构后 arch35 平台 CGDR 算子的完整计算流程，与重构前逐项对比。
> 当前保留 **R2**（strided GM 直读），计划实现 **R4**（beta 列缩放）、**R5**（g_exp 列缩放）和 **R7**（kg 系数移到 v_new）。

---

## 0 重构范围说明

初始重构方案曾包含 R2-R7 六项改动。经与参考实现（`chunk_gated_delta_rule_golden.py`）
逐一比对后：R3 因数学错误回退；R6 原始代码已满足（无需改动）；R4/R5/R7 经修正后方案可行（待实现）；
R2 已实现（纯数据流优化）。

| 编号 | 需求 | 状态 | 原因 |
|:-----|:-----|:-----|:-----|
| R2 | 省略 QKPreProcess，matmul 直接读 strided GM | **已实现** | 纯数据流优化，数学不变 |
| R3 | K 衰减系数 g_exp 移至 Stage2 | **回退** | g_exp 在 matmul 内维（求和下标），不可移到结果行 |
| R4 | beta 从 V/K 左乘移到 A 右乘（列缩放） | **待实现** | 原方案用行缩放错误，改为列缩放后数学等价 |
| R5 | g_exp 从 K 左乘移到 A 右乘（列缩放） | **待实现** | 原方案用逐元素乘 C×C 矩阵错误，改为列缩放 C 维向量后数学等价 |
| R6 | gCum_ 输出到 GM，后续 g 参数从 gCum_ 重算 | **已满足** | 原始代码已是此做法；原 R6 想删除 gCum_ 从原始 g 重算有 strided bug |
| R7 | 删除 kg_，Stage2 直接用 K | **待实现** | 原方案标量缩放错误，改为逐行向量缩放后数学等价 |

### 数学详解

以下用参考实现 `chunk_gated_delta_rule_golden.py:102-149` 的正确公式对照说明。

**正确公式（参考实现）：**
```python
# Stage1
k_beta = key * beta.unsqueeze(-1)          # K 乘 beta
v_beta = value * beta.unsqueeze(-1)        # V 乘 beta
attn = -((k_beta @ key.T) * decay_mask)    # KKT 含 beta，乘 gamma
attn = attn + I                            # A = (I - k_beta@K^T*gamma)^{-1}
value = attn @ v_beta                      # v_inner = A @ (V*beta)
k_cumdecay = attn @ (k_beta * g.exp())     # k_cumdecay = A @ (K*beta*g_exp)

# Stage2 (per chunk)
v_prime = k_cumdecay @ state               # 含 g_exp
v_new = v_inner - v_prime
state_new = state * g[:,-1].exp()
           + (k * (g[:,-1] - g).exp()[...,None]).T @ v_new   # kg = K * d[:,None]
```

矩阵形式（`D = diag(beta)`, `G = diag(g_exp)`, `d = exp(g[-1]-g_cum)`）：
```
KKT 含 beta:  attn = -D @ (K@K^T) * gamma
求逆:         A = (I - D·kk·γ)^{-1}           (beta 在求逆内部, 不可移出)
v_inner:      A @ D @ V                        (beta 乘 V 左边 = A 右边)
k_cumdecay:   A @ D @ G @ K                    (beta+g_exp 乘 K 左边 = A 右边)
state_new:    state·exp(g[-1]) + v_new^T @ diag(d) @ K   (d 乘 K 左边 = v_new^T 右边)
```

---

**R3 为什么不能移到 Stage2：**

g_exp 在 `k_cumdecay = A @ D @ G @ K` 中位于 A 右边（A 的列 = K 的行 = matmul 的**内维/求和下标**）：
```
k_cumdecay[i,d] = Σ_m A[i,m] · g_exp[m] · (-β[m]·K[m,d])
                         ↑ g_exp[m] 在求和内部, 每项 m 不同
```
A 是下三角求逆矩阵（非对角），把不同 m 混在一起求和。求和完成后 g_exp[m] 的逐项权重信息不可逆地丢失，无法事后作为结果行缩放补回。等价地：`A @ diag(g) @ X ≠ diag(g) @ A @ X`（A 非对角不可交换）。

---

**R4 正确方案：** 利用结合律 `A @ D @ V = (A @ D) @ V`，将 beta 从 V/K 左边移到 A 右边。
- `A_beta = A @ diag(beta)`（**列缩放**：第 j 列乘 beta[j]）
- `v_inner = A_beta @ V`（V 不再乘 beta，可直接从 strided GM 读）
- `k_cumdecay = A_beta @ (K * g_exp[:,None])`（K 不再乘 beta，只乘 g_exp）

**R4 原方案错误：**
1. 从 KKT 中删除 beta → `A' = (I-kk·γ)⁻¹` ≠ `A = (I-kk·β·γ)⁻¹`（求逆不分配行缩放）
2. 用行缩放 `diag(β)·A`（第 i 行乘 beta[i]）而非列缩放 `A·diag(β)`（第 j 列乘 beta[j]）
3. `diag(β)·A ≠ A·diag(β)`（方阵不可交换）

反例（`C=2`, `beta=[2,3]`）：
```
A @ diag(β) = [a b] @ [2 0] = [2a 3b]   (列缩放: 第1列×2, 第2列×3)
              [c d]   [0 3]   [2c 3d]

diag(β) @ A = [2 0] @ [a b] = [2a 2b]   (行缩放: 第1行×2, 第2行×3)
              [0 3]   [c d]   [3c 3d]
                                      ≠
```

---

**R5 正确方案：** 在 R4 基础上，对 A_beta 再做一次 g_exp 列缩放，得到 A_beta_gexp 专供 k_cumdecay：
- `A_beta_gexp = A_beta @ diag(g_exp)`（**列缩放**：第 j 列乘 g_exp[j]）
- `k_cumdecay = -A_beta_gexp @ K`（K 不再乘 β·g_exp，直接用 strided GM）
- `v_inner = A_beta @ V`（不变，v_inner 不需要 g_exp）

两步列缩放等价性验证：
```
A_beta_gexp = A @ diag(β) @ diag(g_exp) = A @ diag(β·g_exp)    (对角阵可合并)
k_cumdecay  = -A_beta_gexp @ K = -A @ diag(β·g_exp) @ K = A @ (-K·β·g_exp) ✓
v_inner     = A_beta @ V = A @ diag(β) @ V = A @ (V·β)         ✓
```

**R5 原方案错误：** 用逐元素乘 C×C 矩阵 `gamma`（decay_mask）替代列缩放 C 维向量 `g_exp`。
- 逐元素乘 `A_beta * gamma_matrix`：`A[i,j]·exp(g[i]-g[j])`，依赖 i 和 j → **不等价**
- 列缩放 `A_beta @ diag(g_exp)`：`A[i,j]·exp(g[j])`，只依赖 j → **等价**
- gamma（C×C 矩阵）≠ g_exp（C 维向量），是两个完全不同的量

**R5 与 R3 的关键区别：**
- R3 想把 g_exp 移到 **Stage2 结果行**缩放 → g_exp 在 A 右边（内维），A 非对角不可移
- R5 把 g_exp 移到 **A 列**缩放（仍留在 Stage1 matmul 左矩阵中）→ 矩阵乘对角阵，等价

---

**R7 正确方案：** 利用结合律 `v^T @ diag(d) @ K = (v^T @ diag(d)) @ K = (v·d[:,None])^T @ K`，
将 d 从 K 左边移到 v_new^T 右边（= v_new 的行缩放）。
- `state_new += (v_new * d[:,None])^T @ K`（K 不再乘 d，直接用 strided GM）
- `d[j] = exp(g[-1] - g_cum[j])`（逐行向量，每行 j 不同）

**R7 原方案错误：** 用标量 `exp(g[-1])` 缩放 v_new，正确应为逐行向量 `exp(g[-1]-g_cum[j])`。
- `d[j] = exp(g[-1]-g_cum[j])` 逐行不同（j 是 v_new 的行 = 求和下标）
- `exp(g[-1])` 是标量，所有行相同
- `d` 在 `v_new^T @ diag(d) @ K` 中位于 v_new^T 右边（v_new 的行 = 求和下标），可移到 v_new

**R7 与 R3 的关键区别：**
- R3 的 g_exp 在 `A @ diag(g) @ K` 中位于 A 右边（A 的列），A 非对角混合后不可移
- R7 的 d 在 `v^T @ diag(d) @ K` 中位于 v^T 右边（v 的行），v^T @ diag(d) = (v·d)^T 可直接移

---

**R6 已满足：** 原始代码已采用此做法——Stage1 输出 `gCum`（cumsum 结果）到 GM，
不输出 gCumExp；Stage2/3 后续所有 g 相关参数从 `gCum_` 重算：
- Stage2 `CalGCumExpFp32/Bf16`：从 gCum_ 读，取 `gCum[-1]` 做 exp 得到 `exp(g[-1])`
- Stage3 `CalMaskedQKT`：从 gCum_ 读，算 `decay_mask = exp(g_cum[:,None]-g_cum[None,:])`
- R7 中 `d = exp(g[-1]-g_cum[j])` 同样从 gCum_ 重算

原 R6 想删除 gCum_ 从原始 g(T,Nv) 重算 cumsum，但 g 布局非连续（stride=Nv），
`DataCopyPad{1, C*4, 0, 0, 0}` 读到跨 head 垃圾数据。
正确做法是保留 gCum_ 输出（已是连续 [Nv,Sp] 布局），后续从 gCum_ 重算——即原始代码的做法。

---

## 1 R2 改动点详解

### 1.1 动机

重构前 AIV 通过 `QKPreProcess` 将 Q/K 从 strided GM 搬入连续 workspace
（`queryConGm_`/`keyConGm_`），AIC 再从 workspace 读取做 matmul。
这额外消耗 `2 × coreNum × P × C × Dk × sizeof(bf16_t)` 的 workspace（典型 ~3MB）。

### 1.2 方案

AIC matmul 直接从 strided GM 读取 Q/K，通过 `SetOrgShape(M, N, K, Ka, Kb)` 指定行 stride。
AIV 侧删除 `QKPreProcess` 函数及 `queryConGm_`/`keyConGm_` workspace 分配。

### 1.3 SetOrgShape 用法

```cpp
// strided GM matmul: 用 5 参数 SetOrgShape, orgKc=N (匹配 3 参数行为)
mmRef.SetOrgShape(M, N, Ka, Kb, N);

// 连续 workspace matmul: 用 3 参数 SetOrgShape (Ka=Kb=K, orgKc=N)
mmRef.SetOrgShape(M, N, K);
```

`AICProcess` 根据 `MatmulShapeParams.ka` 是否 >0 自动选择：
- `ka > 0`：strided GM，用 5 参数 `SetOrgShape(m, n, ka, kb, n)`
- `ka == 0`：连续矩阵，用 3 参数 `SetOrgShape(m, n, k)`（保持原始行为）

### 1.4 !gOptional 路径 kg_ 写入

重构前 `QKPreProcess` 在 `!gOptional` 时将 K 拷贝到 `kg_` GM（此时 `kg = K`）。
R2 删除 `QKPreProcess` 后，此逻辑移至 `GBKCompute` 的 `else` 分支：
从 `kUbFloatCon_`（已从 strided GM 读入 UB 的 K）Cast 到 BF16 写出 `kg_`。

---

## 1.5 R4 正确方案详解（待实现）

### 动机

原始代码中 beta 分别乘到 V 和 K 的左边（`v_beta = V*beta`, `k_beta = K*beta`），
AIC matmul5 读 `vBetaWsGm_`（V*beta 的连续 WS），AIV 需额外做 VBetaCompute。
R4 将 beta 从 V/K 左边移到 A 右边（列缩放），使 AIC 直接读原始 V（strided GM），
省掉 `vBetaWsGm_` WS 和 VBetaCompute 计算。

### 数学推导

利用矩阵乘法结合律（`D = diag(beta)`）：
```
原始: v_inner    = A @ D @ V        = A @ (V * beta[:,None])
      k_cumdecay = A @ D @ G @ K    = A @ (K * beta[:,None] * g_exp[:,None])  (G=diag(g_exp))

R4:   A_beta = A @ D                = A * beta[None,:]    (列缩放: 第j列乘beta[j])
      v_inner    = A_beta @ V       (V 不乘 beta)
      k_cumdecay = A_beta @ G @ K   = A_beta @ (K * g_exp[:,None])  (K 不乘 beta, 只乘 g_exp)
```

关键：`A @ diag(beta)` 是**列缩放**（第 j 列乘 beta[j]），
**不是** `diag(beta) @ A`（行缩放，第 i 行乘 beta[i]）。方阵不可交换。

### 不变部分

- **KKBetaCompute**：`attn = kk * beta * gamma`（beta 仍在 KKT 中，求逆内部）
- **求逆**：`A = (I - kk*beta*gamma)^{-1}`（A 本身不变）
- **AttnInverseMMCompute**：操作原始 A（求逆辅助 matmul，必须在列缩放之前）

### 变更部分

| 函数 | 原始 | R4 正确方案 |
|:-----|:-----|:-----------|
| InverseCompute | 求逆后直接写 A 到 attnWs | 求逆后写 A 到 attnWs，**AttnInverseMMCompute 完成后**再做列缩放写 A_beta |
| KKBetaCompute | `attn = kk * beta` | 不变 |
| VBetaCompute | `v_beta = V * beta` → vBetaWsGm_ | **删除**（AIC 直接读原始 V） |
| GBKCompute | `gBKWs = -K * beta * g_exp` | `gBKWs = -K * g_exp`（去掉 beta） |
| AIC matmul4 | `A @ gBKWs` = `A @ (-K*beta*g_exp)` | `A_beta @ gBKWs` = `A_beta @ (-K*g_exp)` |
| AIC matmul5 | `A @ vBetaWs` = `A @ (V*beta)` | `A_beta @ V`（V 从 strided GM 读，R2） |

### 列缩放时机与同步

AttnInverseMMCompute（AIC 求逆辅助 matmul）读写 attnWs 中的**原始 A**，列缩放必须在其之后：

```
AIV: InverseCompute → 写 A 到 attnWs → SetFlag(0x7)
AIC: WaitFlag(0x7) → AttnInverseMMCompute(读/写 attnWs 的 A) → SetFlag(0xB)  // 新增
AIV: WaitFlag(0xB) → 读 attnWs(A) → 列缩放 A*beta[None,:] → 写 A_beta 回 attnWs → SetFlag(0x5/0x6)
AIC: WaitFlag(0x5/0x6) → matmul4/5(读 A_beta)
```

新增同步 flag `0xB`（AIC→AIV，AttnInverseMMCompute 完成后通知 AIV 可做列缩放）。

### 列缩放需要全量 beta

原始 `BetaCopyInWithStride` 每个 sub-block 只读 `halfChunkSize_` 个 beta（自己的行）。
列缩放 `A[i,j] *= beta[j]` 需要**全部 C 个 beta 值**（所有列）。
两个 sub-block 需读同一份全量 beta（C=64，128 字节，开销可忽略）。

### 收益

- 删除 `vBetaWsGm_` WS：`BF16 × coreNum × P × C × Dv`（典型 ~2.6MB）
- 删除 VBetaCompute 计算（V 读入 + beta broadcast + mul + cast + 写出）
- GBKCompute 省 `halfChunkSize_` 次 mul（去掉 beta 乘法）
- 若结合 R2：AIC 直接从 strided GM 读 V

### 代价

- 新增同步 flag 0xB（AIC→AIV，AttnInverseMMCompute 完成后）
- AIV 额外 GM 读写（读 A → 列缩放 → 写 A_beta 回 attnWs）
- 需读全量 beta（C 值，开销极小）

---

## 1.55 R5 正确方案详解（待实现，基于 R4）

### 动机

R4 后 `k_cumdecay = A_beta @ (-K·g_exp)`，K 仍乘着 g_exp，AIV 需在 GBKCompute 中做
`-K * g_exp` 行缩放并写入 `gBKWsGm_` WS。
R5 在 R4 基础上对 A_beta 再做一次 g_exp 列缩放，得到 A_beta_gexp 专供 k_cumdecay，
使 AIC matmul4 直接读原始 K（strided GM），省掉 `gBKWsGm_` WS 和 GBKCompute 的 K 行缩放。

### 数学推导

`k_cumdecay = -A @ diag(β·g_exp) @ K`，利用结合律分两步列缩放：
```
步骤1 (R4):  A_beta = A @ diag(β)                // 列缩放 β
步骤2 (R5):  A_beta_gexp = A_beta @ diag(g_exp)   // 列缩放 g_exp (在 A_beta 基础上)

验证 (对角阵可合并):
A_beta_gexp = A @ diag(β) @ diag(g_exp) = A @ diag(β·g_exp)
k_cumdecay  = -A_beta_gexp @ K = -A @ diag(β·g_exp) @ K = A @ (-K·β·g_exp) ✓
v_inner     = A_beta @ V       (不变, v_inner 不需要 g_exp)        ✓
```

两步都是**列缩放**（矩阵乘对角阵），`diag(β)·diag(g_exp) = diag(β·g_exp)`，完全等价。

### 为什么 R5 可行而 R3 不可行

| | R3 | R5 |
|:---|:---|:---|
| 想法 | 把 g_exp 移到 Stage2 **结果行**缩放 | 把 g_exp 移到 A_beta **列**缩放 |
| g_exp 位置 | `A @ diag(g) @ K`，g 在 A 右边（A 的列=内维） | 同左，但 R5 仍在 A 右边做列缩放 |
| 能否实现 | **不能**（A 非对角，移到结果行=移到 A 左边，不可交换） | **能**（列缩放=矩阵乘对角阵，保持 A 右边） |

R3 的错误在于想把 g_exp 从 A 右边移到 A 左边（结果行缩放 = `diag(g) @ A @ K`），
A 非对角不可交换。R5 保持 g_exp 在 A 右边（`A @ diag(g) @ K`），只是分成两步列缩放，
数学上完全等价。

### R5 原方案错误

原 R5 用逐元素乘 C×C 矩阵 `gamma`（decay_mask）替代列缩放 C 维向量 `g_exp`：
- 逐元素乘 `A_beta * gamma_matrix`：`A[i,j]·exp(g[i]-g[j])`，依赖 i 和 j → **不等价**
- 列缩放 `A_beta @ diag(g_exp)`：`A[i,j]·exp(g[j])`，只依赖 j → **等价**

注意：gamma（decay_mask，C×C 矩阵，用于 KKT）和 g_exp（`exp(g_cum)`，C 维向量，用于 k_cumdecay）
是**两个完全不同的量**，原方案混淆了二者。

### 澄清：两个不同的 g 相关量

| 名称 | kernel 变量 | shape | 定义 | 用在哪 |
|:-----|:-----------|:------|:-----|:-------|
| gamma (decay_mask) | `gammaUbFloat_` | C×C 矩阵 | `exp(g_cum[i]-g_cum[j]) * tril` | KKT: `attn = -(K·β@K^T) * gamma` |
| g_exp | `gCumExpUbFloat_` | C 维向量 | `exp(g_cum[j])` | k_cumdecay: `K * β * g_exp` |

R5 列缩放用的是 **g_exp**（C 维向量），不是 gamma（C×C 矩阵）。

### 变更部分（在 R4 基础上）

| 函数 | R4 后 | R5 后 |
|:-----|:------|:------|
| AIV 列缩放 | β 列缩放 → A_beta → 写 attnWs | β 列缩放 → A_beta → 写 attnWs；**再 g_exp 列缩放 → A_beta_gexp → 写 attnWsNegGm_** |
| GBKCompute | `gBKWs = -K * g_exp` → gBKWsGm_ | **删除**（K 不再乘 g_exp） |
| AIC matmul4 | `A_beta @ gBKWs` = `A_beta @ (-K*g_exp)` | `-A_beta_gexp @ K`（K 从 strided GM 读） |
| AIC matmul5 | `A_beta @ V`（不变） | 不变 |

### 实现流程

A_beta 驻留 UB，连续做两步列缩放，分别写两个 GM WS：

```
AIV: 求逆 A → 写 attnWs (GM)
AIC: AttnInverseMMCompute (读/写 attnWs 的原始 A) → SetFlag(0xB)
AIV: WaitFlag(0xB)
     读 A from GM → 列缩放 β → A_beta (UB)
     写 A_beta → attnWs (GM, 给 matmul5 v_inner)
     列缩放 g_exp → A_beta_gexp (UB, in-place on A_beta 副本)
     写 A_beta_gexp → attnWsNegGm_ (GM, 给 matmul4 k_cumdecay)
     SetFlag(0x5) + SetFlag(0x6)
AIC: matmul5: v_inner = A_beta @ V          (读 attnWs, V 从 strided GM)
     matmul4: k_cumdecay = -A_beta_gexp @ K  (读 attnWsNegGm_, K 从 strided GM)
```

### 收益（在 R4 基础上新增）

- 删除 `gBKWsGm_` WS：`BF16 × coreNum × P × C × Dk`（典型 ~2.6MB）
- 删除 GBKCompute 的 K 行缩放部分（K 读入 + g_exp broadcast + mul + cast + 写出）
- AIC matmul4 直接读原始 K（strided GM，结合 R2）
- 保留 KgCompute 输出 kg_（R7 待实现时再删除）

### 代价（在 R4 基础上新增）

- 新增 `attnWsNegGm_` WS（`BF16 × C×C × P × coreNum`，存 A_beta_gexp）
- AIV 第二步列缩放（C×C 次 mul）+ 一次额外 GM 写出
- 需读全量 g_exp（C 值，开销极小，AIV 已有 `gCumExpUbFloat_`）

### R4+R5 合并后的 WS 布局

```
Stage1 WS (R4+R5 后):
  kkWsGm_      BF16 [P×C×C]     // KKT 结果 (不变)
  attnWsGm_    BF16 [P×C×C]     // A_beta (R4: 原始 A 改为 A_beta)
  attnWsNegGm_ BF16 [P×C×C]     // A_beta_gexp (R5 新增, 供 matmul4)
  // 删除: vBetaWsGm_ (R4), gBKWsGm_ (R5), queryConGm_/keyConGm_ (R2)
```

---

## 1.6 R7 正确方案详解（待实现）

### 动机

原始代码 Stage1 计算 `kg = K * d[:,None]`（`d = exp(g[-1]-g_cum)`）并写入 GM，
Stage2 `CalStateNew` 用 `v_new^T @ kg`。
R7 将 d 从 K 左边移到 v_new^T 右边（= v_new 行缩放），使 Stage2 直接用原始 K（strided GM），
省掉 `kg_` GM workspace 和 Stage1 KgCompute 计算。

### 数学推导

利用矩阵乘法结合律（`d = exp(g[-1]-g_cum)`, 逐行向量）：
```
原始: state_new += v_new^T @ diag(d) @ K        = v_new^T @ (K * d[:,None])

R7:   state_new += (v_new^T @ diag(d)) @ K      = (v_new * d[:,None])^T @ K
```

`v_new^T @ diag(d)` = `(diag(d) @ v_new)^T` = `(v_new * d[:,None])^T`（v_new 的**行缩放**，第 j 行乘 d[j]）。

关键：d 在 `v_new^T @ diag(d) @ K` 中位于 v_new^T 右边（= v_new 的行 = matmul 的**求和下标**），
可以直接移到 v_new。这与 R3 的 g_exp 不同——g_exp 在 `A @ diag(g) @ K` 中位于 A 右边（A 的列），
A 非对角混合后不可移。

### 为什么 R7 可行而 R3 不可行

| | R3 的 g_exp | R7 的 d |
|:---|:---|:---|
| 矩阵位置 | `A @ diag(g) @ K`，g 在 A 右边 | `v^T @ diag(d) @ K`，d 在 v^T 右边 |
| 缩放对象 | A 的列（求和下标），A 混合后丢失 | v^T 的列 = v 的行（求和下标），可直接移 |
| 能否移 | **不能**（A 非对角，混合后不可逆） | **能**（`v^T·diag(d) = (v·d)^T`） |

### 变更部分

| 函数 | 原始 | R7 正确方案 |
|:-----|:-----|:-----------|
| Stage1 KgCompute | `kg = K * d[:,None]` → kg_ GM | **删除**（不输出 kg_） |
| Stage1 GBKCompute | 同时算 kg 和 gBKWs | 只算 gBKWs（去掉 kg 部分） |
| Stage2 AIV | 无 | 计算 `d = exp(g[-1]-g_cum)`，`v_new_scaled = v_new * d[:,None]` |
| Stage2 AIC CalStateNew | `state += v_new^T @ kg` (accum=1) | `state += v_new_scaled^T @ K` (accum=1, K 从 strided GM 读) |

### Stage2 AIV 行缩放实现

d 的计算（从 GM gCum_ 读 g_cum，取 `g_cum[-1] - g_cum[j]`，exp）：
```
d[j] = exp(g_cum[-1] - g_cum[j])    // j ∈ [0, curChunkSize)
```

v_new 行缩放：
```
v_new_scaled[i,j] = v_new[i,j] * d[j]   // 第 j 行乘 d[j]
```

注意：这里 v_new 的行维度是 chunkSize（C），列维度是 Dv。
d 对 v_new 的**行**缩放（每行 j 乘 d[j]），对应 `v_new^T` 的**列**缩放。

### Stage2 同步关系变更

原始 Stage2 AIC `CalStateNew` 用 accum=1 融合 `state += v_new^T @ kg`。
R7 需 AIV 先完成 v_new 行缩放，AIC 才能做 `state += v_new_scaled^T @ K`：

```
AIV: WaitFlag(0x2) ← AIC CalVPrime+CalAttnInter
AIV: 计算 d, v_new_scaled = v_new * d[:,None]
AIV: SetFlag(0x5) → AIC                          // v_new_scaled 就绪
AIC: WaitFlag(0x5)
AIC: CalStateNew(state += v_new_scaled^T @ K, accum=1)  // K 从 strided GM 读
AIC: SetFlag(0x4) → AIV
```

BF16 path 需额外同步（AIV 行缩放完成后 AIC 才能读 vInner）：
```
AIC: SetFlag(0x6) → AIV  // CalVPrime 完成, vInner 就绪
AIV: WaitFlag(0x6)
AIV: v_new_scaled = vInner * d[:,None]
AIV: SetFlag(0x7) → AIC  // 行缩放完成
AIC: WaitFlag(0x7)
AIC: CalStateNew(state += v_new_scaled^T @ K)
```

### 收益

- 删除 `kg_` GM workspace：`BF16 × Nv × Sp × Dk`（典型 Sp=512, Nv=64, Dk=128 → **8MB**）
- 删除 Stage1 KgCompute 计算（broadcast + sub + exp + mul + cast + copyout）
- Stage2 CalStateNew 直接用 strided K（结合 R2 的 SetOrgShape Ka=Nk*Dk）

### 代价

- Stage2 AIV 新增行缩放（C 次 exp + C 次 sub + C×Dv 次 mul）
- Stage2 新增同步 flag（BF16 path: 0x6/0x7；FP32 path: 复用 0x5）
- AIC CalStateNew 改为 strided K matmul（SetOrgShape Ka=Nk*Dk）

---

## 2 GM Workspace 布局对比

### 2.1 重构前

```
offset 0:  gCum_         FP32  [Nv][Sp]
offset 1:  kCumDecay_    BF16  [Nv][Sp][Dk]
offset 2:  vInner_       FP32/BF16 [Nv][Sp][Dv]
offset 3:  vInnerBf16_   BF16  [Nv][Sp][Dv]            (FP32 path)
offset 4:  qPrime_       BF16  [Nv][Sp][Dk]
offset 5:  attnInter_    BF16  [Nv][Sp][Dv]
offset 6:  kg_           BF16  [Nv][Sp][Dk]
offset 7:  qkt_          BF16  [Nv][Sp][C]
offset 8:  curStateBf16_ BF16  [Nv][Dv][Dk]            (FP32 path)
offset 9:  stageOneMask_ FP32  [coreNum*2][C][C]
offset 10: stageThreeMask_ FP32 [coreNum*2][C][C]
offset 11: stageWsAddr_  → gBKWs/queryCon/keyCon/kkWs/attnWs/vBetaWs (6 个 WS buffer)
```

### 2.2 重构后

```
offset 0:  gCum_         FP32  [Nv][Sp]                ← 不变
offset 1:  kCumDecay_    BF16  [Nv][Sp][Dk]            ← 不变
offset 2:  vInner_       FP32/BF16 [Nv][Sp][Dv]        ← 不变
offset 3:  vInnerBf16_   BF16  [Nv][Sp][Dv]            (FP32 path, 不变)
offset 4:  qPrime_       BF16  [Nv][Sp][Dk]            ← 不变
offset 5:  attnInter_    BF16  [Nv][Sp][Dv]            ← 不变
offset 6:  kg_           BF16  [Nv][Sp][Dk]            ← 不变
offset 7:  qkt_          BF16  [Nv][Sp][C]             ← 不变
offset 8:  curStateBf16_ BF16  [Nv][Dv][Dk]            (FP32 path, 不变)
offset 9:  stageOneMask_ FP32  [coreNum*2][C][C]       ← 不变
offset 10: stageThreeMask_ FP32 [coreNum*2][C][C]      ← 不变
offset 11: stageWsAddr_  → gBKWs/kkWs/attnWs/vBetaWs (4 个 WS buffer)
```

**GM workspace 削减（仅 Stage1 WS）：**
- 删除 `queryConGm_` (BF16 × coreNum × P × C × Dk) — R2
- 删除 `keyConGm_` (BF16 × coreNum × P × C × Dk) — R2

**stageWorkspaceSz 变化：**
- 重构前：`sizeLow * C * (2*C + 3*Dk + Dv) * P * coreNum`（6 个 WS buffer）
- 重构后：`sizeLow * C * (2*C + 1*Dk + Dv) * P * coreNum`（4 个 WS buffer）
- `STAGE_ONE_THREE`: 3 → 1

---

## 3 Stage1 重构后数据流伪代码

### 3.1 AIV 侧 (ParaChunkAIV)

```
// ============ 步骤 1: g_cum + gamma (不变) ============
if gOptional:
    for i in [0, curParaNum):
        // g (T,Nv) strided → Gather → UB(C) → CumSum → gCumSumUB(C)
        // Exp → gCumExpUB(C)
        // DataCopyPad → GM gCum_ (供 Stage2/3 读)
        GCumExpCompute(gGm_[bgOffset[i]], gCumGm_[chunkRowBase[i]],
                       gCumSumUB[i*C], gCumExpUB[i*C], validLen[i])
        // gamma = exp(g_cum[:,None] - g_cum[None,:]) * tril_mask → gammaUB(C,C)
        GammaCompute(gBroadUB, gammaUB, gCumSumUB[i*C])

// R2: 不再调用 QKPreProcess, AIC 直接从 strided GM 读取 Q/K
// R2: 不再 SetFlag(0x9) (sync0 删除, AIC 无需等待 Q/K 连续化)

// ============ 步骤 2: beta 搬入 (不变) ============
for i in [0, curParaNum):
    BetaCopyInWithStride(betaGm_[bgOffset[i]], betaUB[i*H], subValidLen[i])

// ============ 步骤 3: KKBetaCompute + 求逆 (不变) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x8)                                  // ← AIC: KKT 就绪
    // attn = kk * beta (含 beta)
    KKBetaCompute(kkWs_[i*C*C], betaUB[i*H])
    // 求逆: A = (I + kk*beta*gamma)^{-1}  → 写入 attnWs (GM)
    InverseCompute(attnWs_[i*C*C], gammaUB)
    CrossCoreSetFlag<PIPE_MTE3>(0x7)                        // → AIC: 求逆完成

// ============ 步骤 4: GBKCompute (R2: K 从 strided GM 读) ============
for i in [0, curParaNum):
    // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
    // k_cumdecay = -1.0 * k * beta * g_cum_exp
    outKgGm_ = kgGm_[chunkRowBase[i]*Dk]
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // R2: K 从 strided GM 读取 (不再从 keyConGm_)
    GBKCompute(gBKWs_[i*C*Dk], outKgGm_, betaUB[i*H],
               gCumSumUB[i*C], gCumExpUB[i*C],
               keyGm_[qk_base + subOffset*qkStride], qkStride)
    CrossCoreSetFlag<PIPE_MTE3>(0x6)                        // → AIC: gBKWs 就绪

// ============ 步骤 5: VBetaCompute (不变) ============
for i in [0, curParaNum):
    vOffset = chunkStartRow[i] * (Nv*Dv) + nId[i]*Dv
    VBetaCompute(valueGm_[vOffset], vBetaWs_[i*C*Dv], betaUB[...], ...)
    CrossCoreSetFlag<PIPE_MTE3>(0x5)                        // → AIC: vBetaWs 就绪

// ============ 步骤 6: q_prime (R2: Q 从 strided GM 读) ============
for i in [0, curParaNum):
    // q_prime = query * scale * g_cum_exp[:,None]
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // R2: Q 从 strided GM 读取 (不再从 queryConGm_)
    QPrimeCompute(qPrimeGm, gCumSumUB[i*C], gCumExpUB[i*C],
                  queryGm_[qk_base + subOffset*qkStride], subValidLen[i], qkStride)

CrossCoreWaitFlag(0xA)                                      // ← AIC: QKT 完成
```

### 3.2 AIC 侧 (ParaChunkAIC)

```
// ============ matmul 1: KKT = K @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    // R2: 不再 WaitFlag(0x9) (sync0 删除)
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    // A=keyGm(C,Dk) stride=Nk*Dk, B=keyGm(C,Dk) transB stride=Nk*Dk
    AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWs[i*C*C],
               {M=C, N=C, K=Dk, sM=C, sN=C, sK=Dk, Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0x8)                         // → AIV: KKT 完成

// ============ matmul 2: QKT = Q @ K^T (R2: 直接读 strided GM) ============
for i in [0, curParaNum):
    qk_base = chunkStartRow[i] * Nk*Dk + nId[i] * (Nk/Nv)*Dk
    AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base], qktGm[...],
               {M=validLen[i], N=validLen[i], K=Dk,
                sM=validLen[i], sN=validLen[i], sK=Dk,
                Ka=Nk*Dk, Kb=Nk*Dk}, transB=true)
    CrossCoreSetFlag<PIPE_FIX>(0xA)                         // → AIV: QKT 完成

// ============ matmul 3: 求逆辅助 (不变, 读 attnWs) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x7)                                  // ← AIV: 求逆完成
    AttnInverseMMCompute(i * C*C)

// ============ matmul 4: k_cumdecay = A @ gBKWs (不变, 读连续 WS) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x6)                                  // ← AIV: GBKCompute 完成
    AICProcess(mmBf16_, attnWs_[i*C*C], gBKWs_[i*C*Dk], kCumDecayGm[...],
               {M=C, N=Dk, K=C, sM=C, sN=Dk, sK=C})        // Ka=0, Kb=0 → 3 参数 SetOrgShape

// ============ matmul 5: vInner = A @ vBeta (不变, 读连续 WS) ============
for i in [0, curParaNum):
    CrossCoreWaitFlag(0x5)                                  // ← AIV: VBetaCompute 完成
    AICProcess(mmVInner_, attnWs_[i*C*C], vBetaWs_[i*C*Dv], vInnerGm[...],
               {M=C, N=Dv, K=C, sM=C, sN=Dv, sK=C})        // Ka=0, Kb=0 → 3 参数 SetOrgShape
```

### 3.3 Stage1 CrossCore 同步关系图（重构后）

```
AIV                                       AIC
────                                       ────
[R2: 不再 QKPreProcess, 不再 SetFlag(0x9)]
GCumExpCompute + GammaCompute (→ gCum_ GM)
beta 搬入
                                            KKT matmul (strided GM → kkWs)  ← R2
                                            SetFlag(0x8) ──────────→ WaitFlag(0x8)
KKBetaCompute (读 kkWs, 含 beta)
InverseCompute (求逆 A → attnWs)
SetFlag(0x7) ─────────────────────────→   WaitFlag(0x7)
                                            AttnInverseMMCompute
GBKCompute (K 从 strided GM 读 → gBKWs)    ← R2
SetFlag(0x6) ─────────────────────────→   WaitFlag(0x6)
                                            k_cumdecay = A @ gBKWs
VBetaCompute (→ vBetaWs)
SetFlag(0x5) ─────────────────────────→   WaitFlag(0x5)
                                            vInner = A @ vBetaWs
                                            QKT matmul (strided GM → qkt_)  ← R2
                                            SetFlag(0xA) ──────────→ WaitFlag(0xA)
QPrimeCompute (Q 从 strided GM 读)         ← R2
```

**与重构前同步关系变化：**
- 删除 sync0 (`0x9`)：AIC 不再等待 AIV 的 QKPreProcess，直接从 strided GM 读 Q/K

### 3.4 Stage1 输出（与重构前完全一致）

| 输出 | 公式 | 变化 |
|:-----|:-----|:-----|
| gCum_ | `g.cumsum()` FP32 Nv×Sp | 不变 |
| kCumDecay_ | `A@(-k*beta*g_exp)` BF16 Nv×Sp×Dk | 不变 |
| vInner_ | `A@(v*beta)` FP32/BF16 Nv×Sp×Dv | 不变 |
| qPrime_ | `q*scale*g_exp` BF16 Nv×Sp×Dk | 不变 (R2: Q 来源从 queryConGm 改为 strided GM) |
| kg_ | `k*exp(g[-1]-g_cum)` BF16 Nv×Sp×Dk | 不变 |
| qkt_ | `Q@K^T` BF16 Nv×Sp×C | 不变 (R2: Q/K 来源从 queryConGm/keyConGm 改为 strided GM) |

---

## 4 Stage2 数据流（与重构前完全一致）

R3-R7 全部回退，Stage2 无任何改动。详见重构前文档 §4。

### 4.1 核心计算公式（正确，与参考实现一致）

```
v_prime = k_cumdecay @ state^T              // k_cumdecay 含 g_exp
v_new = v_inner - v_prime                    // v_inner = A@(V*beta)
attn_inter = q_prime @ state^T               // q_prime 含 g_exp
core_attn_out = attn_inter + attn @ v_new

state_new = state * exp(g[-1]) + v_new^T @ kg
           // kg = K * exp(g[-1]-g_cum)[...,None]
```

### 4.2 同步关系（与重构前一致）

```
AIV                                            AIC
────                                            ────
state CopyIn + Cast
CalGCumExpFp32/Bf16 (从 GM gCum_ 读)
SetFlag(0x3) [FP32 only]
                                                WaitFlag(0x3) [FP32 only]
                                                CalVPrime (vInner += k_cumdecay@state^T)
                                                CalAttnInter (qPrime@state^T → out)
                                                SetFlag(0x2) ─────────→ WaitFlag(0x2)
CopyOutState
[FP32: CastGmToGm (vInner FP32→BF16)]
SetFlag(0x5)                                   WaitFlag(0x5)
                                                CalStateNew (state += v_new^T @ kg)
WaitFlag(0x4)                                  SetFlag(0x4) ─────────→ WaitFlag(0x4)
[下一 chunk]
```

---

## 5 Stage3 数据流（与重构前完全一致）

R6 回退，Stage3 无任何改动。decay_mask 从 GM `gCum_` 读取（Stage1 已写入）。

### 5.1 核心计算公式（正确）

```
decay_mask = exp(g_cum[:,None] - g_cum[None,:]) * tril    // 从 GM gCum_ 读
masked_qkt = qkt * scale * decay_mask * mask
attn_out = masked_qkt @ v_new (accum=1)
```

---

## 6 各级存储数据流总表（重构后）

### 6.1 GM 层

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| query_/key_/value_/beta_ | BF16 | T×Nk×Dk / T×Nv×Dv | 用户输入 | Stage1 | R2: Stage1 直接读 strided GM |
| g_ | FP32 | T×Nv | 用户输入 | Stage1 | 不变 |
| gCum_ | FP32 | Nv×Sp | Stage1 | Stage2/3 | 不变 |
| kCumDecay_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| vInner_ | FP32/BF16 | Nv×Sp×Dv | Stage1→Stage2 | Stage2→Stage3 | 不变 |
| vInnerBf16_ | BF16 | Nv×Sp×Dv | Stage2 | Stage2/3 | 不变 |
| qPrime_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| kg_ | BF16 | Nv×Sp×Dk | Stage1 | Stage2 | 不变 |
| qkt_ | BF16 | Nv×Sp×C | Stage1 | Stage3 | 不变 |
| stateOut_ | FP32/BF16 | Nv×Dv×Dk | Stage2 | Stage2 (递推) | 不变 |
| out_ | BF16 | T×Nv×Dv | Stage2→Stage3 | 用户输出 | 不变 |

### 6.2 Stage1 WS 层（重构后）

| 变量 | dtype | shape | 写入方 | 读取方 | 与重构前变化 |
|:-----|:------|:------|:-------|:-------|:-------------|
| ~~queryConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| ~~keyConGm_~~ | ~~BF16~~ | ~~P×C×Dk~~ | ~~AIV~~ | ~~AIC~~ | **R2 删除** |
| gBKWsGm_ | BF16 | P×C×Dk | AIV GBKCompute | AIC matmul 4 | 不变 |
| kkWsGm_ | BF16 | P×C×C | AIC matmul 1 | AIV KKBetaCompute | 不变 |
| attnWsGm_ | BF16 | P×C×C | AIV InverseCompute | AIC matmul 3/5 | 不变 |
| vBetaWsGm_ | BF16 | P×C×Dv | AIV VBetaCompute | AIC matmul 5 | 不变 |

### 6.3 UB 层

无新增/变更。所有 UB buffer 与重构前一致。

---

## 7 重构前后计算公式对比

**结论：所有计算公式完全一致，R2 不改变任何数学。**

### 7.1 Stage1 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| KKT | `(k@k^T) * beta * gamma` | 不变 |
| 求逆矩阵 M | `I - kk*beta*gamma` | 不变 |
| 求逆结果 A | `(I-kk*beta*gamma)^{-1}` | 不变 |
| v_inner | `A @ (v * beta)` | 不变 |
| k_cumdecay | `A @ (-k * beta * g_cum_exp)` | 不变 |
| kg | `k * exp(g[-1]-g_cum)[...,None]` | 不变 |
| q_prime | `q * scale * g_cum_exp[:,None]` | 不变 |
| QKT | `Q @ K^T` | 不变 |
| gCum | `g.cumsum()` → GM | 不变 |
| Q/K 读取 | 从 queryConGm_/keyConGm_ (WS) | **R2: 从 strided GM (SetOrgShape Ka/Kb)** |

### 7.2 Stage2 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| g_cum_exp 来源 | 从 GM gCum_ 读 | 不变 |
| v_prime | `k_cumdecay @ state^T` | 不变 |
| v_new | `v_inner - v_prime` | 不变 |
| state 衰减 | `state *= exp(gCum[-1])` | 不变 |
| attn_inter | `q_prime @ state^T` | 不变 |
| state_new | `state*exp(g[-1]) + v_new^T @ kg` | 不变 |

### 7.3 Stage3 核心计算

| 计算项 | 公式 | 重构前→后 |
|:-------|:-----|:----------|
| decay_mask | 从 GM gCumExp 读 → broadcast | 不变 |
| masked_qkt | `qkt * scale * decay * mask` | 不变 |
| attn_out | `masked_qkt @ v_new` (accum=1) | 不变 |

---

## 8 数据流图（重构后）

```
原始输入: Q, K, V, beta, g, state, scale
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage1 (per chunk, P 并行)           │
    │                                               │
    │  AIV:                                         │
    │    g → cumsum → g_cum_exp → gCum_ (GM)        │
    │    gamma = decay_mask (UB)                     │
    │    beta → UB                                   │
    │    [R2: 不再 QKPreProcess]                     │
    │    WaitFlag(0x8) ← AIC KKT                     │
    │    attn = kk * beta                            │
    │    A = (I - kk*beta*gamma)^{-1} → attnWs       │
    │    SetFlag(0x7) → AIC                          │
    │    GBKCompute: K 从 strided GM 读 → gBKWs      │  ← R2
    │    SetFlag(0x6) → AIC                          │
    │    VBetaCompute: V*beta → vBetaWs              │
    │    SetFlag(0x5) → AIC                          │
    │    q_prime = q*scale*g_exp (Q 从 strided GM)   │  ← R2
    │    kg = K*exp(g[-1]-g_cum) → kg_ (GM)          │
    │                                               │
    │  AIC:                                         │
    │    KKT = K@K^T (strided GM, Ka=Nk*Dk)         │  ← R2
    │    QKT = Q@K^T (strided GM)                    │  ← R2
    │    求逆辅助 matmul (读 attnWs)                  │
    │    k_cumdecay = A @ gBKWs                      │
    │    vInner = A @ vBetaWs                        │
    │                                               │
    │  输出: gCum_, kCumDecay_, vInner_, qPrime_,    │
    │        kg_, qkt_ (公式不变)                    │
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage2 (per chunk, 递推)             │
    │                                               │
    │  AIV:                                         │
    │    g_cum_exp (从 GM gCum_ 读)                  │
    │    state *= exp(g_cum[-1])                     │
    │    SetFlag(0x3) → AIC                          │
    │    WaitFlag(0x2) ← AIC                         │
    │    [FP32: CastGmToGm (vInner FP32→BF16)]      │
    │    SetFlag(0x5) → AIC                          │
    │                                               │
    │  AIC:                                         │
    │    v_prime = k_cumdecay @ state^T (accum=1)   │
    │    attn_inter = qPrime @ state^T               │
    │    state_new = state*exp(g[-1]) + v_new^T@kg  │
    └───────────────┬───────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────────────┐
    │            Stage3 (per chunk, 并行)             │
    │                                               │
    │  AIV:                                         │
    │    decay_mask (从 GM gCum_ 读 → broadcast)     │
    │    masked_qkt = qkt * scale * decay * mask     │
    │                                               │
    │  AIC:                                         │
    │    attn_out = masked_qkt @ v_new (accum=1)     │
    └───────────────────────────────────────────────┘
```
