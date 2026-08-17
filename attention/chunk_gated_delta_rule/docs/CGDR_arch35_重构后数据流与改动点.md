# CGDR arch35 重构后数据流与改动点

> 本文档描述 arch35 CGDR 算子重构方案。R2 已实现，R4/R5/R7 待实现。
> 数学等价性已用 Python 验证（`/tmp/opencode/verify_r4_r5_r7.py`，误差 < 1e-12）。

---

## 0 重构范围

| 编号 | 需求 | 状态 | 说明 |
|:-----|:-----|:-----|:-----|
| R2 | matmul 直接读 strided GM | **已实现** | 删除 QKPreProcess/queryConGm_/keyConGm_ |
| R3 | g_exp 移至 Stage2 | **不可行** | g_exp 在 matmul 内维（A 的列），A 非对角不可移到结果行 |
| R4 | beta 移到 A 列缩放 | **待实现** | `A_beta = A @ diag(beta)`，删除 VBetaCompute/vBetaWsGm_ |
| R5 | g_exp 移到 A 列缩放 | **待实现** | `A_beta_gexp = A_beta @ diag(g_exp)`，删除 GBKCompute K缩放/gBKWsGm_ |
| R6 | gCum_ 输出到 GM 后续重算 | **已满足** | 原始代码已是此做法 |
| R7 | kg 系数移到 v_new | **待实现** | `v_new *= d[:,None]`，删除 KgCompute/kg_ |

---

## 1 正确公式（参考实现）

基于 `chunk_gated_delta_rule_golden.py:102-149`，矩阵形式（`D=diag(beta)`, `G=diag(g_exp)`, `d=exp(g[-1]-g_cum)`）：

```
Stage1:
  KKT:     attn = -(D @ K @ K^T) * gamma        // beta 在 KKT 内部
  求逆:    A = (I + attn)^{-1}                   // beta 在求逆内部, 不可移出
  v_inner:    A @ D @ V                           // beta 乘 V 左边
  k_cumdecay: A @ D @ G @ (-K)                    // beta+g_exp 乘 K 左边 (kernel 用负号 + accum=1)

Stage2 (per chunk):
  v_new = v_inner + k_cumdecay @ state            // accum=1 融合
  state_new = state * g_exp[-1] + v_new^T @ diag(d) @ K
```

---

## 2 R3 不可行原因

g_exp 在 `A @ D @ G @ K` 中位于 A 右边（A 的列 = matmul 内维/求和下标）：

```
k_cumdecay[i,d] = Σ_m A[i,m] · g_exp[m] · (-β[m]·K[m,d])
                         ↑ g_exp[m] 在求和内部, 每项 m 不同
```

A 是下三角求逆矩阵（非对角），求和后 g_exp[m] 的逐项权重不可逆地丢失。等价地：`A @ diag(g) @ K ≠ diag(g) @ A @ K`（A 非对角不可交换）。g_exp 无法移到结果行（Stage2）。

---

## 3 R4+R5 正确方案：两步列缩放

### 数学推导

利用结合律，将对角阵从 K/V 左边移到 A 右边（**列缩放**）：

```
R4: A_beta = A @ D = A * beta[None,:]            // 第 j 列乘 beta[j]
    v_inner = A_beta @ V                          // V 不乘 beta

R5: A_beta_gexp = A_beta @ G = A_beta * g_exp[None,:]  // 第 j 列再乘 g_exp[j]
    k_cumdecay = -A_beta_gexp @ K                 // K 不乘 beta/g_exp

等价性: A @ D @ G = A @ diag(β·g_exp) = A_beta_gexp  (对角阵可合并)
```

### 关键：列缩放 ≠ 行缩放

```
A @ diag(β)  →  第 j 列乘 β[j]  (列缩放, 正确)
diag(β) @ A  →  第 i 行乘 β[i]  (行缩放, 错误, A 非对角不可交换)
```

### 澄清：g_exp ≠ gamma

| 名称 | shape | 定义 | 用在哪 |
|:-----|:------|:-----|:-------|
| gamma (decay_mask) | C×C | `exp(g[i]-g[j]) * tril` | KKT |
| g_exp | C 维 | `exp(g_cum[j])` | k_cumdecay 列缩放 (R5) |

R5 用的是 g_exp（C 维向量列缩放），不是 gamma（C×C 矩阵逐元素乘）。

### 变更

| 函数 | 原始 | R4+R5 后 |
|:-----|:-----|:---------|
| InverseCompute | 写 A → attnWs | 写 A → attnWs（不变） |
| AttnInverseMMCompute | 读/写 attnWs 的 A | 不变（操作原始 A） |
| **新增 ColumnScale** | — | AttnInverseMMCompute 后：读 A → 列缩放β → A_beta → 写 attnWs；再列缩放g_exp → A_beta_gexp → 写 attnWsNegGm_ |
| KKBetaCompute | `attn = kk * beta` | 不变 |
| VBetaCompute | `v_beta = V*beta` → vBetaWsGm_ | **删除** |
| GBKCompute | `gBKWs = -K*beta*g_exp` → gBKWsGm_ | **删除**（K 不再乘 beta/g_exp） |
| KgCompute | `kg = K*d` → kg_ | 保留（R7 待实现时删除） |
| AIC matmul4 | `A @ gBKWs` | `-A_beta_gexp @ K`（K 从 strided GM） |
| AIC matmul5 | `A @ vBetaWs` | `A_beta @ V`（V 从 strided GM） |

### 同步

```
AIV: InverseCompute → 写 A 到 attnWs → SetFlag(0x7)
AIC: WaitFlag(0x7) → AttnInverseMMCompute → SetFlag(0xB)     // 新增
AIV: WaitFlag(0xB) → 列缩放β → A_beta → 写 attnWs
                    → 列缩放g_exp → A_beta_gexp → 写 attnWsNegGm_
                    → SetFlag(0x5) + SetFlag(0x6)
AIC: WaitFlag(0x5) → matmul5: A_beta @ V
     WaitFlag(0x6) → matmul4: -A_beta_gexp @ K
```

### 收益

- 删除 `vBetaWsGm_`（~2.6MB）+ `gBKWsGm_`（~2.6MB）
- 删除 VBetaCompute + GBKCompute 全部计算
- AIC matmul4/5 直接读原始 V/K（strided GM，结合 R2）
- 新增 `attnWsNegGm_`（C×C，较小）

---

## 4 R7 正确方案：d 移到 v_new 行缩放

### 数学推导

```
原始: state_new += v_new^T @ diag(d) @ K        // d = exp(g[-1]-g_cum), 逐行向量

R7:   state_new += (v_new^T @ diag(d)) @ K
           = (v_new * d[:,None])^T @ K           // d 移到 v_new 行
```

`v^T @ diag(d) = (diag(d) @ v)^T = (v * d[:,None])^T`（v 的行缩放）。

d 在 `v^T @ diag(d) @ K` 中位于 v^T 右边（= v 的行 = 求和下标），可直接移。
（与 R3 的 g_exp 不同：g_exp 在 A 右边=A 的列，A 非对角不可移。）

### 变更

| 函数 | 原始 | R7 后 |
|:-----|:-----|:------|
| Stage1 KgCompute | `kg = K*d` → kg_ GM | **删除** |
| Stage2 AIV | — | 计算 `d=exp(g[-1]-g_cum)`（从 gCum_ 读），`v_new_scaled = v_new * d[:,None]` |
| Stage2 AIC CalStateNew | `state += v_new^T @ kg` | `state += v_new_scaled^T @ K`（K 从 strided GM） |

### 同步（BF16 path）

```
AIV: WaitFlag(0x2) ← AIC CalVPrime
AIV: v_new_scaled = vInner * d[:,None]
AIV: SetFlag(0x5) → AIC
AIC: WaitFlag(0x5) → CalStateNew(state += v_new_scaled^T @ K, accum=1)
AIC: SetFlag(0x4) → AIV
```

### 收益

- 删除 `kg_` GM WS（`BF16 × Nv × Sp × Dk`，典型 ~8MB）
- 删除 Stage1 KgCompute 计算

---

## 5 R6 已满足

原始代码已采用 gCum_ 输出到 GM（连续 [Nv,Sp] 布局），后续从 gCum_ 重算：
- Stage2 `CalGCumExpFp32/Bf16`：取 `gCum[-1]` 做 exp
- Stage3 `CalMaskedQKT`：算 `decay_mask = exp(g_cum[:,None]-g_cum[None,:])`
- R7 中 `d = exp(g[-1]-g_cum[j])` 同样从 gCum_ 重算

---

## 6 重构后 Stage1 计算流程（R2+R4+R5）

### AIV 侧

```
1. g_cum + gamma (不变)
   GCumExpCompute → gCum_ GM, gCumExpUB
   GammaCompute → gammaUB

2. beta 搬入 (改为读全量 C 个 beta, 供列缩放)
   BetaCopyInWithStride → betaUB

3. KKBetaCompute + 求逆 (不变)
   WaitFlag(0x8) ← AIC KKT
   attn = kk * beta * gamma
   A = (I - kk*beta*gamma)^{-1} → 写 attnWs
   SetFlag(0x7) → AIC

4. 列缩放 (R4+R5 新增)
   WaitFlag(0xB) ← AIC AttnInverseMMCompute
   读 A from attnWs
   A_beta = A * beta[None,:]      // R4: 列缩放 β
   写 A_beta → attnWs              // 给 matmul5
   A_beta_gexp = A_beta * g_exp[None,:]  // R5: 列缩放 g_exp
   写 A_beta_gexp → attnWsNegGm_   // 给 matmul4
   SetFlag(0x5) + SetFlag(0x6) → AIC

5. q_prime (R2: Q 从 strided GM)
   QPrimeCompute

6. KgCompute (保留, R7 待删除)
   kg = K * d[:,None] → kg_ GM

WaitFlag(0xA) ← AIC QKT
```

### AIC 侧

```
1. KKT (R2: strided GM)
   keyGm[strided] @ keyGm[strided]^T → kkWs
   SetFlag(0x8) → AIV

2. QKT (R2: strided GM)
   queryGm[strided] @ keyGm[strided]^T → qkt_
   SetFlag(0xA) → AIV

3. 求逆辅助 (不变)
   WaitFlag(0x7) ← AIV 求逆
   AttnInverseMMCompute(读/写 attnWs 的 A)
   SetFlag(0xB) → AIV                    // 新增

4. k_cumdecay (R4+R5: 读 A_beta_gexp, K 从 strided GM)
   WaitFlag(0x6) ← AIV
   -A_beta_gexp @ keyGm[strided] → kCumDecay_

5. vInner (R4: 读 A_beta, V 从 strided GM)
   WaitFlag(0x5) ← AIV
   A_beta @ valueGm[strided] → vInner_
```

### Stage1 WS 布局（R2+R4+R5 后）

```
kkWsGm_      BF16 [P×C×C]     // KKT 结果 (不变)
attnWsGm_    BF16 [P×C×C]     // A_beta (R4 后)
attnWsNegGm_ BF16 [P×C×C]     // A_beta_gexp (R5 新增)
// 删除: queryConGm_/keyConGm_ (R2), vBetaWsGm_ (R4), gBKWsGm_ (R5)
```

---

## 7 重构后 Stage2 计算流程（R7）

### AIV 侧

```
1. state 搬入 + Cast (不变)
2. CalGCumExp (从 gCum_ 读, 不变)
3. SetFlag(0x3) → AIC

4. WaitFlag(0x2) ← AIC CalVPrime+CalAttnInter

5. R7: v_new 行缩放
   d = exp(gCum[-1] - gCum[0:curChunkSize])    // 从 gCum_ 读
   v_new_scaled = vInner * d[:,None]
   [FP32: CastGmToGm vInner→vInnerBf16, 改为 CastGmToGmWithScale]
   SetFlag(0x5) → AIC

6. WaitFlag(0x4) ← AIC CalStateNew
```

### AIC 侧

```
1. WaitFlag(0x3) ← AIV state
2. CalVPrime: vInner += k_cumdecay @ state^T (accum=1)
3. CalAttnInter: out += qPrime @ state^T
4. SetFlag(0x2) → AIV

5. WaitFlag(0x5) ← AIV v_new_scaled
6. CalStateNew: state += v_new_scaled^T @ K (accum=1, K 从 strided GM)
7. SetFlag(0x4) → AIV
```

---

## 8 Python 验证

```python
# /tmp/opencode/verify_r4_r5_r7.py
# C=8, Dk=16, Dv=16, float64
# 结果: v_inner/k_cumdecay/v_new/state_new 全部误差 < 1e-12
#
# R4: A_beta = A * beta[None,:]           (列缩放)
# R5: A_beta_gexp = A_beta * g_exp[None,:] (第二步列缩放)
# R7: state_new += (v_new * d[:,None])^T @ K
```

---

## 9 GM Workspace 布局对比

### 重构前

```
gCum_ kCumDecay_ vInner_ vInnerBf16_ qPrime_ attnInter_ kg_ qkt_ curStateBf16_ masks stageWs
stageWs: gBKWs/queryCon/keyCon/kkWs/attnWs/vBetaWs (6 buffer)
```

### 重构后（R2+R4+R5+R7）

```
gCum_ kCumDecay_ vInner_ vInnerBf16_ qPrime_ attnInter_ qkt_ curStateBf16_ masks stageWs
stageWs: kkWs/attnWs/attnWsNeg (3 buffer)
// 删除: kg_ (R7), queryCon/keyCon (R2), vBetaWs (R4), gBKWs (R5)
// 新增: attnWsNeg (R5, C×C)
```

### WS 削减汇总

| 删除项 | 大小 | 来源 |
|:-------|:-----|:-----|
| queryConGm_ + keyConGm_ | ~3MB | R2 |
| vBetaWsGm_ | ~2.6MB | R4 |
| gBKWsGm_ | ~2.6MB | R5 |
| kg_ | ~8MB | R7 |
| **新增** attnWsNegGm_ | ~0.5MB | R5 |
| **净削减** | **~15.7MB** | |
