# SFA 两遍设计的数学推导与代码验证

## 一、数学推导：证明两遍设计与标准 Flash Attention 等价

### 1.1 标准单遍 Online Softmax（完整推导）

对 Q 的一个 `kTm` 行块，K/V 按 `kTk` 行切块。记 `S_j = Q × K_j^T / √d` 为第 j 个 K block 与 Q 的注意力分数（未归一化 logits）。用 online softmax 逐 K block 迭代，每个 K block 执行四步。

**初始化：**

```
m = -∞       (kTm × 1)    —— 历史行最大值
s = 0        (kTm × 1)    —— 历史分母
O = 0        (kTm × vD)   —— 未归一化输出分子
```

**遍历每个 K block j，对每一行 0 ≤ r < kTm，执行四步：**

```
S_j = Q × K_j^T / √d                                    [S]   Q×K^T → 乘 scale

(a) m_new = max(m_old, rowmax(S_j))                     [M]   更新全局行最大

(b) s_new = s_old × exp(m_old - m_new)                  [S1]  历史 s 乘以 rescale
          + rowsum(exp(S_j - m_new))                    [S2]  加上当前块的贡献

(c) O = O × exp(m_old - m_new)                          [O1]  历史 O 乘以 rescale
       + exp(S_j - m_new) × V_j                         [O2]  加上当前块的 PV

(d) m_old = m_new,  s_old = s_new                            状态继承
```

**关键点说明（内嵌在流程中，不单独拆节）**：

- **rescaling 的原理**：在 (b)[S1] 和 (c)[O1] 中，`exp(m_old - m_new)` 就是 rescale 因子。因为指数函数满足 `exp(a) × exp(b) = exp(a+b)`，所以：

  ```
  exp(S_old - m_old) × exp(m_old - m_new) = exp(S_old - m_new)
  ```

  即把历史项的基准从旧的 `m_old` 对齐到新的 `m_new`，只需乘以一个因子。

- **O 更新的物理含义**：(c)[O1] 将历史输出缩放到新 max 基准，(c)[O2] 加上当前块在新基准下的 PV 贡献。两步合在一起，O 始终保持的不变量为：

  ```
  处理完前 j 个 K block 后: O = Σ_{k=1..j} exp(S_k - m_j) × V_k
                          s = Σ_{k=1..j} exp(S_k - m_j)
  ```

- **最终归一化**：遍历全部 K block 后得到 `m_final`、`s_final`、`O`，输出 `O = O / s_final`：

  ```
  O / s = Σ_j exp(S_j - m_final) × V_j / Σ_j exp(S_j - m_final)
        = Σ_j exp(S_j) × exp(-m_final) × V_j / Σ_j exp(S_j) × exp(-m_final)
        = Σ_j exp(S_j) × V_j / Σ_j exp(S_j)                                 (exp(-m_final) 约去)
        = softmax(QK^T/√d) × V   ✓
  ```

---

### 1.2 SFA 两遍拆解

标准单遍的四步 (a)(b)(c)(d) 中，(c) 要求 O 在整个循环中持续存活（因为它还要被后续的 rescale × 新 PV 反复更新）。在稀疏场景下 K block 数量不固定，tile 寄存器压力有溢出风险。

两遍设计的思路：**把 (a)(b) 和 (c) 拆到两个独立的循环中，用额外一次 Q×K^T 的代价换降低寄存器压力**。

```
Pass 1 (reduce):   只执行 (a)(b)(d)，不产生 O

    S_j = Q × K_j^T / √d                              ← 第一遍算 QK^T
    m_new = max(m_old, rowmax(S_j))                   (a)
    s_new = s_old × exp(m_old - m_new)
          + rowsum(exp(S_j - m_new))                  (b)
    m_old = m_new,  s_old = s_new                     (d)

    遍历完所有活跃 K block 后 → m_final,  s_final

Pass 2 (attend):   直接用 m_final, s_final 归一化，跳过 (a)(b)(d)

    S_j = Q × K_j^T / √d                              ← 第二遍算 QK^T (重复计算)

    p_j = exp(S_j - m_final) / s_final                ← 用最终 m,s 一步归一化！
    O = O + p_j × V_j                                  ← 纯累加，(c) 的 rescale 被消除

    遍历完 → 最终写回 O (已是归一化结果)
```

**为什么 Pass 2 不需要 rescale？** 单遍中 O 被 rescale 是因为每个 K block 后 `m_old` 都会增大，历史 PV 的基准 `m_old` 需要对齐到新的 `m_new`。但 Pass 2 中，p_j 从诞生起就用的就是**全局最终 m_final**——m_final 在 Pass 2 中不更新，不再变了。所以每个 p_j 从一开始就是"基准稳定"的归一化概率，直接乘 V 累加即可，不需要中途 rescale。

**代价**：Q×K^T 算了两次（Pass 1 和 Pass 2 各一次）。好处是 Pass 2 里不存在 `rescale × O` 这种需要 O 长活的模式，tile 寄存器压力显著降低。

---

## 二、对照代码验证

### 2.1 Pass 1 — 对应数学公式 (a)(b)(d)（sfa_pto.hpp:94-146）

```cpp
for (int i = 0; i < Qb; ++i) {
    tileQ tQ;  TLOAD(tQ, gIterQ(i, 0));                          // 加载 Q

    tileMax tMax;  TEXPANDS(tMax, -1e30f);   // m_old = -∞
    tileSum tSum;  TEXPANDS(tSum, 0.0f);    // s_old = 0

    for (p = kv_begin; p < kv_end; p++) {
        j = kv_idx_ptr[p];

        // ---------- [S]  S_j = Q × K_j^T / √d ----------
        tileK tK;  TLOAD(tK, gIterK(j, 0));
        tileW_out tW_out;  TMATMUL(tW_out, tQ, tK);  // Q × K^T → Acc
        tileW tW;  ACCCVT(tW, tW_out);                // Acc → Vec
        TMULS(tW, tW, scale);                         // S_j = S / √d

        // ---------- (a)  m_new = max(m_old, rowmax(S_j)) ----------
        tileMax tLocalMax;  TCOLMAX(tLocalMax, tW);          // rowmax(S_j)
        tileMax tNewMax;    TMAX(tNewMax, tMax, tLocalMax);  // max 合并

        // ---------- rescale = exp(m_old - m_new)  (公共子表达式) ----------
        tileMax tScale;  TSUB(tScale, tMax, tNewMax);
        TEXP(tScale, tScale);

        // ---------- (b)[S1]  s_old × rescale ----------
        tileSum tScaledOldSum;  TMUL(tScaledOldSum, tSum, tScale);

        // ---------- (b)[S2]  rowsum(exp(S_j - m_new)) ----------
        TCOLEXPANDSUB_TEPL(tW, tW, tNewMax);                // S_j - m_new
        TEXP(tW, tW);                                       // exp
        tileSum tLocalSum;  TCOLSUM(tLocalSum, tW);        // rowsum
        tileSum tNewSum;    TADD(tNewSum, tScaledOldSum, tLocalSum); // s_new

        // ---------- (d)  状态继承 ----------
        tMax = tNewMax;  tSum = tNewSum;
    }
    // 至此: tMax = m_final,  tSum = s_final
```

### 2.2 Pass 2 — 对应 p_j 归一化 + O 纯累加（sfa_pto.hpp:152-198）

```cpp
    tileQ tQ2;  TLOAD(tQ2, gIterQ(i, 0));               // 重新加载 Q

    tileSum tInvSum;  TRECIP(tInvSum, tSum);             // 1 / s_final

    tileO tO;  TEXPANDS(tO, 0.0f);                       // O = 0

    for (p = kv_begin; p < kv_end; p++) {
        j = kv_idx_ptr[p];

        // ---------- [S]  S_j = Q × K_j^T / √d (第二遍!) ----------
        tileK tK;  TLOAD(tK, gIterK(j, 0));
        tileW_out tW_out;  TMATMUL(tW_out, tQ2, tK);
        tileW tW;  ACCCVT(tW, tW_out);
        TMULS(tW, tW, scale);

        // ---------- p_j = exp(S_j - m_final) / s_final ----------
        TCOLEXPANDSUB_TEPL(tW, tW, tMax);               // S_j - m_final
        TEXP(tW, tW);                                    // exp
        TCOLEXPANDMUL_TEPL(tW, tW, tInvSum);             // × 1/s  →  归一化完成

        // ---------- 装箱 p_j → Left，加载 V ----------
        tileW_cast tExpW;  TCVT(tExpW, tW);             // float → dtype
        tileW_left tW_left;  TCVT(tW_left, tExpW);      // Vec → Left
        tileV tV;  TLOAD(tV, gIterV(j, 0));

        // ---------- O += p_j × V_j ----------
        tileO_out tPV_out;  TMATMUL(tPV_out, tW_left, tV);  // fresh matmul
        tileO tPV;  ACCCVT(tPV, tPV_out);                    // Acc → Vec
        TADD(tO, tO, tPV);                                    // O += PV (纯加)
    }

    // ---------- 最终写回 (O 已归一化) ----------
    tileO_cast tO_cast;  TCVT(tO_cast, tO);
    TSTORE(gIterO(i, 0), tO_cast);
```

### 2.3 Pass 2 的 O 为什么不需要 rescale？(简短验证)

标准单遍中，每个 K block 后 `m_old` 都会变大，历史的 O 贡献需要乘以 `exp(m_old - m_new)` 对齐基准。

但在 Pass 2 中，p_j 从第一步起就用了最终的 `m_final` 和 `s_final` 完成归一化：

```
p_j = exp(S_j - m_final) / s_final
```

此时 p_j 已经直接等于标准 softmax 权重——因为 `m_final` 在 Pass 2 整个循环中不更新、不改变。所以每个 `p_j × V_j` 天然就是归一化输出的一部分，直接累加即得最终 O：

```
O = Σ_j p_j × V_j = Σ_j exp(S_j - m_final) / s_final × V_j
  = Σ_j exp(S_j) × V_j / Σ_k exp(S_k)  ✓
```

Pass 1 已经把 `m_final` 和 `s_final` 算好了，Pass 2 直接"套用"最终结果，用 p_j 事前归一化替掉了 (c) 中的在线 rescale。

---

### 2.4 公式与代码完整映射表

| 数学步骤 | 代码 Pass 1 | 代码 Pass 2 |
|----------|------------|------------|
| `S_j = Q×K_j^T/√d` | `TMATMUL → ACCCVT → TMULS(scale)` | 同上（第二遍） |
| (a) `m_new = max(m_old, rowmax(S_j))` | `TCOLMAX → TMAX` | — |
| rescale `= exp(m_old - m_new)` | `TSUB → TEXP` | — |
| (b) `s_new = s_old × rescale + rowsum(exp(S_j - m_new))` | `TMUL → TCOLEXPANDSUB_TEPL → TEXP → TCOLSUM → TADD` | — |
| (c) `O = O × rescale + exp(S_j - m_new) × V_j` | — | 被拆成两步：`p_j = exp(S_j - m_final) / s_final` 提前归一化 + `O += p_j × V_j` 纯累加 |
| `p_j = exp(S_j - m_final) / s_final` | — | `TCOLEXPANDSUB_TEPL → TEXP → TCOLEXPANDMUL_TEPL` |
| `O += p_j × V_j` | — | `TCVT(装箱) → TMATMUL → ACCCVT → TADD` |
| (d) `m → m_new, s → s_new` | `=` 赋值 | — |
| 最终写回 `O` | — | `TCVT → TSTORE` |
