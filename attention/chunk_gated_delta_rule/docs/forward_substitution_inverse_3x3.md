# 前代法求逆 3×3 逐步推导

针对下三角单位对角矩阵 L(对角线全 1,上三角全 0),求其逆矩阵 X = L⁻¹,使得 L·X = I。
本文档逐步展示公式推导与下标说明,对应 `InverseAIVVFImpl` 的实现逻辑。

---

## 1. 矩阵定义(含下标)

设 L 为 3×3 下三角单位对角矩阵,X = L⁻¹ 同样为下三角单位对角:

$$
L = \begin{pmatrix}
L_{00} & 0 & 0 \\
L_{10} & L_{11} & 0 \\
L_{20} & L_{21} & L_{22}
\end{pmatrix}
= \begin{pmatrix}
1 & 0 & 0 \\
a & 1 & 0 \\
b & c & 1
\end{pmatrix}
$$

$$
X = L^{-1} = \begin{pmatrix}
X_{00} & 0 & 0 \\
X_{10} & X_{11} & 0 \\
X_{20} & X_{21} & X_{22}
\end{pmatrix}
$$

满足约束:

$$
L \cdot X = I = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

展开矩阵乘法:

$$
\begin{pmatrix}
1 & 0 & 0 \\
a & 1 & 0 \\
b & c & 1
\end{pmatrix}
\begin{pmatrix}
X_{00} & 0 & 0 \\
X_{10} & X_{11} & 0 \\
X_{20} & X_{21} & X_{22}
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

---

## 2. 按行前代法的通用公式推导

### 2.1 出发点:L·X = I

设 X = L⁻¹,满足 L·X = I。展开矩阵乘法,结果矩阵第 i 行等于 L 的第 i 行乘以 X:

$$
(L \cdot X)_{i,:} = L_{i,:} \cdot X = I_{i,:} = e_i
$$

即 L 的第 i 行与整个 X 相乘,等于单位阵第 i 行 $e_i$(第 i 位为 1,其余为 0)。

### 2.2 利用 L 下三角特性展开

#### 矩阵乘法第 i 行的本质

L·X 的第 i 行 = L 的第 i 行 × 整个 X(行向量 × 矩阵):

$$
(L \cdot X)_{i,:} = L_{i,:} \cdot X = \sum_{j=0}^{N-1} L_{ij} \cdot X_{j,:}
$$

展开就是 L 第 i 行的每个元素,分别乘以 X 对应行,再求和:

$$
L_{i,:} \cdot X = L_{i0} \cdot X_{0,:} + L_{i1} \cdot X_{1,:} + \dots + L_{i,N-1} \cdot X_{N-1,:}
$$

#### 下三角特性的作用

L 下三角意味着 **$j > i$ 时 $L_{ij} = 0$**,且 **$L_{ii} = 1$**(单位对角)。所以求和项分三类:

| j 的范围 | $L_{ij}$ 的值 | 项 | 处理 |
|---------|--------------|-----|------|
| $j < i$ | 非零(下三角元素) | $L_{ij} \cdot X_{j,:}$ | 保留,进入求和 |
| $j = i$ | 1(对角) | $1 \cdot X_{i,:}$ | 单独提出 |
| $j > i$ | 0(上三角) | $0 \cdot X_{j,:} = 0$ | 消去 |

#### 展开过程

L 第 i 行与 X 相乘:

$$
L_{i,:} \cdot X = \sum_{j=0}^{N-1} L_{ij} \cdot X_{j,:}
$$

利用下三角特性($j>i$ 时 $L_{ij}=0$),并分离对角项 $L_{ii}=1$:

$$
e_i = \sum_{j=0}^{i-1} L_{ij} \cdot X_{j,:} \;+\; L_{ii} \cdot X_{i,:} \;+\; \sum_{j=i+1}^{N-1} 0 \cdot X_{j,:}
$$

$$
e_i = \sum_{j=0}^{i-1} L_{ij} \cdot X_{j,:} \;+\; 1 \cdot X_{i,:}
$$

#### N=3、i=2 的具体示例

$$
e_2 = \underbrace{L_{20} \cdot X_{0,:} + L_{21} \cdot X_{1,:}}_{j<i,\text{保留}} + \underbrace{L_{22} \cdot X_{2,:}}_{j=i,\text{对角=1}} + \underbrace{0 \cdot X_{\text{无}}}_{j>i,\text{N=3 无此项}}
$$

$$
e_2 = L_{20} \cdot X_{0,:} + L_{21} \cdot X_{1,:} + 1 \cdot X_{2,:}
$$

#### 关键点

1. **为什么求和上限从 N-1 变成 i-1**:因为 $j>i$ 时 $L_{ij}=0$,那些项直接消去,求和只需到 $i-1$
2. **为什么能解出 $X_{i,:}$**:对角项 $L_{ii}=1$ 使 $X_{i,:}$ 以系数 1 出现在等式中,可移项孤立出来
3. **串行依赖的根源**:$X_{i,:}$ 的表达式中含 $X_{0,:} \dots X_{i-1,:}$,必须先求出前面的行

### 2.3 解出 X 第 i 行

移项,解出 $X_{i,:}$:

$$
X_{i,:} = e_i - \sum_{j=0}^{i-1} L_{ij} \cdot X_{j,:}
$$

记 `inv[i]` = $X_{i,:}$,即得到按行前代法的通用公式:

$$
\text{inv}[i] = e_i - \sum_{j=0}^{i-1} L_{ij} \cdot \text{inv}[j], \quad i = 0, 1, \dots, N-1
$$

### 2.4 递推关系与边界条件

- **i=0**:求和上限为空,无累加项,直接得 `inv[0] = e₀`
- **i=k**:求和项 `inv[j]`(j=0..k-1)均已在之前步骤求出,可按序计算
- **串行依赖**:`inv[i]` 依赖 `inv[0..i-1]`,外层循环必须按序执行,无法并行

### 2.5 各项的数据类型(标量 vs 向量)

公式中各项的数据类型需明确区分:

$$
\text{inv}[i] = \underbrace{e_i}_{\text{向量}} - \sum_{j=0}^{i-1} \underbrace{L_{ij}}_{\text{标量}} \cdot \underbrace{\text{inv}[j]}_{\text{向量}}
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $e_i$ | 向量 | 单位阵第 i 行,如 $(0,0,1,0,\dots)$ |
| $L_{ij}$ | 标量 | L 矩阵第 i 行第 j 列的单个元素 |
| $\text{inv}[j]$ | 向量 | X 第 j 行 $(X_{j0}, X_{j1}, \dots, X_{j,N-1})$ |
| $L_{ij} \cdot \text{inv}[j]$ | 向量 | 标量乘向量,每个分量都乘 $L_{ij}$ |
| $\sum$ | 向量 | 多个向量逐分量相加 |
| $\text{inv}[i]$ | 向量 | 结果,X 第 i 行 |

整个运算是 **"标量 × 向量"的累加,再向量相减**:

- $L_{ij}$ 是单个标量,但需广播成"所有 lane 相同"的向量,才能与 $\text{inv}[j]$ 逐元素相乘
- $\text{inv}[j]$ 是已求出的 X 第 j 行,整体参与乘加
- $\sum$ 是向量逐分量累加,最终得到一个向量
- $e_i - \text{acc}$ 是向量相减,得到 X 第 i 行

对应代码:

| 公式操作 | 代码 | 说明 |
|---------|------|------|
| 取标量 $L_{ij}$ 并广播 | `Gather(lijBrc, li, idxReg)` | 从寄存器取单个元素,广播成向量 |
| 向量乘加 $\text{acc} += \text{inv}[j] \cdot L_{ij}$ | `MulAddDst(acc, invj, lijBrc)` | 逐 lane 乘加 |
| 向量相减 $\text{inv}[i] = e_i - \text{acc}$ | `Sub(invi, ei_i, acc)` | 逐 lane 相减 |

### 2.6 物理意义

通用公式可理解为"**单位阵第 i 行减去 L 第 i 行下三角部分与已求逆行的加权累加**":

- $e_i$:目标(单位阵第 i 行,即 X 第 i 行的对角线约束)
- $L_{ij} \cdot \text{inv}[j]$:L 第 i 行第 j 列元素,乘以已求出的 X 第 j 行(向量)
- $\sum$:把所有 j<i 的贡献累加起来
- 相减:消除下三角耦合,得到 X 第 i 行

### 2.7 用 N=3 验证公式来源

以 L·X = I 的第 2 行(i=2)为例,展开矩阵乘法:

$$
I_{2,:} = L_{20} \cdot X_{0,:} + L_{21} \cdot X_{1,:} + L_{22} \cdot X_{2,:}
$$

代入 $L_{22}=1$、$I_{2,:}=e_2$:

$$
e_2 = L_{20} \cdot X_{0,:} + L_{21} \cdot X_{1,:} + X_{2,:}
$$

移项:

$$
X_{2,:} = e_2 - L_{20} \cdot X_{0,:} - L_{21} \cdot X_{1,:}
$$

即:

$$
\text{inv}[2] = e_2 - \sum_{j=0}^{1} L_{2j} \cdot \text{inv}[j]
$$

与通用公式 $i=2$ 完全一致。

---

## 3. 逐步推导

### 3.1 i = 0:inv[0] = e₀

i=0 时无累加项(求和上限为空),直接取单位阵第 0 行:

$$
\text{inv}[0] = e_0 = (1, 0, 0)
$$

对应 X 第 0 行:

$$
(X_{00}, 0, 0) = (1, 0, 0) \implies X_{00} = 1
$$

---

### 3.2 i = 1:inv[1] = e₁ − L₁₀·inv[0]

i=1 时,j 从 0 到 0,只有一项累加。

**Step 1:初始化累加器**

$$
\text{acc} = (0, 0, 0)
$$

**Step 2:j=0 累加**

取出 L 第 1 行第 0 列元素 $L_{10} = a$,与 inv[0] 相乘:

$$
L_{10} \cdot \text{inv}[0] = a \cdot (1, 0, 0) = (a, 0, 0)
$$

累加到 acc:

$$
\text{acc} = (0, 0, 0) + (a, 0, 0) = (a, 0, 0)
$$

**Step 3:求 inv[1]**

$$
\text{inv}[1] = e_1 - \text{acc} = (0, 1, 0) - (a, 0, 0) = (-a, 1, 0)
$$

对应 X 第 1 行:

$$
(X_{10}, X_{11}, 0) = (-a, 1, 0) \implies X_{10} = -a,\ X_{11} = 1
$$

---

### 3.3 i = 2:inv[2] = e₂ − L₂₀·inv[0] − L₂₁·inv[1]

i=2 时,j 从 0 到 1,两项累加。

**Step 1:初始化累加器**

$$
\text{acc} = (0, 0, 0)
$$

**Step 2:j=0 累加**

取出 L 第 2 行第 0 列元素 $L_{20} = b$,与 inv[0] 相乘:

$$
L_{20} \cdot \text{inv}[0] = b \cdot (1, 0, 0) = (b, 0, 0)
$$

累加到 acc:

$$
\text{acc} = (0, 0, 0) + (b, 0, 0) = (b, 0, 0)
$$

**Step 3:j=1 累加**

取出 L 第 2 行第 1 列元素 $L_{21} = c$,与 inv[1] 相乘:

$$
L_{21} \cdot \text{inv}[1] = c \cdot (-a, 1, 0) = (-ac, c, 0)
$$

累加到 acc:

$$
\text{acc} = (b, 0, 0) + (-ac, c, 0) = (b - ac, c, 0)
$$

**Step 4:求 inv[2]**

$$
\text{inv}[2] = e_2 - \text{acc} = (0, 0, 1) - (b - ac, c, 0) = (-b + ac, -c, 1)
$$

对应 X 第 2 行:

$$
(X_{20}, X_{21}, X_{22}) = (-b + ac, -c, 1)
$$

---

## 4. 最终结果

$$
X = L^{-1} = \begin{pmatrix}
1 & 0 & 0 \\
-a & 1 & 0 \\
-b+ac & -c & 1
\end{pmatrix}
$$

---

## 5. 验证 L·X = I

矩阵乘法 $I_{ij} = \sum_k L_{ik} X_{kj}$,因 L 下三角,k 从 0 到 i:

| 结果元素 | 计算式 | 化简 | 结果 |
|---------|--------|------|------|
| $I_{00}$ | $L_{00} \cdot X_{00}$ | $1 \cdot 1$ | 1 ✓ |
| $I_{10}$ | $L_{10} \cdot X_{00} + L_{11} \cdot X_{10}$ | $a \cdot 1 + 1 \cdot (-a)$ | 0 ✓ |
| $I_{11}$ | $L_{10} \cdot 0 + L_{11} \cdot X_{11}$ | $a \cdot 0 + 1 \cdot 1$ | 1 ✓ |
| $I_{20}$ | $L_{20} \cdot X_{00} + L_{21} \cdot X_{10} + L_{22} \cdot X_{20}$ | $b \cdot 1 + c \cdot (-a) + 1 \cdot (-b+ac)$ | $b - ac - b + ac = 0$ ✓ |
| $I_{21}$ | $L_{20} \cdot 0 + L_{21} \cdot X_{11} + L_{22} \cdot X_{21}$ | $c \cdot 1 + 1 \cdot (-c)$ | 0 ✓ |
| $I_{22}$ | $L_{20} \cdot 0 + L_{21} \cdot 0 + L_{22} \cdot X_{22}$ | $1 \cdot 1$ | 1 ✓ |

验证通过,L·X = I。

---

## 6. 与代码的下标对应

以 i=2、j=1 这一步为例,展示代码变量与矩阵下标的对应关系:

| 代码 | 矩阵含义 | 下标 | 值(N=3 示例) |
|------|---------|------|-------------|
| `LoadAlign(li, attnUb + 2·chunkSize)` | L 第 2 行 $(L_{20}, L_{21}, L_{22})$ | i=2 | $(b, c, 1)$ |
| `Duplicate(idxReg, 1)` | 取第 j=1 个元素 | j=1 | 索引 = 1 |
| `Gather(lijBrc, li, idxReg)` | $L_{21}$ 广播成向量 | i=2, j=1 | $(c, c, c)$ |
| `LoadAlign(invj, invRes + 1·chunkSize)` | X 第 1 行 $(X_{10}, X_{11}, 0)$ | j=1 | $(-a, 1, 0)$ |
| `MulAddDst(acc, invj, lijBrc)` | acc += $L_{21} \cdot X_{1,:}$ | | acc += $(c \cdot (-a), c \cdot 1, 0) = (-ac, c, 0)$ |
| `Sub(invi, ei_i, acc)` | $X_{2,:} = e_2 - \text{acc}$ | i=2 | $(0,0,1) - (b-ac, c, 0) = (-b+ac, -c, 1)$ |

### 关键规律

- **外层循环 i**:遍历 X 的行,逐行求解
- **内层循环 j**:遍历 L 第 i 行的下三角元素 $L_{ij}$(j 从 0 到 i-1)
- **Gather 的作用**:从已加载的 L 第 i 行寄存器中取出单个标量 $L_{ij}$,广播成"所有 lane 相同"的向量
- **MulAddDst 的作用**:`acc += inv[j] * L[i,j]`,即把 $L_{ij} \cdot X_{j,:}$ 累加进 acc
- **Sub 的作用**:`inv[i] = e_i - acc`,得到 X 第 i 行
- **LocalMemBar**:保证 inv[i] 写入 UB 后对下一轮 i+1 的 LoadAlign 可见

---

## 7. VF 函数详细注释

以下是 `InverseAIVVFImpl` 与外层 `InverseAIVVF` 的完整源码,逐行配注释说明。

### 7.1 内层实现 `InverseAIVVFImpl`

```cpp
template <uint32_t N>
__simd_vf__ inline void InverseAIVVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb,
                                         __ubuf__ float *eiUb, uint32_t offset, uint32_t chunkSize)
{
    // ---------- 掩码准备 ----------
    // N<=64, 单个矢量寄存器(64 lane)可容纳一行。
    // UpdateMask 生成掩码: 前 N 个 lane 有效(=1), 其余关闭(=0)。
    // 后续所有 Load/Store/MulAddDst 带 maskN, 只操作前 N 个元素, 避免越界污染。
    uint32_t maskLen = N;
    MaskReg maskN = UpdateMask<float>(maskLen);

    // ---------- inv[0] = e_0 ----------
    // 从 eiUb 加载单位阵第 0 行(行 0 = e_0, 首位 1 其余 0)。
    RegTensor<float> inv0;
    LoadAlign(inv0, eiUb);                                          // inv0 ← eiUb[0..chunkSize-1]
    // 写入 invResUb 第 0 行(带 offset 列偏移, 即对角块所在子块起始)。
    StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset, inv0, maskN);
    // store→load 屏障: 保证 inv[0] 写入 UB 后, 后续 LoadAlign 能读到。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // ---------- 前代法主循环 ----------
    // idxReg: Gather 索引寄存器, 每轮设为 j, 用于从 li 取第 j 个元素。
    // acc:    累加寄存器, 存放 sum_{j<i} L[i,j] * inv[j] (向量)。
    RegTensor<uint32_t> idxReg;
    RegTensor<float> acc;
    for (uint16_t i = 1; i < static_cast<uint16_t>(N); ++i) {
        // acc 清零, 准备累加当前行 i 的内积 sum_{j<i} L[i,j] * inv[j]
        Duplicate(acc, static_cast<float>(0.0));

        // 加载 L 第 i 行到寄存器 li (整行 chunkSize 个元素, 但 maskN 屏蔽后只前 N 有效)。
        // 地址 = attnUb + offset(子块列偏移) + i*chunkSize(行偏移)。
        RegTensor<float> li;
        LoadAlign(li, attnUb + offset + i * chunkSize);

        // ---------- 内层累加循环 ----------
        // 对 j=0..i-1, 逐个累加 L[i,j] * inv[j]:
        //   1. Duplicate(idxReg, j)        — 索引寄存器置为 j
        //   2. Gather(lijBrc, li, idxReg) — 从 li 取第 j 个标量 L[i,j], 广播成向量(所有 lane = L[i,j])
        //   3. LoadAlign(invj, ...)        — 从 invResUb 加载已求出的 inv[j] (X 第 j 行)
        //   4. MulAddDst(acc, invj, lijBrc)— acc += invj * L[i,j] (逐 lane 乘加)
        for (uint16_t j = 0; j < i; ++j) {
            Duplicate(idxReg, j);
            RegTensor<float> lijBrc;
            Gather(lijBrc, li, idxReg);
            RegTensor<float> invj;
            LoadAlign(invj, invResUb + offset + j * chunkSize);
            MulAddDst(acc, invj, lijBrc, maskN);
        }

        // ---------- inv[i] = e_i - acc ----------
        // 加载单位阵第 i 行 e_i (对角位 1, 其余 0)。
        RegTensor<float> ei_i;
        LoadAlign(ei_i, eiUb + i * chunkSize);
        // 向量相减: inv[i] = e_i - acc (消除下三角耦合, 得到 X 第 i 行)。
        RegTensor<float> invi;
        Sub(invi, ei_i, acc, maskN);
        // 写回 invResUb 第 i 行(带 offset 列偏移)。
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize, invi, maskN);
        // store→load 屏障: 保证 inv[i] 写入对下一轮 i+1 的 LoadAlign(invj) 可见。
        // (下一轮 i+1 的内层 j 循环会读取 inv[i], 故必须同步。)
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}
```

### 7.2 外层包装 `InverseAIVVF`

```cpp
template <uint32_t N>
__aicore__ inline void InverseAIVVF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                    const LocalTensor<float> &eiUb, uint32_t offset, uint32_t chunkSize)
{
    // 将 LocalTensor 转为 __ubuf__ 原始指针, 供 __simd_vf__ 内层使用。
    // (编译器要求 __simd_vf__ 函数的指针参数必须显式标注 __ubuf__ 地址空间。)
    __ubuf__ float *attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    __ubuf__ float *inv  = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    __ubuf__ float *ei   = reinterpret_cast<__ubuf__ float *>(eiUb.GetPhyAddr());
    // 转发到内层向量函数。
    InverseAIVVFImpl<N>(attn, inv, ei, offset, chunkSize);
}
```

### 7.3 关键 API 速查

| API | 作用 | 对应公式 |
|-----|------|---------|
| `UpdateMask<float>(N)` | 生成前 N lane 有效的掩码 | 控制每行仅前 N 元素参与运算 |
| `LoadAlign(reg, ptr)` | 从 UB 内存整行加载到寄存器 | 取 L 第 i 行 / inv[j] / e_i |
| `Duplicate(reg, val)` | 标量填充整个寄存器 | acc 清零; idxReg 置索引 j |
| `Gather(dst, src, idx)` | 从 src 寄存器取第 idx 个元素,广播成向量 | 取标量 $L_{ij}$ 并广播 |
| `MulAddDst(acc, a, b, mask)` | `acc += a * b`(逐 lane) | acc += inv[j] * L[i,j] |
| `Sub(dst, a, b, mask)` | `dst = a - b`(逐 lane) | inv[i] = e_i - acc |
| `StoreAlign(ptr, reg, mask)` | 寄存器整行写回 UB | 写入 inv[i] |
| `LocalMemBar<STORE, LOAD>` | store→load 屏障 | 保证写入对后续读取可见 |
