# MX 量化算法原理及 OCP / cuBLAS 实现说明

本文档分两部分：

1. **与具体代码无关的量化算法原理**
2. **本 kernel 中 OCP / cuBLAS 的实现与区别**

---

# 第一部分：量化算法原理

## 1. 什么是低比特量化

量化的目标是把高精度数据（如 FP16/BF16）用更低比特表示，从而降低存储、带宽和计算开销。

最简单的低比特格式是 FP8/FP4，但它们能表示的数据范围较小：

- FP8 比 FP16/BF16 的指数和尾数位数少；
- FP4 能表示的范围和精度更小。

如果直接把原始数据转成 FP8/FP4，很多数值会因为超出范围或精度不足而严重失真。

---

## 2. Block Quantization：分块量化

为了在有限位宽下尽量保留动态范围，通常不是对整张张量只用一个 scale，而是：

> 把一个轴上的数据分成很多个小 block，每个 block 使用一个自己的 scale。

例如：

```text
原始一行数据：
[ x0 x1 x2 ... x31 | x32 ... x63 | ... ]

block size = 32
每个 block 单独计算一个 scale
```

这种思路称为：

```text
Block Quantization / Per-block Quantization
```

---

## 3. Shared Scale 的作用

每个 block 内部：

```text
scale = 根据 block 内数据的绝对值最大值计算出来的一个数
y_i  = cast(x_i / scale)
```

由于同一 block 内共享同一个 scale，所以需要记录：

```text
量化后的 y_i
block 的 scale
```

推理/反量化时：

```text
x_approx_i = y_i * scale
```

也就是说：

```text
量化过程损失的是尾数/低位精度，
scale 保存了 block 的“整体动态范围”。
```

---

## 4. 为什么 scale 通常取 2 的幂

如果 scale 可以取任意浮点数，反量化需要做一次浮点乘法，精度和实现都比较直接。

但很多硬件/格式（如 MX）希望 scale 本身也尽量简单，甚至用 E8M0 这种格式保存。

E8M0 的特点：

```text
只有 8 bit 指数，没有尾数
表示的值可以理解为 power-of-two 形式
```

所以算法通常会计算：

```text
scale = 2^e
```

这样：

- scale 存储简单；
- 除 scale 变成乘 `2^{-e}`；
- 反量化变成乘 `2^e`；
- 适合硬件实现。

---

## 5. 量化核心公式

给定一个 block：

```text
x_i, i = 1..N
```

其中 N 是 block size，通常是 32。

量化目标输出：

```text
scale = 2^e
y_i  = cast(x_i * 2^{-e})
```

关键问题变成：

> 如何根据 block 内的数据选择共享指数 e？

常见做法是让 `2^e` 尽量覆盖 block 内的最大绝对值，同时不要过大导致精度浪费。

---

## 6. 从 block max 推导 scale

### 第一步：求绝对最大值

```text
amax = max_i |x_i|
```

### 第二步：根据目标类型确定 scale

目标低比特类型能表示的最大有限值记作：

```text
dtype_max
```

一种通用思路是：

```text
s = amax / dtype_max
```

然后选择最合适的 2 的幂作为 scale：

```text
scale = 2^e ≈ s
```

由于 scale 是 2 的幂，通常有两种取法：

```text
取 floor：保证 scale ≤ s，尽量利用低比特类型范围，但可能溢出
取 ceil：保证 scale ≥ s，尽量不溢出，但可能损失一些精度
```

不同规范会选择不同的取法。

---

## 7. 两种常见计算 e 的方式

### 7.1 基于指数直接减

如果输入本身就是 FP16/BF16，可以直接利用其指数位。

定义：

```text
emax = floor(log2(dtype_max))
```

例如：

```text
FP8_E4M3FN 的 max = 448
log2(448) ≈ 8.807
emax = 8
```

则：

```text
e = floor(log2(amax)) - emax
```

这种方式的优点是：

- 不需要真实除法；
- 直接用 BF16/FP16 的指数域做移位；
- 实现快，适合低精度高吞吐场景。

代表实现：

```text
OCP
```

### 7.2 先除以真实 dtype_max，再做 ceil(log2)

另一种方式是：

```text
s = amax / dtype_max
e = ceil(log2(s))
```

由于：

```text
dtype_max 是真实的有限最大值
```

所以这种归一化更贴近参考实现的数值行为。

代表实现：

```text
cuBLAS / NV FP8 scale
```

---

## 8. floor 与 ceil 的差别

对于 `amax`，如果：

```text
2^{k} ≤ amax < 2^{k+1}
```

那么：

```text
floor(log2(amax)) = k
ceil(log2(amax)) = k+1（当 amax 不是 2 的幂）
```

当和 `dtype_max` 结合时，OCP 与 cuBLAS 的差异主要体现在边界位置：

- OCP 用 `floor(log2(dtype_max))` 作为 emax；
- cuBLAS 用真实的 `dtype_max` 做分母。

因此两者在某些边界 block 上会得到不同的 `e`。

---

## 9. 特殊值处理

任何实际量化算法都必须处理：

```text
0
Inf
NaN
```

常见约定：

- block 全 0：scale 输出 0；
- block 含 NaN/Inf：scale 输出特殊 NaN；
- 量化后元素中的 NaN 往往被置为 0。

---

## 10. 量化的精度/性能权衡

| 方向 | 说明 |
|---|---|
| block 越大 | scale 数量少，存储小，但共享 scale 的数据越多，精度损失越大 |
| block 越小 | scale 数量多，动态范围更准，但 scale 存储和计算开销越大 |
| floor 类 | 偏向更大利用率，但可能溢出 |
| ceil 类 | 偏向避免溢出，但可能放大尾数损失 |
| 指数直接减 | 快，近似 |
| 先除 dtype_max | 更贴近参考实现，慢一些 |

---

# 第二部分：本 kernel 中 OCP / cuBLAS 的实现与区别

## 1. 背景

本 kernel 是：

```text
CLA gate / SwiGLU 融合 + 双轴 MX 量化
```

SwiGLU 结果在 UB 中按 `64 × 256` 的 tile 处理：

- `-1 轴量化`：每行内每 32 列一个 block；
- `-2 轴量化`：每列上每 32 行一个 block。

两个 scale 算法对应：

```text
scaleAlg = 0：ComputeScaleOcp
scaleAlg = 1：ComputeScaleCuBLAS
```

---

## 2. OCP 实现

### 公式

```text
amax = max_i |x_i|
e    = floor(log2(amax)) - emax
scale = 2^e
y_i  = cast(x_i * 2^{-e})
```

其中：

```text
emax = 目标类型最大正规数的指数
```

### 代码路径

- 输入 FP16 会先转成 BF16，因为 BF16 指数位更易提取；
- 用 interleaved load 把一行拆成偶/奇两部分；
- 对偶/奇 abs 求 max；
- 通过 `ReduceDataBlock<MAX>` 得到每 32 个元素一个 max；
- 从 max 中提取 BF16 指数；
- 减去目标类型 emax；
- 得到 scale 和 reciprocal。

### 支持类型

```text
FP8 和 FP4
```

### 特点

- 快；
- 使用输入 BF16/FP16 的指数结构；
- 不显式做 `amax / dtype_max`；
- 更贴近 OCP MX 的通用语义。

---

## 3. cuBLAS 实现

### 公式

```text
amax = max_i |x_i|
M    = dtype_max
s    = amax / M
e    = ceil(log2(s))
scale = 2^e
y_i  = cast(x_i * 2^{-e})
```

### 代码路径

- 先求 block 内 amax；
- 乘以预计算 `invDtypeMax = 1 / dtype_max`；
- 在 FP32 中提取 exponent 和 mantissa；
- 根据 mantissa 判断是否需要指数 +1，实现 `ceil(log2(s))`；
- 特殊处理 0 / NaN / Inf；
- 生成 E8M0 scale 和 reciprocal；
- 用 reciprocal 乘 x 再 cast 成 FP8。

### 支持类型

```text
只支持 FP8
```

### 特点

- 更接近 cuBLAS / NV FP8 的参考数值行为；
- 使用真实 `dtype_max` 做归一化；
- 计算比 OCP 多，但 scale 边界行为更明确；
- FP4 不走此路径。

---

## 4. OCP 与 cuBLAS 对比

| 项目 | OCP | cuBLAS |
|---|---|---|
| 是否除以真实 dtype_max | 否 | 是 |
| 核心公式 | `floor(log2(amax)) - emax` | `ceil(log2(amax / dtype_max))` |
| 运算域 | BF16/FP16 指数域 | FP32 指数/尾数域 |
| 是否做尾数进位 | 基本不做 | 做 |
| FP8 | 支持 | 支持 |
| FP4 | 支持 | 不支持 |
| 性能 | 更快 | 相对更慢 |
| 数值边界 | 偏向 OCP 兼容 | 偏向参考实现 |

---

## 5. 适用场景建议

### 使用 OCP

- 需要 OCP MX 兼容性；
- 输出可能是 FP4；
- 对性能更敏感；
- 对边界 scale 数值不是严格需要匹配 cuBLAS。

### 使用 cuBLAS

- 输出是 FP8；
- 需要对齐 cuBLAS / NV 风格参考；
- 对 block 是否溢出、scale 边界更敏感；
- 可以接受稍多的计算。

---

## 6. 一句话总结

```text
OCP：
    快、简单，从输入指数直接算 scale，支持 FP4/FP8。

cuBLAS：
    更精细，除以真实 dtype_max 再做 ceil(log2)，
    只支持 FP8，适合需要对齐 FP8 参考实现的场景。
```
