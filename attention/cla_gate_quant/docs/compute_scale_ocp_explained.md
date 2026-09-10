# ComputeScaleOcp 详解：伪代码、原理与知识点

本文用尽量通俗的方式解释 `ComputeScaleOcp`：

- 它是 `scaleAlg=0` 时的 OCP 量化核心函数；
- 它计算 -1 轴和 -2 轴的 scale / reciprocal；
- 它不生成 y1/y2，y1/y2 由后面的 `ComputeY1ToFP8/FP4`、`ComputeY2ToFP8/FP4` 完成。

---

## 一、这个函数在做什么

`ComputeScaleOcp` 处理一个：

```text
32 × 256
```

的输入块。

它输出：

```text
-1 轴：
    每行 256 列，每 32 列一个 scale
    所以每行 8 个 scale
    以及对应的 reciprocal

-2 轴：
    当前 32 行作为一个 slot
    每一列计算一个 scale
    以及对应的 reciprocal
```

注意：OCP 路径不像 cuBLAS 一次处理 64 行两个 slot。

OCP 路径一次只处理 32 行一个 slot，后续由 `Process()` 把两个 32 行 slot 交织成 scale2 输出。

---

## 二、OCP 的核心原理

OCP 的公式是：

```text
amax = max_i |x_i|
e    = floor(log2(amax)) - emax
scale = 2^e
```

其中：

```text
emax = 目标类型的最大指数
```

例如：

```text
FP8_E4M3FN：emax = 8
FP8_E5M2：emax = 15
FP4_E2M1：emax = 2
FP4_E1M2：emax = 0
```

它和 cuBLAS 最大的区别是：

```text
OCP 不计算 amax / dtype_max
OCP 直接从 BF16/FP16 的指数位提取信息
```

---

## 三、为什么 OCP 可以“只看指数”

输入是 FP16 或 BF16。

这两种格式本身已经用指数表示了一个数的大致范围。

例如 BF16：

```text
x = 1.xxx × 2^{指数}
```

所以我们不需要精确求 `log2(amax)`，只需要看 BF16 的指数位，就能得到：

```text
floor(log2(amax))
```

这就是 OCP 能做得更快的原因：

```text
不需要 FP32 除法
不需要 FP32 尾数舍入
只需要指数域的移位和比较
```

---

## 四、前置知识点

### 1. BF16 / FP16 指数 mask

BF16 的指数位：

```text
BF16 = 1 bit 符号 + 8 bit 指数 + 7 bit 尾数
```

提取指数用：

```text
EXP_MASK_BF16 = 0x7f80
```

因为 BF16 的指数位在 bit14~bit7。

FP16 的指数位：

```text
FP16 = 1 bit 符号 + 5 bit 指数 + 10 bit 尾数
```

提取指数用：

```text
EXP_MASK_FP16 = 0x7c00
```

### 2. 为什么 FP16 要转 BF16

OCP 希望统一在 BF16 指数域处理。

FP16 指数范围比 BF16 小，但转成 BF16 后指数范围更大，而且 BF16 指数位提取更简单。

所以代码中：

```text
FP16 -> 先检测 Inf/NaN
     -> Cast 成 BF16
     -> 再提取 BF16 指数
```

### 3. yMaxExp 是什么

`yMaxExp` 是目标 FP8/FP4 类型的最大指数。

OCP 用：

```text
expMax - yMaxExp
```

得到最终的 scale 指数。

如果 block 的最大指数比 `yMaxExp` 还小，说明数值太小，按 subnormal/零处理。

### 4. reciprocal 还是 2^{-e}

OCP 的 scale 仍然是：

```text
scale = 2^e
```

所以 reciprocal：

```text
reciprocal = 2^{-e}
```

依然可以用指数域减法得到。

---

## 五、伪代码实现

下面是 `ComputeScaleOcp` 的完整伪代码。

```text
函数 ComputeScaleOcp(
    输入:
        dataLen   : 每行元素数，通常 256
        blockCount: 行数，通常 32
        xAddr     : 输入数据
        mxScale1Addr          : -1 轴 E8M0 scale 输出
        mxScale1ReciprocalAddr: -1 轴 reciprocal 输出
        mxScale2Addr          : -2 轴 E8M0 scale 输出
        mxScale2ReciprocalAddr: -2 轴 reciprocal 输出
    )

    # 常量
    BLOCK_SIZE = 32

    # -2 轴按列累积的最大指数，分成偶数列/奇数列
    expMax1Dim2 = 0   # 偶数列累积 max exponent
    expMax2Dim2 = 0   # 奇数列累积 max exponent

    # 对当前 32 行逐行处理
    for i in 0 .. blockCount-1:

        # 读取一行 256 个元素，按偶/奇拆分
        x0, x1 = deinterleave_load(xAddr)
        xAddr += dataLen

        # --------------------------------------------------
        # 提取指数
        # --------------------------------------------------
        if 输入是 FP16:
            # 先检测 Inf/NaN
            x0IsInfNan = (x0_exp_fp16 == FP16_EXP_MASK)
            x1IsInfNan = (x1_exp_fp16 == FP16_EXP_MASK)

            # FP16 转 BF16
            x0BF16 = cast_half_to_bf16(x0)
            x1BF16 = cast_half_to_bf16(x1)

            # 提取 BF16 指数
            x0Exp = x0BF16 & EXP_MASK_BF16
            x1Exp = x1BF16 & EXP_MASK_BF16

            # 如果是 Inf/NaN，把指数设为全 1，便于后续按异常处理
            x0Exp = select(if Inf/NaN, EXP_MASK_BF16, x0Exp)
            x1Exp = select(if Inf/NaN, EXP_MASK_BF16, x1Exp)
        else:
            # BF16 输入直接提取指数
            x0Exp = x0 & EXP_MASK_BF16
            x1Exp = x1 & EXP_MASK_BF16

        # --------------------------------------------------
        # -1 轴：每 32 列一个 max exponent
        # --------------------------------------------------
        # x0Exp/x1Exp 是相邻两个元素的指数
        expMaxDim1 = max(x0Exp, x1Exp)

        # reduce 成每 32 列一个最大指数
        expMaxDim1 = reduce_block_max(expMaxDim1)

        # --------------------------------------------------
        # -2 轴：逐列累积 32 行的 max exponent
        # --------------------------------------------------
        expMax1Dim2 = max(expMax1Dim2, x0Exp)   # 偶数列
        expMax2Dim2 = max(expMax2Dim2, x1Exp)   # 奇数列

        # --------------------------------------------------
        # 生成这一行的 -1 轴 scale
        # --------------------------------------------------
        # Inf/NaN 检测
        isInfNan = (expMaxDim1 == EXP_MASK_BF16)
        # 0 检测
        isZero   = (expMaxDim1 == 0)
        # 指数太小，小于目标类型最大指数时，按 subnormal 处理
        isTooSmall = (expMaxDim1 <= yMaxExp)

        # 如果太小，先把指数抬到 yMaxExp
        expMaxDim1 = select(isTooSmall, yMaxExp, expMaxDim1)

        # 计算 scale 指数：
        # scaleExp = expMaxDim1 - yMaxExp
        scaleExp = expMaxDim1 - yMaxExp

        # 右移 7 位，得到 E8M0 scale 对应的低 8 bit
        scale1 = scaleExp >> 7

        # Inf/NaN -> E8M0 NaN
        scale1 = select(isInfNan, E8M0_NAN, scale1)
        # 0 -> scale 0
        scale1 = select(isZero, 0, scale1)

        # 写出这一行的 8 个 scale
        store_align(mxScale1Addr, scale1)
        mxScale1Addr += 8

        # --------------------------------------------------
        # 生成这一行的 -1 轴 reciprocal
        # --------------------------------------------------
        # reciprocal = 2^{-e}
        # 用 bias - scaleExp 得到 reciprocal 指数
        recip1 = biasE8M0 - scaleExp

        # Inf/NaN -> NaN
        recip1 = select(isInfNan, BF16_NAN, recip1)
        # 0 -> 0
        recip1 = select(isZero, 0, recip1)
        # subnormal 场景特殊处理
        recip1 = select(isTooSmall, specialExp, recip1)

        store_align(mxScale1ReciprocalAddr, recip1)
        mxScale1ReciprocalAddr += recip_row_len

    # ------------------------------------------------------------
    # 所有 32 行处理完后，计算 -2 轴 scale
    # ------------------------------------------------------------

    # ---------- 偶数列部分 ----------
    # 以下流程和 -1 轴类似，只是数据来自 expMax1Dim2

    isInfNan   = (expMax1Dim2 == EXP_MASK_BF16)
    isZero     = (expMax1Dim2 == 0)
    isTooSmall = (expMax1Dim2 <= yMaxExp)

    expMax = select(isTooSmall, yMaxExp, expMax1Dim2)
    scaleExp = expMax - yMaxExp
    scale2Zero = scaleExp >> 7

    scale2Zero = select(isInfNan, E8M0_NAN, scale2Zero)
    scale2Zero = select(isZero, 0, scale2Zero)

    recip2Zero = biasE8M0 - scaleExp
    recip2Zero = select(isInfNan, BF16_NAN, recip2Zero)
    recip2Zero = select(isZero, 0, recip2Zero)
    recip2Zero = select(isTooSmall, specialExp, recip2Zero)

    # ---------- 奇数列部分 ----------
    # 处理方式和偶数列完全一样，只是输入是 expMax2Dim2
    ...

    # ---------- 交织输出 ----------
    # -2 轴输出需要把偶数列和奇数列交织成正常列顺序
    interleave_store(mxScale2Addr,
                     scale2Zero,
                     scale2One)

    interleave_store(mxScale2ReciprocalAddr,
                     recip2Zero,
                     recip2One)

结束函数
```

---

## 六、OCP 和 cuBLAS 的核心差异

| 对比项 | OCP | cuBLAS |
|---|---|---|
| 公式 | `floor(log2(amax)) - emax` | `ceil(log2(amax / dtype_max))` |
| 是否除法 | 否 | 是 |
| 运算域 | BF16/FP16 指数域 | FP32 指数/尾数域 |
| 尾数处理 | 基本不看尾数 | 看尾数决定是否进位 |
| FP8 | 支持 | 支持 |
| FP4 | 支持 | 不支持 |
| 速度 | 更快 | 更慢 |
| 边界行为 | 偏向 OCP 标准 | 偏向参考实现 |

---

## 七、OCP 路径整体流程

```text
32x256 输入块
    |
    v
逐行扫描
    |
    |-- 每行 256 列
    |     |
    |     |-- FP16 -> 检测 Inf/NaN -> 转 BF16
    |     |-- 提取 BF16 exponent
    |     |-- -1 轴：每 32 列 reduce 出 max exponent
    |     |-- -2 轴：逐列累积 32 行 max exponent
    |
    v
逐行生成 -1 轴 scale / reciprocal
    |
    v
扫描完 32 行后
    |
    |-- 用 expMax1Dim2 生成偶数列 -2 轴 scale
    |-- 用 expMax2Dim2 生成奇数列 -2 轴 scale
    |-- 交织输出 scale2 / reciprocal2
```

---

## 八、为什么 OCP 输出 y1/y2 需要后续函数

`ComputeScaleOcp` 只负责 scale 和 reciprocal。

真正的量化输出：

```text
y1 = cast(x * reciprocal1)
y2 = cast(x * reciprocal2)
```

是在后面由：

```text
ComputeY1ToFP8 / ComputeY1ToFP4
ComputeY2ToFP8 / ComputeY2ToFP4
```

完成的。

这样分工更清晰：

```text
ComputeScaleOcp       -> 生成 scale / reciprocal
ComputeY1ToFP8/FP4    -> 生成 -1 轴量化结果
ComputeY2ToFP8/FP4    -> 生成 -2 轴量化结果
```

---

## 九、容易混淆的点

### 1. OCP 不是不做特殊值处理

OCP 同样处理：

- 0；
- Inf / NaN；
- 指数太小导致的 subnormal。

只是实现方式是在 BF16 exponent 域直接比较。

### 2. OCP 的 reciprocal 仍然是 2 的幂

OCP 的 scale 仍然是 `2^e`，所以 reciprocal 是 `2^{-e}`。

### 3. OCP 一次处理 32 行

OCP 函数本身的 `blockCount=32`。

如果整体有 64 行，会调用两次 `ComputeScaleOcp`，然后在 `Process()` 中把两个 32 行结果交织成 scale2。

---

## 十、一句话总结

```text
ComputeScaleOcp 是 OCP 风格的 scale 计算函数：

它利用 BF16/FP16 的指数位直接得到
block 的最大指数，
然后通过：
    e = max_exponent - emax
生成 scale = 2^e 和 reciprocal = 2^{-e}。

它比 cuBLAS 更快，
因为它不需要 FP32 除法，
也不需要看尾数做 ceil(log2) 舍入。
```
