# ComputeScaleCuBLAS 详解：伪代码、原理与知识点

本文用尽量通俗的方式解释 `ComputeScaleCuBLAS` 这个 kernel 函数：

- 它做什么；
- 为什么这样做；
- 背后的浮点数知识点；
- 一版带详细注释的伪代码；
- 完整流程。

---

## 一、这个函数在做什么

`ComputeScaleCuBLAS` 是 `scaleAlg=1` 时的量化核心函数。

它处理一个 `64 × 256` 的 SwiGLU 结果 tile，完成：

```text
1. 计算 -1 轴 scale：
   每行 256 个元素，每 32 个元素共享一个 scale
   所以每行有 256/32 = 8 个 scale

2. 计算 -1 轴 reciprocal：
   reciprocal = 1 / scale

3. 使用 reciprocal 生成 -1 轴量化结果 y1：
   y1 = cast(x * reciprocal)

4. 计算 -2 轴 scale：
   每列上每 32 行共享一个 scale
   64 行会形成两个 32 行 slot

5. 计算 -2 轴 reciprocal，供后续 y2 使用
```

简单说：

```text
输入：原始高精度数据 x
输出：
  - E8M0 scale
  - 16-bit reciprocal
  - 量化后的 y1 / y2
```

---

## 二、背后的核心原理

### 1. Block Quantization

量化不是对每个元素单独给 scale，而是对一小块数据共用一个 scale。

本函数中：

```text
-1 轴 block size = 32 列
-2 轴 block size = 32 行
```

### 2. scale 取 2 的幂

量化采用：

```text
scale = 2^e
```

所以除法可以变成：

```text
x / scale = x * 2^{-e}
```

也就是乘以 reciprocal。

### 3. E8M0 scale

E8M0 是一种只有 8-bit exponent、没有尾数的格式。

它适合保存：

```text
2^e
```

这种值。

### 4. cuBLAS scale 算法

cuBLAS 风格计算 scale 的公式：

```text
amax = max |x|
M    = 目标 FP8 类型的最大有限值
s    = amax / M
e    = ceil(log2(s))
scale = 2^e
```

例如：

```text
FP8_E4M3FN：M = 448
FP8_E5M2：M = 57344
```

代码不是真的调用 `log2`，而是利用 FP32 的 bit 表示快速完成。

---

## 三、前置知识点

### 1. FP32 bit 布局

FP32 用 32 bit 表示一个数：

```text
| 1 bit 符号 | 8 bit 指数 | 23 bit 尾数 |
```

指数是 biased 的：

```text
存储指数 = 真实指数 + 127
真实指数 = 存储指数 - 127
```

例如：

```text
1.0 的存储指数 = 127
2.0 的存储指数 = 128
0.5 的存储指数 = 126
```

### 2. 从 FP32 中取指数和尾数

```text
exp = bits >> 23
man = bits & 0x007fffff
```

- `exp` 是 biased exponent；
- `man` 是 23 bit 尾数。

### 3. Normal 与 Subnormal

普通 normal 数：

```text
value = 1.man × 2^{exp-127}
```

最小 normal 是 `2^{-126}`。

比它还小的非零数叫 subnormal：

```text
value = 0.man × 2^{-126}
```

subnormal 的 `exp = 0`。

### 4. 为什么尾数非 0 时指数要加 1

对于 normal 数：

```text
value = 1.man × 2^{exp-127}
```

如果 `man != 0`，说明：

```text
2^{exp-127} < value < 2^{exp-126}
```

所以：

```text
floor(log2(value)) = exp-127
ceil(log2(value))  = exp-126
```

因此 `ceil(log2)` 比 `floor(log2)` 大 1，需要把 biased exponent 加 1。

### 5. Reciprocal 为什么能用指数减法得到

因为 scale 是 `2^e`，reciprocal 是 `2^{-e}`。

BF16/FP32 的指数都是 biased：

```text
scale_field      = bias + e
reciprocal_field = bias - e
```

因此：

```text
reciprocal_field = 2*bias - scale_field
```

代码里的：

```text
scaleBiasReg = 0x7F00
```

就是 `2*bias` 放到 BF16 exponent field 后的编码。

---

## 四、伪代码实现

下面是一版偏“人类可读”的伪代码，重点展示逻辑，不是完整 kernel 指令。

```text
函数 ComputeScaleCuBLAS(
    输入:
        dataLen   : 每行元素数，通常 256
        blockCount: 处理行数，通常 64
        xAddr     : UB 中 64x256 的输入数据
        y1Addr    : 输出的 -1 轴量化结果
        mxScale1Addr          : 输出的 -1 轴 E8M0 scale
        mxScale1ReciprocalAddr: 输出的 -1 轴 reciprocal
        mxScale2Addr          : 输出的 -2 轴 E8M0 scale
        mxScale2ReciprocalAddr: 输出的 -2 轴 reciprocal
    )

    # ------------------------------------------------------------
    # 常量
    # ------------------------------------------------------------
    BLOCK_SIZE     = 32          # -1 轴 block 大小
    FP32_MANTISSA_MASK = 0x007fffff
    FP32_INF_PATTERN    = 0x7f800000
    FP32_NAN_MARK       = 0x000000ff   # E8M0 NaN
    ZERO                = 0
    HALF_FOR_MAN        = 0x00400000   # 2^22
    EXP_254             = 0x000000fe
    SHR_FP32_EXP        = 23
    SHL_BF16_EXP        = 7

    # amax / dtype_max 的预计算倒数
    invDtypeMax = 1 / dtype_max
    # 例如 E4M3: invDtypeMax = 1/448
    #      E5M2: invDtypeMax = 1/57344

    # ------------------------------------------------------------
    # 阶段 1：遍历行，收集 -1 轴和 -2 轴需要的 max
    # ------------------------------------------------------------

    # 临时 buffer 用来存 -1 轴每个 block 的 max
    tempMaxAddr = mxScale2ReciprocalAddr   # 临时借用

    # -2 轴需要两个 32 行 slot
    slot0Rows = min(blockCount, 32)

    # 清零 -2 轴累加器
    scale2Slot0Part0 = 0   # slot0 偶数列 max
    scale2Slot0Part1 = 0   # slot0 奇数列 max
    scale2Slot1Part0 = 0   # slot1 偶数列 max
    scale2Slot1Part1 = 0   # slot1 奇数列 max

    # 第一个循环：处理 slot0 的行，通常 row 0~31
    for i in 0 .. slot0Rows-1:

        # 读取一行 256 个元素，按偶/奇拆成 x0/x1
        x0, x1 = deinterleave_load(xAddr)
        xAddr += dataLen

        # 取绝对值
        x0Abs = abs(x0)
        x1Abs = abs(x1)

        # ---------- 计算这一行的 -1 轴 block max ----------
        # x0/x1 对应位求 max，得到相邻两个元素的 max
        pairMax = max(x0Abs, x1Abs)

        # 把每 16 个 pair max reduce 成 1 个
        # 最终得到 dataLen/32 = 8 个 -1 轴 max
        rowMax = reduce_block_max(pairMax, block_size=32)

        # 把这 8 个 max 紧凑存到 tempMaxAddr
        unalign_store(tempMaxAddr, rowMax, count=8)
        tempMaxAddr += 8

        # ---------- 累加 slot0 的 -2 轴逐列 max ----------
        scale2Slot0Part0 = max(scale2Slot0Part0, x0Abs)
        scale2Slot0Part1 = max(scale2Slot0Part1, x1Abs)

    # 第二个循环：处理 slot1 的行，通常 row 32~63
    for i in slot0Rows .. blockCount-1:

        x0, x1 = deinterleave_load(xAddr)
        xAddr += dataLen

        x0Abs = abs(x0)
        x1Abs = abs(x1)

        # -1 轴处理同上
        pairMax = max(x0Abs, x1Abs)
        rowMax  = reduce_block_max(pairMax, block_size=32)
        unalign_store(tempMaxAddr, rowMax, count=8)
        tempMaxAddr += 8

        # 累加 slot1 的 -2 轴逐列 max
        scale2Slot1Part0 = max(scale2Slot1Part0, x0Abs)
        scale2Slot1Part1 = max(scale2Slot1Part1, x1Abs)

    # 结束非对齐 store，flush 残留数据
    store_unalign_post(tempMaxAddr)

    # ------------------------------------------------------------
    # 阶段 2：把缓存的 -1 轴 max 转成 scale / reciprocal
    #         并生成 y1
    # ------------------------------------------------------------

    # 总共缓存的 -1 轴 max 数量
    totalMaxCount = blockCount * (dataLen / BLOCK_SIZE)

    # 一批处理 64 个 max
    batchCount = ceil(totalMaxCount / 64)

    for j in 0 .. batchCount-1:

        # 读取 64 个 16-bit max
        max16 = load_unaligned_or_unpack(tempMaxAddr, count=64)
        tempMaxAddr += 64

        # 转成 FP32，方便后续位运算
        max32 = cast_to_fp32(max16)

        # 判断是否为有限数
        isFinite = (max32 < FP32_INF_PATTERN)
        # false 表示 Inf/NaN

        # 判断是否为非 0
        isNonZero = (max32 != 0)

        # 归一化：
        # s = amax / dtype_max
        max32 = max32 * invDtypeMax

        # 提取 biased exponent 和 mantissa
        exp32 = max32 >> 23
        man32 = max32 & 0x007fffff

        # 判断是否需要指数 +1，以实现 ceil(log2(s))
        needRoundUp = False

        # 情况 1：普通 normal 数
        if exp32 > 0 and exp32 < 254 and man32 != 0:
            needRoundUp = True

        # 情况 2：subnormal 数但已经接近最小 normal
        if exp32 == 0 and man32 > 2^22:
            needRoundUp = True

        # 如果 needRoundUp，则取 exp32+1，否则取 exp32
        extractExp = exp32 + 1 if needRoundUp else exp32

        # Inf/NaN 时，E8M0 scale 标记为 NaN
        if not isFinite:
            extractExp = E8M0_NAN   # 0xFF

        # 全 0 时，scale 为 0
        if not isNonZero:
            extractExp = 0

        # 生成 8-bit E8M0 scale
        scale8 = pack_low8(extractExp)

        # 生成 reciprocal：2^{-e}
        # 将 extractExp 左移 7 位，放到 BF16 指数位置
        # 然后用 2*bias 减去它
        reciprocalField = (2 * BF16_BIAS) - (extractExp << 7)

        # Inf/NaN -> reciprocal 也置为 NaN
        if not isFinite:
            reciprocalField = BF16_NAN

        # 0 -> reciprocal 也置为 0
        if not isNonZero:
            reciprocalField = 0

        # 打包成 16-bit reciprocal
        recip16 = pack_low16(reciprocalField)

        # 当前这 64 个 max 对应 8 行：
        # row0 的 8 个 scale
        # row1 的 8 个 scale
        # ...
        # row7 的 8 个 scale
        for k in 0 .. 7:

            # 取出第 k 行的 8 个 scale
            scaleRow = gather(scale8, row=k)

            # 写入 mxScale1Addr，然后指向下一行
            store_align(mxScale1Addr, scaleRow)
            mxScale1Addr += row_stride_scale1

            # 取出第 k 行的 reciprocal
            recipRow = gather(recip16, row=k)

            # 写入 mxScale1ReciprocalAddr
            store_align(mxScale1ReciprocalAddr, recipRow)
            mxScale1ReciprocalAddr += row_stride_recip

    # 生成 y1：对每一行做 x * reciprocal，再 cast 到 FP8
    for i in 0 .. blockCount-1:

        x0, x1 = deinterleave_load(xAddrBase + i * dataLen)

        # 读取这一行的 reciprocal
        recip = load_reciprocal(mxScale1ReciprocalBase + i * recipRowLen)

        # 如果是 FP16 输入，转 FP32 乘；如果是 BF16 输入，可以直接 BF16 乘
        if xDtype == FP16:
            y = cast_to_fp8_fp32(x * cast_fp32(recip))
        else:
            y = cast_to_fp8_fp32(x * recip)

        # 打包写 y1
        store_align(y1Addr, y)

    # ------------------------------------------------------------
    # 阶段 3：处理 -2 轴 scale
    # ------------------------------------------------------------

    # 把 slot0 的偶/奇 max 交织成完整列顺序
    interleave_store(mxScale2ReciprocalAddr,
                     scale2Slot0Part0,
                     scale2Slot0Part1)

    # 如果有 slot1，则放到第二段
    if blockCount > 32:
        interleave_store(mxScale2ReciprocalAddr + dataLen,
                         scale2Slot1Part0,
                         scale2Slot1Part1)

    # 调用 SecondLast 函数，把逐列 max 转成 scale2
    ComputeScaleCuBLASSecondLast(
        dataLen,
        invDtypeMax,
        mxScale2ReciprocalAddr,
        mxScale2Addr)

结束函数
```

---

## 五、`ComputeScaleCuBLASSecondLast` 伪代码

```text
函数 ComputeScaleCuBLASSecondLast(dataLen, invDtypeMax,
                                  mxScale2ReciprocalAddr,
                                  mxScale2Addr):

    # 每次处理 VF_LEN_FP32=64 列
    times = dataLen / 64

    for i in 0 .. times-1:

        colOffset = i * 64

        # slot0 的 64 个逐列 max
        slot0Addr = mxScale2ReciprocalAddr + colOffset

        # slot1 的 64 个逐列 max
        slot1Addr = mxScale2ReciprocalAddr + dataLen + colOffset

        # 对 slot0 做 scale 转换
        scale8Slot0 = convert_max_to_scale(slot0Addr)

        # 对 slot1 做 scale 转换
        scale8Slot1 = convert_max_to_scale(slot1Addr)

        # 两个 slot 交织写 mxScale2Addr
        interleave_store(mxScale2Addr + 2*colOffset,
                         scale8Slot0,
                         scale8Slot1)
```

---

## 六、`convert_max_to_scale` 伪代码

这个函数就是“单个 64 列向量”的 scale 转换：

```text
函数 convert_max_to_scale(maxAddr):

    # 读 64 个 16-bit max
    max16 = load(maxAddr)

    # 转 FP32
    max32 = cast_to_fp32(max16)

    # 特殊判断
    isFinite   = max32 < 0x7f800000
    isNonZero  = max32 != 0

    # amax / dtype_max
    max32 = max32 * invDtypeMax

    # 提取指数和尾数
    exp32 = max32 >> 23
    man32 = max32 & 0x007fffff

    # ceil(log2) 进位
    needRoundUp = False
    if 0 < exp32 < 254 and man32 != 0:
        needRoundUp = True
    if exp32 == 0 and man32 > 2^22:
        needRoundUp = True

    extractExp = exp32 + 1 if needRoundUp else exp32

    # 特殊值
    if not isFinite:
        extractExp = 0xFF      # E8M0 NaN
    if not isNonZero:
        extractExp = 0         # 0

    # 生成 scale
    scale8 = pack_low8(extractExp)

    # 生成 reciprocal
    reciprocalField = (2*127 - (extractExp << 7))
    if not isFinite:
        reciprocalField = NaN
    if not isNonZero:
        reciprocalField = 0

    store(reciprocalAddr, reciprocalField)

    return scale8
```

---

## 七、完整流程图

```text
64x256 tile
    |
    v
Phase 1: 逐行扫描
    |
    |-- 每行 256 列
    |     |
    |     |-- 偶/奇拆分
    |     |-- 每 32 列 reduce 出一个 max  -> -1 轴
    |     |-- 每列累积 32 行的 max       -> -2 轴
    |
    v
缓存了所有 -1 轴 block max
    |
    v
Phase 2: 每 64 个 max 一批
    |
    |-- 转 FP32
    |-- amax * invDtypeMax
    |-- 提取 exponent/mantissa
    |-- 根据 mantissa 决定 exp+1
    |-- 生成 E8M0 scale
    |-- 用 2*bias - scale_field 生成 reciprocal
    |-- scatter 到 8 行
    |
    v
生成 y1
    |
    v
Phase 3: 把 -2 轴两个 slot 的 max
        转成 scale2 并交织输出
```

---

## 八、容易混淆的知识点汇总

### 1. 为什么 `exp32 >> 23` 是取指数

因为 FP32 尾数占低 23 bit，右移 23 后指数落到低位。

### 2. 为什么 `0x7f800000` 能判断 Inf/NaN

```text
0x7f800000 是 +Inf 的 bit pattern
有限正数 < 0x7f800000
Inf/NaN >= 0x7f800000
```

### 3. 为什么尾数非 0 时指数要加 1

因为实现的是 `ceil(log2(s))`。

如果 `s` 不是精确的 2 的幂，`ceil` 比 `floor` 大 1。

### 4. 为什么 reciprocal 用 16-bit

因为 reciprocal 是 `2^{-e}`，是 2 的幂，16-bit BF16 可以精确表示。

### 5. 为什么代码中 `scaleBiasReg=0x7F00`

`0x7F00 >> 7 = 254`

254 是 `2 × BF16 bias`。

用它减去 scale 的 biased exponent，得到 reciprocal 的 biased exponent。

---

## 九、一句话总结

```text
ComputeScaleCuBLAS 做的事情是：

对每个 32 元素 block：
    求 amax
    计算 e = ceil(log2(amax / dtype_max))
    得到 scale = 2^e
    得到 reciprocal = 2^{-e}
    用 x * reciprocal 量化成 FP8

其中所有“取整、特殊值、reciprocal”的计算
都是通过 FP32/BF16 的 exponent bit 操作完成的。
```
