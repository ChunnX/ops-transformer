# FP8 DN 场景 rescaleO 上移 CUBE 方案设计

> 状态：**已实现（未上板实测）**——代码完成于 2026-08-03 会话，修复了实现期发现的 4 类缺陷（见 §10）；待 950 板端 UT/精度验证
> 日期：2026-08-01
> 涉及算子：BlockSparseAttention（950 / arch35 / AtlasA5）
> 目标场景：FP8 输入、FP16 输出、transposedMm1=true（DN）、qBaseTile=128、kvBaseTile=256、D=128

## 1. 背景与动机

现状（arch35 主循环，四级流水 PRE_LAUNCH=2）：

```
MM1(CUBE): K×Q^t → S^t [kvS,qS] → L0C → FixPipe → UB S
softmax(VEC): S → P（nZ 散布）→ L1；同时产出 lm/gm/ll/gl/dm
MM2(CUBE): P×V → OTmp [qS,D] → L0C → FixPipe → UB OTmp(lo)
rescaleO(VEC): go = go·dm + lo（逐行 broadcast 系数乘 + 加），末块 ÷gl → cast → GM O
```

### 1.1 性能动机：算子为 vector-bound，FP8 的 CUBE 收益被 VEC 掩盖

- 主循环的耗时瓶颈在 VEC 单元（mask2idx + online softmax + rescaleO），CUBE 侧 MM1/MM2 相对空闲；
- FP8 场景 Q/K/V 均为 fp8，CUBE 计算时间较 FP16/BF16 明显下降，但 VEC 侧工作量**不降反增**（FP8 的 P 量化路径比 half 多出 ×448、cast 拆分、Or/DeInterleave 等操作，见 `ComputePAndLocalSumFP8`）；
- 结果：FP8 相对非量化的执行时间收益没有体现出来；
- 因此把 rescale 阶段从 VEC 迁移到 CUBE：削掉每 tile 一次 `[qS,D]` 逐行乘加的 VEC 循环，同时消除 OTmp 的 FixPipe→UB 搬运链，压低 VEC 关键路径。

### 1.2 问题与目标

问题：rescaleO 的 `go·dm + lo` 在 VEC 上逐行执行，`lo`（OTmp）需要完整的 FixPipe→UB 搬运链和 2 份 UB 缓冲；VEC 侧 rescale 循环占用向量算力。

方案目标：把 rescale 重解释为**对角矩阵乘**，融合进 MM2 的 L0C 累加器（bias），在 CUBE 上完成 `GO_{i+1} = diag(dm)·GO_i + P·V`，从而：

- 删除 OTmp（lo）的 UB staged 缓冲与逐 tile 的 FixPipe→UB 搬运；
- VEC rescaleO 只剩末块的 ÷gl/cast/写 GM；
- rescale 的矩阵乘按转置分块（方法 3）实现，算量减半；并按 16×16 对角块裁剪（只搬非零块）后再降 1/4（见 §2.2）。

## 2. 数学基础

### 2.1 矩阵乘等价

```
现状：  GO_{i+1} = GO_i · dm_{i+1} + LO_{i+1}，其中 LO_{i+1} = P_{i+1}·V_{i+1}（逐元素系数乘 + 矩阵加）
等价：  GO_{i+1} = diag(dm_{i+1}) · GO_i + P_{i+1}·V_{i+1}
```

`diag(dm)` 为对角矩阵 `[qS,qS]`，行系数 = 每行（每个 q）的 rescale 系数。

### 2.2 转置分块（方法 3）

基于 `(D·R)^T = R^T·D^T`，整个融合模块在**转置空间**运算（DN 世界的天然形态，与 MM1 的 K×Q^t 同构）：

```
(GO_{i+1})^T = MMAD( A = (GO_i)^T        [D, qS]
                    B = diag(dm/vScale)^T [qS, qS]
                    bias = (V)^T·(P)^T    [D, qS]  ← PV^T 结果留在 L0C 作累加器 )
                   · vScale
```

CUBE 单实例全量计算（与现有 MM1/PV 同风格——mmad 全量、双子核划分只发生在 FixPipe 的 Sub0/Sub1 显式调用）：

```
(GO_{i+1})^T = MMAD( A = (GO_i)^T        [D, qS]
                    B = diag(dm/vScale)  [qS, qS]（双子核块拼接）
                    bias = (V)^T·(P)^T   [D, qS] ) · vScale
```

- 跳过了对角矩阵的零区域，算量降为 **4×2×128×16×16 = 0.26M MACs**（方法 1 为 128³ = 2.1M）；
- K=qS 按 16 切（见下文，K=16 合法性见 §6.1）；
- diag 双子核块（每块 zN [64,64]，L1 顺序排布）：RE 按 fr/4 选块、块内偏移 (fr%4)×16。

**16×16 块级优化（方法 3 的进一步裁剪）**：B_s=diag 块 [64,64] fp32 = 4×8 个 16×8 fractal，非零的只有对角斜带 8 个（fractal 行 fr 的列 2fr、2fr+1）；按 16×16 块（= 2 个 fractal）合并为 4×4 块网格后，**只有 4 个对角块 (fr,fr) 非零**（其余 12 块全零）。因此：

- **B 只搬 4 个 16×16 对角块**：`GetTile` 定位 + `copyL1ToL0B`（zN→nZ 转置特化，内部 LoadData 一次 `mStep=1、kStep=2` 即"一次搬斜带上两个分形"），搬运量 32 → 8 个 fractal（1/4）；
- **A（GO）竖着切成 4 个 [D,16]**，**C（L0C）按 16 列切**（2 个 fractal 列）；
- **RE = 4 次 `[D,16]×[16,16]` 小 mmad**，K=16 合法（`MmadParams.k ∈ [0,4095]` 无对齐约束，fp32 粒度 8 元素，见 §6.1 修正）；4 次在 **C 列方向拼接**（各写各的 16 列区，互不重叠），每次 `initMmad=false` **累加**到 PV^T 留下的 bias 上；
- 算量：**2×4×128×16×16 = 0.26M MACs**（方法 3 的 1/4）；
- 跳过零块 = 只对 4 个非零块发 copy：A/B 均用 `GetTile` 子块 + 现有 copy 特化（copy 内部按子块 coord 自动推导 LoadData 参数：`mStartPosition` 按 16 元素、`kStartPosition` 按 32B、`srcStride/dstStride` 单位 512B；GO 走 `copyL1ToL0A` zN→zN 恒等、diag 走 `copyL1ToL0B` zN→nZ 转置（ifTranspose=true，fp32 转置要求 kStep 为 2 的倍数 ✓ 满足））；
- PV^T 段不受影响（仍用现有 copyL1ToL0A/L0B 封装）。

### 2.3 反量化 scale 记账（关键正确性论证）

FP8 场景 bias 是 raw fp8 累加（未反量化），而 GO 存反量化后的 float 值。方案通过 **B 里先除 vScale、整体最后乘 vScale** 让两项同时正确：

```
方案计算:  GO_{i+1} = [ GO_i^T·diag(dm/vScale) + V^T·P^T_raw ] · vScale
逐项展开:  = GO_i·diag(dm)  +  PV_raw·vScale        ✓✓ 与原始语义逐项相等
```

- 两项量纲不同（一项已反量化、一项 raw），靠一个 `×vScale` 同时摆平；
- 对 **vScale 逐 tile 变化**鲁棒（`vScale/vScale` 恒消掉），不要求 vScale 跨 tile 常量；
- 末次的 `×vScale` 即 FixPipe 现有的 `deqScalar` 参数——反量化零新增成本；
- GO 存反量化 float，保持现状"go 为 float"的数值语义（不做每 tile 截断）。

## 3. 目标场景参数

| 参数 | 值 | 来源 |
|---|---|---|
| 平台 | 950（AtlasA5，arch35） | — |
| 输入/输出 | fp8_e4m3fn_t / half | BsaInferInterfaceFullQuant |
| transposedMm1 | true（host 恒置） | TransposedMatmulConfig950 |
| qBaseTile / kvBaseTile | 128 / 256 | CalcSplitCoreTilingParams950（innerPrecise==4 且 D≤128） |
| D (embed) | 128 | CheckQKVDimVal |
| L1 tile 参数 | mm1TileM/KLeft=128、mm1TileN/KRight=256、qL1=1、kL1=2；mm2TileM/KLeft=128/256、mm2TileN/KRight=128/256、pL1=3、vL1=2 | CalcMatmulPhaseL1TileInfo950（D=128 分支） |
| PRE_LAUNCH | 2（MAX_CROSS_CORE_BUF_STAGES=3） | 常量 |

## 4. 总体数据流（融合 PV^T + Rescale）

### 4.1 四种 tile 类型

| tile 类型 | 条件 | CUBE 动作 | FixPipe 目标 | 说明 |
|---|---|---|---|---|
| 单块（首=末） | kvSLoopNum==1 | PV^T（init） | NZ2DN → UB（deqScalar） | 无 diag、无 rescale；VEC ÷gl/cast → GM |
| 首块 | i==0 | PV^T（init） | NZ → L1（deqScalar） | 结果即 GO^T₁，作下块 A |
| 中间块 | 0 < i < last | PV^T（init）→ **RE（累加）** | NZ → L1（deqScalar） | bias 留 L0C，RE 累加后出 |
| 末块 | i == last | PV^T（init）→ **RE（累加）** | NZ2DN → UB（deqScalar） | VEC ÷gl ×(1/448) cast → GM O |

### 4.2 流水结构（PRE_LAUNCH=2）

```
迭代 t（CUBE）:
  MM1(t):  K gather→L1→L0A，Q→L0B → S^t→L0C → FixPipe(Nz2ND)→UB S
  PV^T(t-2): V gather 提前发射（早于跨核等待）→ PV^T → L0C [D,qS]（bias 保留）
  RE(t-2):   A=GO^T_{t-1}（L1→L0A，同核 FIX 后同步）、B=diag(t-2)（L1→L0B）
             → 累加到同一 L0C tile
  FixPipe(t-2): 中间块 CFG_NZ→L1；末块 CFG_COLUMN_MAJOR→UB

迭代 t（VEC）:
  softmax(t): S→P→L1；lm/gm/ll/gl；读 vScale(t)（当前索引，无 De）→ 生成 diag 块→L1
  rescaleO(t-2): 仅末块执行
```

### 4.3 跨核同步

| flag | id | 方向 | 说明 |
|---|---|---|---|
| Mm1ToSm | ubSBufId（0/1） | CUBE→VEC | 现有，不变 |
| smToMm2 | l1PBufId + 2（2/3/4） | VEC→CUBE | 现有，不变 |
| Mm2ToRe | 0 + UB_S_OTMP_BUF_STAGES + pL1BufNum（5） | CUBE→VEC | go 单份（仅末块），现有语义收窄 |
| **diag 消费完成** | 6/7/8（3 份） | CUBE→VEC | 新增：CUBE 读 diag 块完成（最后一块 RE mmad 后）set、**仅子核 0 单发**（避免双 set 残留）、VEC 覆写同 buffer 前 wait |

- GO^T 由 CUBE 自己写（迭代 t）自己读（迭代 t+1），同核顺序访问，**无需跨核 flag**，只需同核 FixPipe→MTE 同步；
- diag 块 3 份轮转（VEC 迭代 t 写、CUBE 迭代 t+2 读），与现有 dm 3 份语义一致。

## 5. 内存分配

### 5.1 L1（A1）：400KB / 512KB

| 段 | 偏移 | 每份 | 份数 | 合计 | 说明 |
|---|---|---|---|---|---|
| Q | 0 | 128×128×1B = 16KB | 1 | 16KB | ColumnMajor [D,qS]，常驻 |
| K | 16KB | 256×128×1B = 32KB | 2 | 64KB | RowMajor [kvS,D]，gather 轮转 |
| P | 80KB | 128×256×1B = 32KB | 3 | 96KB | zN（[kvS,qS] 视角），softmax 写/MM2 读 |
| V | 176KB | 256×128×1B = 32KB | 2 | 64KB | ColumnMajor [D,kvS] 进 L1（nZ 落 L1），gather 轮转 |
| GO^T（float） | 240KB | 128×128×4B = 64KB | 1 | 64KB | 转置空间 [D,qS]，1 份（同核串行读写） |
| diag 块（float） | 304KB | [64,64]×4B = 16KB | 2×3 | 96KB | 每子核 1 个块 × 3 份流水 |
| 合计 | | | | **400KB** | 剩 112KB |

要点：

- 全部数据段为 fp8（1B），比 FP16（480KB）省一半，是方案可行性的容量基础；
- **GO^T 1 份即可**：写（FixPipe→L1）与读（下块 L1→L0A）在同一 CUBE 核串行，无跨核竞争；
- **diag 只需两个 [64,64] 对角块**而非整张 [128,128]（方法 3 的结构红利），3 份对齐 PRE_LAUNCH 流水；
- `mm2L1AddrStart_` 之后的排布顺延：P/V 后接 GO^T、diag。

### 5.2 UB（VECCALC）：每子核 ~148KB / 256KB

| 段 | 大小 | 份数 | 变化 |
|---|---|---|---|
| S | 64×256×2B = 32KB | 2 | 不变（half） |
| P | (64×256+256)×1B = 16.25KB | 2 | 不变（fp8，含 pExtraElemNum 填充） |
| **OTmp (lo)** | — | — | **删除**（-64KB，PV^T 结果留 L0C） |
| go | 64×128×4B = 32KB | 1 | 保留（float，仅末块） |
| **diag 块中转** | [64,64]×4B = 16KB | 1 | **新增**（float，VF 算完 → UB → MTE3 → L1） |
| lm / gm / ll / gl | 128B / 256B / 128B / 256B | 1 | 不变 |
| **dm 向量** | 64×4B = 256B | 3（PRE_LAUNCH+1 轮转） | 保留（`UpdateDeltaMax` 先算 dm 存 dmUb，`GenDiag` 据此生成 diag） |
| lse | 64×8×4B = 2KB | 1 | 不变 |
| 合计 | | | **≈ 148KB** |

要点：

- 删除 OTmp 的 2 份 staged 缓冲后，diag 中转可放在原 loStartOffset 位置；
- mask2idx 的硬编码区（0~237568B）与主循环区时间复用，不受影响；
- 本表为每 AIV 子核一份（AIV0/AIV1 各自独立 UB）。

### 5.3 L0A / L0B / L0C

| 缓冲 | 分配 | 每 task 阶段计数（新增 RE 后） |
|---|---|---|
| L0A（64KB，2×32KB） | MM1/PV^T/RE 共享，全局 stage 计数取模 | MM1: 2；PV^T: 2；RE: 1（4 次小 mmad 轮流复用：A 块 [D,16] fp32 仅 8KB） |
| L0B（64KB，2×32KB） | 同上 | MM1: 1；PV^T: 2；RE: 1（B 块 [16,16] fp32 仅 1KB，4 个对角块依次搬入） |
| L0C（256KB） | MM1 前半 [0,128K)，MM2+RE 后半 [128K,256K)，各 2×64KB | — |

- RE 的 stage 计数并入 `CalcCrossMm1Mm2PrefixSumL0ABStages`（每 task 总计数 L0A 5、L0B 4）；
- RE 的 mmad：4 次 `M=128、N=16、K=16`，`initMmad=false` 累加到 PV^T 的 L0C 结果上（K=16 合法，见 §6.1）。

## 6. 硬件能力确认（已查证）

### 6.1 fp32 mmad（CUBE）

- Ascend 950PR/950DT 的 Mmad 明确支持 **float×float→float**（带/不带 bias 组合表均有）；
- fp32 fractal 为 **16×8**（half 为 16×16）；`MmadParams.k ∈ [0,4095]` **无对齐约束**（官方文档，K=70 非 16 倍数亦合法），fp32 的 K 粒度由 `kDirectionAlign=false`（950 仅支持 false）决定为 **8 元素** → RE 的 K=16 小块 mmad 合法；
- `SetHF32Mode` 可切换 HF32 舍入模式（默认关闭 = 原生 FP32），可用于精度问题定位。

### 6.2 FixPipe（L0C 搬出）——已按官方文档确认（Fixpipe L0C→L1 / L0C→UB）

- **L0C→L1 通路正式支持**（dst 为 L1 Buffer / TPosition C1；950 支持 NZ/ND/DN 三种输出格式）；`FixpipeConfig{CO2Layout format; bool isToUB}`，isToUB 指定目标是否 UB；
- **NZ2DN 是 950（351x）新增**；`TransformParams<CO2Layout::COLUMN_MAJOR>::PARAMS = Nz2DnParams{dnNum=1, srcNzMatrixStride=0, dstDnMatrixStride=0, srcNzC0Stride=0}`：
  - dnNum=1 时 `srcNzMatrixStride`/`dstDnMatrixStride` 置 0 不生效；
  - **`srcNzC0Stride` 启用 NZ2DN 时必须非 0（填 1），且 =1 时才兼容 unitFlag**；
- **L0C→L1 通路无 `dualDstCtrl`/`subBlockId`**（二者仅 L0C→UB 通路有效）→ 中间块 GO^T 是**单次整块搬运**（nSize=qS=128、mSize=D=128），无需按 AIV 切分；末块 NZ2DN→UB 仍走两次单目标调用（subBlockId 0/1）；
- **反量化**：950 的 QuantMode_t 支持 `QF322F32_PRE`（float→float scalar 反量化）——中间块 GO^T 落 L1 与末块落 UB 均使用 + `deqScalar=vScale`；
- 关键参数（FixpipeParamsArch3510）：nSize∈[0,4095] 且为 16 倍数（且 nSize×sizeof(T) 为 32 倍数）；mSize∈[0,65535] 且 NZ2DN 时 mSize×sizeof(T) 为 32 倍数；srcStride = mSize 对 16 向上取整（C0_Size=16×sizeof(T) 单位）；dstStride（NZ2ND/NZ2DN）= 目的矩阵每行元素数；
- NZ2DN 输出语义：DN[D,qS] 的列主序内存序 ≡ 行主序 [qS,D] 的内存序。**末块必须用 NZ2DN 而非 NZ2ND**：整个循环中 GO^T 保持倒置（[D,qS]）状态，NZ2ND 输出 row-major [D,qS] 仍是倒置的，而 rescaleO 的 ÷gl 按 q 行（[qS,D] 的行方向）广播，方向会错；NZ2DN 相当于"把倒置的再倒置"转回 ND，搬到 UB 的 go **直接就是 O 的 row-major [qS,D]**，现有 rescaleO 的逐行 ÷gl/cast 逻辑可原样复用。

## 7. 代码改动清单

### 7.1 Dispatch policy（模板参数）

`MmadAtlasA5BsaPV` 增加第二模板参数 `rescaleOnCube`，特化仅两种：

| policy 实例 | 特化文件 |
|---|---|
| `MmadAtlasA5BsaPV<transposedMm1, false>` | 现有 `block_mmad_pv_arch35_ABf16_C_to_UB.hpp`（特化键补 `false`，唯一改动） |
| `MmadAtlasA5BsaPV<true, true>` | **新文件** `block_mmad_pv_transposed_rescale_arch35.hpp` |

- `rescaleOnCube=false` 时全链路与现状一致，保证兼容性；
- **interface 的 LayoutPDummy/LayoutV 不改**（仅用于 TileCopy 推导 L1A/L1B tag；新 PV 文件全部硬编码布局 tag、不依赖这两个推导）；
- **kernel 类 LayoutV 条件化**（full_quant kernel）：rescaleOnCube 时 V 以 ColumnMajor [D,kvS] 进 L1（GM→L1 走 ColumnMajor→nZ 特化，与 Q 的 DN 加载同构），L1→L0A 走 nZ→zN 转置加载（vShape 同步交换）；
- 新 PV 文件内布局 tag 全部硬编码：V 的 gather 声明 **nZ**、B=(P)^T 声明 **zN [kvS,qS]**（P 在 L1 从 [kvS,qS] 视角即 zN，走 copyL1ToL0B zN→nZ 特化，与 B=V、diag 同构）、GO^T/diag 声明 **zN**。

### 7.2 新 PV 文件（`<true, true>`）

- **结构模板**：ND 与转置的对照参考 `bsa_block_mmad_qk.hpp`（ND QK）与 `block_mmad_k_qt_st_arch35.hpp`（转置 QK：L0C 结果保持 Nz，仅搬出时转 row-major）这一对文件的写法；
- PV^T：C=[D,qS] 全量（CUBE 单实例，无子核分摊），K=kvS=256 分 2 块；A=(V)^T 的 L1 声明硬编码 **nZ**（不依赖 LayoutTagL1A/L1B 推导，interface 保持原样）；
- RE：PV^T 后 **8 次 `[D,16]×[16,16]` 小 mmad**（K=16，C 列方向拼接 + `initMmad=false` 累加到 bias，全量覆盖 qS）；A/B 加载用 `GetTile` 子块 + 现有 copy 特化（GO 走 zN→zN 恒等、diag 走 zN→nZ 转置，fr/4 选双子核块、只搬非零对角块，见 §2.2）；
- FixPipe 三路径：首/中间块 `CFG_NZ→L1`（保持倒置的 Nz 状态——下块 rescale 的 A 同为倒置，零转换闭环）、末块 `CFG_COLUMN_MAJOR→UB`（**Sub0/Sub1 两次显式调用**，各搬 [D,qS/2] 半区到对应 AIV 的 UB，subBlockId 0/1 路由，带 deqScalar）；
- 单块特判：仅 PV^T + `CFG_COLUMN_MAJOR→UB`（同样 Sub0/Sub1 两次）。

### 7.3 CATLASS FixPipe 特化

新增两类（现有 `CopyL0CToUBTla` 的 6 个特化均硬编码 `CFG_ROW_MAJOR_UB`，不动）：

1. **新增独立 copy 类 `CopyL0CToL1Tla`**：`CFG_NZ` + isToUB=false → L1，**单次整块搬运**（无 subBlockId），quantPre=`QF322F32_PRE` + deqScalar（首/中间块 GO^T 反量化落 L1）；dstStride = `mSize × C0_Size(32B) / sizeof(T)`（fixpipe_l0c2l1 样例：`dstStride = baseM × c0Size`，c0Size 单位 B——已按样例确认）；
2. `CopyL0CToUBTla` 新增 `CFG_COLUMN_MAJOR`（NZ2DN）特化：`Nz2DnParams{dnNum=1, srcNzMatrixStride=0, dstDnMatrixStride=0, srcNzC0Stride=1}`，两次单目标调用（subBlockId 0/1），dstStride = DN 目的矩阵每行元素数；
3. 配置值（[D,64] 每子核 / [D,128] 整块）：nSize=64/128、mSize=128、srcStride=128、dstStride=128。

### 7.4 OnlineSoftmax（VEC）

- 生成 **diag(dm/vScale) 块** [64,64] fp32（本子核 64 个 q）：`UpdateDeltaMaxGlobalMaxAndGlobalSum` 先算 dm 存 dmUb（3 份轮转）→ `GenDiagBlockAndCopyL1` 读 dmUb 生成对角块 → MTE3 → L1（3 份轮转）；
- **VEC 每 task 读 vScale**（GM，当前索引无 De）：`yBlockIndex = (gatheredKvSTileIdx·kvBaseTile_)/blockShapeY_` → `gSparseIdx` → `vDequantScale`，与 CUBE 侧 FixPipe 用 De 索引各取各的；
- 保留 dm 向量缓冲（3 份轮转）：先经过 `UpdateDeltaMaxGlobalMaxAndGlobalSum` 得到 dm 结果，再据此生成 diag(dm/vScale) 块。

### 7.5 RescaleO（VEC）

- **删除原逐 tile 的 rescale 计算**（`block_epilogue_rescale_o_arch35_reg_high_prec.hpp` 中的 `RescaleFunc` / `RescaleFuncLastNotFirst` / `DivFuncLastAndFirst` 及 `SubCoreCompute` 的非末块分支）；
- 仅保留末块路径：`go/gl`、FP8 时 ×(1/448)、cast、写 GM O；go 缓冲保留；
- 输入即 row-major [qS,D]（NZ2DN 输出），末块处理逻辑本身不变。

### 7.6 跨核同步

- 新增 diag 块**消费完成** flag（6/7/8，3 份）：CUBE 读 diag 块完成 set、VEC 覆写同 buffer 前 wait——沿用"每份 dst 缓冲一个 id、顺序递增"的加法规律；写完成无需 flag（VEC 迭代 t 写、CUBE 迭代 t+2 读，流水深度天然满足，与 smToMm2 同理）；
- 末块 go 的 CUBE→VEC 同步（rescaleO 等待末次 FixPipe 完成）沿用 Mm2ToRe 的语义；
- GO^T 无需跨核 flag：同核 FIX→MTE 同步已实现（`FIX_M, GO_READY_EVENT_ID=4`，中间块 set / 非首块 RE wait，见 §8 待验证项 3）。

### 7.7 Kernel 主循环（block_sparse_attention_kernel_arch35_full_quant.h）

- L1/UB 偏移表更新（`CalcUBufTileInfo` 与 L1 排布）；
- 首/中/末块分支 + 单块特判；
- `CalcCrossMm1Mm2PrefixSumL0ABStages` 计入 RE 的 stage 计数（L0A 6、L0B 4/task）；
- VEC 侧读取 vScale（当前索引，无 De）传入 softmax。

### 7.8 内存分配改动文件清单（所有涉及内存分配的文件同步修改）

| 文件 | 改动 |
|---|---|
| `block_sparse_attention_kernel_arch35_full_quant.h` | `CalcUBufTileInfo`（删 lo 段、保留 dm 段、增 diag 中转段）、L1 排布（Q/K/P/V 后接 GO^T 64KB + diag 块 96KB）、`InitCrossCoreDstBuf`、`mm2L1AddrStart_` 语义、**LayoutV 条件化 + vShape 交换**（rescaleOnCube 时 V 以 ColumnMajor [D,kvS] 进 L1） |
| `block_epilogue_arch35_utils.hpp`（UBufTileHelper） | 偏移字段调整：保留 dm、增 diag 中转 |
| `block_epilogue_online_softmax_arch35_reg_low_prec_S_trans.hpp` | 构造缓冲：保留 dmUb（3 份轮转）、增 diag 块中转缓冲；`UpdateDeltaMax` 先算 dm 存 dmUb，再生成 diag(dm/vScale) 块 |
| `block_epilogue_rescale_o_arch35_reg_high_prec.hpp` | go 缓冲偏移随新布局；删除原 rescale 计算（见 §7.5） |
| 新 PV 文件（`<true,true>`） | 构造函数：diag 块的 L1 起始地址传入（沿用 `mm2L1AddrStart_` 式传参）、L0 缓冲；GO^T 与 go 的 UB/L1 TLA 均由 kernel 构造传入（模板参数 TensorGo/TensorC，与现有 PV 的 TLA 传入风格一致） |
| 现有 PV 文件（`<*,false>`） | 特化键补 `false`；缓冲逻辑不动 |

> 仅全量化（FP8）路径涉及；常规路径（`rescaleOnCube=false`）的 kernel/softmax/rescaleO 全部不动。

## 8. 待验证项

| # | 项 | 说明 |
|---|---|---|
| 1 | ~~FixPipe `CFG_NZ` 输出与 tla 布局 tag 的对应~~ **已确认** | 昇腾社区术语 **NZ ≡ tla zN**（同一种 fractal 布局的两种写法）；FixPipe `CFG_NZ` 保形输出 = L0C NZ 物理 = tla zN → GO^T 声明 zN、RE 走 zN→zN 恒等特化正确（dump 实验可作为上线前保险，非必需） |
| 2 | ~~CANN 头文件确认 NZ2DN 参数~~ **已完成** | `Nz2DnParams` 结构体与语义已按官方 Fixpipe 文档确认（dnNum=1 时 matrix stride 置 0、srcNzC0Stride=1 且 =1 才兼容 unitFlag）；L0C→L1 通路无 dualDstCtrl/subBlockId，单次整块搬运 |
| 3 | ~~FixPipe→L1 的同步语义~~ **已实现** | GO^T 的 L1 写完成用独立同核事件（`FIX_M, GO_READY_EVENT_ID=4`，l0C 释放用 0-3）：中间块 FixPipe(CFG_NZ→L1) 后 set、非首块 RE 读 GO^T 前 wait（严格交替，单 id 即可）；**不能尾部统一 set**（末块 NZ2DN→UB 与 GO^T 无关，会污染跨 task 的 wait 提前通过）；l0C 释放时序不变（每 tile 仍 FixPipe + FIX_M 释放，仅目标分级） |
| 4 | ~~尾块 RE 丢列~~ **已修复** | `blockNum = CeilDiv(qPerSubCore,16)` + 尾块 `nAct`（`min(16, qPerSubCore-fr*16)`），mmad 用 `n=nAct, k=nAct`（k/n 实际值有先例：PV^T 的 K 尾块同惯例）；**对角零贡献保证无污染**：k 对齐填充读的 B 行在 C 的列区（< fr*16+nAct）内为 0（对角矩阵行 k 非零在列 k），`A×0=0` 恒成立——即使硬件按 8 对齐读 A 的填充列；A/B 块固定 [16] 取（GO^T 有效列到 qS、diag 物理够），仅 mmad 参数用 nAct。待实测确认 |

## 10. 实现期修复记录（2026-08-03）

实现审查（对照旧 PV/QK 惯例逐事件核对）发现的缺陷及修复：

| # | 位置 | 缺陷 | 修复 |
|---|---|---|---|
| 1 | 新 PV K/RE 循环 mmad 前 | M 侧 wait 用了 `M_MTE1`（应为 `MTE1_M`），每轮迭代 M_MTE1 事件 0 set 2 wait | 改 `WaitFlag<MTE1_M>`（mmad 后 `SetFlag<M_MTE1>` 原本正确） |
| 2 | 新 PV FixPipe 段 | 额外 `WaitFlag(FIX_M)`（mmad 循环 kL0Itr==0 已有），L0C 事件双消费 | 删除 FixPipe 段的重复 wait |
| 3 | half/bf16 S_trans diag DataCopy | 单缓冲 diagUbTensor 无 MTE3 围栏，下轮 Duplicate 覆盖在飞 MTE3 读 | 新增 `MTE3_V EVENT_ID5`：覆写前 wait（UpdateDeltaMax 之前）、DataCopy 后 set；kernel Init/ReleaseSyncFlags 同步预置/收尾 |
| 4 | full_quant kernel | rescaleOnCube 专属改动全部无条件：`CalcUBufTileInfo` 新布局、`BlockMmadPV` 4 参构造、softmax 8 参调用、rescaleO 仅末块、`Mm2ToReFlagId` 固定、InitSyncFlags 删 FLAG_ID6 —— **FP8 ND 12 活跃键（rescaleOnCube=false）编译失败 + 运行破坏** | 引入 `RESCALE_ON_CUBE = transposedMm1 && BlockMmadPV::rescaleOnCube`，全部 if constexpr 守卫；旧路径恢复 lo/dm/ll 原布局、ubOTmpTensor 2 份、flag 5/6 交替、逐 tile rescaleO |
| 5 | 旧 PV 构造 | kernel 统一 4 参调用与 3 参构造不兼容 | 旧 PV 加 4 参委托构造（默认参数不可行：3 参调用的第 3 参类型不匹配） |
| 6 | kernel `mm2L0ATotalStages_` | 未计入 RE 的 8 次小 mmad（每 tile L0A/L0B 实际 10 次，计数 2） | rescaleOnCube 分支：`kvBaseTile_/L0_TILE_K + CeilDiv(qBaseTile_,16)` |
| 7 | bf16 S_trans softmax | 上一会话只改了 half 版，bf16 版未加 rescaleOnCube 支持（FP8 OBF16 DN 6 活跃键编译失败） | 同步移植：模板参数、diag 缓冲、`GenDiagBlockAndCopyL1`、operator() 尾参默认值 |

## 9. 参考

- 代码：`block_mmad_k_qt_st_arch35.hpp`（DN 转置 + n 切分参考）、`block_mmad_qk_arch35_ABf16_C_to_UB.hpp`（ND 对照）、`block_mmad_pv_arch35_ABf16_C_to_UB.hpp`（现有 PV）、`copy_l0c_to_ub_a5.hpp`（FixPipe 封装）、`block_epilogue_online_softmax_arch35_reg_low_prec_S_trans.hpp`、`block_epilogue_rescale_o_arch35_reg_high_prec.hpp`、`block_sparse_attention_kernel_arch35_full_quant.h`
- 官方样例：`fixpipe_l0c2gm`、`fixpipe_nz2dn_tensor_quantization_f322f16`（asc-devkit）
- 文档：SetFixPipeConfig（CO1→GM/CO1→A1）、220x→351x 架构变更（NZ2DN 新增）、asc-devkit Mmad.md（950PR/DT float×float→float）
