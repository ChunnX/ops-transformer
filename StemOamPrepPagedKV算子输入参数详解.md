# stem_oam_prep_paged_kv 算子输入参数详解

## 背景

本算子在 Decode 阶段每个新 token 生成时被调用，负责从 Paged KV Cache 中提取 K/V 数据，经过压缩计算生成摘要（kFlat/vBias），供后续 Stem OAM 打分模块快速判断哪些 token 块值得做完整注意力计算。

---

## 输入参数

### 1. kCache

**类型**: 输入张量

**数据类型**: FP8_E4M3FN（8位浮点数，精度低、体积小）

**含义**: Paged K cache，存储所有历史 token 的 Key 向量。Key 向量是每个 token 的"检索标签"，描述"我是谁、有什么特征"，供其他 token 来查找匹配。

**Shape**:
- Layout A: `[total_blocks, kvBlockSize, H_kv, 128]`
- Layout B: `[total_blocks, H_kv, kvBlockSize, 128]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | total_blocks | 内存池中物理页的总数。多个请求共享内存池，所有请求的 KV 数据都存在这些物理页里。例如同时处理8个请求，总共需要9649个物理页 |
| 第1维 | kvBlockSize（Layout A）或 H_kv（Layout B） | Layout A: 每页内有多少个 token 行（64或128）；Layout B: 有多少个注意力头（如2个头） |
| 第2维 | H_kv（Layout A）或 kvBlockSize（Layout B） | Layout A: 有多少个注意力头；Layout B: 每页内有多少个 token 行 |
| 第3维 | 128 (D) | 每个头每个 token 的向量维度，固定为128。即一个 Key 向量有128个数值 |

**为什么不连续**: KV Cache 采用分页存储，多个请求共享内存池。kCache 的数据散落在物理页上，页与页之间不连续，页内数据连续。

**FP8 的原因**: 为了节省内存。原始 K 值是 FP32（每个数占4字节），量化成 FP8（每个数占1字节）存储，读取时乘 kScaleCache 还原真实值。

**示例**:

```
典型场景: total_blocks=9649, H_kv=2, kvBlockSize=128
内存池中共有 9649 个物理页

Layout B 下:
kCache shape = [9649, 2, 128, 128]
  第0维=9649: 内存池有9649个物理页
  第1维=2: 2个注意力头
  第2维=128: 每页128个token行
  第3维=128: 每个向量128维

物理页3的数据（第0个头）:
  token0的head0: [fp8值 × 128维]  ← 第0行
  token1的head0: [fp8值 × 128维]  ← 第1行
  ...（128行连续）
  token127的head0: [fp8值 × 128维] ← 第127行

物理页3的数据（第1个头）:
  token0的head1: [fp8值 × 128维]
  token1的head1: [fp8值 × 128维]
  ...（128行连续）
```

**Layout A vs Layout B 区别**:

```
Layout A（交织）: 不同头的行交替排列
  行0: token0_head0 [128维]
  行1: token0_head1 [128维]
  行2: token1_head0 [128维]
  行3: token1_head1 [128维]
  ...
  同一个头的数据每隔一行才有一条，读取跳跃，效率低

Layout B（连续）: 同一个头的所有行连续排列
  head0区域:
    行0: token0_head0 [128维]
    行1: token1_head0 [128维]
    ...（128行连续）
  head1区域:
    行128: token0_head1 [128维]
    行129: token1_head1 [128维]
    ...（128行连续）
  同一个头的数据排在一起，读取连续，效率高
```

---

### 2. vCache

**类型**: 输入张量

**数据类型**: FP8_E4M3FN

**含义**: Paged V cache，存储所有历史 token 的 Value 向量。Value 向量是每个 token 的"内容摘要"，描述"如果你关注我，我能提供什么信息"。与 kCache shape/layout 完全一致。

**Shape**: 同 kCache

**Shape 各维度含义**: 与 kCache 完全相同，只是存储的是 V 数据而不是 K 数据：
| 维度 | 变量名 | 含义 |
|------|--------|------| 
| 第0维 | total_blocks | 内存池中物理页的总数 |
| 第1维 | kvBlockSize 或 H_kv | 取值取决于 kvLayout |
| 第2维 | H_kv 或 kvBlockSize | 取值取决于 kvLayout |
| 第3维 | 128 (D) | 每个头每个 token 的 V 向量维度，固定为128 |

**与 kCache 的关系**: K 和 V 是同一个 token 在同一层产出的两个向量，K 用于匹配检索，V 用于提供内容。两者存储方式相同，但用途不同。

**示例**:

```
与 kCache 完全同结构，只是存的是 V 数据而不是 K 数据
vCache shape = [9649, 2, 128, 128]  (Layout B)
```

---

### 3. kvIndices

**类型**: 输入张量

**数据类型**: INT32

**含义**: 页目录，记录每个请求的逻辑页号对应的物理页号。因为多个请求共享同一个内存池，每个请求的 KV 数据散落在不同的物理页上，kvIndices 就是告诉你去哪找。

**Shape**: `[batch, max_kv_blocks]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | batch | 同时处理的请求数量。例如同时处理2个用户的请求，batch=2 |
| 第1维 | max_kv_blocks | 每个请求最多可能占多少个逻辑页。所有请求共用这个维度长度，取所有请求中的最大值。例如请求A需要8页，请求B需要4页，max_kv_blocks取较大值如2000 |

**为什么需要**: kCache/vCache 的物理页不连续，要读取某个请求的 KV 数据，必须先通过 kvIndices 找到对应的物理页号，再去那个物理页读数据。

**示例**:

```
场景: 同时处理2个请求（batch=2）
内存池共100个物理页

请求A的数据散落在物理页3、7、15、22
请求B的数据散落在物理页1、5、9

kvIndices:
  kvIndices[0] = [3, 7, 15, 22, 0, 0, ...]  ← 请求A，前4个有效，后面0是填充
  kvIndices[1] = [1, 5, 9, 0, 0, ...]       ← 请求B，前3个有效

kvIndices shape = [2, 2000]
  第0维=2: 2个请求
  第1维=2000: 最大可能的逻辑页数

读取过程:
  要读请求A的逻辑页0 → kvIndices[0][0] = 3 → 去物理页3读数据
  要读请求A的逻辑页1 → kvIndices[0][1] = 7 → 去物理页7读数据
```

---

### 4. kvSeqLens

**类型**: 输入张量

**数据类型**: INT32

**含义**: 每个请求当前有多少个有效 token（KV 序列长度）。用于区分有效数据和填充数据——分配的页可能比实际 token 数多，超出部分要填零。

**Shape**: `[batch]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | batch | 同时处理的请求数量。每个请求有一个序列长度值，例如 kvSeqLens=[1000, 500] 表示请求A有1000个token，请求B有500个token |

**为什么需要**: 确定哪些数据是真正有效的、哪些是空的。超出 kvSeqLens 的 token 位置在计算中要填零，避免无效数据影响结果。

**示例**:

```
场景: 2个请求

请求A已生成1000个token，分配了8页（8×128=1024个位置）
请求B已生成500个token，分配了4页（4×128=512个位置）

kvSeqLens = [1000, 500]
  第0维=2: 2个请求
  kvSeqLens[0]=1000: 请求A有1000个有效token
  kvSeqLens[1]=500: 请求B有500个有效token

效果:
  请求A: 前1000行有效，第1000~1023行填零（padding）
  请求B: 前500行有效，第500~511行填零

特殊边界:
  kvSeqLens[b] = 0 → 该请求的kFlat/vBias输出全零（没有token，没有摘要）
```

**在算子中的作用**:
- De-page 时：只搬运有效 token 的数据，超出部分填零
- kScaleCache 读取时：超出 kv_len 的行填零
- V Processing 时：超出 kv_len 的 L2 Norm 填零
- 计算输出 shape 时：max_Kb = ceil(max(kvSeqLens) / stemBlockSize)

---

### 5. kScaleCache

**类型**: 输入张量（FP8 时必填）

**数据类型**: FP32

**含义**: 每个 token、每个头的 K 缩放系数。因为 K 数据存的是 FP8（精度低），存入时做了量化压缩，读取时需要乘 kScaleCache 还原真实大小。粒度是 per-token per-head（每个 token 每个头各一个缩放系数）。

**Shape**:
- Layout A: `[total_blocks, kvBlockSize, H_kv, 1]`
- Layout B: `[total_blocks, H_kv, kvBlockSize, 1]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | total_blocks | 内存池中物理页的总数，与 kCache 第0维相同。kScaleCache 的数据也按页存储，每个物理页都有对应的缩放系数 |
| 第1维 | kvBlockSize（Layout A）或 H_kv（Layout B） | 与 kCache 同布局含义：Layout A 是每页的 token 行数，Layout B 是注意力头数 |
| 第2维 | H_kv（Layout A）或 kvBlockSize（Layout B） | 与 kCache 同布局含义：Layout A 是注意力头数，Layout B 是每页的 token 行数 |
| 第3维 | 1 | 一个 token 一个头只需要一个缩放系数（整个128维向量共用同一个系数），所以最后一维是1而不是128 |

**为什么跟 kCache 同布局**: kScaleCache 和 kCache 一起存、一起读，布局一致方便定位。最后一维是1而不是128，因为一个 token 一个头只需要一个缩放系数（整个128维向量共用同一个系数）。

**示例**:

```
Layout B 下:
kScaleCache shape = [9649, 2, 128, 1]
  第0维=9649: 内存池有9649个物理页
  第1维=2: 2个注意力头
  第2维=128: 每页128个token行
  第3维=1: 每token每头一个缩放系数

原始K值 (FP32): [3.14, -0.87, 2.56, ...]  ← 精度高但占4字节/元素

存入KV Cache时量化成FP8 (占1字节/元素):
  kScaleCache[token0, head0] = 4.0   ← token0在head0的缩放系数
  存储值: [0.785, -0.218, 0.64, ...] = 原始值 / 4.0

读取时还原:
  还原值 = FP8存储值 × kScaleCache
         = [0.785×4.0, -0.218×4.0, 0.64×4.0, ...]
         = [3.14, -0.87, 2.56, ...]

为什么每个token的kScale不同:
  不同token的数值范围不同
  token0的值可能在 [-3, 3] 范围 → kScale=4.0
  token1的值可能在 [-100, 100] 范围 → kScale=127.0
  各自缩放到FP8能表示的范围，各自需要各自的还原系数

边界处理:
  超出kvSeqLens的token位置 → kScaleCache值填零
  （乘零后K数据也变零，不影响计算）
```

---

### 6. vScale

**类型**: 输入张量（FP8 时必填）

**数据类型**: FP32

**含义**: 每个头的 V 缩放系数。与 kScaleCache 作用相同（还原 FP8 量化后的真实值），但粒度不同——是 per-head（每个头只有一个系数，所有 token 共享）。

**Shape**: `[H_kv]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | H_kv | 注意力头数。每个头有一个 V 缩放系数，所有 token 共享。例如 H_kv=2 时 vScale=[4.0, 8.0]，表示 head0 的所有 token 用 4.0 还原，head1 的所有 token 用 8.0 还原 |

**为什么粒度比 kScaleCache 粗**: K 的量化是按每个 token 的数值范围单独缩放的（per-token quantization），V 的量化是按每个头统一缩放的（per-head quantization），同一个头的所有 token 用同一个还原系数就够了。

**示例**:

```
vScale shape = [2]
  第0维=2: 2个注意力头，每个头一个缩放系数

vScale = [4.0, 8.0]

含义:
  vScale[0]=4.0: head0的所有token共用缩放系数4.0
  vScale[1]=8.0: head1的所有token共用缩放系数8.0

还原过程:
  token5在head0的V值还原 = FP8存储值 × vScale[0] = FP8值 × 4.0
  token5在head1的V值还原 = FP8存储值 × vScale[1] = FP8值 × 8.0

与kScaleCache对比:
  kScaleCache: per-token per-head → [total_blocks, kvBlockSize, H_kv, 1] → 每token每头一个
  vScale:      per-head           → [H_kv]                                → 每头一个
```

---

## 属性参数（ATTR）

### 7. lambdaMag

**类型**: 属性（ATTR）

**数据类型**: DOUBLE

**含义**: V Processing 最终输出的强度控制系数。在全局归一化 + ReLU 之后，乘上 lambdaMag 调低整体评分幅度，避免 vBias 值过大。

**默认值**: 0.3

**约束**: 0 ~ 1

**示例**:

```
V Processing 计算链:
  log值 → 全局归一化 → ReLU（负数截零）→ 乘 lambdaMag → block average → vBias

  归一化后某值 = 2.5
  ReLU后 = 2.5（正数不变）
  乘 lambdaMag = 2.5 × 0.3 = 0.75
  → 最终vBias中该块的贡献是0.75

  lambdaMag越小 → vBias整体越低 → 后续打分模块认为更多块不重要 → 稀疏性更强
  lambdaMag越大 → vBias整体越高 → 更多块被认为重要 → 稀疏性更弱
```

---

### 8. kvLayout（cacheLayout）

**类型**: 属性（ATTR）

**数据类型**: INT64

**含义**: KV Cache 的存储布局方式，决定 kCache/vCache/kScaleCache 的数据排列顺序和偏移计算公式。

**可选值**:
- 0 = Layout A（交织布局）——不同头的行交替排列，同一头的数据不连续
- 1 = Layout B（连续布局）——同一头的所有行连续排列，读取效率高

**默认值**: 0

**示例**:

```
kvLayout = 0 (Layout A):
  kCache shape = [total_blocks, kvBlockSize, H_kv, 128]
    第0维=total_blocks: 物理页总数
    第1维=kvBlockSize:  每页token行数（64或128），不同头的行交织排列
    第2维=H_kv:         注意力头数
    第3维=128:          向量维度
  页内排列: token0_head0, token0_head1, token1_head0, token1_head1, ...
  读某个head的数据要跳跃读取

kvLayout = 1 (Layout B):
  kCache shape = [total_blocks, H_kv, kvBlockSize, 128]
    第0维=total_blocks: 物理页总数
    第1维=H_kv:         注意力头数，同一头的行连续排列
    第2维=kvBlockSize:  每页token行数（64或128）
    第3维=128:          向量维度
  页内排列: token0~token127_head0连续, token0~token127_head1连续, ...
  读某个head的数据连续读取，效率高

对偏移计算的影响:
  Layout A偏移: phys × kvBlockSize × H_kv × D + row × H_kv × D + h × D
  Layout B偏移: phys × H_kv × kvBlockSize × D + h × kvBlockSize × D + row × D
```

---

### 9. kvBlockSize

**类型**: 属性（ATTR）

**数据类型**: INT64

**含义**: Paged KV Cache 中一个页包含多少个 token。每个页的大小固定，要么64个token一页，要么128个token一页。对应 kCache/vCache shape 中 kvBlockSize 维度的值。

**可选值**: 64 或 128

**示例**:

```
kvBlockSize = 128:
  每页存储128个token的K/V数据
  请求A有1000个token → 需要 ceil(1000/128) = 8 个页
  每页大小 = 128 × H_kv × 128 维 × 1字节(FP8) = 128×2×128×1 = 32KB (Layout B)

kvBlockSize = 64:
  每页存储64个token
  请求A有1000个token → 需要 ceil(1000/64) = 16 个页
  页更小，更灵活，但页数更多，kvIndices更长

对本算子的影响:
  De-page偏移计算依赖kvBlockSize
  模式2的bulk gather: kv_blocks_per_stem_block = stemBlockSize / kvBlockSize
```

---

### 10. stemBlockSize

**类型**: 属性（ATTR）

**数据类型**: INT64

**含义**: Stem 块大小——把 KV 数据按多少个 token 分一组做摘要压缩。每 stemBlockSize 个 token 产出一个 kFlat 向量和一个 vBias 数值。对应输出 kFlat/vBias shape 中 max_Kb 维度的划分依据。

**约束**: 必须是32的倍数，最大256

**推荐值**: 128

**示例**:

```
stemBlockSize = 128:
  每128个token划为一组
  请求A有1000个token → num_Kb = ceil(1000/128) = 8 组
  产出8个kFlat向量和8个vBias数值

  分组方式:
    组0: token 0~127   → kFlat[0], vBias[0]
    组1: token 128~255 → kFlat[1], vBias[1]
    ...
    组7: token 896~999 + 24个填充零 → kFlat[7], vBias[7]
```

---

### 11. stemStride

**类型**: 属性（ATTR）

**数据类型**: INT64

**含义**: Stem 步长——在 K Processing 的分组求和里，每 stemStride 个 token 为一条，一个 stemBlock 内有 R = stemBlockSize / stemStride 组，每组内对 R 条求和。对应输出 kFlat 中 kflat_dim = stemStride × 128 的计算。

**约束**: 必须是16的倍数，最大64，且 stemStride ≤ stemBlockSize

**推荐值**: 16

**示例**:

```
stemStride = 16, stemBlockSize = 128:
  R = 128 / 16 = 8

  一个stemBlock内128个token按"8组×16条"排列:
    组0: token0, token16, token32, ..., token112  (8条，每组内步长16)
    组1: token1, token17, token33, ..., token113
    ...
    组15: token15, token31, token47, ..., token127

  K Processing:
    每组内8条求和 → 得到16个向量（每个128维）
    反斜线翻转 → 拼成 kflat_dim = 16×128 = 2048 维的 kFlat

  V Processing:
    k_down_len = num_Kb × R = 8 × 8 = 64
    （64个位置需要做L2 Norm + Max + Log）
```

---

## 输出参数

### 12. kFlat

**类型**: 输出张量

**数据类型**: BFLOAT16

**含义**: K 侧摘要向量。每 stemBlockSize 个 token 的 K 数据经过缩放+分组求和+反斜线翻转后压缩成一个短向量，供后续 Q × kFlat 打分使用。

**Shape**: `[batch, H_kv, max_Kb, kflat_dim]`

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | batch | 同时处理的请求数量。每个请求独立产出自己的 kFlat 数据 |
| 第1维 | H_kv | 注意力头数。每个头独立计算摘要，不同头的 kFlat 不同 |
| 第2维 | max_Kb | 所有请求中最多的 stem block 组数。max_Kb = ceil(max(kvSeqLens) / stemBlockSize)。组数少的请求后面填零 |
| 第3维 | kflat_dim | 每组的摘要向量长度。kflat_dim = stemStride × 128。例如 stemStride=16 时 kflat_dim=2048，即每组压缩成一个2048维的 BF16 向量 |

**示例**:

```
stemStride=16, D=128:
  kflat_dim = 16 × 128 = 2048

  batch=2, H_kv=2, max_Kb=8:
  kFlat shape = [2, 2, 8, 2048]
    第0维=2: 2个请求
    第1维=2: 2个注意力头
    第2维=8: 最多的stem block组数（ceil(1000/128)=8）
    第3维=2048: 每组摘要向量长度

  kFlat[0][0][0] = 请求A、head0、第0组(token0~127)的2048维摘要向量
  kFlat[0][0][1] = 请求A、head0、第1组(token128~255)的2048维摘要向量
  ...
```

---

### 13. vBias

**类型**: 输出张量

**数据类型**: FP32

**含义**: V 侧块评分。每 stemBlockSize 个 token 的 V 数据经过缩放+L2 Norm+Max+Log+全局归一化+ReLU+乘λ+块平均后压缩成一个数值，代表该块的重要性。

**Shape**: `[batch, H_kv, max_Kb]`（实际存储stride可能更大，padded到8以上）

**Shape 各维度含义**:
| 维度 | 变量名 | 含义 |
|------|--------|------|
| 第0维 | batch | 同时处理的请求数量。每个请求独立产出自己的 vBias 数据 |
| 第1维 | H_kv | 注意力头数。每个头独立计算评分，不同头的 vBias 不同 |
| 第2维 | max_Kb | 所有请求中最多的 stem block 组数。每组产出一个 FP32 数值，代表该组的重要性评分。组数少的请求后面填零。实际存储可能 padded 到更大值（如 vbias_stride）以保证对齐 |

**示例**:

```
batch=2, H_kv=2, max_Kb=8:
vBias shape = [2, 2, 8]
  第0维=2: 2个请求
  第1维=2: 2个注意力头
  第2维=8: 最多的stem block组数，每组一个评分数值

vBias[0][0][0] = 请求A、head0、第0组(token0~127)的重要性评分，比如0.75
vBias[0][0][1] = 请求A、head0、第1组(token128~255)的重要性评分，比如0.12
vBias[0][0][7] = 请求A、head0、第7组的评分，比如0.0（不重要，ReLU截零了）

vBias越大 → 该块越重要 → 后续更可能被选中做完整注意力计算
vBias=0   → 该块不重要 → 直接跳过，不做完整注意力
```