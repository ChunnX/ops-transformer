# Flash Attention：ND 与 DN 排布格式深度解析

> 适用文件：`flash_attn_block_cube_noquant_gqa.h`、`flash_attn_block_vec_noquant_gqa.h`

---

## 一、背景：Flash Attention 的两次矩阵乘

Flash Attention 每个 tile 做两次矩阵乘：

```
BMM1:  Q × Kᵀ  →  分数矩阵 S         [M×S2]
Vec1:  softmax(S) → P                 [M×S2]
BMM2:  P × V   →  输出 O             [M×Dv]
```

其中：
- **M**  = 本 tile 处理的 Query token 数（`actMSize`，最大 128）
- **D**  = head 维度（`dSize`，通常 128）
- **S2** = 本 tile 处理的 Key/Value token 数（`actSingleLoopS2Size`，最大 128）
- **Dv** = output head 维度（`dSizeV`，通常 128）

**ND / DN 的名字来源**：
| 格式 | 含义 | BMM1 左矩阵 | BMM1 右矩阵 |
|------|------|------------|------------|
| **ND** | 结果是 \[N行 × D列\] 即 \[M×S2\] | Q \[M×D\] | K \[D×S2\] |
| **DN** | 结果是 \[D行 × N列\] 即 \[S2×M\]（转置） | K \[S2×D\] | Q \[D×M\] |

---

## 二、ND 格式：数据排布全流程图

### 2.1 BMM1 输入：Q 是左矩阵，K 是右矩阵

```
L1 里 Q（左矩阵，M×D）:          L1 里 K（右矩阵，D×S2）:

       d0  d1  d2 ... d127             s0  s1  s2 ... s127
q0  [  ·   ·   ·  ...  ·  ]    d0  [  ·   ·   ·  ...  ·  ]
q1  [  ·   ·   ·  ...  ·  ]    d1  [  ·   ·   ·  ...  ·  ]
q2  [  ·   ·   ·  ...  ·  ]    d2  [  ·   ·   ·  ...  ·  ]
...                             ...
q127[  ·   ·   ·  ...  ·  ]   d127[  ·   ·   ·  ...  ·  ]

 形状：[128 × 128]                形状：[128 × 128]
 Q[i,d] = Query token i 的第 d 维  K[d,j] = Key token j 的第 d 维
```

### 2.2 BMM1 输出：L0C 存 Score[M×S2]，Fixpipe 写 UB

```
L0C 结果（Score，M×S2）：              UB 经 Fixpipe（dualDstCtl=1，按 M 行切半）：

       s0  s1  s2 ... s127            Sub-block 0（AIV sub0）：
q0  [  ·   ·   ·  ...  ·  ]          → Score 的前 M/2 行（q0..q63）
q1  [  ·   ·   ·  ...  ·  ]               s0  s1 ... s127
...                                   q0 [ ·   ·  ...  · ]
q63 [  ·   ·   ·  ...  ·  ]          q1 [ ·   ·  ...  · ]
q64 [  ·   ·   ·  ...  ·  ]          ...
...                                   q63[ ·   ·  ...  · ]
q127[  ·   ·   ·  ...  ·  ]
                                      Sub-block 1（AIV sub1）：
  形状：[128 × 128]                   → Score 的后 M/2 行（q64..q127）
                                      q64 [ ·   ·  ...  · ]
                                      ...
                                      q127[ ·   ·  ...  · ]
```

**关键参数（FixpipeMm1）**：
```
nSize   = (actSingleLoopS2Size + 7) >> 3 << 3   // S2 对齐到 8
mSize   = (actMSize + 1) >> 1 << 1               // M 仅对齐到 2（最小）
dualDstCtl = 1                                   // 按 M 维切半
```

> ⚠️ **问题**：`mSize` 只对齐到 2，对 BF16 的 32-element 向量操作**不对齐**。

### 2.3 Vec1（softmax）：逐行处理，每行是一个 query token

```
Sub-block 0 中 UB 的内存视图（sub0 处理 q0..q63）：

逻辑视图：[M/2 行 × S2 列] = [64 × 128]

─── 内存地址增大方向 ──────────────────────────────────────────►
| q0的s0 | q0的s1 | ... | q0的s127 | q1的s0 | q1的s1 | ... | q1的s127 | ...
└────────── q0 的 128 个分数（一行）──────────┘└────────── q1 的 128 个分数 ──┘

softmax 沿行方向计算，每次处理 1 行（1 个 query token，128 个分数）。
每个 softmax 调用处理 actVecMSize = (actMSize+1)>>1 个 token → 串行！
```

### 2.4 Vec1 后 DataCopy UB→L1（为 BMM2 准备 P 矩阵）

```
UB（softmax 后，[M/2 × S2]）→ L1（P 矩阵的一半）

DataCopy 参数：
  blockCount = s2BaseSize / 16 = 8
  blockLen   = actVecMSize          = 64（每块连续 64 个元素）
  srcStride  = vec1Srcstride - actVecMSize = 65 - 64 = 1  ← 每块间只有 1 个元素间隔
  dstStride  = mBaseSize - actVecMSize = 128 - 64 = 64

内存访问示意：
源（UB）: [64 个][1 gap][64 个][1 gap][64 个] ...  ← 碎片化！DMA 不友好
目标（L1）: [64 个][64 gap][64 个][64 gap] ...
```

> ⚠️ **问题**：srcStride=1 导致 DMA 每次只能搬 64 个元素后跳过 1 个再继续，**硬件 burst 利用率低**。

---

## 三、DN 格式：数据排布全流程图

### 3.1 BMM1 输入：K 是左矩阵，Q 是右矩阵（角色互换！）

```
L1 里 K（左矩阵，S2×D）:         L1 里 Q（右矩阵，D×M）:

       d0  d1  d2 ... d127             q0  q1  q2 ... q127
s0  [  ·   ·   ·  ...  ·  ]    d0  [  ·   ·   ·  ...  ·  ]
s1  [  ·   ·   ·  ...  ·  ]    d1  [  ·   ·   ·  ...  ·  ]
s2  [  ·   ·   ·  ...  ·  ]    d2  [  ·   ·   ·  ...  ·  ]
...                             ...
s127[  ·   ·   ·  ...  ·  ]   d127[  ·   ·   ·  ...  ·  ]

 形状：[128 × 128]                形状：[128 × 128]
 K[j,d] = Key token j 的第 d 维   Q[d,i] = Query token i 的第 d 维（转置存储）
```

### 3.2 BMM1 输出：L0C 存 ScoreT[S2×M]，Fixpipe 写 UB

```
L0C 结果（ScoreT = Scoreᵀ，S2×M）：    UB 经 Fixpipe（dualDstCtl=2，按 M 列切半）：

       q0  q1  q2 ... q127            Sub-block 0（AIV sub0）：
s0  [  ·   ·   ·  ...  ·  ]          → ScoreT 的前 M/2 列（q0..q63）
s1  [  ·   ·   ·  ...  ·  ]               q0  q1 ... q63
...                                   s0 [ ·   ·  ...  · ]
s127[  ·   ·   ·  ...  ·  ]          s1 [ ·   ·  ...  · ]
                                      ...
  形状：[128 × 128]                   s127[ ·   ·  ...  · ]
                                      形状：[S2=128 × M/2=64]

                                      Sub-block 1（AIV sub1）：
                                      → ScoreT 的后 M/2 列（q64..q127）
                                      形状：[S2=128 × M/2=64]
```

**关键参数（FixpipeMm1Dn）**：
```
nSize   = (actMSize + 31) >> 5 << 5   // M 对齐到 32（BF16 向量宽度！）
mSize   = actSingleLoopS2Size         // S2 天然是 16 的倍数，无需额外对齐
dualDstCtl = 2                        // 按 N(=M) 维切半
```

### 3.3 Vec1（softmax）：列方向即为 S2 维，向量操作天然连续

```
Sub-block 0 中 UB 的内存视图（sub0 处理 q0..q63）：

逻辑视图：[S2 行 × M/2 列] = [128 × 64]

─── 内存地址增大方向 ───────────────────────────────────────────►
| s0的q0 | s0的q1 | ... | s0的q63 | s1的q0 | s1的q1 | ... | s1的q63 | ...
└───── s0 这行包含所有 q0..q63 的分数（64 个，一个向量块）─────┘

softmax 沿列方向计算（所有 Key token 对每个 Query 的分数）。
向量操作一次处理 S2=128 行 × actMSizeAlign32/2=64 列 → 并行处理所有 query！
```

> ✅ **优势**：64 = 2×32，完美对齐 BF16 向量宽度，**一次向量调用覆盖所有 M/2 个 query token**。

### 3.4 Vec1 后 DataCopy UB→L1（为 BMM2 准备 P 矩阵）

```
UB（softmax 后，[S2 × M/2]）→ L1（P 矩阵）

DataCopy 参数：
  blockCount = mBaseSize / 32 = 4   （128/32 = 4 组）
  blockLen   = s2BaseSize / 2 = 64  （每组 64 个连续元素）
  srcStride  = 1                     （每组后 1 个元素间隔，对应 32 行的布局）
  dstStride  = actSingleLoopS2SizeAlign - 64

内存访问示意：
源（UB）: [64][1][64][1][64][1][64] ...  ← 4 组 64 个，高度连续
                                            硬件可合并为大 burst！
```

> ✅ **优势**：blockLen=64 elements = 128B，每 burst 传输 128B，DMA 效率远高于 ND 的 srcStride=1 碎片访问。

---

## 四、ND vs DN 核心差异一览表

```
                        ND 格式                     DN 格式
                ─────────────────────────  ───────────────────────────
BMM1 左矩阵      Q [M × D]                  K [S2 × D]
BMM1 右矩阵      K [D × S2]                 Q [D × M]
L0C 结果形状     Score [M × S2]             ScoreT [S2 × M]
Fixpipe M对齐    向 2 对齐（最小）            向 32 对齐（BF16向量宽度）
Fixpipe 切分轴   按 M 切半（行切分）          按 N(M) 切半（列切分）
UB 中每行含义    1 个 query 对所有 key 的分数  S2 个 key 对 M/2 个 query 的分数
Vec1 并行度      每次处理 1 个 query token    一次处理全部 M/2 个 query token
UB→L1 srcStride  1 element 间隔（碎片）      紧密的 64-element 块（高效）
BMM2 mSize       actMSize（动态，每tile变）    mBaseSize=128（编译期常量！）
是否有 mask 开销  是（hasAtten 时有）          否（IsDn 要求 hasAtten=false）
```

---

## 五、性能差异的根本原因

### 5.1 M 的对齐粒度（最核心）

BF16 向量指令每次处理 **32 个元素**（64B）。

```
举例：actMSize = 48（尾块，非 128 的倍数）

ND 模式：
  mSize = (48 + 1) >> 1 << 1 = 48（对齐到 2）
  向量操作每次 32 个，处理 48 个需要：
    第 1 组 [0..31]  → 正常处理
    第 2 组 [32..47] → 只有 16 个有效，需要 mask 掉后 16 个！
  额外 mask 指令 = 一轮 Select/Muls/Add 来清零无效元素。

DN 模式：
  nSize = (48 + 31) >> 5 << 5 = 64（对齐到 32）
  向量操作每次 32 个，处理 64 个需要：
    第 1 组 [0..31]  → 正常
    第 2 组 [32..63] → 正常（16 个有效 + 16 个 padding，padding 在后续不参与输出）
  无需 mask，补零即可。
```

### 5.2 softmax 向量并行度（图示）

```
ND 模式 Vec1（subBlock 处理 M/2=64 个 query）：

time→  t1          t2          t3    ...  t64
       softmax(q0) softmax(q1) softmax(q2) ... softmax(q63)
       [S2=128元素] [S2=128元素] ...
       ↓           ↓
       串行，每次调用消耗 1 query 的向量计算资源

DN 模式 Vec1（subBlock 处理 M/2=64 个 query）：

time→  t1
       softmax(q0..q63) 同时计算！
       [S2 × M/2 = 128×64 元素，向量一次性铺满]
       ↓
       并行，M/2 个 query 共享一次大向量操作
```

### 5.3 BMM2 mSize 固定化的意义

```
ND 模式 FixpipeMm2：
  fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;  // 每 tile 动态计算
  // actMSize 在尾 tile 可能是 1~127 的任意值
  // → 硬件每次 tile 都要重新配置 Fixpipe 参数，增加调度开销

DN 模式 FixpipeMm2：
  fixpipeParams.mSize = mBaseSize;  // 始终 = 128，编译期常量
  // 原因：BMM1 DN 的结果已经是 [S2 × M_align32]，softmax 输出同形状
  //       所以 BMM2 的左矩阵（P^T）m 维始终满 128，无尾块问题
  // → Fixpipe 参数恒定，硬件调度更规律
```

---

## 六、IsDn 使能条件（代码路径）

```cpp
// flash_attn_entry_regbase.h:73
constexpr bool useDn = BaseApi::IsDn(
    false,              // isFp32         → false（BF16/FP16 才能用 DN）
    false,              // isValidFp8
    PSE_NONE_TYPE,      // pseMode        → 无 PSE
    hasAttenMask,       // hasAtten       → false（有 mask 则禁用！）
    false,              // hasDrop
    s1TemplateType==64, // isS1Base64     → false（S1=64 时禁用）
    dTemplateType,      //                → D <= 256
    false,              // hasRope        → 无 RoPE
    ...
);
```

```
IsDn = true  要求同时满足：
  ✓ 数据类型 = BF16 或 FP16（不是 FP32）
  ✓ hasAttenMask = false（无 attention mask）
  ✓ s1BaseSize = 128（不是 64）
  ✓ dSize <= 256
  ✓ 无 RoPE
  ✓ 无 PSE / Dropout
```

| 典型 case | useDn | 说明 |
|-----------|-------|------|
| BF16 decode，无 mask，BNSD | ✅ true | 走 DN 全路径 |
| BF16 prefill，有 causal mask | ❌ false | mask 存在 → 强制 ND |
| TND_MIXED_02（有 mask） | ❌ false | 用户测试场景 → ND |
| FP32 任意场景 | ❌ false | FP32 不支持 DN |
| BF16 + RoPE | ❌ false | RoPE 禁用 DN |

---

## 七、总结：一句话概括 ND vs DN

```
ND：Q 作矩阵乘左矩阵，结果 [M×S2] 存 UB，
    softmax 逐行（每行=1个query），M 对齐到 2，
    向量操作需 mask 尾块，DMA 访存有碎片。

DN：K 作矩阵乘左矩阵（Q 作右矩阵），结果 [S2×M] 存 UB，
    softmax 按列并行（每列=1个query），M 对齐到 32，
    向量无 mask、DMA 高效、BMM2 参数恒定，
    但仅限无 mask/PSE/RoPE 的推理场景。
```

---

*文档生成时间：2026-05-22*  
*覆盖文件：`flash_attn_block_cube_noquant_gqa.h`（IterateBmm1Nd/Dn, FixpipeMm1/Dn, FixpipeMm2）*  
*         `flash_attn_block_vec_noquant_gqa.h`（ProcessVec1Nd, ProcessVec1Dn）*  
*         `flash_attention_score_common_regbase.h`（IsDn 判断逻辑）*
