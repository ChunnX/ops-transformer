# 13 新芯片 Bank 冲突解决方案

## 1. 新芯片 Bank 架构 vs 950

### 1.1 950 (arch35) Bank 布局

```
shape:  (bank_depth=512, banks=2, bank_groups=8, block=32)
stride: (512B, 256B, 32B, 1B)

宏定义:
UB_BLOCK = 32           // Bank block = 32B
UB_BANK_GROUPS = 8      // 8 个 bank group
UB_BANKS = 2            // 每个 group 有 2 个 bank
UB_BANK_DEPTH = 512     // 256KB / 512B = 512 行
UB_BANK_STRIDE = 256    // 同一 bank、下一 group 的 stride = 256B
UB_BANK_DEPTH_STRIDE = 512 // 一整行（2 banks × 8 groups × 32B）= 512B
```

关键语义：`UB_BANK_STRIDE=256B` = 跳转到**同一 bank 索引、下一个 bank group**。

### 1.2 新芯片 Bank 布局

```
核心参数:
- 8 个 bank group，每个 group 有 4 个 bank
- 每个 bank 32B，包含 2 个 sub-bank × 16B（950 是 8B sub-bank）
- 每 1KB 地址范围内：bank0=[0,255B], bank1=[256,511B], bank2=[512,767B], bank3=[768,1023B]

推导:
- 每行（一个 bank group 的完整 stride）= 4 banks × 256B = 1024B = 1KB
- Vector UB = 384KB → bank_depth = 384KB / 1KB = 384
```

**新宏定义：**

```cpp
#define UB_BLOCK              32    // Bank block = 32B（不变）
#define UB_BANK_GROUPS        8     // 8 个 bank group（不变）
#define UB_BANKS              4     // 每组 4 个 bank（950 是 2）
#define UB_BANK_DEPTH         256   // 256KB / 1024B = 256 行

#define UB_BANK_STRIDE        256   // 一个 bank 宽度 = 256B（值不变，语义变）
#define UB_BANK_GROUP_STRIDE  1024  // 一个 bank group 宽度 = 4 × 256B
#define UB_BANK_DEPTH_STRIDE  1024  // 完整深度行 stride = 4 banks × 8 groups × 32B = 1024B
```

### 1.3 关键语义变化

| 宏 | 950 值 | 新芯片值 | 950 语义 | 新芯片语义 |
|----|--------|----------|----------|-----------|
| `UB_BANK_STRIDE` | 256 | **256** | 下一 group、同一 bank | 下一 bank、同一 group |
| `UB_BANK_DEPTH_STRIDE` | 512 | **1024** | 1 depth 行 | 1 depth 行 |
| `UB_BANKS` | 2 | **4** | 每 group 2 bank | 每 group 4 bank |

**核心变化**：`UB_BANK_STRIDE=256B` 在新芯片上不再是"同一 bank 跨 group"，而是"**同一 group 内跨 bank**"。这意味着用 UB_BANK_STRIDE 做 ping-pong 偏移时，两个 ping-pong buffer 自然落在 **不同 bank** 上 → 消除了同一 buffer 内部的 bank 冲突。

---

## 2. Buffer 冲突分析与 Bank 分配策略

### 2.1 同时访问的 Buffer 对（MXFP8 场景）

在 `ProcessVec1` → `BatchMulWeightAndReduceSum_MxFP8` 内层循环中，同时访问：

| Buffer | 操作 | 每行大小 | 行 stride |
|--------|------|----------|-----------|
| **QK (resMm1)** | 读 (LoadAlign) | 128 floats = 512B (2行×256B) | `s2BaseSize × sizeof(QK_T)` = 256B |
| **Weight** | 读 (BroadcastLane) | 64 floats = 256B | `gSize × sizeof(W_T)` = 256B |
| **vec1Out** | 写 (StoreAlign) | 128 uint16_t = 256B | `s2BaseSize × sizeof(SCORE_T)` = 256B |

### 2.2 冲突条件

在新芯片上，两个地址 A、B 映射到同一 bank 当且仅当：
```
(A / 256) % 4 == (B / 256) % 4  且  (A / 1024) % 8 == (B / 1024) % 8
```

即：同一 bank group 内的同一 bank index。

### 2.3 逐 Buffer Bank 分配方案

为每个 buffer 指定目标 bank index，通过额外偏移量确保同时访问的 buffer 落在不同 bank：

| Buffer | 目标 bank | 额外偏移 | 说明 |
|--------|-----------|----------|------|
| **resMm1 (QK)** | Bank 0 | 0 | 基准，不做偏移 |
| **weight** | Bank 1 | +256B | 与 QK 错开 1 个 bank |
| **vec1Out** | Bank 2 | +512B | 与 QK、weight 均错开 |
| **weightTemp** | Bank 3 | +768B | 重用 weight bank，与 vec1 顺序访问不冲突 |

**Ping-pong 自然消解**：由于 `UB_BANK_STRIDE=256B` 在新芯片上等于"同一 group 内下一 bank"，ping-pong 偏移（pingpong × 256B）使得 ping=0 和 pong=1 落在相邻 bank 上。对于 4-bank 系统，这天然避免了同一 buffer 的 ping-pong 冲突。

### 2.4 多 bank 轮转分析

以 pingpong=0 为例，QK 行 0 在 bank 0，并发访问的 weight 行 0 在 bank 1，vec1Out 行 0 在 bank 2：

```
QK[0*256B]    → bank 0, group X
Weight[256B]  → bank 1, group X   ← 不同 bank，无冲突 ✓
Vec1Out[512B] → bank 2, group X   ← 不同 bank，无冲突 ✓
```

当 pingpong=1 时：
```
QK[1*256B]    → bank 1, group X   （QK 内部 ping-pong 偏移）
Weight[1*256B + 256B] → bank 2, group X  ← 不同 bank ✓
Vec1Out[1*256B + 512B] → bank 3, group X ← 不同 bank ✓
```

与 QK 的 read 和 weight 的 read 之间：QK 在 bank 1，weight 在 bank 2 → 不冲突 ✓

---

## 3. 代码修改方案

### 3.1 修改文件清单

| 文件 | 修改项 |
|------|--------|
| `op_kernel/arch35/quant_lightning_indexer_v2_common.h` | 更新 bank 宏定义 |
| `op_kernel/arch35/quant_lightning_indexer_v2_service_vector.h` | 修改 buffer 分配和访问 stride |
| `op_kernel/arch35/quant_lightning_indexer_v2_service_cube.h` | 调整 Fixpipe dstStride |
| `op_kernel/arch35/vf/quant_lightning_indexer_v2_vector1.h` | 更新 row stride 注释，确认无隐式依赖 |

### 3.2 common.h 修改

```cpp
// === 修改前（950）===
// #define UB_BLOCK              32
// #define UB_BANK_GROUPS        8
// #define UB_BANKS              2
// #define UB_BANK_DEPTH         512
// #define UB_BANK_GROUP_STRIDE  UB_BLOCK                                   // 32B
// #define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B
// #define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B

// === 修改后（新芯片）===
// 新芯片 bank 布局:
//   8 bank groups × 4 banks/group × 32B/bank, 每 1KB bank0=0~255B, bank1=256~511B, bank2=512~767B, bank3=768~1023B
//   sub-bank: 16B（950 为 8B）
//   stride: depth=1024B, bank=256B, group=1024B, block=1B
//   注意: UB_BANK_DEPTH_STRIDE 从 512B 翻倍到 1024B，需确认 UB 空间充足
#define UB_BLOCK              32
#define UB_BANK_GROUPS        8
#define UB_BANKS              4      // 每组 4 bank（原 2）
#define UB_BANK_DEPTH         384   // 384KB / 1024B = 384（原 512）

#define UB_BANK_STRIDE        256    // 一个 bank 宽度 = 256B（值不变，语义变为"下一 bank 同 group"）
#define UB_BANK_GROUP_STRIDE  1024   // 一个 bank group 宽度（新增用于明确语义）
#define UB_BANK_DEPTH_STRIDE  1024   // 完整深度行 stride（原 512）

// Buffer bank 分配偏移（用于确保不同 buffer 落在不同 bank）
#define QLIV2_BANK_OFFSET_W      256    // weight 偏移到 bank 1
#define QLIV2_BANK_OFFSET_OUT    512    // vec1Out 偏移到 bank 2
#define QLIV2_BANK_OFFSET_WTEMP  768    // weightTemp 偏移到 bank 3
```

### 3.3 service_vector.h 修改（InitBuffers）

当前所有带 bank padding 的 buffer 使用 `UB_BANK_DEPTH_STRIDE` 做行 stride。由于新芯片 `UB_BANK_DEPTH_STRIDE = 1024`（翻倍），buffer 大小也会翻倍。需要评估 UB 空间是否充足。

**方案 A：保持 UB_BANK_DEPTH_STRIDE 作为行 stride（简单，空间换时间）**

```cpp
// weightBuf_ 原: 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE
// = 2 * 4 * 512 = 4KB
// 改后: 2 * 4 * 1024 = 8KB（多 4KB）
pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE
                 + QLIV2_BANK_OFFSET_W);  // 额外 256B 实现 bank 偏移
weightUB_ = weightBuf_.Get<W_T>()[QLIV2_BANK_OFFSET_W / sizeof(W_T)];

// weightTempBuf_ 同理
pipe->InitBuffer(weightTempBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE
                 + QLIV2_BANK_OFFSET_WTEMP);
weightTempUB_ = weightTempBuf_.Get<float>()[QLIV2_BANK_OFFSET_WTEMP / sizeof(float)];

// qScaleBuf_（非 MXFP8）
pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
qScaleUB_ = qScaleBuf_.Get<SCALE_T>();

// outBuf_: 原本无 bank padding（紧凑），需要加偏移
// 原: 2 * CeilDiv(s1BaseSize_, CVRATIO) * s2BaseSize_ * sizeof(uint16_t)
//    = 2 * 4 * 128 * 2 = 2KB
// 改后增加 QLIV2_BANK_OFFSET_OUT 额外空间
pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * s2BaseSize_ * sizeof(uint16_t)
                 + QLIV2_BANK_OFFSET_OUT);
vec1OutUB_ = outBuf_.Get<uint16_t>()[QLIV2_BANK_OFFSET_OUT / sizeof(uint16_t)];
```

**方案 B：使用紧凑 layout + bank 偏移（更省空间，需更多改动）**

如果 UB 空间紧张，可以将所有 buffer 改为紧凑分配（按实际数据大小），只加 bank 偏移：

```cpp
// weight: 紧凑 = 2 * 4 * 64 * sizeof(float) = 2KB
pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * gSize_ * sizeof(W_T)
                 + QLIV2_BANK_OFFSET_W);
weightUB_ = weightBuf_.Get<W_T>()[QLIV2_BANK_OFFSET_W / sizeof(W_T)];
```

**但此方案需要同步修改所有 stride 参数**（当前用 `UB_BANK_DEPTH_STRIDE / sizeof(T)` 做行 stride，紧凑后需改为 `gSize_` 或 `s2BaseSize_`）。

**推荐方案 A**（最小改动，先验证正确性，后续再优化空间）。

### 3.4 service_vector.h 修改（访问 stride）

当前代码中 stride 参数：
```cpp
// ProcessVec1 (MXFP8 分支)
auto qkVLstride = QLIV2Common::Align((uint64_t)info.actMBaseSize, 2 * constInfo_.gSize)
    * (UB_BANK_DEPTH_STRIDE / sizeof(QK_T));

vector1::BatchMulWeightAndReduceSum_MxFP8(
    outBase, UB_BANK_DEPTH_STRIDE / sizeof(uint16_t),          // outStride: 512→1024
    qkBase, qkVLstride,                                         // qkVLStride: 自动变化
    (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),  // qkStride: 64*256→64*512
    weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T),             // weightStride: 128→256
    weightTempBase,
    gSize_, curAivS1ProcNum);
```

关键影响：`qkStride = gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)`。
- 原：64 × 512/4 = 64 × 128 = 8192 floats
- 新：64 × 1024/4 = 64 × 256 = 16384 floats

`qkStride` 用于 `BatchMulWeightAndReduceSum_MxFP8` 中按 S1=2 分批跳转：
```cpp
auto curQk = qk + p * 2 * qkStride;   // 跳到下一对 S1 行
auto curQk1 = curQk + qkStride;        // 跳到同对内的第 2 行
```

由于 `qkStride` 已经包含 `UB_BANK_DEPTH_STRIDE`，自动适配新值后内部跳转仍然正确。

**同理**，`outStride` 和 `weightStride` 自动适配新值。

**DataCopy 的 srcStride 也需要更新：**
```cpp
// 原: copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
//    = (512 - 256) / 32 = 8
// 新: (1024 - 256) / 32 = 24
copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
```

同样：
```cpp
// qwDataCopyExtParams.dstStride 原: (512 - 256) / 32 = 8 → 新: 24
qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
```

### 3.5 service_cube.h 修改（Fixpipe）

```cpp
// Fixpipe dstStride: 确保每行落在同一 bank
// 原: fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T);
//    = 512 / 4 = 128 (float) 或 512 / 2 = 256 (bfloat16)
// 新: = 1024 / 4 = 256 (float) 或 1024 / 2 = 512 (bfloat16)

// 对于 bfloat16 QK_T（MXFP8 场景）：
fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T);  // 自动变为 512 元素
```

**注意**：Fixpipe 输出每行 s2BaseSize=128 元素，dstStride=512 意味着只有 128/512=25% 利用率，浪费 384 个元素/行 × 2B = 768B/行。对于 mBaseSize=4×G=256 行 → 浪费约 192KB，UB 总容量 256KB，**不可接受！**

**解决方案：使用紧凑 stride + 接受跨 bank 分布。**

由于新芯片上 256B 正好是一个 bank 宽度，用紧凑 stride（128 元素 = 256B）时，连续行自动落在 bank 0,1,2,3,0,1,2,3... 上。Vector1 核按行读取时，相邻行在不同 bank → **天然无冲突**。

```cpp
// === 修改后：紧凑 Fixpipe stride ===
if constexpr (std::is_same_v<QK_T, float>) {
    // float: 每行 128 floats = 512B，跨越 2 个 bank
    // 分行方案：每行 64 floats = 256B（对半拆），ndNum=2
    fixpipeParams.dstStride = s2BaseSize_ / 2;  // 64 元素 = 256B
    fixpipeParams.params.ndNum = CeilDiv(s2L0RealSize, s2BaseSize_ / 2);
    // ... 调整 ndNdStride
} else {
    // bfloat16: 每行 128 bf16 = 256B，正好一个 bank
    fixpipeParams.dstStride = s2BaseSize_;  // 128 元素 = 256B（紧凑，不浪费）
    fixpipeParams.params.ndNum = 1;
    // ndStride = 0（单 ND 块）
}

// 对应的 Vector1 读取 stride 也需改为紧凑值
```

**对应的 Vector1 端修改：**

```cpp
// qkVLstride: 原来 = Align(actMBaseSize, 2*gSize) * (UB_BANK_DEPTH_STRIDE / sizeof(QK_T))
// 紧凑版本:
auto qkVLstride = QLIV2Common::Align((uint64_t)info.actMBaseSize, 2 * constInfo_.gSize)
    * s2BaseSize_;  // 改为 s2BaseSize_（=128 元素）而非 UB_BANK_DEPTH_STRIDE/sizeof

// qkStride:
uint32_t qkStride = gSize_ * s2BaseSize_;  // 改为 gSize_ * s2BaseSize_ 而非 gSize_ * UB_BANK_DEPTH_STRIDE/sizeof

// outStride: 原 = UB_BANK_DEPTH_STRIDE / sizeof(uint16_t) = 512
// 紧凑: 128
uint32_t outStride = s2BaseSize_;  // 紧凑

// weightStride: 原 = UB_BANK_DEPTH_STRIDE / sizeof(W_T) = 512/4 = 128
// 紧凑: gSize_ = 64
uint32_t weightStride = gSize_;  // 紧凑
```

### 3.6 vector1.h 修改

`MulWeightAndReduceSum_MxFP8` 中 QK 加载：
```cpp
// 原注释: // RowStride是128, 行都落在一个bank上
// 实际 stride = 128 floats × 4B = 512B（float 版本）
// 新芯片紧凑布局：stride = 128 floats × 4B = 512B（跨 2 个 bank）

// bfloat16 版本：stride = 128 bf16 × 2B = 256B（跨 1 个 bank）
// 原注释: // RowStride是256, 行都落在一个bank上
// 新芯片注释: // RowStride是256(元素), 行跨bank但相邻行在不同bank, 无冲突
```

`MulWeightAndReduceSum2_MxFP8` 中：
```cpp
// qkStride 为紧凑值 gSize_ * s2BaseSize_ = 64 * 128 = 8192 元素
// 对应 8192 * 4B = 32KB 或 8192 * 2B = 16KB
```

---

## 4. 子 Bank (sub-bank) 16B 对齐

### 4.1 影响

新芯片每个 bank 有 2 个 16B sub-bank（950 是 4 个 8B sub-bank）。两个 32B 访问如果落在同一 sub-bank 会冲突。

在当前代码中，所有 LoadAlign/StoreAlign 都是 32B 对齐的（`128 * sizeof(float) = 512B = 16×32B`），且使用 128 元素粒度，天然避免了 sub-bank 冲突：

```
qk + 128*i        → 偏移 512B*i → 始终 512B 对齐 → 同一 sub-bank 内无冲突
qk + 128*i + 128  → 偏移 512B*i + 512B → bank 1, sub-bank 0
```

由于 16B sub-bank 的粒度加大（原来是 8B），实际上对 32B 对齐的访问**更友好**——两个 32B Load 更容易落在不同 sub-bank。

### 4.2 结论

当前代码已满足 16B sub-bank 对齐要求，**无需额外修改**。

---

## 5. 两种方案对比与选择

### 5.1 对比总表

| 维度 | 方案 A：保持 UB_BANK_DEPTH_STRIDE=1024B | 方案 B：紧凑 + 偏移 |
|------|----------------------------------------|---------------------|
| **Vector UB 占用 (MXFP8)** | 195.75 KB (51.0%) | **183.75 KB (47.9%)** |
| **Vector UB 占用 (非MXFP8, float QK)** | 347.75 KB (90.6%) | **329.75 KB (85.9%)** |
| **Fixpipe 效率** | 25% (128/512 元素利用率) | **100%** (128/128) |
| **代码改动量** | 小：改宏 + 加 bank 偏移 | 中：需改所有 stride 参数 |
| **正确性风险** | 低（stride 语义与旧代码一致） | 中（需仔细验证所有 stride） |
| **非 MXFP8 余量** | 仅 ~36KB（风险较高） | ~54KB（更安全） |

### 5.2 结论与推荐

**两方案均不超 384KB**，但：

- 方案 A 在非 MXFP8 场景占用 **90.6%**，余量仅 ~36KB，考虑编译器栈/临时变量后存在风险
- 方案 B 更省空间（省 ~18KB），Fixpipe 无浪费，非 MXFP8 余量 ~54KB 更安全

**推荐采用方案 B（紧凑 + 偏移）**，理由：
1. 非 MXFP8 场景余量更充足（86% vs 91%）
2. Fixpipe 100% 利用率，无浪费
3. 从 384KB 角度看，紧凑方案才是真正匹配新芯片带宽的最优解

---

## 6. 完整修改对照表

### 6.1 common.h

```diff
- #define UB_BANKS              2
- #define UB_BANK_DEPTH         512
- #define UB_BANK_DEPTH_STRIDE  (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)    // 512B
+ #define UB_BANKS              4
+ #define UB_BANK_DEPTH         256
+ #define UB_BANK_GROUP_STRIDE  1024
+ #define UB_BANK_DEPTH_STRIDE  1024
+
+ // Buffer bank 分配偏移
+ #define QLIV2_BANK_OFFSET_W      256
+ #define QLIV2_BANK_OFFSET_OUT    512
+ #define QLIV2_BANK_OFFSET_WTEMP  768
```

### 6.2 service_vector.h InitBuffers

```diff
- pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
- weightUB_ = weightBuf_.Get<W_T>();
+ pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * gSize_ * sizeof(W_T) + QLIV2_BANK_OFFSET_W);
+ weightUB_ = weightBuf_.Get<W_T>()[QLIV2_BANK_OFFSET_W / sizeof(W_T)];

- pipe->InitBuffer(weightTempBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
- weightTempUB_ = weightTempBuf_.Get<float>();
+ pipe->InitBuffer(weightTempBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * gSize_ * sizeof(float) + QLIV2_BANK_OFFSET_WTEMP);
+ weightTempUB_ = weightTempBuf_.Get<float>()[QLIV2_BANK_OFFSET_WTEMP / sizeof(float)];

- pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * s2BaseSize_ * sizeof(uint16_t));
- vec1OutUB_ = outBuf_.Get<uint16_t>();
+ pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * s2BaseSize_ * sizeof(uint16_t) + QLIV2_BANK_OFFSET_OUT);
+ vec1OutUB_ = outBuf_.Get<uint16_t>()[QLIV2_BANK_OFFSET_OUT / sizeof(uint16_t)];
```

### 6.3 service_vector.h ProcessVec1

```diff
- auto qkVLstride = QLIV2Common::Align((uint64_t)info.actMBaseSize, 2 * constInfo_.gSize)
-     * (UB_BANK_DEPTH_STRIDE / sizeof(QK_T));
+ auto qkVLstride = QLIV2Common::Align((uint64_t)info.actMBaseSize, 2 * constInfo_.gSize)
+     * s2BaseSize_;

+ // MXFP8 分支 stride 用紧凑值
+ uint32_t qkStrideCompact = gSize_ * s2BaseSize_;
+ uint32_t weightStrideCompact = gSize_;        // 64 floats = 256B
+ uint32_t outStrideCompact = s2BaseSize_;       // 128 uint16 = 256B

  vector1::BatchMulWeightAndReduceSum_MxFP8(outBase, outStrideCompact,
                                              qkBase, qkVLstride,
-                                             (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
+                                             qkStrideCompact,
-                                             weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T),
+                                             weightBase, weightStrideCompact,
                                              weightTempBase,
                                              gSize_, curAivS1ProcNum);
```

### 6.4 DataCopy strides

```diff
  // weight/qScale 加载
- qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
+ qwDataCopyExtParams.dstStride = 0;  // 紧凑，无需填补

  // vec1Out 写出
- copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
+ copyOutParams.srcStride = 0;  // 紧凑
```

### 6.5 service_cube.h Fixpipe

```diff
  // bfloat16 QK_T 分支 (MXFP8)
- fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T);
+ fixpipeParams.dstStride = s2BaseSize_;  // 128 元素 = 256B，紧凑无填充
```

---

## 7. UB 空间预算验证（Vector 核 384KB）

### 7.1 参数设定

典型 MXFP8 场景：`CVRATIO=1, s1BaseSize=4, s2BaseSize=128, gSize=64, topkCount=2048, trunkLen=16384`

```cpp
// mBaseSize = s1BaseSize × gSize = 4 × 64 = 256
// CeilDiv(mBaseSize, CVRATIO) = 256   (CVRATIO=1)
// CeilDiv(s1BaseSize_, CVRATIO) = 4
```

### 7.2 Vector 核完整 Buffer 清单（逐行 InitBuffer）

| # | Buffer | 旧大小 (950, 512B stride) | 方案 B: 紧凑+偏移 | 方案 A: 1024B stride |
|---|--------|--------------------------|-------------------|---------------------|
| 1 | **resMm1Buf_** (QK, bf16) | `2×256×128×2` = **128.00 KB** | 同左 = **128.00 KB** | 同左 = **128.00 KB** |
| 2 | **weightBuf_** (W_T=float) | `2×4×512+0` = **4.00 KB** | `2×4×64×4+256` = **2.25 KB** | `2×4×1024+256` = **8.25 KB** |
| 3 | **weightTempBuf_** (float) | `2×4×512+0` = **4.00 KB** | `2×4×64×4+768` = **2.75 KB** | `2×4×1024+768` = **8.75 KB** |
| 4 | **outBuf_** (uint16) | `2×4×128×2` = **2.00 KB** | `2×4×128×2+512` = **2.50 KB** | 同左 = **2.50 KB** |
| 5 | **mrgValueBuf_** (uint16) | `(2048+16384)×2` = **36.00 KB** | 同左 = **36.00 KB** | 同左 = **36.00 KB** |
| 6 | **indicesOutBuf_** (uint32) | `(2048+64)×4` = **8.25 KB** | 同左 = **8.25 KB** | 同左 = **8.25 KB** |
| 7 | **valueOutBuf_** (bf16) | `2048×2` = **4.00 KB** | 同左 = **4.00 KB** | 同左 = **4.00 KB** |
| | **MXFP8 总计** | **186.25 KB** | **183.75 KB** ✅ | **195.75 KB** ✅ |
| | **占 384KB 比例** | **48.5%** | **47.9%** | **51.0%** |

> 注：`scoreOutLocal_` 和 `topkSharedTmpLocal_` 复用 `resMm1UB_` 的空间（第 188-203 行），不计入额外分配。

### 7.3 非 MXFP8 场景（float QK_T，含 qScale/kScale）

非 MXFP8 场景下 `QK_T=float`，resMm1Buf_ 翻倍，且额外需要 qScaleBuf_ 和 kScaleBuf_：

| # | Buffer | 方案 B: 紧凑+偏移 | 方案 A: 1024B stride |
|---|--------|-------------------|---------------------|
| 1 | **resMm1Buf_** (QK, float) | `2×256×128×4` = **256.00 KB** | 同左 = **256.00 KB** |
| 2-7 | 同上表 MXFP8 buffers | **55.75 KB** | **67.75 KB** |
| 8 | **kScaleBuf_** (float) | `2×128×16×4` = **16.00 KB** | 同左 = **16.00 KB** |
| 9 | **qScaleBuf_** (float) | `2×4×64×4` = **2.00 KB** | `2×4×1024` = **8.00 KB** |
| | **非 MXFP8 总计** | **329.75 KB** ✅ | **347.75 KB** ✅ |
| | **占 384KB 比例** | **85.9%** | **90.6%** |

### 7.4 极端场景：topkCount=4096, s1BaseSize=3

| 参数 | 影响 |
|------|------|
| s1BaseSize=3 | mBaseSize=192 → resMm1Buf_ = `2×192×128×4` = **192KB** (float), **96KB** (bf16) |
| topkCount=4096 | mrgValueBuf_ → (4096+16384)×2 = **40KB** (+4KB); indicesOutBuf_ → (4096+64)×4 = **16.25KB** (+8KB) |

非 MXFP8 极端：192 + 55.75 + 16 + 2 + 4 + 8 = **277.75 KB** → **72.3%** ✅
MXFP8 极端：96 + 55.75 + 4 + 8 = **163.75 KB** → **42.6%** ✅

### 7.5 结论

**两种方案均不会超出 384KB Vector UB**：

- **方案 B（紧凑+偏移）**：MXFP8 仅用 ~184KB (48%)，非 MXFP8 用 ~330KB (86%)，极端场景 ~278KB (72%)
- **方案 A（UB_BANK_DEPTH_STRIDE=1024B）**：MXFP8 用 ~196KB (51%)，非 MXFP8 用 ~348KB (91%)

方案 A 在非 MXFP8 场景下达到 91%，虽不溢出但余量仅 ~36KB，考虑到编译器可能分配额外栈/临时空间，**推荐方案 B** 作为最终方案。

---

## 8. 验证要点

### 8.1 功能性验证

1. **单 batch 精度**：与非 bank 冲突版本的 golden 对比
2. **多 batch 精度**：连续多 batch 无精度漂移
3. **边界条件**：G=1/64、s2BaseSize 不满 128、topk 各种取值

### 8.2 Bank 冲突验证

1. **Profile 对比**：对比新旧芯片的 bank conflict stall cycles
2. **同 buffer ping-pong**：确认 pingpong=0/1 落在不同 bank
3. **跨 buffer**：确认 QK、weight、vec1Out 同一时刻在不同 bank

### 8.3 Fixpipe 验证

1. **float QK**：ndNum=2 时 dstNdStride 正确
2. **bfloat16 QK**：紧凑 stride 下 Vector1 读取正确
3. **ReLU 随路**：Fixpipe reluEn=true 仍正常工作

### 8.4 回归验证

1. **非 MXFP8 路径**：含 qScale/kScale 的流程不受影响
2. **arch22**：不受影响（无 bank 宏）
3. **V2 Metadata**：Metadata 输出不受影响

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Fixpipe 紧凑 stride 导致 QK 行跨 bank | Vector1 内层循环读不同 bank 的行可能冲突 | 内层循环同时读 2 行（i 和 i+1），2 行间隔 128 元素=512B（float）跨 2 个 bank，无冲突 |
| Bank 偏移导致 buffer 空间越界 | InitBuffer 申请空间不足 | 已在偏移量上额外增加空间 |
| DataCopyPad srcStride=0 不符合 DataCopyPad 语义 | MTE 搬运异常 | 确认 DataCopyPad 支持 srcStride=0（等价于 blockCount × blockLen 连续搬） |
| weightTemp 与 vec1Out Bank 冲突 | 交换 buffer 策略未覆盖 | weightTemp 紧跟 weight 使用，与 vec1Out 不并发，无冲突 |
