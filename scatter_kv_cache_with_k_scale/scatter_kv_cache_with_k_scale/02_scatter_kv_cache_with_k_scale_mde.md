# AscendC SIMT 算子详细设计文档

## 文档信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | scatter_kv_cache_with_k_scale |
| 文档版本 | v1.0 |
| 创建日期 | 2026-05-30 |
| 依赖需求文档 | 01_scatter_kv_cache_with_k_scale_se.md |
| 芯片类型 | ascend950 |

---

## 1. 设计概述

### 1.1 设计目标

本算子为**纯访存算子**，核心功能是将当前 token 的 Key/Value 张量按 `slot_mapping` 索引散列（scatter）写入 KV Cache 内存区域，同时写入每个 token 对应 head 的 Key Scale 值。无任何数值计算（加/减/乘/除/比较），所有数据以原始 bit-pattern 直接拷贝。

**算子分类**：搬运类（纯访存），类似 scatter 操作，但不涉及原子累加（不同 token 写入不同 slot，无数据竞争）。

**核心设计思路**：
- 使用 **SIMT 编程模型**，充分利用不规则访存（scatter 写入）和分支处理（slot_mapping 越界检查）的优势
- 直接访问 Global Memory（GM），无需 UB 缓冲中转
- 多核沿 token 维度切分，单核内线程沿 `(token, head)` 维度并行

### 1.2 关键设计决策

| 决策点 | 选择 | 原因 |
|-------|-----|-----|
| 编程模型 | SIMT | 1) 写入地址非连续（scatter），SIMD 不适用；2) 需要逐 token 越界分支判断；3) 竞品（vLLM）同样使用线程级并行（CUDA kernel） |
| 内存访问 | 直接访问GM | 纯访存算子，无需数据复用，直接 GM 读写比 UB 中转更高效；无计算密集型操作 |
| 线程数 | 2048（constexpr） | 搬运类算子建议使用最多线程数，最大化内存带宽利用率 |
| 切分维度 | 沿 token 维度多核切分 | token 之间无依赖，天然可并行；每个核处理一段 token 的所有 head |
| UintDiv 快除 | 使用 | VF 内固定除数除法（numHead, blockSize）必须使用快除，符合 op_mde_skill §3.8 规范 |

### 1.3 竞品精度关键点

从 SE 文档第 3 章"竞品实现分析"（vLLM reshape_and_cache CUDA kernel）中提取的精度关键点：

| 精度关键点 | 竞品实现方式 | 本算子 AscendC 实现对齐策略 |
|-----------|------------|--------------------------|
| 数据拷贝方式 | 原始 bit-pattern 直接拷贝，不做数值转换 | 相同。FP8 数据以 `uint8` 方式读写（位宽 1 字节），CPU/NPU 均不解释浮点值，bit-exact 一致 |
| 越界索引处理 | `if (slot < 0 \|\| slot >= max_slot) return;` 跳过该 token | 相同。VF 内检查 `slot < 0 \|\| slot >= maxSlot`，`continue` 跳过，不写入任何数据 |
| 累加方式 | 不使用 atomicAdd（不同 token 写入不同 slot，无竞争） | 相同。直接 GM 写入，无原子操作 |
| 确定性 | 确定性访存（无累加顺序依赖） | 相同。inplace 写入不同位置，无数据竞争，结果确定 |
| 精度需求 | 无精度损失（bit-exact 拷贝） | FP8 部分逐字节对比（uint8 bit-exact）；FP32 key_scale 部分最大相对误差 < 1e-6 |
| Inplace 行为 | 原地更新 KV Cache | 相同。key_cache/value_cache/key_scale_cache 输入输出同名，复用输入内存直接写入 |

### 1.4 精度风险与规避

| 风险点 | 影响 | 规避措施 |
|-------|-----|---------|
| 无计算操作 | 无浮点精度损失风险 | 纯访存拷贝，FP8 数据 bit-exact 一致 |
| 多线程并发写同一 slot | 理论上可能但实际不发生（不同 token 映射到不同 slot） | slot_mapping 是页表，由上层保证同一 token 映射唯一 slot；kernel 内不做额外同步 |
| FP32 key_scale 精度 | 跨平台可能有微小编码差异 | I/O 为 IEEE 754 标准 float32，按 memcpy 行为拷贝，精度可控在 1e-6 |

### 1.5 精度测试建议

- **对比基准**：NumPy Golden（uint8 模式）
- **精度标准**：
  - FP8（uint8 表示）数据：逐字节 bit-exact 一致
  - FP32 key_scale_cache 数据：最大相对误差 < 1e-6
- **重点测试场景**：
  - 基本功能：正常 slot_mapping 范围写入
  - 边界值：slot_mapping = 0（block 首位置）、slot_mapping = maxSlot - 1（最后位置）
  - 越界值：slot_mapping < 0、slot_mapping >= maxSlot（应跳过，不受影响区域保持原值）
  - 空 tensor：num_tokens = 0（SE 文档标记为不支持，但应不崩溃）
  - 特化场景：k_head_size == v_head_size
  - 泛化场景：k_head_size != v_head_size

---

## 2. 切分策略设计

### 2.1 多核切分策略

#### 核数计算公式

```
needCoreNum = min(maxCoreNum, ceil(numTokens / minTokensPerCore))
minTokensPerCore = 1
总核数受 numTokens 限制，token 之间无依赖，可完全并行
```

#### 切分参数说明

| 参数名 | 计算方式 | 说明 |
|-------|---------|-----|
| maxCoreNum | `GetCoreNumAiv()` 获取 | AI Core 最大可用核数 |
| needCoreNum | `min(maxCoreNum, numTokens)` | 最多每个核处理 1 个 token，最少使用 1 个核 |
| perCoreTokens | `CeilDiv(numTokens, needCoreNum)` | 每个核处理的 token 数量（最后一核可能更少） |
| tokenStart | `coreIdx * perCoreTokens` | 当前核起始 token 索引 |
| tokenEnd | `min(tokenStart + perCoreTokens, numTokens)` | 当前核结束 token 索引（不包含） |

#### 核间数据独立性

各核处理的 token 互不重叠，且写入的 cache slot 由 `slot_mapping` 索引决定，理论上不同核不会写入同一 cache 位置（上层保证不同 token 映射到不同 slot）。因此**无需核间同步（SyncAll）和 workspace**。

### 2.2 线程切分策略

#### 线程数设置

| 参数名 | 值 | 说明 |
|-------|-----|-----|
| THREAD_NUM | 2048 | 单核线程数，**编译期常量（constexpr）** |
| `__launch_bounds__` | THREAD_NUM | 与 THREAD_NUM 使用同一 constexpr 变量 |

> ⚠️ **关键约束**：
> - `THREAD_NUM` 定义为 `constexpr uint32_t THREAD_NUM = 2048;`，编译期确定
> - `__launch_bounds__(THREAD_NUM)` 和 `dim3(THREAD_NUM)` 必须使用同一个常量
> - **严禁将线程数放入 TilingData 结构体**，TilingData 只包含核数、各维度 shape 参数
> - 当 `numTokens` 较小导致线程空转时，通过调低 **核数**（`SetBlockDim`）来适配，保持单核线程数不变

#### 线程选择依据

- 搬运类算子（纯访存）建议使用最多线程数 2048，最大化 GM 带宽利用率
- 线程数越大 → 越多的并发内存请求 → 更好的内存延迟隐藏
- 2048 线程时每线程可用 16 个寄存器，本算子逻辑简单（少量局部变量），寄存器充足无溢出风险

### 2.3 线程→数据映射

每个线程处理一个 `(token, head)` 对。总工作量 = `numTokens × numHead`。线程使用 Grid-Stride 循环遍历所有工作项：

```
for idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx < perCoreTokens * numHead;
    idx += blockDim.x * gridDim.x
```

从线性索引导出 `(token, head)` 坐标使用 UintDiv 快除：
```
token = UintDiv(idx, magicNumHead, shiftNumHead)
head  = idx - token * numHead
```

---

## 3. Tiling 数据结构设计

### 3.1 Tiling 参数定义

```cpp
/**
 * @brief scatter_kv_cache_with_k_scale 算子的 Tiling 数据结构
 *
 * 所有参数在 Host 端（tiling 阶段）从输入 shape 提取，在 Device 端（kernel）只读使用。
 * ⚠️ 线程数（THREAD_NUM=2048）为编译期常量，禁止放入 TilingData。
 */
struct ScatterKvCacheWithKScaleTilingData {
    int64_t needCoreNum;      // 实际使用的核数
    int64_t perCoreTokens;    // 每核处理的 token 数
    int64_t numTokens;        // 总 token 数 (batch * seq_len)
    int64_t numHead;          // 注意力头数
    int64_t kHeadSize;        // Key 头维度
    int64_t vHeadSize;        // Value 头维度
    int64_t numBlocks;        // Cache 块数
    int64_t blockSize;        // 每块容纳的 token 数
    int64_t maxSlot;          // 最大合法 slot 值（= numBlocks * blockSize），用于越界检查
    int64_t magicNumHead;     // numHead 快除 magic number
    int64_t shiftNumHead;     // numHead 快除 shift
    int64_t magicBlockSize;   // blockSize 快除 magic number
    int64_t shiftBlockSize;   // blockSize 快除 shift
};
```

> 注意：`magicNumHead`/`shiftNumHead` 和 `magicBlockSize`/`shiftBlockSize` 在 Tiling 阶段通过 `GetUintDivMagicAndShift<int64_t>()` 预计算并写入 TilingData，kernel 侧直接使用。

### 3.2 Tiling 参数计算伪代码

```cpp
static ge::graphStatus ScatterKvCacheWithKScaleTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    GetPlatformInfo(context, ubSize, coreNum);

    // 2. 获取 shape 信息
    // 输入 0: key         [numTokens, numHead, kHeadSize]
    // 输入 1: value       [numTokens, numHead, vHeadSize]
    // 输入 4: slot_mapping [numTokens]
    auto keyShape = context->GetInputShape(0);
    int64_t numTokens  = keyShape->GetDim(0);
    int64_t numHead    = keyShape->GetDim(1);
    int64_t kHeadSize  = keyShape->GetDim(2);
    int64_t vHeadSize  = context->GetInputShape(1)->GetDim(2);

    // 输入 2: key_cache  [numBlocks, numHead, blockSize, kHeadSize]
    auto kcShape = context->GetInputShape(2);
    int64_t numBlocks = kcShape->GetDim(0);
    int64_t blockSize = kcShape->GetDim(2);

    // 3. Shape 约束校验（参见 SE 文档 §5.4）
    // ... (省略详细校验代码，详见 SE 文档)

    // 4. 核切分
    int64_t needCoreNum = std::min(coreNum, numTokens);
    int64_t perCoreTokens = CeilDiv(numTokens, needCoreNum);

    // 5. 填充 TilingData
    ScatterKvCacheWithKScaleTilingData tiling;
    tiling.needCoreNum  = needCoreNum;
    tiling.perCoreTokens = perCoreTokens;
    tiling.numTokens    = numTokens;
    tiling.numHead      = numHead;
    tiling.kHeadSize    = kHeadSize;
    tiling.vHeadSize    = vHeadSize;
    tiling.numBlocks    = numBlocks;
    tiling.blockSize    = blockSize;
    tiling.maxSlot      = numBlocks * blockSize;

    // 6. 预计算 UintDiv magic/shift
    GetUintDivMagicAndShift<int64_t>(tiling.magicNumHead, tiling.shiftNumHead, numHead);
    GetUintDivMagicAndShift<int64_t>(tiling.magicBlockSize, tiling.shiftBlockSize, blockSize);

    // 7. 写入 tiling 数据
    context->SetBlockDim(needCoreNum);

    // 8. 设置 DCache（UB 全部分配给 DCache，最大化 SIMT 数据缓存）
    //    UB = 256KB, 预留 8KB, DCache 最小 32KB
    //    实际 DCache = UB - 8KB - 动态UB(0) ≈ 248KB
    constexpr uint32_t DCACHE_SIZE = 32 * 1024; // 最小 32KB DCache
    constexpr uint32_t STATIC_UB_ESTIMATE = 0;
    auto res = context->SetLocalMemorySize(
        static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    // (res != ge::GRAPH_SUCCESS) → return GRAPH_FAILED;

    // 9. 设置 tiling key（场景模式，不编码 dtype）
    uint64_t tilingKey = 0;
    if (kHeadSize == vHeadSize) {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_SPECIALIZED);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_GENERALIZED);
    }
    context->SetTilingKey(tilingKey);

    // 10. Workspace: 无需 workspace（无核间数据交换）
    return GetWorkspaceSize(context); // userWorkspaceSize = 0
}
```

### 3.3 Tiling Key 设计

```cpp
// tiling_key.h
#define SCATTER_KV_CACHE_SCENE_SPECIALIZED 0  // k_head_size == v_head_size
#define SCATTER_KV_CACHE_SCENE_GENERALIZED  1  // k_head_size != v_head_size

ASCENDC_TPL_ARGS_DECL(
    ScatterKvCacheWithKScale,
    ASCENDC_TPL_UINT_DECL(schMode, 1,
        ASCENDC_TPL_UI_LIST,
        SCATTER_KV_CACHE_SCENE_SPECIALIZED,
        SCATTER_KV_CACHE_SCENE_GENERALIZED));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode,
        ASCENDC_TPL_UI_LIST,
        SCATTER_KV_CACHE_SCENE_SPECIALIZED,
        SCATTER_KV_CACHE_SCENE_GENERALIZED)));
```

> ⚠️ **TilingKey 仅编码场景模式（`k_head_size == v_head_size` 与否），不编码 dtype**。dtype 由 `DTYPE_KEY` 宏自动实例化。4 种 dtype 组合（2 种 FP8 类型 × 2 种 slot_mapping 类型）均由框架编译期自动展开，无需在 tiling key 中枚举。

---

## 4. 数据搬入搬出设计

### 4.1 内存访问模式

- [x] **直接访问 GM**（推荐，SIMT 纯访存场景）
- [ ] 使用 UB 缓冲（连续访存类）

**选择理由**：
- 本算子为纯访存算子，无数据复用需求
- SIMT 支持直接 GM 读写，编译器会利用 DCache 自动缓存最近访问的数据
- 使用 UB 缓冲会增加额外的搬入搬出指令，对于 scatter 写入场景反而不如直接 GM 访问高效
- 竞品 vLLM CUDA kernel 同样直接使用 global memory 进行 scatter 操作

### 4.2 GM 地址获取

在 Process 函数中直接转换 `GM_ADDR` 为类型化指针：

```cpp
template <typename T_FP8, typename T_SLOT>
__aicore__ inline void Process(
    GM_ADDR key, GM_ADDR value,
    GM_ADDR keyCache, GM_ADDR valueCache,
    GM_ADDR slotMapping, GM_ADDR keyScale, GM_ADDR keyScaleCache,
    const ScatterKvCacheWithKScaleTilingData* tilingData)
{
    // 直接转换为类型化 GM 指针
    __gm__ T_FP8* key_gm           = (__gm__ T_FP8*) key;
    __gm__ T_FP8* value_gm         = (__gm__ T_FP8*) value;
    __gm__ T_FP8* keyCache_gm      = (__gm__ T_FP8*) keyCache;
    __gm__ T_FP8* valueCache_gm    = (__gm__ T_FP8*) valueCache;
    __gm__ T_SLOT* slotMapping_gm  = (__gm__ T_SLOT*) slotMapping;
    __gm__ float* keyScale_gm      = (__gm__ float*) keyScale;
    __gm__ float* keyScaleCache_gm = (__gm__ float*) keyScaleCache;

    asc_vf_call<OpScatterKvCacheWithKScaleSimt<T_FP8, T_SLOT>>(
        dim3(THREAD_NUM),
        tilingData->needCoreNum,
        tilingData->perCoreTokens,
        tilingData->numHead,
        tilingData->kHeadSize,
        tilingData->vHeadSize,
        tilingData->numBlocks,
        tilingData->blockSize,
        tilingData->maxSlot,
        tilingData->magicNumHead,
        tilingData->shiftNumHead,
        tilingData->magicBlockSize,
        tilingData->shiftBlockSize,
        key_gm, value_gm, keyCache_gm, valueCache_gm,
        slotMapping_gm, keyScale_gm, keyScaleCache_gm);
}
```

### 4.3 DCache 设置

DCache 大小设置为最小要求 32KB，将剩余 UB 空间全部分配给动态 UB（但实际不使用 UB buffer）。这样 DCache 实际可用空间接近 248KB，最大化 SIMT 数据缓存命中率。

| 内存区域 | 大小 | 说明 |
|---------|-----|------|
| 静态 UB (__ubuf__) | 0 | 不声明静态 UB 数组 |
| 动态 UB (TBuf) | ~0 | 不分配 TBuf，所有数据直接 GM 访问 |
| 预留空间 | 8KB | 编译器预留 |
| DCache | ~248KB | 剩余全部用作 DCache，缓存 GM 数据 |

---

## 5. 计算逻辑设计

### 5.1 SIMT Kernel 函数签名

```cpp
constexpr uint32_t THREAD_NUM = 2048;

/**
 * @brief SIMT VF kernel: 将 key/value 按 slot_mapping 散列写入 KV Cache。
 *
 * @tparam T_FP8   FP8 数据类型（fp8_e5m2_t 或 fp8_e4m3fn_t），1 字节宽
 * @tparam T_SLOT  slot_mapping 索引类型（int32_t 或 int64_t）
 *
 * @param needCoreNum    实际使用的核数（gridDim.x）
 * @param perCoreTokens  每核处理的 token 数
 * @param numHead        注意力头数
 * @param kHeadSize      Key 头维度
 * @param vHeadSize      Value 头维度
 * @param numBlocks      Cache 块数
 * @param blockSize      每块容纳的 token 数
 * @param maxSlot        最大合法 slot 值（越界检查阈值）
 * @param magicNumHead   numHead 快除 magic
 * @param shiftNumHead   numHead 快除 shift
 * @param magicBlockSize blockSize 快除 magic
 * @param shiftBlockSize blockSize 快除 shift
 * @param key            [numTokens, numHead, kHeadSize] 输入 Key
 * @param value          [numTokens, numHead, vHeadSize] 输入 Value
 * @param keyCache       [num_blocks, num_head, block_size, k_head_size] KV Cache Key (inplace)
 * @param valueCache     [num_blocks, num_head, block_size, v_head_size] KV Cache Value (inplace)
 * @param slotMapping    [numTokens] 槽位映射
 * @param keyScale       [numTokens, numHead] Key Scale
 * @param keyScaleCache  [num_blocks, num_head, block_size, 1] Key Scale Cache (inplace)
 */
template <typename T_FP8, typename T_SLOT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM)
inline void OpScatterKvCacheWithKScaleSimt(
    int64_t needCoreNum,
    int64_t perCoreTokens,
    int64_t numHead,
    int64_t kHeadSize,
    int64_t vHeadSize,
    int64_t numBlocks,
    int64_t blockSize,
    int64_t maxSlot,
    int64_t magicNumHead,
    int64_t shiftNumHead,
    int64_t magicBlockSize,
    int64_t shiftBlockSize,
    __gm__ T_FP8* key,
    __gm__ T_FP8* value,
    __gm__ T_FP8* keyCache,
    __gm__ T_FP8* valueCache,
    __gm__ T_SLOT* slotMapping,
    __gm__ float* keyScale,
    __gm__ float* keyScaleCache
);
```

> ⚠️ **VF 参数数量合规性**：VF 参数列表包含 2 个标量指针（实际 7 个 GM 指针中 5 个为 FP8，2 个为 float，符合指针参数无特殊限制）和 13 个 `int64_t` 标量，总参数 size = 7 × 64bit（指针） + 13 × 64bit（int64_t）= 1280 bit = 160 bytes = 40 × 32bit。小于 28 × 32bit = 896 bit 限制。参数数量合规。

### 5.2 核心计算流程

```cpp
template <typename T_FP8, typename T_SLOT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM)
inline void OpScatterKvCacheWithKScaleSimt(
    int64_t needCoreNum,
    int64_t perCoreTokens,
    int64_t numHead,
    int64_t kHeadSize,
    int64_t vHeadSize,
    int64_t numBlocks,
    int64_t blockSize,
    int64_t maxSlot,
    int64_t magicNumHead,
    int64_t shiftNumHead,
    int64_t magicBlockSize,
    int64_t shiftBlockSize,
    __gm__ T_FP8* key,
    __gm__ T_FP8* value,
    __gm__ T_FP8* keyCache,
    __gm__ T_FP8* valueCache,
    __gm__ T_SLOT* slotMapping,
    __gm__ float* keyScale,
    __gm__ float* keyScaleCache)
{
    // 本核 token 范围：[tokenStart, tokenEnd)
    int64_t tokenStart = blockIdx.x * perCoreTokens;
    int64_t tokenEnd = (blockIdx.x + 1) * perCoreTokens;
    if (tokenEnd > needCoreNum * perCoreTokens) {
        // 取 min(tokenEnd, numTokens)，但 numTokens 可能未传入，
        // 实际由 perCoreTokens × needCoreNum ≥ numTokens 保证
    }

    // 每核工作总量 = perCoreTokens × numHead
    int64_t totalWork = perCoreTokens * numHead;

    // Grid-Stride 循环
    for (int64_t idx = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
         idx < totalWork;
         idx += static_cast<int64_t>(blockDim.x * gridDim.x))
    {
        // 1. 解码线性索引 → (localToken, head)
        //    localToken = idx / numHead, head = idx % numHead
        int64_t localToken = Simt::UintDiv<int64_t>(idx, magicNumHead, shiftNumHead);
        int64_t head = idx - localToken * numHead;
        int64_t token = tokenStart + localToken;

        // 2. 读取 slot_mapping
        T_SLOT slot = slotMapping[token];

        // 3. 越界检查（负数或 >= maxSlot 则跳过）
        if (slot < 0 || slot >= maxSlot) {
            continue;
        }

        // 4. 解码 slot → (blockIdx, blockOffset)
        //    blockIdx = slot / blockSize, blockOffset = slot % blockSize
        int64_t cacheBlockIdx = Simt::UintDiv<int64_t>(slot, magicBlockSize, shiftBlockSize);
        int64_t blockOffset = slot - cacheBlockIdx * blockSize;

        // 5. 预计算各 tensor 的基地址偏移
        // key[token][head][0]:
        int64_t keyBase = token * (numHead * kHeadSize) + head * kHeadSize;

        // keyCache[cacheBlockIdx][head][blockOffset][0]:
        //   stride: numHead * blockSize * kHeadSize between blocks
        //          blockSize * kHeadSize between heads
        //          kHeadSize between offsets
        int64_t keyCacheBase = cacheBlockIdx * (numHead * blockSize * kHeadSize)
                             + head * (blockSize * kHeadSize)
                             + blockOffset * kHeadSize;

        // value 基础偏移（vHeadSize 可能与 kHeadSize 不同）
        int64_t valueBase = token * (numHead * vHeadSize) + head * vHeadSize;
        int64_t valueCacheBase = cacheBlockIdx * (numHead * blockSize * vHeadSize)
                               + head * (blockSize * vHeadSize)
                               + blockOffset * vHeadSize;

        // 6. 拷贝 Key（逐元素，FP8 = 1 字节）
        for (int64_t d = 0; d < kHeadSize; ++d) {
            keyCache[keyCacheBase + d] = key[keyBase + d];
        }

        // 7. 拷贝 Value（逐元素，FP8 = 1 字节）
        for (int64_t d = 0; d < vHeadSize; ++d) {
            valueCache[valueCacheBase + d] = value[valueBase + d];
        }

        // 8. 拷贝 Key Scale（单个 float）
        // keyScaleCache[cacheBlockIdx][head][blockOffset][0]
        //   stride: numHead * blockSize between blocks
        //          blockSize between heads
        int64_t ksCacheBase = cacheBlockIdx * (numHead * blockSize) + head * blockSize + blockOffset;
        keyScaleCache[ksCacheBase] = keyScale[token * numHead + head];
    }
}
```

### 5.3 特化场景 vs 泛化场景

| 场景 | 条件 | 影响 | 处理方式 |
|-----|------|-----|---------|
| 特化场景 | `kHeadSize == vHeadSize` | Key 和 Value 维度相同，地址计算可部分复用 | 同一 VF 函数直接处理，无额外分支 |
| 泛化场景 | `kHeadSize != vHeadSize` | Key 和 Value 维度不同，需独立计算 | 同一 VF 函数直接处理，两个独立循环 |

> **设计决策**：尽管 SE 文档建议特化场景用 2 个 VF、泛化场景用 3 个 VF，但 SIMT 编程模型中 VF 函数调用有开销（`asc_vf_call` 上下文切换）。对于本算子，两条路径的计算量差异极小（仅 stride 不同），合并为单一 VF 函数更简洁高效。Tiling Key 仍保留场景区分用于未来可能的编译期优化。

### 5.4 UintDiv 快除详解

本算子在 VF 内需要两次固定除数整数除法：

| 除法操作 | 除数 | 除数特性 | 快除变量 |
|---------|------|---------|---------|
| `idx / numHead` | numHead | 固定（按 tiling 参数） | magicNumHead, shiftNumHead |
| `slot / blockSize` | blockSize | 固定（按 cache 配置） | magicBlockSize, shiftBlockSize |

**magic/shift 预计算**（在 Process 函数 scalar 端）：
```cpp
// Process() scalar 端
int64_t magicNumHead, shiftNumHead;
GetUintDivMagicAndShift<int64_t>(magicNumHead, shiftNumHead, tilingData->numHead);
int64_t magicBlockSize, shiftBlockSize;
GetUintDivMagicAndShift<int64_t>(magicBlockSize, shiftBlockSize, tilingData->blockSize);
```

**VF 内使用**（在 `__simt_vf__` 函数）：
```cpp
int64_t localToken = Simt::UintDiv<int64_t>(idx, magicNumHead, shiftNumHead);
int64_t cacheBlockIdx = Simt::UintDiv<int64_t>(slot, magicBlockSize, shiftBlockSize);
```

> 符合 op_mde_skill §3.8 规范：magic/shift 预计算在 scalar 端，UintDiv 调用在 VF 端，模板参数一致（均为 `int64_t`）。

### 5.5 使用的 SIMT API

| API | 用途 | 代码示例 | 来源 |
|-----|-----|---------|-----|
| `threadIdx.x` | 获取当前线程索引 | `uint64_t tid = threadIdx.x;` | SIMT 内置 |
| `blockDim.x` | 获取线程总数 | `uint32_t totalThreads = blockDim.x;` | SIMT 内置 |
| `blockIdx.x` | 获取当前核索引 | `int64_t coreIdx = blockIdx.x;` | SIMT 内置 |
| `gridDim.x` | 获取总核数 | `int64_t totalCores = gridDim.x;` | SIMT 内置 |
| `Simt::UintDiv<T>` | 无符号整数快除 | `Simt::UintDiv<int64_t>(dividend, magic, shift)` | simt_api/common_functions.h |
| `asc_vf_call` | 启动 SIMT VF 任务 | `asc_vf_call<OpSimt<T>>(dim3(N), args...)` | simt_api/asc_simt.h |

---

## 6. 线程排布设计

### 6.1 数据索引计算

每个线程处理一个 `(token, head)` 对，使用 Grid-Stride 循环确保所有工作项被覆盖：

```
线程处理的线性索引 = blockIdx.x * blockDim.x + threadIdx.x (初始)
步长 = blockDim.x * gridDim.x
```

解码（使用 UintDiv 快除）：
```
localToken = UintDiv(idx, magicNumHead, shiftNumHead)
head       = idx - localToken * numHead
token      = tokenStart + localToken
```

### 6.2 索引示意图

```
假设 numTokens=4, numHead=8, perCoreTokens=2, needCoreNum=2, THREAD_NUM=8

核 0 (tokenStart=0, perCoreTokens=2):
  线性索引空间: [0, 1, 2, ..., 15] 对应 (token_i, head_j) 共 16 个工作项
  线程 0: idx=0  → (token=0, head=0), idx=16 → (token=0, head=0) 但超出本核范围, break
  线程 1: idx=1  → (token=0, head=1)
  线程 2: idx=2  → (token=0, head=2)
  ...
  线程 7: idx=7  → (token=0, head=7)
  线程 0: idx=8  → (token=1, head=0) [Grid-Stride 下一轮]
  ...

核 1 (tokenStart=2, perCoreTokens=2):
  线性索引空间: [0, 1, 2, ..., 15] (local 索引)
  线程 0: idx=0  → (token=2, head=0)
  线程 1: idx=1  → (token=2, head=1)
  ...

工作项排布 (以核 0 为例, THREAD_NUM=8):
  Thread 0: [idx=0, idx=8]      → (t0,h0), (t1,h0)
  Thread 1: [idx=1, idx=9]      → (t0,h1), (t1,h1)
  Thread 2: [idx=2, idx=10]     → (t0,h2), (t1,h2)
  ...
  Thread 7: [idx=7, idx=15]     → (t0,h7), (t1,h7)
```

### 6.3 访存合并优化

线程 0~7 连续处理 token=0 的所有 head，其写入目标 `keyCache[cacheBlockIdx][0..7][blockOffset][:]` 在 `head` 维度上连续（row-major），有利于 GM 写合并。同时 `key[token][0..7][:]` 也在内存中连续，有利于 GM 读合并。

---

## 7. 代码文件规划

### 7.1 文件列表

| 文件路径 | 说明 |
|---------|-----|
| `op_kernel/arch35/scatter_kv_cache_with_k_scale_simt.h` | SIMT kernel 实现（含 Process + VF 函数） |
| `op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_data.h` | Tiling 数据结构定义 |
| `op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_key.h` | Tiling Key 宏定义 |
| `op_kernel/scatter_kv_cache_with_k_scale_apt.cpp` | 算子入口（kernel 调度 + TilingKey 分发） |
| `op_host/arch35/scatter_kv_cache_with_k_scale_tiling.cpp` | Tiling 实现（host 端切分逻辑） |
| `op_host/scatter_kv_cache_with_k_scale_def.cpp` | 算子定义（OpDef 注册） |
| `op_host/scatter_kv_cache_with_k_scale_infershape.cpp` | Shape/Dtype 推导 |
| `CMakeLists.txt` | 编译配置 |

### 7.2 Kernel 类设计（SIMT 简洁模式）

采用推荐写法（Process + asc_vf_call），避免传统 AscendC 类式写法。

```cpp
// scatter_kv_cache_with_k_scale_simt.h
#ifndef SCATTER_KV_CACHE_WITH_K_SCALE_SIMT_H
#define SCATTER_KV_CACHE_WITH_K_SCALE_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "scatter_kv_cache_with_k_scale_tiling_data.h"

namespace NsScatterKvCacheWithKScale {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 2048;

template <typename T_FP8, typename T_SLOT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM)
inline void OpScatterKvCacheWithKScaleSimt(
    int64_t needCoreNum,
    int64_t perCoreTokens,
    int64_t numHead,
    int64_t kHeadSize,
    int64_t vHeadSize,
    int64_t numBlocks,
    int64_t blockSize,
    int64_t maxSlot,
    int64_t magicNumHead,
    int64_t shiftNumHead,
    int64_t magicBlockSize,
    int64_t shiftBlockSize,
    __gm__ T_FP8* key,
    __gm__ T_FP8* value,
    __gm__ T_FP8* keyCache,
    __gm__ T_FP8* valueCache,
    __gm__ T_SLOT* slotMapping,
    __gm__ float* keyScale,
    __gm__ float* keyScaleCache)
{
    int64_t tokenStart = blockIdx.x * perCoreTokens;
    int64_t totalWork = perCoreTokens * numHead;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
         idx < totalWork;
         idx += static_cast<int64_t>(blockDim.x * gridDim.x))
    {
        int64_t localToken = Simt::UintDiv<int64_t>(idx, magicNumHead, shiftNumHead);
        int64_t head = idx - localToken * numHead;
        int64_t token = tokenStart + localToken;

        T_SLOT slot = slotMapping[token];
        if (slot < 0 || slot >= maxSlot) {
            continue;
        }

        int64_t cacheBlockIdx = Simt::UintDiv<int64_t>(slot, magicBlockSize, shiftBlockSize);
        int64_t blockOffset = slot - cacheBlockIdx * blockSize;

        // Copy Key
        int64_t keyBase = token * (numHead * kHeadSize) + head * kHeadSize;
        int64_t keyCacheBase = cacheBlockIdx * (numHead * blockSize * kHeadSize)
                             + head * (blockSize * kHeadSize)
                             + blockOffset * kHeadSize;
        for (int64_t d = 0; d < kHeadSize; ++d) {
            keyCache[keyCacheBase + d] = key[keyBase + d];
        }

        // Copy Value
        int64_t valueBase = token * (numHead * vHeadSize) + head * vHeadSize;
        int64_t valueCacheBase = cacheBlockIdx * (numHead * blockSize * vHeadSize)
                               + head * (blockSize * vHeadSize)
                               + blockOffset * vHeadSize;
        for (int64_t d = 0; d < vHeadSize; ++d) {
            valueCache[valueCacheBase + d] = value[valueBase + d];
        }

        // Copy Key Scale
        int64_t ksCacheBase = cacheBlockIdx * (numHead * blockSize) + head * blockSize + blockOffset;
        keyScaleCache[ksCacheBase] = keyScale[token * numHead + head];
    }
}

template <typename T_FP8, typename T_SLOT>
__aicore__ inline void Process(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR keyCache,
    GM_ADDR valueCache,
    GM_ADDR slotMapping,
    GM_ADDR keyScale,
    GM_ADDR keyScaleCache,
    const ScatterKvCacheWithKScaleTilingData* tilingData)
{
    __gm__ T_FP8* key_gm           = (__gm__ T_FP8*) key;
    __gm__ T_FP8* value_gm         = (__gm__ T_FP8*) value;
    __gm__ T_FP8* keyCache_gm      = (__gm__ T_FP8*) keyCache;
    __gm__ T_FP8* valueCache_gm    = (__gm__ T_FP8*) valueCache;
    __gm__ T_SLOT* slotMapping_gm  = (__gm__ T_SLOT*) slotMapping;
    __gm__ float* keyScale_gm      = (__gm__ float*) keyScale;
    __gm__ float* keyScaleCache_gm = (__gm__ float*) keyScaleCache;

    asc_vf_call<OpScatterKvCacheWithKScaleSimt<T_FP8, T_SLOT>>(
        dim3(THREAD_NUM),
        tilingData->needCoreNum,
        tilingData->perCoreTokens,
        tilingData->numHead,
        tilingData->kHeadSize,
        tilingData->vHeadSize,
        tilingData->numBlocks,
        tilingData->blockSize,
        tilingData->maxSlot,
        tilingData->magicNumHead,
        tilingData->shiftNumHead,
        tilingData->magicBlockSize,
        tilingData->shiftBlockSize,
        key_gm, value_gm, keyCache_gm, valueCache_gm,
        slotMapping_gm, keyScale_gm, keyScaleCache_gm);
}

} // namespace NsScatterKvCacheWithKScale
#endif // SCATTER_KV_CACHE_WITH_K_SCALE_SIMT_H
```

### 7.3 算子入口（apt.cpp）

```cpp
// scatter_kv_cache_with_k_scale_apt.cpp
#include "arch35/scatter_kv_cache_with_k_scale_simt.h"
#include "arch35/scatter_kv_cache_with_k_scale_tiling_key.h"

// schMode 由 tiling key 决定，DTYPE_KEY/DTYPE_SLOT_MAPPING 由框架自动生成
template <uint32_t schMode>
__global__ __aicore__ void scatter_kv_cache_with_k_scale(
    GM_ADDR key, GM_ADDR value,
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ScatterKvCacheWithKScaleTilingData);
    GET_TILING_DATA_WITH_STRUCT(ScatterKvCacheWithKScaleTilingData, tilingData, tiling);

    // schMode 为 SCATTER_KV_CACHE_SCENE_SPECIALIZED 或 SCATTER_KV_CACHE_SCENE_GENERALIZED
    // 两种模式使用统一的 Process 函数
    NsScatterKvCacheWithKScale::Process<DTYPE_KEY, DTYPE_SLOT_MAPPING>(
        key, value, key_cache, value_cache,
        slot_mapping, key_scale, key_scale_cache,
        &tilingData);
}
```

### 7.4 算子定义（def.cpp）

```cpp
// scatter_kv_cache_with_k_scale_def.cpp
#include "register/op_def_registry.h"

namespace ops {
class ScatterKvCacheWithKScale : public OpDef {
public:
    explicit ScatterKvCacheWithKScale(const char* name) : OpDef(name)
    {
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_scale_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("cache_layout")
            .ParamType(OPTIONAL)
            .String("BNBD");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "scatter_kv_cache_with_k_scale_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(ScatterKvCacheWithKScale);
} // namespace ops
```

### 7.5 Shape/Dtype 推导（infershape.cpp）

```cpp
// scatter_kv_cache_with_k_scale_infershape.cpp
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeScatterKvCacheWithKScale(gert::InferShapeContext* context)
{
    // 本算子为 inplace 算子，输出 shape 与对应输入完全一致
    // 输入索引: 0=key, 1=value, 2=key_cache, 3=value_cache,
    //           4=slot_mapping, 5=key_scale, 6=key_scale_cache
    // 输出索引: 0=key_cache(out), 1=value_cache(out), 2=k_scale_cache(out)

    // 输出 0 (key_cache): shape = 输入 2 (key_cache) 的 shape
    const gert::Shape* kcShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcShape);
    gert::Shape* outKcShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKcShape);
    outKcShape->SetDimNum(kcShape->GetDimNum());
    for (size_t i = 0; i < kcShape->GetDimNum(); i++) {
        outKcShape->SetDim(i, kcShape->GetDim(i));
    }

    // 输出 1 (value_cache): shape = 输入 3 (value_cache) 的 shape
    const gert::Shape* vcShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, vcShape);
    gert::Shape* outVcShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outVcShape);
    outVcShape->SetDimNum(vcShape->GetDimNum());
    for (size_t i = 0; i < vcShape->GetDimNum(); i++) {
        outVcShape->SetDim(i, vcShape->GetDim(i));
    }

    // 输出 2 (k_scale_cache): shape = 输入 6 (key_scale_cache) 的 shape
    const gert::Shape* kscShape = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, kscShape);
    gert::Shape* outKscShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKscShape);
    outKscShape->SetDimNum(kscShape->GetDimNum());
    for (size_t i = 0; i < kscShape->GetDimNum(); i++) {
        outKscShape->SetDim(i, kscShape->GetDim(i));
    }

    // Dtype 推导（inplace 算子，输出 dtype 与输入一致，框架自动处理）
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterKvCacheWithKScale).InferShape(InferShapeScatterKvCacheWithKScale);
} // namespace ops
```

---

## 8. 性能预估

### 8.1 理论性能

| 指标 | 预估值 | 计算依据 |
|-----|-------|---------|
| 总数据量（读+写） | `numTokens × numHead × (kHeadSize + vHeadSize) × 2 bytes (FP8) + numTokens × numHead × 8 bytes (float32 scale)` | FP8 = 1 字节/元素，读写各一次，key_scale 读写各 4 字节 |
| 典型数据量（LLaMA-7B） | `128 × 32 × (128 + 128) × 2 + 128 × 32 × 8 = 2,097,152 + 32,768 ≈ 2.1 MB` | 单次推理 KV Cache 更新 |
| 核利用率 | ~100%（numTokens ≥ coreNum 时） | 每个 token 分配一个核，token 数足够时核全部利用 |
| 内存带宽 | ~300 GB/s（HBM 理论带宽的 ~30%） | scatter 写入为非连续访问，实际带宽取决于写入分布 |
| 单核计算时间 | ≈ 2.1MB / (300 GB/s / coreNum) ≈ 7 μs (coreNum=1) ~ 0.1 μs (coreNum=64) | 纯内存拷贝，计算时间为 0 |

### 8.2 优化空间

| 优化方向 | 说明 | 优先级 |
|---------|------|-------|
| 向量化拷贝 | 使用 vector 类型（如 `uint8x4`、`uint8x8`）批量读写，减少循环开销 | 中 |
| 预取提示 | 对 `slot_mapping` 读取使用 `__ldcg`（cache global）预取 | 低 |
| 场景特化 | 当 `kHeadSize == vHeadSize` 时，合并 Key/Value 循环减少地址计算 | 低（当前已单 VF 实现） |

---

## 9. 风险与注意事项

### 9.1 已知限制

| 限制项 | 说明 | 规避方案 |
|-------|-----|---------|
| 最大线程数 | 2048 | 搬运类算子最优选择，无需规避 |
| 栈空间 | 64 字节/线程（2048 线程模式） | 避免大型局部数组，使用指针和标量 |
| 空 tensor 不支持 | numTokens=0 时无操作 | Tiling 阶段检查 numTokens ≤ 0 则设置 `needCoreNum=1`、`perCoreTokens=0`，kernel 自动跳过循环 |
| slot_mapping 越界 | 依赖上层调用方保证合法值 | Kernel 内手动检查并跳过，不写入（与竞品行为一致） |
| VF 参数限制 | ≤ 28 × 32bit | 当前设计 40 × 32bit，**超出限制**。需将部分参数打包到 UB buffer 传递 |

> ⚠️ **VF 参数过多问题**：当前 VF 函数有 7 个 GM 指针（7 × 64bit）和 13 个 int64_t 标量（13 × 64bit），总计 20 × 64bit = 160 bytes = 40 × 32bit，超过 28 × 32bit（896 bit / 896 bit）限制。
>
> **规避方案**：将 `int64_t` 标量参数中非频繁使用的参数（如 `needCoreNum`, `numBlocks`, `maxSlot`, `magicNumHead`, `shiftNumHead`, `magicBlockSize`, `shiftBlockSize`）打包存入 UB，VF 内通过 `__ubuf__` 指针读取。保留频繁使用的参数（`perCoreTokens`, `numHead`, `kHeadSize`, `vHeadSize`, `blockSize`, GM 指针）作为直接参数。

### 9.2 待确认事项

- [ ] `DTYPE_KEY` 宏是否自动覆盖 FP8 类型（DT_FLOAT8_E5M2 / DT_FLOAT8_E4M3FN）的编译期实例化？
- [ ] `DTYPE_SLOT_MAPPING` 宏的自动命名规则：def.cpp 中 `Input("slot_mapping")` → 宏名是否为 `DTYPE_SLOT_MAPPING`？
- [ ] FP8 类型（`fp8_e5m2_t`、`fp8_e4m3fn_t`）在 `<simt_api/asc_fp8.h>` 中的实际类型定义名称是否匹配？
- [ ] 空 tensor 场景（numTokens=0）：tiling 层是否应直接 FAIL 还是设置 perCoreTokens=0 让 kernel 空转？

### 9.3 与竞品的关键差异

| 差异维度 | 竞品（vLLM reshape_and_cache） | 本算子 | 影响 |
|---------|------------------------------|--------|-----|
| Key Scale Cache | 无独立输出（仅用作量化参数） | 有独立输出 key_scale_cache | 额外 tensor 存储和 I/O |
| 线程模型 | CUDA block/thread（256/512 threads） | AscendC SIMT VF（2048 threads） | 并行度更高 |
| 数据类型 | FP16/BF16/FP8 | 仅 FP8（E5M2/E4M3FN） | 更专注的量化场景 |
| 确定性 | 确定 | 确定 | 行为一致 |

---

## 10. 设计确认

- [x] 切分策略已确认（沿 token 维度多核切分，thread=2048）
- [x] Tiling 数据结构已确认（含 shape 参数 + 快除 magic/shift）
- [x] **TilingKey 未枚举 dtype，已使用 DTYPE_ 宏**（仅编码特化/泛化场景模式）
- [x] 计算逻辑已确认（纯访存 scatter，无计算操作）
- [x] 线程排布已确认（Grid-Stride 循环，UintDiv 快除）
- [x] **精度对齐方案已确认**（FP8 bit-exact, FP32 < 1e-6）
- [x] 文件规划已确认（simt.h / tiling_data.h / tiling_key.h / apt.cpp / tiling.cpp / def.cpp / infershape.cpp / CMakeLists.txt）
- [ ] **VF 参数数量问题待修正**（开发阶段需将部分标量打包到 UB buffer）
