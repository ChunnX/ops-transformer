# quant_lightning_indexer_v2 LD 阶段 UB 占用分析

> 涉及文件：`attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_vector.h`

## 一、问题背景

最新 commit `897e02470` "LD topk" 为 LD（Load-balance Decode）阶段独立分配 UB Buffer，但存在两个问题：

1. **UB 内存溢出**：`InitLDBuffers` 用固定常量 `TRUNK_LEN_LD = 102400` 分配 buffer，仅 `ldIndexBuf_` 一个就约 401KB，远超 UB 物理上限，导致执行失败。
2. **topk 内部指针悬垂**：`pipe->Reset()` 释放了 `topkSharedTmpBuf_`，但未重新分配，topk 算子内部指针（`tmpIndexLocal`、`histogramsLocal` 等）指向已释放地址，与新分配的 `ldIndexBuf_` 内存重叠，存在数据踩踏隐患。

本文档分析 LD 阶段各 buffer 的 UB 占用，给出根据 `sparseCount` 动态计算 `trunkLenLd` 的方案。

## 二、UB 可用容量

- UB 物理容量：256KB
- 系统保留：8KB
- **实际可用：248KB = 253952 字节**

目标：让 LD 阶段 UB 占用尽量接近 248KB 且不超限，以最大化每次处理的 trunk 长度、减少 LD 循环次数。

## 三、LD 阶段各 Buffer 明细

`InitLDBuffers` 中分配的 buffer（设 `A = topkCountAlign256_ = Align(sparseCount, 256)`，`L = trunkLenLd_`，`SCORE_T = uint16_t` 占 2 字节）：

| Buffer | 元素数 | 字节大小 | 说明 |
|--------|--------|----------|------|
| `ldIndexBuf_` | (A + L) | (A + L) × 4 | LD 索引暂存 |
| `indicesOutBuf_` | (A + 64) | (A + 64) × 4 | topk 输出索引（64 为 duplicate 刷 -1 额外空间） |
| `mrgValueBuf_` | (A + L) | (A + L) × 2 | 待排序 value，SCORE_T=uint16 |
| `scoreOutBuf_` | (A + 64) | (A + 64) × 2 | topk 输出 score |
| `valueOutBuf_` | (A + 64) | (A + 64) × 2 | 返回值 bf16 |
| `topkSharedTmpBuf_` | - | 6A + 2L + 3328 | topk 算子内部 tmp（见下方推导） |

### topkSharedTmpBuf_ 推导

来自 `quant_lightning_indexer_v2_topk.h` 的 `GetSharedTmpBufferSize()`（uint16 特化）：

```
bufferSize1 = (2·Align(topK,256) + 3·256 + 64) × 4   // hisIndex[1]+histograms+idxHigh+idxLow+nkValue
bufferSize2 = (Align(topK,256) + trunkLen) × 2         // tmpIndexLocal (uint16)
reuseSize   = Align(topK,256) × 4                      // hisIndex[0] 复用外部 indicesOutLocal_
total = bufferSize1 + bufferSize2 - reuseSize
     = (2A + 3·256 + 64)·4 + (A + L)·2 - A·4
     = 8A + 3072 + 256 + (A + L)·2 - 4A
     = 6A + 2L + 3328
```

### 总占用公式

```
Total = (A+L)·4 + (A+64)·4 + (A+L)·2 + (A+64)·2 + (A+64)·2 + (6A+2L+3328)
      = 4A+4L + 4A+256 + 2A+2L + 2A+128 + 2A+128 + 6A+2L+3328
      = 20A + 8L + 3840  (字节)
```

其中 **3840 字节**为各 buffer 的固定开销（`TOPK_FIXED_BYTES + OUTPUT_PAD_BYTES`），由两部分组成：
- `TOPK_FIXED_BYTES = (3×256+64)×4 = 3328`：topk 内部固定区（histograms、idxHigh、idxLow 各 256，nkValue 64，均为 uint32）
- `OUTPUT_PAD_BYTES = 64×(4+2+2) = 512`：三个输出 buffer 各自的 +64 余量（indicesOut uint32、scoreOut SCORE_T、valueOut bf16）

## 四、动态计算方案

### 公式

令 `Total = LD_UB_TARGET_BYTES = 248KB = 253952`，反解 `L`：

```
trunkLenLd_ = (LD_UB_TARGET_BYTES - 20·A - 3840) / 8
```

保底：`trunkLenLd_` 至少为 `topkCountAlign16_`（保证能装下一个 workspace 块）。

### 代码实现

```cpp
constexpr uint32_t LD_UB_TARGET_BYTES = 248 * 1024;

// topk内部固定区: histograms(256)+idxHigh(256)+idxLow(256)+nkValue(64)，均为uint32
constexpr uint32_t TOPK_FIXED_BYTES = (3 * 256 + 64) * sizeof(uint32_t);
// 三输出buffer的+64余量: indicesOut(uint32)+scoreOut(SCORE_T)+valueOut(bf16)
constexpr uint32_t OUTPUT_PAD_BYTES = 64 * (sizeof(uint32_t) + sizeof(SCORE_T) + sizeof(bfloat16_t));
const uint32_t fixedBytes = TOPK_FIXED_BYTES + OUTPUT_PAD_BYTES;

uint32_t availForTrunk = (LD_UB_TARGET_BYTES > 20 * topkCountAlign256_ + fixedBytes)
    ? (LD_UB_TARGET_BYTES - 20 * topkCountAlign256_ - fixedBytes) : 0;
trunkLenLd_ = availForTrunk / 8;
if (trunkLenLd_ < topkCountAlign16_) {
    trunkLenLd_ = topkCountAlign16_;
}
```

### 各 sparseCount 下的占用（目标 248KB）

| sparseCount | A (Align256) | fixedBytes | trunkLenLd_ (L) | Total | 占用率 |
|-------------|-------------|------------|-----------------|-------|--------|
| 128 | 256 | 8960 | 30624 | 248.0 KB | 100% |
| 512 | 512 | 14080 | 29984 | 248.0 KB | 100% |
| 1024 | 1024 | 24320 | 28704 | 248.0 KB | 100% |
| 2048（当前上限） | 2048 | 44800 | 26144 | 248.0 KB | 100% |
| 4096 | 4096 | 85760 | 21024 | 248.0 KB | 100% |
| 8192（远期规划） | 8192 | 167680 | 10784 | 248.0 KB | 100% |

可见动态方案在任何 `sparseCount` 下都精确命中 248KB，充分利用 UB。

## 五、安全性分析

### 5.1 tmpIndexLocal 不越界

`LiTopKVF`（16 位版本）最坏情况下将整个 `s2SeqLen` 范围的索引写入 `tmpIndexLocal`。`ProcessLD` 中 `s2SeqLen` 最大值：

```
s2LenAlign_max = trunkLenLd_ + topkCountAlign256_   // i!=0 时
```

而 `tmpIndexLocal` 容量（来自 `GetSharedTmpBufferSize`）= `Align(topK,256) + topkOp_.trunkLen`。

关键：`InitLDBuffers` 中先调用 `topkOp_.Init(topkCount_, trunkLenLd_)`，使 `topkOp_.trunkLen == trunkLenLd_`，因此：

```
tmpIndexLocal 容量 = A + trunkLenLd_ >= s2LenAlign_max = trunkLenLd_ + A   (等号成立)
```

**结论：不越界。** 前提是 `trunkLenLd_` 与 `topkOp_.trunkLen` 必须同步（已由代码保证）。

### 5.2 ldIndexLocal_ gather 不越界

`LiTopKLDGatherVF` 用 `tmpIndexLocal` 的值作索引 gather `ldIndexLocal_`。索引值范围 `[0, s2SeqLen)`，而 `ldIndexLocal_` 容量 = `A + trunkLenLd_ > s2SeqLen`。**不越界。**

### 5.3 topk 指针重新绑定

`pipe->Reset()` 后重新分配 `topkSharedTmpBuf_` 并调用 `topkOp_.InitBuffers()`，确保 `tmpIndexLocal`、`histogramsLocal`、`idxHighLocal`、`idxLowLocal`、`nkValueLocal` 指向新地址，消除悬垂指针隐患。

### 5.4 非 LD 路径不受影响

`ProcessTopK`（非 LD 路径）仍使用 `trunkLen_ = TRUNK_LEN_16K = 16384`，与 `InitBuffers`（非 Reset 路径）分配的 `mrgValueBuf_` 容量匹配，不受 LD 动态计算影响。

## 六、修改清单

| 位置 | 修改 |
|------|------|
| 常量区 | 删除 `TRUNK_LEN_LD = 102400`；新增 `LD_UB_TARGET_BYTES = 248KB` |
| 成员区 | 新增 `trunkLenLd_` 成员 |
| `InitLDBuffers` | 动态计算 `trunkLenLd_`；补齐 `topkSharedTmpBuf_` 分配 + `topkOp_.Init/InitBuffers`；刷新所有成员 LocalTensor |
| `ProcessLD` | 4 处 `trunkLen_` → `trunkLenLd_`（line 888/889/911/913） |

## 七、测试验证

使用 `pt_path_1` 下 6 个 LD 相关用例验证（覆盖 sm0/sm3、bsnd/tnd、ld/weakld、k4096~k32768）：

```
QLIV2_PT_FILE_LIST="pt_path_1/tnd_tnd_b8_q4_k4096_cmp1_sm0_ld.pt,...,pt_path_1/tnd_pabb_b8_q4_k32768_cmp1_sm0_ld.pt" \
  python3 -m pytest test_quant_lightning_indexer_v2_batch.py -v -m ci
```

结果：**6 passed**，全部通过。
