# TopK 融合各版本实现详解

> 对应 `TOPK_OPTIMIZATION_JOURNEY.md` 的 v1→v9，逐个版本展开实现方式、数据布局与关键代码。

---

## v1：Q token 按 SM 分配（修 148× 冗余）

**目标**：原代码每个 block 遍历全部 624 token，需按 SM 均分。

**实现**：Phase 2 外层循环从 warp 内 strided 改为跨 SM strided：

```cpp
const uint32_t num_sms = gridDim.x;
for (uint32_t q_token = sm_idx * kNumTopkWarps + warp_idx;
     q_token < num_q_tokens;
     q_token += num_sms * kNumTopkWarps) {
    // 该 token 的 TopK
}
```

**语义**：SM `s` 的 warp `w` 处理 `{s*8+w, s*8+w+148*8, ...}`，每个 token 恰好被处理一次。

---

## v2：8-bit radix select 替换 bubble sort

**目标**：消除 bubble sort 的 O(K²)=O(512²) 串行比较。

**实现**（`warp_bucket_topk` 主体）：

```cpp
// float → uint32 可比较序（单调映射）
auto to_sortable = [](float f) -> uint32_t {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
};

// 4 轮，从最高字节到最低字节
for (int round = 0; round < 4 && remaining > 0; ++round) {
    const int shift = (3 - round) * 8;
    const uint32_t prefix_mask = (round == 0) ? 0u : (0xFFFFFFFFu << (32 - round * 8));
    const uint32_t prefix_val = kth_value & prefix_mask;

    // ① 清 256 桶直方图（SMEM）
    for (i = lane_idx; i < 256; i += 32) histogram[i] = 0;

    // ② 统计当前字节：只数前几轮 prefix 命中的元素
    for (i = lane_idx; i < k_len; i += 32) {
        u = to_sortable(load(i));
        if ((u & prefix_mask) == prefix_val)
            atomicAdd(&histogram[(u >> shift) & 0xFF], 1);
    }

    // ③ 找 cutoff：每 lane 读 8 桶 → warp 前缀和 → 从高桶扫
    //    ca=高于该桶的存活数, cge=含该桶的存活数
    //    条件: ca < remaining && cge >= remaining

    // ④ 拼接 kth 字节、累计 definite、更新 remaining
    kth_value |= (cutoff_bucket << shift);
    definite_count += count_above_cutoff;
    remaining = kSparseCount - definite_count;
}

// final：u > kth → definite 直出；u == kth → boundary 补足
```

**正确性要点**：`to_sortable` 保证 uint32 大小关系与 float 一致（负数取反、正数翻符号位）；4 轮后 kth 是第 K 大值的精确位模式。

**代价**：round-0 全命中，热点桶（top 值集中的字节）atomicAdd 同地址串行——这是后续 v9 想解决、最终放弃的残留点。

---

## v3：8-warp 分段 + 候选合并

**目标**：8192 元素拆 8 段并行处理，段间只做一次候选级归并。

**正确性依据**：元素不在本段 top-512 → 本段有 ≥512 个更大元素 → 必不在全局 top-512。

**SMEM 布局**（复用 Phase 1 的 smem_buffer，共 ~105KB）：

```
[0, 16KB)        merge_vals   float[4096]
[16K, 32KB)      merge_idx    int32[4096]
[32K, 33KB)      ghist        uint32[256]
[33K, ~33.1KB)   gslots       uint32[16]
[33.1KB, ...)    warp regions ×8，每 warp:
                    wcache  float[1024]  (4KB)
                    whist   uint32[256]  (1KB)
                    wstage  int32[...]   (2KB, boundary staging)
                    wcand   int32[512]   (2KB)
```

**stage 1 — 每 warp 局部 TopK**：

```cpp
// 段 [my_start, my_start+my_len), my_len ≤ 1024
for (i = lane_idx; i < my_len; i += 32)          // 32 lane coalesced 载入
    st_shared(wcache + i, __ldcg(logits_row + k_start + my_start + i));
warp_bucket_topk<kSparseCount, true>(wcache, k_start + my_start, my_len,
                                     wcand, whist, wstage, nullptr, lane_idx);
// wcand[0..511] = 本段 top-512 的绝对 KV 索引（out_base = k_start + my_start）
```

**stage 2 — 合并候选**：

```cpp
// 8 warp × 32 lane = 256 线程分担 4096 项
for (j = warp_idx*32 + lane_idx; j < kNumCand; j += 256) {
    w = j / 512; pos = j % 512;
    c = cand_base[w * stride + kCandOff + pos];   // 绝对索引
    merge_idx[j] = c;
    cache_pos = c - k_start - w * segL;           // 回推段内位置
    merge_vals[j] = ld_sh(cache_base + w * stride + cache_pos);  // 值从段缓存取
}
// 无效候选: merge_vals[j] = -INF, merge_idx[j] = -1
```

**stage 3+4（v3 版）**：warp0 对 merge_vals 跑完整 `warp_bucket_topk`（`kUseMap=true`，输出取 merge_idx[i]）。

**代价**：最终 radix 是 warp0 单干 4096（6 遍 × 128 iters = 768 次串行迭代），成为下一版的目标。

---

## v4：merge 值读改 SMEM + writer threadfence

**问题 1**：merge 值初始用 `__ldcg(logits_row + c)` 随机读 GMEM（4096 次散读）。

**修法**：值改从该 warp 的段缓存取（上一版代码中的 `ld_sh(cache_base + ...)`），GMEM 只在段载入时读一次。

**问题 2**：跨 SM 读别的 block 刚写的 logits 行，偶发 5M-cycle stall。

**修法**：writer 侧在 Phase 1 结束处 flush：

```cpp
__threadfence();    // 把本 block 的 logits 写推送到 L2
__syncthreads();
```

---

## v5：移除调试 printf（测量修正）

**发现**：设备侧 printf ~100K cycles/次，且仅 block0 打印 → block0 拖成最后完成块。

**改动**：删除全部 `[TOPK8]`/`[SPLIT]` 打印与配套 clock64 计时变量，计时改回 bench 脚本的 CUDA event。

**结果**：这是"零计算改动、-40% 耗时"的一步——测量本身有系统性偏差。

---

## v6：分布式 selection

**目标**：final 三遍扫描（count/write/boundary）从 warp0 单干（128 iters/遍）拆到 8 warp × 512 片（16 iters/遍）。

**stage 3（v6 版）**：warp0 只做 rounds，通过新增的 `kth_slot` 参数导出 kth：

```cpp
if (warp_idx == 0)
    warp_bucket_topk<kSparseCount, true, true>(merge_vals, 0, kNumCand,
        out_ptr, whist, wstage, merge_idx, lane_idx, gslots);  // gslots[0] = kth
__syncthreads();
const uint32_t kth_full = gslots[0];
```

**stage 4 — 块级前缀的三段式**：

```cpp
// ① 每 warp 数自己的片
my_def = count(u > kth);  my_col = stage(u == kth);

// ② 块级前缀（8 warp）
warp_def = xor-reduce(my_def);            // 关键：归约到独立变量!
if (lane==0) gslots[8+warp] = warp_def;
__syncthreads();
warp0: 8 项串行前缀 → gslots[0] = total, 各槽改写为 exclusive;

// ③ 写 definite
run = gslots[8+warp] + shfl_excl(my_def);   // 块基址 + warp 内 exclusive
for (i...) if (u > kth) out_ptr[run++] = merge_idx[warp*512+i];

// ④ boundary 同构（warp 内 xor-reduce → 块级前缀 → 写 stage 区）
// ⑤ warp0 从 filled 位置起填 -1
```

**三个 bug 修复记录**：

| Bug | 修法 |
|-----|------|
| `gslots[8+w]=my_def` 只存 lane0 值 | 先 `xor_reduce(my_def)` 再写槽 |
| reduce 破坏 `my_def` 原值 | 归约存 `warp_def` 独立变量 |
| barrier 发散 | 所有 `__syncthreads` 提升到全 block 统一执行 |

---

## v7：分布式 rounds

**目标**：rounds（warp0 单干 512 迭代）也拆到 8 warp。

**实现**：

```cpp
for (round = 0; round < 4 && rem > 0; ++round) {
    // ① 各 warp 清自己的 256 槽 hist（whist 区域）
    // ② 各 warp 扫自己的 512 片（16 iters），prefix 命中则 atomicAdd 到自有 hist
    __syncthreads();
    // ③ warp0 归并 8×256 → ghist[256]
    for (i = lane; i < 256; i += 32) {
        s = Σ_w wh[w*stride + kSegFloats + i];
        ghist[i] = s;
    }
    __syncthreads();
    // ④ warp0 从 ghist 找 cutoff（32 lane × 8 桶 + shfl 前缀）
    //    写 gslots[0]=cutoff, gslots[1]=cao, gslots[2]=valid
    __syncthreads();
    if (gslots[2] == 0) break;
    kth |= gslots[0] << shift;  def_cnt += gslots[1];  rem = K - def_cnt;
}
if (warp_idx == 0) gslots[0] = kth;   // 发布给 stage 4
```

**要点**：barrier 统一执行（warps 8-11 也到达）；cutoff 广播走 gslots 而非 shfl（跨 warp 无法 shuffle）。

---

## v8：BF16 logits

**动机**：logits 写/读减半（GMEM 流量 + Phase 2 载入）。

**6 处改动**：

```cpp
// ① host 分配
torch::empty({...}, q_fp.options().dtype(torch::kBFloat16));

// ② kernel 签名（core_impl + 两个 entry）
nv_bfloat16* __restrict__ logits_buffer;

// ③ Phase 1 写入（两个分支）
logits_buffer[q_offset + kv_offset] = __float2bfloat16(result);

// ④ Phase 2 读入
wcache[i] = __bfloat162float(logits_row[k_start + my_start + i]);

// ⑤ JIT dispatch 的 data_ptr（避免模板实例化未定义符号）
reinterpret_cast<nv_bfloat16*>(logits_buffer.data_ptr());

// ⑥ warp_bucket_topk 参数类型
const void* logits;   // 缓存路径按 uint32 视图、GMEM 兜底按 bf16 视图
```

**结果**：持平（写仅占 Phase 1 流量的 ~1%，Phase 2 载入省 ~4us，均在噪声内）。保留改动作为可选项。

---

## v9：nibble 原子消除（已回退，仅记录）

**动机**：round-0 热点桶 atomicAdd 串行化（单桶数百元素同地址原子）。

**实现**：8 轮 4-bit + 每 lane 私有 16 槽（零原子）+ lane j 归并桶 j。

**实测**：槽步长 16 → 490us（16-way bank conflict）；步长 17 → 235us；仍差于 182us 基线。

**失败原因**：
1. 扫描遍数翻倍（8 轮 × 32 iters = 256 vs 8-bit 的 4 轮 × 32 = 128）
2. 归并每轮 lane<16 串行读 32 槽，其余 lane 空等

**教训**：换掉瓶颈操作前先算替代方案的等量成本——nibble 用 2× 扫描 + 归并串行换了原子，净亏。

---

## 版本间数据流速览

```
Phase 1 (未变): Q@K^T → relu → ×W → sum → logits_buffer (bf16, GMEM)
                ↓ __threadfence + __syncthreads
Phase 2:
  token 循环 (SM-strided, v1)
    ├─ stage1: 8 warp 各载 1024 段 → 局部 8-bit radix (v2) → wcand[512] (v3)
    ├─ stage2: 4096 候选合并 → merge_vals/merge_idx (v3, 值读 SMEM v4)
    ├─ stage3: 分布式 rounds → kth (v7, 原 warp0 单干)
    └─ stage4: 分布式 selection → out (v6, 原 warp0 单干)
```
