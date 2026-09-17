# TopK 融合优化全流程记录

> 对象：DeepGEMM `sm100_mqa_logits` + TopK 融合算子（paged 模式）
> Shape：156 batch × 4 next_n × 64 heads × 128 dim，KV=8192，K=512，block_kv=64
> 平台：NVIDIA B200
> 目标：融合 TopK 后整体耗时接近 Phase 1 水平

---

## 0. 版本演进总表

| 版本 | 关键改动 | Phase 2 | 总耗时 | 主要问题 |
|------|---------|---------|--------|---------|
| v0 | bubble sort + 全量遍历 | ~17ms | — | 148× 冗余 |
| v1 | Q token 按 SM 分配 | ~57ms | — | bubble O(K²) |
| v2 | 8-bit radix select (SMEM atomicAdd) | ~412us | ~518us | atomic 争用 |
| v3 | 8-warp 分段 + 局部 TopK + 候选合并 | ~1082us | — | merge 随机 GMEM 读 |
| v4 | merge 值读改 SMEM + writer `__threadfence` | ~500us | — | 5M-cycle stall |
| v5 | 移除调试 printf | — | **322us** | 测量修正 |
| v6 | 分布式 selection | — | **275us** | 3 个精度 bug |
| v7 | 分布式 rounds | ~132us | **182us** | 当前基线 |
| v8 | BF16 logits | ~132us | ~183us | 持平 |
| v9 | nibble 原子消除 | 490us→235us | — | 2×扫描+归并串行，回退 |

Phase 1 全程 ≈ 50us（起点 594us → 最终 **182us**，3.3×；Phase 2 545us → 132us，4.1×）。

---

## 1. v0 → v1：修复 token 分配（148× 冗余）

**改动点**：Phase 2 的 token 循环从
```cpp
for (q_token = warp_idx; q_token < num_q_tokens; q_token += kSpecWarpStart)
```
改为按 SM 分配：
```cpp
for (q_token = sm_idx * kNumTopkWarps + warp_idx;
     q_token < num_q_tokens;
     q_token += num_sms * kNumTopkWarps)
```

**现象**：原来每个 block 处理全部 624 token → 92,352 次 TopK（148 块 × 624），正确只需 624 次。

**性能**：17ms →（bubble sort 主导）57ms。TopK 算法本身（O(K²)=O(512²) 串行 bubble）成为新瓶颈。

## 2. v1 → v2：radix select 取代 bubble sort

**改动点**：`warp_bucket_topk` 算法整体替换：

- 4 轮 8-bit radix select（值域字节级分桶）
- `to_sortable`：float → uint32 可比较序：
  ```cpp
  u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
  ```
- 256 桶 SMEM histogram（atomicAdd）+ 前缀和 + cutoff 定位
- final：count definite（`u > kth`）+ collect boundary（`u == kth`）

**性能**：57ms → 412us（Phase 2），total 518us。

## 3. v2 → v3：8-warp 分段 + 候选合并

**改动点**：
- 8 个 warp 各取 1024 元素段 → SMEM cache（4KB/段） → 局部 radix TopK-512 → `wcand`
- 合并 8×512 = 4096 候选 → `merge_vals`/`merge_idx`（SMEM 32KB）
- warp0 对候选做最终 radix + 写出

**正确性依据**：元素不在本段 top-512 → 本段有 ≥512 个更大元素 → 必不在全局 top-512。

**性能**：首次 1082us——merge 循环对候选值用 `__ldcg` 随机读 GMEM。

## 4. v3 → v4：merge 值读改 SMEM + 跨 SM 一致性

**改动点**：
1. merge 循环的值读取从 `__ldcg(logits_row + c)` 改为从该 warp 的 SMEM 段缓存读：
   ```cpp
   merge_vals[j] = __uint_as_float(ptx::ld_shared(cache_base + w*stride + cache_pos));
   ```
2. Phase 1→2 边界 writer 侧加 `__threadfence()`（把 logits 写推送到 L2）

**现象**：跨 SM 读产生偶发 5M-cycle（2.5ms）stall。

**性能**：1082us → ~500us。

## 5. v4 → v5：移除调试 printf（测量修正）

**发现**：`[TOPK8]` 等设备侧 printf 每次开销 ~100K cycles，且只有 block 0 打印 → block 0 变最后完成块，拉高整个 event 计时。分阶段数据揭示"隐藏开销"全部来自 printf。

**性能**：~500us → **322us**（Phase 2 ~272us）。

## 6. v5 → v6：分布式 selection

**改动点**：warp0 单干的最终 selection（count/write/boundary 3 遍扫描）拆到 8 warp × 512 片，块级前缀协调。

**踩坑（3 个）**：

| Bug | 现象 | 根因 | 修复 |
|-----|------|------|------|
| total_def=19 | 精度 3.17% | `gslots[8+w]=my_def` 只存 lane0 值 | warp 内 xor-reduce 后写槽 |
| 输出留洞、垃圾值 1065353216 | 精度 11.84% | reduce 破坏 `my_def` 原值 | 归约结果存独立变量 |
| barrier 发散 | 精度 3.17% | `__syncthreads` 在 `if(warp<8)` 内 | barrier 全 block 统一执行 |

**性能**：322us → **275us**。

## 7. v6 → v7：分布式 rounds

**改动点**：radix rounds 从 warp0 单干（4 轮 × 128=512 迭代）改为 8 warp × 512 片（各 16 iter/轮）+ 各自 SMEM histogram + warp0 归并 8×256 → 全局 cutoff + `gslots` 广播。

**性能**：275us → **182us**（Phase 2 132us；起点 545us → 4.1×）。

## 8. v8：BF16 logits（持平）

**改动点**（6 处）：
1. host 分配 `kFloat32` → `kBFloat16`
2. kernel 签名 `float* logits_buffer` → `nv_bfloat16*`
3. Phase 1 写入 `__float2bfloat16(result)`
4. Phase 2 读入 `__bfloat162float(...)`
5. `data_ptr<nv_bfloat16>()` → `reinterpret_cast<nv_bfloat16*>(data_ptr())`（避免未定义符号）
6. `warp_bucket_topk` 参数 `float*` → `void*`，内部分视图

**性能**：182us → 183.3us（paged）/ 178.3us（non-paged）——写减半仅省 ~1% Phase 1、~4us Phase 2，噪声级。

## 9. Phase 1 四个代码级方向验证

| 方向 | 结果 |
|------|------|
| Stage 深度（Q2/KV6） | SMEM 231KB > 227KB，不可行 |
| BF16 logits | 落地，持平 |
| Weight 向量读（ld.shared.v4） | 落地，中性 |
| KV 页调度/L2 复用 | 上游已按 request 连续分组，无空间（A/B 实验 0.48 vs 理论 0.25，差值源于 B 场景 request 开销） |

## 10. v9：nibble 原子消除（已实施并验证，最终回退）

**实施**：`warp_bucket_topk` 的 rounds 从 4 轮 8-bit atomicAdd 换成 8 轮 4-bit 每 lane 私有 16 槽。

**定位依据**：空扫描校准 24c/it、8×syncwarp 20 cycles，残余 ~10K/segment 来自 round-0 热点桶 atomicAdd 串行化（top 值集中，单桶数百元素同地址原子）。

**实测结果**：

| 阶段 | 结果 |
|------|------|
| 初版（槽步长 16） | 490us —— 32 lane 槽位只落 2 个 bank，16-way 系统冲突 |
| 修步长 17 | 235us —— 冲突消除但仍差于基线 |
| **最终回退** | **182us 基线** |

**失败原因**（两条都成立）：

1. **扫描遍数翻倍**：8-bit × 4 轮 = 128 次扫描迭代；4-bit × 8 轮 = 256 次。nibble 用 2× 的扫描换原子，得不偿失
2. **归并串行化**：每轮 lane j<16 串行读 32 槽（其余 lane 空等），8 轮累计 ~8K cycles/token

**结论**：8-bit 原子版的"热点桶风暴"确实存在，但 nibble 的替代成本更高。`warp_bucket_topk` 维持 8-bit atomic 版本（182us 基线）为最终方案。

---

## 11. 关键 Bug 总清单

1. **148× token 冗余**：Phase 2 全 block 遍历全 token
2. **发散 `__syncthreads`**：warps 8-11 不执行 barrier → prefix 竞态 → 精度 3.17%
3. **warp reduce 复用被归约变量**：`my_def` 自身被改成 warp 和 → 后面前缀用错
4. **block 前缀只写 lane0**：缺 warp 内 reduce → total_def=19
5. **跨 SM 读 stall**：Phase 2 读别块刚写的 logits → writer 侧 `__threadfence`
6. **printf 污染测量**：单 block 打印拖慢整体 event 计时
7. **合并偏移错**：cand/cache 区域偏移计算错误 → 3.18% 精度
8. **寄存器直方图数据所有权**：lane 只数自己桶 → 漏数 31/32（早期 4-bit 尝试全 0）
9. **越界 insertion**（最早期）：binary search `lo==K` 时写入 `vvals[K]` 越界 → float 位模式污染 idx

---

## 12. 方法论沉淀

1. **分阶段计时先行**：每次结构改动前用 `clock64` 确认热点
2. **性能测试不带 printf**
3. **校准实验**：空扫描/空 syncwarp 建立基线，排除环境因素
4. **行为对比实验**：A/B（共享 vs 无共享）绕开 GPU counter 权限反推复用率
5. **回退纪律**：每个实验保留锚点版本，失败即回退
6. **精度失败看数字特征**：3.17% ≈ 16/512（kPerLane 边界 cap）、1065353216 = float 1.0 位模式——数值特征直接指向 bug 类型

---

## 13. 各步骤代码修改量

（行数为估算，按最终演化线 v0→v9）

| 步骤 | 改动内容 | 文件/函数 | 修改量 | 性质 |
|------|---------|----------|--------|------|
| v1 修复 148× 冗余 | Phase 2 token 循环改 SM-strided | core_impl Phase 2 | ~3 行 | 改循环 |
| v2 radix select | bubble sort + 流式插入 → 8-bit radix（直方图+前缀+cutoff+final） | `warp_bucket_topk` 整体 | ~100 行重写 | 算法替换 |
| v3 8-warp 分段+合并 | SMEM 布局（cache/hist/stage/cand/merge）、stage1 局部 TopK、stage2 合并、warp0 final | Phase 2 块整体 | ~150 行 | 结构重构 |
| v4 merge 读 SMEM + threadfence | merge 值来源改 ld_shared；Phase 1→2 加 fence | 2 处 | ~5 行 | 小修 |
| v5 去 printf | 删 `[TOPK8]`/`[SPLIT]` 及 clock64 计时 | 分散 | 删 ~15 行 | 清理 |
| v6 分布式 selection | stage4：8-warp count/stage/write/boundary + 块级前缀 | Phase 2 stage 3/4 | ~130 行 | 重构 |
| v6 的 3 个 bug 修复 | warp-reduce、独立归约变量、barrier 统一 | 各 | 各 1~3 行 | 修正 |
| v7 分布式 rounds | stage3：每 warp 私有 hist + warp0 归并 8×256 + gslots 广播 | Phase 2 stage 3 | ~100 行 | 重构 |
| v8 BF16 logits | 分配 dtype、3 个签名、2 处写入、2 处读入、data_ptr cast、void* 参数 | .cuh + 2 个 .hpp | ~15 行/6 处 | 类型迁移 |
| weight 向量读 | 64 标量 → 16 条 ld.shared.v4 | Phase 1 prelude | ~10 行 | 微优化 |
| v9 nibble（已回退） | rounds 换 4-bit 私有槽 + 步长修复 | `warp_bucket_topk` | ~90 行（净 0） | 实验+回退 |

### 辅助改动（非演化线）

| 内容 | 修改量 |
|------|--------|
| 部署文件（hpp JIT dispatch、attention API、python_api_additions） | ~600 行（一次性基础设施） |
| 测试/bench 脚本 | ~400 行（test_topk、bench_topk、bench_paged_mqa、L2 实验） |
| 文档（OPTIMIZATION_SUMMARY + JOURNEY） | ~500 行 |

### 统计

- **kernel 核心逻辑累计重写**：~500 行（v2/v3/v6/v7 四个重构是大头）
- **净增代码**（相对原始 DeepGEMM）：~800 行（kernel + host + API + 脚本）
- **已丢弃的实验代码**（two-level bucket、1-bit radix、寄存器直方图、协作版、nibble 等）：约 600 行，全部未进入最终版

**特征**：收益最大的三步（v3 分段、v6/v7 分布式化）都是"大改"；失败最多的也是"大改"——真正省时的是每个大改前后的分阶段计时和小步验证。