# DeepGEMM MQA Logits + TopK 融合算子优化总结

## 1. 背景与目标

在 DeepGEMM 的 `sm100_mqa_logits`（FP8 MQA Logits, BLOCK_Q=2, SPLIT_KV=256）基础上，**融合 TopK 选取流程**，使其输出与 QLI（quant_lightning_indexer）一致的 sparse_indices。

目标 shape（QLI 对应参数）：

| 参数 | 值 |
|------|-----|
| batch | 156 |
| next_n | 4 |
| num_heads | 64 |
| head_dim | 128 |
| seq_len_kv (s2) | 8192 |
| sparse_count (topk) | 512 |
| block_kv | 64 |

优化目标：融合 TopK 后整体耗时接近原始算子（Phase 1）水平。

## 2. 最终性能

| 阶段 | 起点 | 最终 | 提升 |
|------|------|------|------|
| Phase 1 (matmul) | ~50us | ~50us | 不变 |
| Phase 2 (TopK) | 545us | **132us** | 4.1x |
| 总计 | 594us | **182us** | 3.3x |

测试环境：B200，paged 模式，`fp8_paged_mqa_logits_with_topk` 接口。

## 3. 方案总览

kernel 分为两阶段：

```
Phase 1（原始 DeepGEMM 逻辑）:
    Q @ K^T → relu → × weights → sum over heads → logits_buffer (GMEM, float)

Phase 2（新增 TopK, 8-warp 分段 + 合并）:
    token (8192 floats)
      ├─ warp 0..7: 各取 1024 段 → SMEM cache(4KB) → 局部 radix TopK-512 → 候选
      ├─ 8 warps 协作: 合并 8×512=4096 候选 → merge_vals/merge_idx (SMEM 32KB)
      ├─ 分布式 radix rounds: 每 warp 扫自己的 512 片 + 各自 SMEM histogram
      │     → warp0 合并 8×256 hist → 全局 cutoff → 广播 kth
      └─ 分布式 selection: 8 warp × 512 片 count+stage → 块级前缀 → 写 definite + boundary
```

## 4. 优化步骤与证据链

### 4.1 分段测量（诊断基础）

在 Phase 1→2 边界用 `clock64()` 分段计时，先量化各阶段占比，拒绝盲改。

### 4.2 8-warp 分段 + 局部 TopK + 候选合并

- 原理：8192 元素的全局 TopK 拆成 8 个 1024 段的局部 TopK-512，再对 4096 候选做最终选择。
- 正确性依据：若元素不在本段 top-512 内，则本段有 ≥512 个比它大的元素，它必然不在全局 top-512 内。
- 收益：消除 token 级全量串行处理。

### 4.3 logits 行一次载入 SMEM

- 原理：Phase 2 的 4 轮 radix + 3 遍 final 全部扫描同一数据。把整行从 GMEM 读一次进 SMEM(每 warp 4KB cache)，后续全部读 SMEM。
- 收益：GMEM 读取 7 遍 → 1 遍 + 7 遍 SMEM 扫描。

### 4.4 显式 shared 指令

- `ptx::ld_shared/st_shared` 替代裸指针解引用，确保编译器生成真正的 `ld.shared`/`st.shared`，避免 generic load。
- 附带验证（后续 A/B 测试）：plain deref vs ld_shared vs 非 volatile 版本性能无差异——此条是"保险"而非收益来源。

### 4.5 消除 printf 对测量的污染

- 发现：5 个 token 的各阶段合计 ~115K cycles/token，但 phase2 实测 217K/token。
- 根因：设备侧 printf 极慢（~100K cycles/次），且只有 block0 打印 → block0 成为最后完成 block，拖长整个 event 计时。
- 移除全部调试打印后：phase2 1.08M → 实测真实值，收益巨大且常被忽视。

### 4.6 最终 radix 分布式化

- 原：warp0 单干 4096 候选（6 遍 × 128 iters = 768 串行迭代，83K cycles/token）。
- 改：rounds 由 8 warp 各扫 512 片（16 iters/轮）+ 各自 SMEM histogram + warp0 合并 8×256 → 全局 cutoff；selection 同样跨 8 warps。
- 收益：final 83K → ~15K/token。

### 4.7 barrier 统一性修复

- 根因：`__syncthreads()` 写在 `if (warp_idx < 8)` 内，warps 8-11（specialized）不执行 → CUDA 发散 barrier = UB，表现为块级前缀的读写竞态（精度 3.17% 的根源之一）。
- 修复：所有 `__syncthreads()` 提升到全 block 统一执行，仅具体工作用 `if` 守护。

## 5. 三个关键 Bug（踩坑记录）

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | 精度 3.17%（≈16/512 的边界收集特征） | `__syncthreads()` 发散，前缀竞态 | barrier 全 block 统一执行 |
| 2 | total_def=19（实际应 511） | block 前缀写 `gslots[8+w]=my_def` 只存了 lane0 的值，缺 warp 内 shfl 归约 | warp 内 xor-reduce 后再写槽位 |
| 3 | 输出稀疏留洞、垃圾值 1065353216 | 归约把 `my_def` 自身改成 warp 总和，后续 lane 内前缀用错值 | 归约结果存独立变量，保留 per-lane 原值 |

调试方法论：对矛盾数据（manual 重数 511 vs 前缀 19）下手，而非猜结构。空扫描校准（766 cyc/32 iters）证明基础扫描健康，把矛头指向原子争用。

## 6. 剩余差距与已定位根因

Phase 2 132us vs Phase 1 50us。校准实验结论：

- 空 SMEM 扫描：24c/it（健康）
- 8×syncwarp：20 cycles（便宜）
- **残余 ~10K/segment 来源于 round-0 热点桶 atomicAdd 串行化**：候选 top 值高度集中（单桶可达数百个元素），`atomicAdd` 打到同一地址完全串行化。

未实施的解法（已有明确设计，预期再降 ~2x）：

- 4-bit nibble（16 bin）+ 每 lane 私有 16 槽 SMEM histogram（零原子）+ shfl 归并：
  - 每轮扫描仍为 32 iters × ~26c（私有写不加锁）
  - 归并：lane j 归并桶 j（32 次读，无争用）
  - 8 轮合计 ≈ 12K vs 当前 23.7K

## 6.1 Phase 1 代码级优化验证记录

四个方向逐一实测：

| 方案 | 结果 | 数据依据 |
|------|------|---------|
| 1. Stage 深度（Q2/KV6） | 不可行 | SMEM 预算 231KB > 227KB 硬件上限 |
| 2. BF16 logits | 已落地，性能持平 | Phase 1 写减半仅省 ~1%，Phase 2 载入省 ~4us，均在噪声内 |
| 3. Weight 向量读（ld.shared.v4） | 已落地，中性 | 权重载入本非瓶颈（64→16 条指令，无可见收益） |
| 4. KV 页调度/L2 复用 | **空间 ≈ 0** | 见下方 A/B 实验 |

### KV 复用 A/B 实验（无 GPU counter 权限下的行为对比）

- A：156 batch × 4 token 共享 KV（156MB 唯一数据），Phase 1 = **51.8us**
- B：624 batch × 1 token 各自独立 KV（624MB，无共享可能），Phase 1 = **108.3us**
- A/B = 0.48，理论完美分组下界 = 0.25

上游源码核查（`scheduler/sm100_paged_mqa_logits.cuh` 的 metadata kernel）：

```
1. 按 REQUEST 计算工作量: work[r] = num_kv_splits × num_q_tokens
2. requests 工作量前缀和
3. 每 SM 分配一段连续的 requests 区间
```

**结论：一个 request 的全部 4 个 token 天然落在同一 SM**，调度级 KV 复用已最优。A/B=0.48 偏高是因为 B 场景本身含 4 倍 request 固定开销（~30us），而非 A 复用不足。A 的 51.8us 距综合下界（max(37us 算力, 19.5us 带宽) 重叠后 ≈ 42-45us）已到 ~85-90% 效率。

**BF16 logits 改动清单**（已落地，供参考）：

| 位置 | 改动 |
|------|------|
| `attention_with_topk.hpp` | logits_buffer 分配 `kFloat32` → `kBFloat16` |
| `.cuh` kernel 签名 | `float* logits_buffer` → `nv_bfloat16*`（core_impl + 2 个 entry） |
| Phase 1 写入 | `= result` → `= __float2bfloat16(result)`（compressed/非 compressed） |
| Phase 2 读入 | `__ldcg` → `__bfloat162float(logits_row[...])`（段缓存 + merge 兜底） |
| `.hpp` JIT | `data_ptr<nv_bfloat16>()` → `reinterpret_cast<nv_bfloat16*>(data_ptr())`（避免未定义符号） |
| `warp_bucket_topk` | logits 参数 `float*` → `void*`，内部按 kCached 分视图 |

## 7. 文件结构

| 文件 | 作用 |
|------|------|
| `sm100_mqa_logits_with_topk.cuh` | device kernel：分段局部 TopK + 候选合并 + 分布式 rounds/selection |
| `sm100_mqa_logits_with_topk.hpp` | host 侧 JIT dispatch（模板实例化 + launch） |
| `attention_with_topk.hpp` | C++ pybind11 API 注册 |
| `python_api_additions.py` | Python API 补丁（追加到 `deep_gemm/__init__.py`） |
| `bench_topk.py` | 分阶段性能测试（paged/non-paged、Phase1/Phase2 分离计时） |
| `test_topk.py` | 精度验证（与 PyTorch reference 对拍） |

## 8. 部署流程（新环境适配）

1. 复制 `sm100_mqa_logits_with_topk.cuh` → `deep_gemm/include/deep_gemm/impls/`
2. 复制 `sm100_mqa_logits_with_topk.hpp` → `csrc/jit_kernels/impls/`
3. 复制 `attention_with_topk.hpp` → `csrc/apis/embedded/attention_with_topk.hpp`
4. 在 `csrc/python_api.cpp` 注册 `register_topk_apis(m)`
5. Python API 追加到 `deep_gemm/__init__.py`
6. `./develop.sh && PYTHONPATH=. python test_topk.py` 验证精度，`bench_topk.py` 测性能

## 9. 经验教训

1. **数据先行**：每次结构改动前先用 clock64 分阶段计时确认热点，避免盲改。
2. **性能测试别带 printf**：设备侧 printf 的开销可彻底扭曲测量结论。
3. **CUDA 语法纪律**：发散 `__syncthreads` 是 UB；warp reduce 不要复用被归约变量。
4. **正确性优先于速度**：分布式实现失败多次只因两处代码笔误，先小步验证再扩展。