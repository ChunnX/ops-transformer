# 0706.py Double Buffer 优化方案

> 参考：`tilelang-ascend/examples/sparse_flash_attention/bench_sfa/sparse_mla_performance_optimization_zh.md`
> 约束：手动管理 double buffer（`set_flag`/`wait_flag`），不使用 `T.Pipelined` 做 inter-core 流水

## 1. 硬件约束

| 存储层级 | 容量 | 说明 |
|---------|------|------|
| UB | 192KB (保守) / 256KB (910B2 实测) | Vector 核本地存储 |
| L1 | 1MB | Cube 核本地存储 |
| L0A | 64KB | Cube 矩阵乘 A 矩阵缓存 |
| L0B | 64KB | Cube 矩阵乘 B 矩阵缓存 |
| L0C | 512KB | Cube 矩阵乘结果缓存 |
| MTE2 | GM→UB | 与 MTE3 可并发 |
| MTE3 | UB→GM | 与 MTE2 可并发 |

## 2. 当前各 Scope 内存使用

测试参数：`block_n=64, N1=128, Nidx1=64, G=128, vector_baseg=32, n_num=32, d_num_query_index=1`

### 2.1 V0 (Gather) — 可忽略

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| key_row_ub | (1, 512) | fp16 | 1KB |
| ki_row_ub | (1, 128) | fp16 | 256B |
| kr_row_ub | (1, 64) | fp16 | 128B |
| **合计** | | | **~1.4KB** |

### 2.2 C1 (Matmul1 + Matmul2)

**Matmul1 (T.Pipelined, num_stages=3):**

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| Q_L1 | (128, 128) | fp16 | 32KB |
| GK_L1 | (64, 128) | fp16 | 16KB |
| C_L0C | (128, 64) | fp32 | 32KB |
| L1 合计 | | | 48KB / 1MB |
| L0C 合计 | | | 32KB / 512KB |

**Matmul2 (serial):**

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| QI_L1 | (64, 128) | fp16 | 16KB |
| GKI_L1 | (64, 128) | fp16 | 16KB |
| C2_L0C | (64, 64) | fp32 | 16KB |

✅ C1 内存充裕，Matmul2 可加 Pipelined（对大 DQueryIndex 场景有帮助）

### 2.3 V1 (核心计算) — 关键瓶颈

**全局 buffer（s1_id 循环外分配）：**

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| bmm1_ub | (128, 64) | fp32 | 32KB |
| bmm2_ub | (64, 64) | fp32 | 16KB |
| weight_2d_ub | (64, 64) | fp32 | 16KB |
| sm_2d_ub | (128, 64) | fp32 | 32KB |
| rsy_tmp_ub | (32, 64) | fp32 | 8KB |
| rsy_all_ub | (32, 64) | fp32 | 8KB |
| 其余 1D buffer | ~14 个 | fp32 | ~2KB |
| **全局小计** | | | **~114KB** |

**s1_id 循环内分配：**

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| dw_chunk_ub | (32, 64) | fp32 | 8KB |
| rg_chunk_ub | (32, 64) | fp32 | 8KB |
| rg_half_ub | (32, 64) | fp16 | 4KB |
| ones_2d_ub | (32, 64) | fp32 | 8KB |
| diff_ub | (64,) | fp32 | 256B |
| dw_reduce_ub | (32,) | fp32 | 128B |
| dw_half_ub | (32,) | fp16 | 64B |
| loss 系列 (5×) | (64,) × 5 | fp32 | 1.25KB |
| **s1 内小计** | | | **~30KB** |

| **V1 总计** | | | **~144KB** |
| **UB 余量 (192KB)** | | | **~48KB** |
| **UB 余量 (256KB)** | | | **~112KB** |

### 2.4 C2 (Matmul4)

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| RG_L1 | (64, 64) | fp16 | 8KB |
| GKI2_L1 | (64, 128) | fp16 | 16KB |
| DQ_L0C | (64, 128) | fp32 | 32KB |

✅ 内存充裕

### 2.5 V3 (Cast) — 可忽略

| Buffer | Shape | dtype | Size |
|--------|-------|-------|------|
| dq_fp32_ub | (4, 128) | fp32 | 2KB |
| dq_fp16_ub | (4, 128) | fp16 | 1KB |
| **合计** | | | **~3KB** |

---

## 3. Double Buffer 机会排序（按 ROI）

### 🔴 #0: V0 Gather Chunked Double Buffer — 最高优先级（用户指定）

**现状** (行 233-248)：逐行 gather，串行 MTE2→MTE3：

```python
for k_id in T.serial(k_start, k_end):        # 64 行
    idx = SparseIndices[bz, s1_id, 0, k_id]
    T.copy(Key[idx], key_row_ub)             # MTE2: 随机读 1KB
    T.copy(key_row_ub, GatherKey_ws[k_id])   # MTE3: 顺序写 1KB
    T.copy(KeyIndex[idx], ki_row_ub)          # MTE2: 随机读 256B
    T.copy(ki_row_ub, GatherKeyIndex_ws[k_id]) # MTE3: 顺序写 256B
    ...KeyRope 同理...
```

**问题**：
1. 每行只有 ~1.4KB DMA，MTE3 写带宽利用率低（burst 太小）
2. MTE2 随机读和 MTE3 顺序写完全串行，无重叠
3. 64 行 × 6 次 T.copy = 384 次 DMA 调用

**方案**：Chunked gather + MTE2/MTE3 ping-pong（参考 SparseMLA `sparse_flash_attn_pa.py:169-210`）

```
将 n_block 行按 CHUNK_SIZE=16 切分为多个 chunk。
MTE2 逐行 gather 填 chunk buffer[0]，填满后 MTE3 整块写出。
同时 MTE2 开始填 chunk buffer[1]。
```

**流水示意**：
```
Chunk 0: [MTE2: read 16 rows → buf[0]] [signal] [MTE3: write buf[0] → ws]
Chunk 1:                                   [MTE2: read 16 rows → buf[1]] [signal] [MTE3: write buf[1] → ws]
Chunk 2:                                                                     [MTE2: read 16 rows → buf[0]] ...
                                            ↑ MTE2 fill buf[N+1] 与 MTE3 write buf[N] 并发
```

**Chunk 参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| CHUNK_SIZE | 16 | 整除所有 block_n (64/128/256/...) |
| 每 block chunk 数 | 4 (block_n=64) | k_end - k_start = 64 |
| MTE3 单次写大小 | 16×512×2 = 16KB (Key) | 比逐行 1KB 大 16× |

**UB 分配**（V0 scope 独占，不与其他 scope 冲突）：

| Buffer | Shape | dtype | 单份 | ×2 |
|--------|-------|-------|------|-----|
| key_chunk_ub | (16, 512) | fp16 | 16KB | 32KB |
| ki_chunk_ub | (16, 128) | fp16 | 4KB | 8KB |
| kr_chunk_ub | (16, 64) | fp16 | 2KB | 4KB |
| **合计** | | | **22KB** | **44KB** |

✅ V0 独占 UB，44KB / 192KB = 23%，充裕

**实现伪代码**：

```python
CHUNK_SIZE = 16
num_chunks = n_block // CHUNK_SIZE  # block_n 必定整除 CHUNK_SIZE

with T.Scope("V"):
    key_chunk_ub = T.alloc_ub((2, CHUNK_SIZE, DQuery), input_dtype)
    ki_chunk_ub = T.alloc_ub((2, CHUNK_SIZE, DQueryIndex), input_dtype)
    kr_chunk_ub = T.alloc_ub((2, CHUNK_SIZE, max(d_rope, 1)), input_dtype)

    for s1_id in T.serial(s1_start, s1_end):
        # ---- 预加载 chunk 0 ----
        for i in T.serial(CHUNK_SIZE):
            k_id = k_start + i
            idx = SparseIndices[bz, s1_id, 0, k_id]
            if idx >= 0:
                T.copy(Key[bz, idx, 0, 0:DQuery], key_chunk_ub[0, i, 0:DQuery])
                T.copy(KeyIndex[bz, idx, 0, 0:DQueryIndex], ki_chunk_ub[0, i, 0:DQueryIndex])
                if d_rope > 0:
                    T.copy(KeyRope[bz, idx, 0, 0:d_rope], kr_chunk_ub[0, i, 0:d_rope])
        T.set_flag("mte2", "mte3", 0)  # MTE2 → MTE3: buf[0] 就绪

        # ---- Ping-pong: chunk 0, 1, 2, ... ----
        for chunk_id in T.serial(num_chunks):
            task_id = chunk_id % 2
            nxt_id = 1 - task_id
            chunk_start = k_start + chunk_id * CHUNK_SIZE

            # 等待 MTE2 填完当前 chunk
            T.wait_flag("mte2", "mte3", task_id)

            # MTE3: 整块写出到 workspace (与下一次 MTE2 fill 并发!)
            T.copy(key_chunk_ub[task_id, 0:CHUNK_SIZE, 0:DQuery],
                   GatherKey_ws[bz, s1_id, chunk_start:chunk_start+CHUNK_SIZE, 0:DQuery])
            T.copy(ki_chunk_ub[task_id, 0:CHUNK_SIZE, 0:DQueryIndex],
                   GatherKeyIndex_ws[bz, s1_id, chunk_start:chunk_start+CHUNK_SIZE, 0:DQueryIndex])
            if d_rope > 0:
                T.copy(kr_chunk_ub[task_id, 0:CHUNK_SIZE, 0:d_rope],
                       GatherKeyRope_ws[bz, s1_id, chunk_start:chunk_start+CHUNK_SIZE, 0:d_rope])

            # 通知 MTE2: buf[task_id] 已写出，可复用
            T.set_flag("mte3", "mte2", task_id)

            # MTE2: 预加载下一个 chunk (与 MTE3 write 并发!)
            if chunk_id < num_chunks - 1:
                next_start = k_start + (chunk_id + 1) * CHUNK_SIZE
                for i in T.serial(CHUNK_SIZE):
                    k_id = next_start + i
                    idx = SparseIndices[bz, s1_id, 0, k_id]
                    if idx >= 0:
                        T.copy(Key[bz, idx, 0, 0:DQuery], key_chunk_ub[nxt_id, i, 0:DQuery])
                        T.copy(KeyIndex[bz, idx, 0, 0:DQueryIndex], ki_chunk_ub[nxt_id, i, 0:DQueryIndex])
                        if d_rope > 0:
                            T.copy(KeyRope[bz, idx, 0, 0:d_rope], kr_chunk_ub[nxt_id, i, 0:d_rope])
                T.set_flag("mte2", "mte3", nxt_id)  # 通知 MTE3: buf[nxt] 就绪

    T.set_cross_flag("V", 10)
```

**Flag 使用**：

| Flag ID | 方向 | 含义 |
|---------|------|------|
| 0, 1 | MTE2→MTE3 | chunk buffer 填完，可以写出 |
| 0, 1 | MTE3→MTE2 | chunk buffer 已写出，可以覆盖 |

**预期收益**：
- MTE3 写出从 64 次 ×1.4KB → 4 次 ×22KB，burst 效率大幅提升
- MTE2 填 chunk[N+1] 与 MTE3 写 chunk[N] 完全重叠 → V0 时间约减半
- MTE2 连续 16 次随机读（同 buffer 不同行），cache 命中率可能略有改善

**风险**：

| 风险 | 缓解 |
|------|------|
| `T.copy` 切片 `chunk_start:chunk_start+CHUNK_SIZE` 需编译期确定 shape | 源/目标 shape 均为静态 `(CHUNK_SIZE, D)`，仅偏移量为运行时变量。当前逐行 `T.copy` 已有运行时 `k_id` 偏移，应兼容 |
| `SparseIndices[..., k_id]` 中 `k_id` 在两处出现（预加载 + 循环内），可能导致 flag 冲突 | 预加载只执行一次（第一个 chunk），后续 chunk 统一在循环体末尾预加载 |
| idx < 0 的行在 chunk buffer 中为未初始化数据 | 预置条件：workspace 已由 host `torch.zeros` 初始化。或增加 `else: T.tile.fill(row, 0.0)` |

---

### 🥇 #1: Global SoftmaxI bmm2 循环 — ROI 最高

**目标循环** (行 361-372):
```python
for n_rsy in T.serial(n_num):  # 32 次迭代!
    T.copy(bmm2Res[...], bmm2_ub)     # DMA load: 16KB
    T.tile.mul(bmm2_ub, bmm2_ub, weight_2d_ub)  # VEC
    T.reduce_sum(bmm2_ub, rsy_reduce_ub, 0)     # VEC
    T.copy(rsy_reduce_ub, rsy_all_ub[n_rsy, :]) # UB copy
    T.copy(rsy_reduce_ub, ReduceSumY[...])       # DMA store
    ...max update...
```

**方案**：将 `bmm2_ub` 翻倍为 `bmm2_ub[2]`，ping-pong 交替：

```
迭代 N:   [DMA load→buf[0]] [VEC compute buf[0]] [DMA store result]
迭代 N+1:                     [DMA load→buf[1]]   [VEC compute buf[1]] [DMA store]
                              ↑ MTE2 load 与 VEC compute 重叠
```

**同步机制**（手动 `set_flag`/`wait_flag`，参考 SparseMLA `no_cv_pipeline.py:241-261`）:

`T.set_flag(producer, consumer, flag_id)` — producer 通知 consumer 数据就绪
`T.wait_flag(producer, consumer, flag_id)` — consumer 等待 producer

```python
# ping-pong buffer: bmm2_ub[0] 和 bmm2_ub[1] 交替
bmm2_ub = T.alloc_ub((2, Nidx1, n_block), calc_dtype)  # 2×(64,64) fp32 = 32KB

# 预加载第一块: MTE2 → UB[0]
T.copy(bmm2Res[bz, s1_id, 0:Nidx1, 0:n_block], bmm2_ub[0, :, :])
T.set_flag("mte2", "v", 0)  # MTE2 通知 VEC: buf[0] 就绪

for n_rsy in T.serial(n_num):
    task_id = n_rsy % 2
    nxt_id = 1 - task_id

    # 等待 MTE2 把当前块搬完
    T.wait_flag("mte2", "v", task_id)

    # === VEC 计算当前块 (与下一块的 MTE2 load 并发) ===
    T.tile.mul(bmm2_ub[task_id, :, :], bmm2_ub[task_id, :, :], weight_2d_ub)
    T.reduce_sum(bmm2_ub[task_id, :, :], rsy_reduce_ub, 0)
    T.copy(rsy_reduce_ub, rsy_all_ub[n_rsy, :])

    # MTE3 写出当前结果 (异步，与 VEC 计算下一块并发)
    T.copy(rsy_reduce_ub, ReduceSumY[bz, s1_id, n_rsy * n_block:(n_rsy + 1) * n_block])
    T.set_flag("v", "mte3", task_id)

    # ... max update (VEC) ...

    # 预加载下一块: MTE2 → UB[nxt] (与 VEC 计算当前块并发!)
    if n_rsy < n_num - 1:
        next_start = (n_rsy + 1) * n_block
        T.copy(bmm2Res[bz, s1_id, 0:Nidx1, next_start:next_start + n_block],
               bmm2_ub[nxt_id, :, :])
        T.set_flag("mte2", "v", nxt_id)  # 通知 VEC: buf[nxt] 就绪
```

**流水示意**：
```
n_rsy=0: MTE2→buf[0] | VEC wait | VEC compute buf[0] | MTE3 store
n_rsy=1:               MTE2→buf[1] (与↑并发)          | VEC wait | VEC compute buf[1] | MTE3 store
n_rsy=2:                             MTE2→buf[0] (与↑并发)          | VEC wait | VEC compute buf[0] | MTE3 store
```

> **关键**：`T.copy` 在 Ascend 上默认是异步的（由 DMA 引擎执行），`T.set_flag`/`T.wait_flag` 提供精确的 producer→consumer 同步。`T.copy` 发起后立即返回，VEC 可以同时执行 `T.tile.*` 操作。

**额外 UB 开销**：`+1 × (64, 64) fp32 = +16KB`

**V1 总计后**：144 + 16 = **160KB** ✅ 192KB 安全

**预期收益**：
- 32 次 DMA load 各 ~16KB，总 DMA 时间 ~32×18ns ≈ 0.6μs
- 如果 VEC compute 时间 ≥ DMA load 时间，DMA 完全被掩盖
- 节省 ~3-5% V1 scope 时间

---

### 🥈 #2: dWeight/reluGrad g_id 循环 — 次高 ROI

**目标循环** (行 426-459):
```python
for g_id in range(g_num):  # g_num=2 (随 Nidx1 增大而增加)
    T.copy(bmm2Res[...], dw_chunk_ub)  # DMA load: 8KB
    
    for i in range(vector_baseg):      # 32 次向量迭代
        ...VEC compute...
    
    T.tile.cast(rg_half_ub, rg_chunk_ub, ...)
    T.barrier_all()
    T.copy(rg_half_ub, reluGradRes[...])    # DMA store: 4KB
    T.tile.cast(dw_half_ub, weight_ub, ...)
    T.barrier_all()
    T.copy(dw_half_ub, dWeight_partial[...]) # DMA store: 64B
```

**方案**：将 chunk buffer 翻倍，MTE2 load 下一块与 VEC compute + MTE3 store 当前块三路并发：

```
g_id=0:      [MTE2 load→dw_A] [VEC 32×→rg_A, tmp_A] [cast→rg_half_A] [MTE3 store A]
g_id=0→1:                                                          [MTE2 load→dw_B] ← 并发!
g_id=1:                                                             [VEC→rg_B] [cast→rg_half_B] [MTE3 store B]
```

**同步示意**：
```python
dw_chunk_ub = T.alloc_ub((2, vector_baseg, n_block), calc_dtype)
rg_chunk_ub = T.alloc_ub((2, vector_baseg, n_block), calc_dtype)
rg_half_ub = T.alloc_ub((2, vector_baseg, n_block), input_dtype)
rsy_tmp_ub = T.alloc_ub((2, vector_baseg, n_block), calc_dtype)

# 预加载第一块
T.copy(bmm2Res[..., g_start:g_start+vector_baseg, n_idx_k*n_block:(n_idx_k+1)*n_block],
       dw_chunk_ub[0, :, :])
T.set_flag("mte2", "v", 10)

for g_id in range(g_num):
    task_id = g_id % 2
    nxt_id = 1 - task_id
    g_start = g_id * vector_baseg

    T.wait_flag("mte2", "v", 10 + task_id)

    # VEC compute on dw_chunk_ub[task_id] → rg_chunk_ub[task_id], rsy_tmp_ub[task_id]
    for i in range(vector_baseg):
        idx = g_start + i
        T.tile.mul(rsy_tmp_ub[task_id, i, :], dw_chunk_ub[task_id, i, :], -10000.0)
        ...

    T.tile.cast(rg_half_ub[task_id, :, :], rg_chunk_ub[task_id, :, :], ...)
    T.barrier_all()
    T.copy(rg_half_ub[task_id, :, :], reluGradRes[...])  # MTE3 store
    T.set_flag("v", "mte3", 20 + task_id)

    # 预加载下一块
    if g_id < g_num - 1:
        next_start = (g_id + 1) * vector_baseg
        T.copy(bmm2Res[..., next_start:next_start+vector_baseg, ...],
               dw_chunk_ub[nxt_id, :, :])
        T.set_flag("mte2", "v", 10 + nxt_id)
```

**需要翻倍的 buffer**：

| Buffer | Shape | dtype | 单份 | ×2 |
|--------|-------|-------|------|-----|
| dw_chunk_ub | (32, 64) | fp32 | 8KB | 16KB |
| rg_chunk_ub | (32, 64) | fp32 | 8KB | 16KB |
| rg_half_ub | (32, 64) | fp16 | 4KB | 8KB |
| rsy_tmp_ub | (32, 64) | fp32 | 8KB | 16KB |
| **合计新增** | | | | **+32KB** |

> `rsy_tmp_ub` 需从全局区移入循环内，因为它与 g_id 绑定。

**V1 总计后**：144 + 32 = **176KB** ✅ 192KB 安全

如果同时做 #1 + #2：144 + 16 + 32 = **192KB** ⚠️ 刚好顶满 192KB（编译器临时变量可能导致超限）

**建议**：910B2 如果是 256KB UB，两者都做。如果是 192KB，优先 #1 + 简化版 #2（只 double dw_chunk_ub 和 rg_chunk_ub，rsy_tmp_ub 保持共享 → 144 + 16 + 16 = 176KB）。

---

### #4: C1 Matmul2 Pipelined — 低 ROI（当前测试用例）

当前 Matmul2 是 `T.serial(d_num_query_index)`。对于测试用例 DQueryIndex=128 → d_num_query_index=1，无收益。但对于 DQueryIndex=512 场景有益。

**方案**：仿照 Matmul1，改为 `T.Pipelined(d_num_query_index, num_stages=2)`。这是 `T.Pipelined` 的核内用法（不是 inter-core CV 流水），可以参考。

**L1 额外开销**：需 double QI_L1 或 GKI_L1 → +16KB L1（L1 有 1MB，充裕）。

---

## 4. 实施顺序

```
Phase 0 (当前优先 — V0 double buffer):
  #0: V0 Gather chunked double buffer           → +44KB UB (V0 独占)
  CHUNK_SIZE=16, MTE2↔MTE3 ping-pong

Phase 1 (低风险):
  #1: Global SoftmaxI bmm2 循环 double buffer    → +16KB UB (V1)
  #4: C1 Matmul2 Pipelined (对多 DQueryIndex)    → +16KB L1

Phase 2 (需要确认 UB 容量):
  #2: dWeight/reluGrad g_id 循环 double buffer    → +32KB UB (V1)
  前提: #1 完成且 UB 有余量

Phase 3 (后续):
  #3: 其他精细化优化
```

## 5. 关键技术点

### 5.1 set_flag / wait_flag API（参考 SparseMLA 实现）

API 签名：`T.set_flag(producer, consumer, flag_id)` / `T.wait_flag(producer, consumer, flag_id)`

| 单元名 | 含义 | 说明 |
|--------|------|------|
| `"mte2"` | MTE2 DMA 引擎 | GM→UB/L1 数据搬运 |
| `"mte3"` | MTE3 DMA 引擎 | UB→GM 数据搬运 |
| `"v"` | Vector 核 | V scope 内的向量计算 |
| `"mte1"` | MTE1 DMA 引擎 | L1→L0A/L0B 数据搬运 |
| `"m"` | Cube 核 | C scope 内的矩阵计算 |

**Producer→Consumer 方向**：
- `T.set_flag("mte2", "v", id)` — MTE2 通知 VEC: "数据已搬到 UB"
- `T.wait_flag("mte2", "v", id)` — VEC 等待 MTE2: "等数据就绪"
- `T.set_flag("v", "mte3", id)` — VEC 通知 MTE3: "结果已算出，可以搬出"
- `T.set_flag("m", "mte1", id)` — Cube 通知 MTE1: "已消费完 L0 数据，可以覆盖"

**flag_id 约定**：使用不同的 ID 范围避免不同流水线冲突（如 SparseMLA 中 BMM1 用 0-1, BMM2 用 2-3, Gather 用 0-1）。

### 5.2 T.copy 异步语义

`T.copy` 在 Ascend 上是**异步**的 — 调用后立即返回，DMA 引擎在后台执行搬运。VEC 计算（`T.tile.*`）可以与 DMA 并发。`T.set_flag`/`T.wait_flag` 提供显式的生产者→消费者同步。

`T.barrier_all()` 是**同步**的 — 等待所有在途 DMA 完成。Double buffer 需要移除中间的 `T.barrier_all()`，改为精确的 `set_flag`/`wait_flag`。

### 5.3 MTE2 / MTE3 并发

Ascend 架构中 MTE2 (GM→UB) 和 MTE3 (UB→GM) 是独立的 DMA 引擎，可并发执行。利用这一点：
- 同时发起 MTE3 store（当前结果写出）和 MTE2 load（下一块数据读入）
- 两者与 VEC compute 三路并发

### 5.4 UB Buffer 生命周期管理

Double buffer 引入后，需要注意：
- ping-pong buffer 的 `task_id = iteration % 2` 在每次迭代正确翻转
- 首尾迭代的边界处理（第一个块需预加载，最后一个块不需预加载下一块）
- 全局 buffer 移入循环内需改为 2× 版本
- `weight_ub`（跨 g_id 累加）和 `weight_all_ub`（跨 g_id 不变）保持单份

## 6. 风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| `T.set_flag`/`T.wait_flag` 在 V scope 的行为与 SparseMLA 不完全一致 | 🟡 中 | 先用最简单的 #1 方案验证，成功后再扩展 |
| UB 超限 (192KB) | 🟡 中 | 分阶段实施；910B2 若为 256KB UB 则安全 |
| g_num=2 场景下 #2 double buffer 收益不明显 | 🟢 低 | #2 对大 Nidx1 (g_num 大) 场景更有效；先做 #1 |
| flag_id 冲突（多个流水线同时用相同 ID） | 🟡 中 | #1 用 flag_id=0,1（MTE2↔VEC）；#2 用 flag_id=10,11（MTE2↔VEC）, 20,21（VEC↔MTE3） |
