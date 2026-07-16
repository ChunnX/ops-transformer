# 支持Token粒度一致性方案

🎯

* 目标：支持token粒度一致性；例如：

    * prefill\(input\) \-> 1个token

    * n \* decode\(tokenX\) \-> n个token

    * response = 1\*prefill阶段token + n\*decode阶段token

    * 目标：prefill\(input\) + n\*decode\(tokenX\) = prefill\(input+response\)

* 思路

    * 每个token的切分不受其他token影响

    * 为了兼顾性能，默认kv seqlen开多核

## 1. 背景

**输入输出张量形状：**

```auth
Q:          [B, N1, S1, D]    输入
K:          [B, N2, S2, D]    输入 (S2 为最大KV序列长度，实际有效长度per batch不同)
V:          [B, N2, S2, D]    输入
O:          [B, N1, S1, D]    输出
seqused_kv: [B]               每个batch的实际KV序列长度
```

|符号|含义|
|---|---|
|B|batch数|
|N1|Query的head数|
|N2|KV的head数|
|S1|Q的序列长度|
|S2|KV的最大序列长度（规约维度）|
|S2\_*b = seqused\_*kv\[b\]|第 b 个 batch 的实际KV序列长度；对于smla，应是参与attention计算的token数量；后面为了书写方便，不再强调；|
|D|head维度 \(head dim\)|
|G = N1/N2|每个KV head对应的Q head数（group size），作为矩阵乘的m轴|

每个 batch 的实际 KV 长度通过 `seqused_kv[b]` 指定，不同 batch 的 S2\_b 可以不同，因此各 batch 的规约块大小和数量也不同。

在 GQA/MQA 场景下，每 G 个 Q head 共享同一组 KV head。将这 G 个 Q head 作为矩阵乘法的 m 轴进行批量计算。因此，每个 `(b, n2, s1)` 位置独立执行相同的 FA 计算规则：对 `q[G, D]`（G = N1/N2）与完整的 `K[S2_b, D]`、`V[S2_b, D]` 做 attention。目标是对 S2\_b 维做多级 tiling，实现分块计算与多核并行。

## 2. Tiling层级划分

### 2.1 全局视图

```auth
Batch 维 [B]  (每个batch的S2_b可不同)
└── KV Head 维 [N2] (每个KV head独立计算)
    └── Q序列维 [S1] (每个query position独立计算)
        │
        │  每个计算单元: q[G, D] (G = N1/N2 个Q head作为m轴)
        │
        └── KV序列维 S2_b (规约维，per batch不同，做多级tiling)
            ├── Level-1: 规约级 (Reduction Tile)
            │   大小 = R_b = align_up(S2_b/32, 128)
            │   ┌────────────────────────────────┐
            │   │  Level-2   Level-2   Level-2   │
            │   │  [128]     [128]     [128]     │
            │   └────────────────────────────────┘
            ├── Level-1: 规约级
            │   ...
            └── Level-1: 规约级
```

### 2.2 Per\-Batch 切分公式

每个 batch 的规约块大小和数量独立计算：

```auth
S2_b = seqused_kv[b]                    // 第 b 个 batch 的实际 KV 长度
R_b  = align_up(S2_b / 32, 128)         // 第 b 个 batch 的规约块大小
num_tiles_b = S2_b / R_b                 // 第 b 个 batch 的规约块数量
```

|层级|切分维度|切分大小|约束|
|---|---|---|---|
|Level\-1|S2\_b \(规约维\)|`R_b = align_up(S2_b / 32, 128)`|多核调度的最小单位，per batch|
|Level\-2|S2\_b \(规约维\)|`128` \(固定\)|片上计算的基本块|

**示例：**

|batch|S2*b \(seqused*kv\)|S2\_b / 32|R\_b \(对齐128\)|num*tiles*b|
|---|---|---|---|---|
|0|4096|128|128|32|
|1|8192|256|256|32|
|2|2048|64|128|16|
|3|5120|160|256|20|

### 2.3 总负载与任务总量

**总负载不是规约块的数量，而是所有 batch 的 负载 累加和，可沿用当前模板的总负载计算；**

```auth
# 示例：
total_workload = sum(S2_b for b in range(B))      // 真实计算量的度量
total_tasks    = sum(num_tiles_b for b in range(B)) × N2 × S1
```

每个 `(b, n2, s1, tile)` 四元组对应一个独立的计算任务，产出一个 partial result（形状为 `[G, D]`）。不同 batch 的任务计算量可能不同（R\_b 不同）。

## 3. 计算流程

### 3.1 单规约块内部（Level\-2 循环）

一个规约块包含 `R / 128` 个基本块，在单核内串行执行：

```auth
输入: q[G, D], K_tile[R, D], V_tile[R, D]    (G = N1/N2)
输出: partial_O[G, D], partial_m[G], partial_l[G]

m = -inf, l = 0, O = 0          # 初始化，形状均含G维

for j in range(R // 128):       # Level-2 循环
    Kj = K_tile[j*128 : (j+1)*128, :]    # [128, D]
    Vj = V_tile[j*128 : (j+1)*128, :]    # [128, D]

    Sj = q @ Kj^T / sqrt(D)              # [G, 128]  mm, m轴=G
    m_new = max(m, rowmax(Sj))            # online softmax [G]
    Pj = exp(Sj - m_new)                 # [G, 128]
    l = exp(m - m_new) * l + rowsum(Pj)
    O = exp(m - m_new) * O + Pj @ Vj     # 累加 [G, D]  mm, m轴=G
    m = m_new

partial_O = O      # [G, D]
partial_m = m      # [G]
partial_l = l      # [G]
```

` 备注 ` 这里的partial\_l使用的业界写法，可仍保留当前的sum方式；

### 3.2 跨规约级合并（Combine）

所有规约级完成后，合并各 partial result：

```auth
输入: {partial_O_i[G, D], partial_m_i[G], partial_l_i[G]}, i = 0..num_tiles-1
输出: final_O[G, D]

# 全局最大值
M = max(partial_m_0, partial_m_1, ..., partial_m_{n-1})   # [G]

# 重新缩放并合并
L = 0, O = 0
for i in range(num_tiles):
    scale_i = exp(partial_m_i - M)        # [G]
    L += scale_i * partial_l_i
    O += scale_i * partial_O_i            # [G, D]

final_O = O / L                           # [G, D]
```

## 4. 多核切分策略

**约束：多核只能以规约块（Level\-1 tile）为调度单位，一个核可承载多个规约块。**

### 4.1 分配原则

**负载度量：以 S2 累加和为基准，而非规约块数量。**

不同 batch 的规约块大小 R\_b 不同，因此不能简单按规约级个数均分。以实际计算量（S2 累加和）均衡分配：

```auth
total_workload = sum(S2_b for b in range(B))   // 所有batch的KV长度之和
workload_per_core = total_workload / C          // C = 32

// 分配时，按规约级粒度贪心填充，使每核负载接近 workload_per_core
// 每个规约级的负载 = R_b (该规约级覆盖的KV长度)
```

每个核串行处理分配到的多个规约级，每个规约级独立输出一个 partial result，核内根据情况决定是否就地 combine（见 4.5 节）。

🎯

仅为示例，可沿用当前的负载均衡算法；唯一不同：以规约块为粒度；

### 4.2 示例：变长 batch 的负载分配

```auth
假设 B=3, N2=1, S1=1, C=4 核 (简化为单head单token):
  seqused_kv = [4096, 2048, 5120]
  R_0 = align_up(4096/32, 128) = 128,  num_tiles_0 = 32, 负载 = 4096
  R_1 = align_up(2048/32, 128) = 128,  num_tiles_1 = 16, 负载 = 2048
  R_2 = align_up(5120/32, 128) = 256,  num_tiles_2 = 20, 负载 = 5120

  total_workload = 4096 + 2048 + 5120 = 11264
  workload_per_core = 11264 / 4 = 2816

按负载贪心分配:
  Core 0: batch0 tile[0..21]   负载 = 22×128 = 2816
  Core 1: batch0 tile[22..31] + batch1 tile[0..5]   负载 = 10×128 + 6×128 = 2048 (欠)
  Core 2: batch1 tile[6..15] + batch2 tile[0..2]    负载 = 10×128 + 3×256 = 2048 (欠)
  Core 3: batch2 tile[3..19]  负载 = 17×256 = 4352 (超)

  注: 由于规约级粒度约束，实际分配无法完美均衡，
      但以 workload_per_core 为目标贪心填充可最小化最大负载差异。
```

### 4.3 示例：单 batch 场景

```auth
S2_0 = 2048, R_0 = 128, 16个规约块, C = 4核:

workload_per_core = 2048 / 4 = 512 (即每核 4 个规约块)

Core 0: batch0 tile[0..3]   [K[0:512],    V[0:512]]    → partial_0..3
Core 1: batch0 tile[4..7]   [K[512:1024], V[512:1024]]  → partial_4..7
Core 2: batch0 tile[8..11]  [K[1024:1536],V[1024:1536]] → partial_8..11
Core 3: batch0 tile[12..15] [K[1536:2048],V[1536:2048]] → partial_12..15
                              ↓
               Combine: 合并 partial_0..15 → final_O
```

### 4.4 核内执行流程（一核多规约块）

```python
# Core i 负责规约级 [start_tile, end_tile)，处理某个 (b, n2, s1) 的计算

q = Q[b, n2*G:(n2+1)*G, s1, :]           # [G, D]  (G = N1/N2 个Q head)

for t in range(start_tile, end_tile):       # 遍历本核的各规约块
    # --- Level-2 循环：计算当前规约块 ---
    m = -inf, l = 0, O = 0
    for j in range(R // 128):
        Kj, Vj = load_block(b, n2, t, j)   # [128, D]
        Sj = q @ Kj^T / sqrt(D)            # [G, 128]
        m_new = max(m, rowmax(Sj))
        Pj = exp(Sj - m_new)
        l = exp(m - m_new) * l + rowsum(Pj)
        O = exp(m - m_new) * O + Pj @ Vj   # [G, D]
        m = m_new

    # --- 直接写回当前规约块的 partial result ---
    write_back(partial_O[t] = O, partial_m[t] = m, partial_l[t] = l)
```

### 4.5 Combine 策略（基于 Workspace 优化）

根据某个 token 的规约级是否全部落在同一个核，分两种 combine 路径：

**假定核数 C = 32。**

#### 情况一：同核 Combine（规约块全在同一核内）

当某个 `(b, n2, s1)` 的所有规约块均分配在同一个核时，该核在计算完所有规约块后，**就地完成 combine**，直接输出最终结果，无需跨核通信。

```python
# Core i 负责某 (b, n2, s1) 的全部规约块
# 计算完所有 partial result 后，立即在核内 combine

M = max(partial_m[0], ..., partial_m[num_tiles-1])  # [G]

L = 0, O = 0
for t in range(num_tiles):
    scale_t = exp(partial_m[t] - M)
    L += scale_t * partial_l[t]
    O += scale_t * partial_O[t]

O_final[b, n2*G:(n2+1)*G, s1, :] = O / L   # 直接写回最终结果
```

#### 情况二：跨核 Combine（规约块分布在多个核上）

当某个 `(b, n2, s1)` 的规约块分布在多个核时，各核将 partial result 写入共享 workspace，待所有核执行完成后再统一 combine。

```python
# 各核将 partial result 写入 workspace
workspace_O[core_id, tile_local_idx] = O     # [G, D]
workspace_m[core_id, tile_local_idx] = m     # [G]
workspace_l[core_id, tile_local_idx] = l     # [G]

# --- 同步屏障：等待所有核完成 ---

# Combine 阶段：汇总 workspace 中所有 partial result
M = max(所有相关 partial_m)                   # [G]
L = 0, O = 0
for each partial in workspace:
    scale = exp(partial_m - M)
    L += scale * partial_l
    O += scale * partial_O

O_final[b, n2*G:(n2+1)*G, s1, :] = O / L
```

### 4.6 Workspace 空间设计

```auth
┌─────────────────────────────────────────────────────────────┐
│  核内 Workspace (per core)                                   │
│                                                             │
│  大小: C × C × partial_result_size                           │
│      = 32 × 32 × (G×D + G + G)                                  │
│                                                             │
│  用途: 存放同核 token 的各规约块 partial result,              │
│        计算完即刻 combine, 空间可复用                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  跨核 Workspace (共享)                                       │
│                                                             │
│  大小: C × C × partial_result_size                           │
│      = 32 × 32 × (G×D + G + G)                             │
│                                                             │
│  用途: 跨核 token 的 partial result 暂存,                     │
│        每个跨核token最多贡献 C 个 partial result,              │
│        共 C 个核 → C×C 个槽位                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**空间汇总 \(C = 32\)：**

|Workspace 类型|大小|说明|
|---|---|---|
|核内 workspace \(per core\)|`32 × (G×D + 2G)`|同核 combine 的中间存储，可复用|
|跨核 workspace \(全局共享\)|`32 × 32 × (G×D + 2G)`|跨核 partial result 暂存|
|总计|`32 × 32 × (G×D + 2G) + 32 × 32 × (G×D + 2G)`|核内部分合计 = 全局部分|

简化计算（以 G=8, D=128 为例）：

```auth
partial_result_size = G×D + G + G = 8×128 + 8 + 8 = 1040 元素

核内 workspace (per core): 32 × 1040 = 33,280 元素
跨核 workspace (全局):     32 × 32 × 1040 = 1,064,960 元素
```

### 4.7 核内执行流程（含 Workspace 优化）

```python
# Core i 的执行流程，假定核数 C = 32

for each (b, n2, s1, tile) assigned to this core:
    # --- Level-2 循环：计算当前规约块 ---
    q = Q[b, n2*G:(n2+1)*G, s1, :]         # [G, D]
    m = -inf, l = 0, O = 0
    for j in range(R // 128):
        Kj, Vj = load_block(b, n2, tile, j) # [128, D]
        Sj = q @ Kj^T / sqrt(D)             # [G, 128]
        m_new = max(m, rowmax(Sj))
        Pj = exp(Sj - m_new)
        l = exp(m - m_new) * l + rowsum(Pj)
        O = exp(m - m_new) * O + Pj @ Vj    # [G, D]
        m = m_new

    # --- 根据 token 的规约块分布选择路径 ---
    if token_all_on_this_core(b, n2, s1):
        # 情况一: 存入核内 workspace，累积完后就地 combine
        local_buf[local_idx] = (O, m, l)
        if is_last_tile_of_token(b, n2, s1):
            combine_and_write_final(local_buf, b, n2, s1)
    else:
        # 情况二: 存入跨核 workspace，等待全局 combine
        cross_workspace[core_id, slot_idx] = (O, m, l)

# --- 同步屏障 ---
# 对跨核 token 执行全局 combine
```

### 4.8 调度规则

1. **分配粒度**：以完整规约块为最小调度单位，不可拆分 Level\-1 tile 跨核

2. **负载均衡基准**：以`sum(S2_b) / C`为每核目标负载，按规约块粒度做负载均衡

3. **一核多级**：每核可承载来自不同 batch 的多个规约块

4. **同核 combine 优化**：若某 token`(b, n2, s1)`的全部规约块落在同一核，核内即刻 combine，节省跨核通信与 workspace

5. **跨核 combine**：规约块跨核的 token，partial result 写入共享 workspace，待所有核完成后统一 combine

## 5. 参数选择建议

|参数|选择依据|
|---|---|
|`G` \(m轴大小\)|`N1 / N2`（每个KV head对应的Q head数）|
|`R_b` \(Level\-1大小\)|`align_up(S2_b / 32, 128)`（per batch）|
|`num_tiles_b`|`S2_b / R_b`（第 b 个 batch 每个 \(n2,s1\) 的规约块数）|
|`D` \(head dim\)|模型固定，常见值 64/128/256|
|`C` \(核数\)|32|
|`workload_per_core`|`sum(S2_b) / C`（负载均衡目标）|

## 6. 存储布局

```auth
片上 (per core, 处理单个 (b, n2, s1) 的单个规约块):
  Q_buf:       [G, D]         — G = N1/N2 个Q head，常驻 (m轴)
  KV_buf:      [128, D] × 2   — Level-2 ping-pong
  Acc_O:       [G, D]         — 累加器
  Acc_m/l:     [G]            — softmax状态

Workspace (C = 32):
  核内 workspace (per core):
    local_O:   [C, G, D]      — 同核 token 的 partial result 暂存
    local_m:   [C, G]
    local_l:   [C, G]

  跨核 workspace (全局共享, HBM):
    cross_O:   [C, C, G, D]   — 跨核 token 的 partial result
    cross_m:   [C, C, G]      — 32个核 × 每核最多32个 partial
    cross_l:   [C, C, G]
```

## 7. 总结

```auth
整体流程 (对每个 batch b, KV head n2, query position s1):

  for b in range(B):
    for n2 in range(N2):
      for s1 in range(S1):
        q = Q[b, n2*G:(n2+1)*G, s1, :]             # [G, D]  (G = N1/N2)
        K_full = K[b, n2, :, :]                       # [S2, D]
        V_full = V[b, n2, :, :]                       # [S2, D]

        S2维 ──┬── Level-1 tile 0 → Core → partial[b,n2,s1,0] ──┐
              ├── Level-1 tile 1 → Core → partial[b,n2,s1,1] ──┤
              ├── ...                                            │ Combine → O[b,n2*G:(n2+1)*G,s1,:]
              └── Level-1 tile n → Core → partial[b,n2,s1,n] ──┘

单核内 (处理一个 (b,n2,s1) 的一个规约块):
  q[G,D] × Level-1 tile ──→ [128][128]...[128] (Level-2串行) ──→ partial[G,D]
```

核心设计原则：

* 每个 \(b, n2, s1\) 三元组独立执行完整的 FA 计算，互不依赖

* G = N1/N2 个 Q head 共享同一组 KV，作为矩阵乘的 m 轴批量计算

* Level\-1 保证多核调度粒度，大小 =`align_up(S2/32, 128)`

* Level\-2 是片上计算的原子单元，固定 128

* 同一 \(b, n2, s1\) 的各规约级之间无数据依赖，天然并行

* 核内不做 combine，所有 partial result 统一合并

* Combine 开销为 O\(G × D × num\_tiles\)（per \(b,n2,s1\)），远小于主计算
