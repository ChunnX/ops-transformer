# 全手动同步重写蓝图

> 目标：关闭 `TL_ASCEND_AUTO_SYNC`，逐 scope 手写 `T.set_flag`/`T.wait_flag`，消除冗余 PIPE_ALL，并加 V0/V1 double buffer。
> 参照：`source_code.cpp`（auto-sync 生成的已知正确同步）+ SparseMLA `sparse_flash_attn_pa.py`。

## 0. 决策前提（已确认）

- `TL_ASCEND_AUTO_SYNC` 是**全局开关**，无法只对某个 scope 关闭（`ascend_sync_insert.cc:67-71`）。
- 手动 flag 与 auto flag **不能混用**（pass 不识别手写 flag → 重复同步 + event ID 冲突）。
- → 一旦关闭，**5 个 scope（V0/C1/V1/C2/V3）全部**要手写 intra-core 同步。
- inter-core 同步（`T.set_cross_flag`/`T.wait_cross_flag`）**已经是手动的**，不受影响。

## 1. 同步原语速查

| API | 含义 |
|-----|------|
| `T.set_flag(producer, consumer, id)` | producer 完成，通知 consumer |
| `T.wait_flag(producer, consumer, id)` | consumer 阻塞等 producer |
| `T.pipe_barrier(pipe)` | 单 pipe 内屏障（如 `"V"`, `"MTE3"`） |
| `T.barrier_all()` | 全 pipe 屏障（= PIPE_ALL，尽量避免） |

Pipe 名：`"mte2"`(GM→UB/L1), `"mte3"`(UB→GM), `"mte1"`(L1→L0), `"m"`(Cube MMA), `"v"`(Vector), `"fix"`(L0C→GM), `"s"`(scalar)。

事件 ID 范围约定（避免冲突，硬件仅 8 个 event）：
- V0: 0-1（MTE2↔MTE3 ping-pong）
- C1: 0-3
- V1: 0-7（复用，注意跨循环不重叠）
- C2: 0-2
- V3: 3-7

## 2. PIPE_ALL 根因与消除策略

| 位置 | 静态数 | 动态数/row | 根因 | 消除策略 |
|------|:---:|:---:|------|---------|
| **V1 广播循环** | 3 | **320** | 标量 `GetValue` + 逐行 `Muls` | ⭐见 §2.1 |
| V0 gather | 3 | 128 | 标量 `idx` + `if idx>=0` | chunked gather 已减少；标量读用 `T.pipe_barrier("s")` 局部化 |
| C1 Pipelined | 3 | 3 | 旋转 buffer `(d_idx+2)%3` 取模 | 手写 double buffer，固定 flag |
| V1 cast 别名 | 3 | ~12 | `rg_half_ub`/`rsy_tmp_ub` UB 地址重叠 | 拆开 buffer 地址，用 `set_flag("v","mte3")` |

### 2.1 ⭐ V1 广播循环消除（最高收益）

**现状**（CPP 203-210, 215-222, 241-248）：
```python
for g_id in range(G):                        # G=128
    T.tile.mul(sm_2d_ub[g_id, :], p_reduced_tmp_ub, sm_max_ub[g_id])  # 标量 sm_max_ub[g_id]
    # → 每次 GetValue 强制 PIPE_ALL
```

三个这样的循环：`sm_max`(128) + `sm_sum`(128) + `weight`(64) = 320 次 PIPE_ALL/row。

**根因**：`sm_max_ub[g_id]` 是从 UB 读标量（scalar pipe），后续 `Muls` 在 V pipe，auto-sync 判定标量读 sliced → PIPE_ALL。

**消除策略（两选一）**：

**方案 A（手动同步，保留标量循环）**：标量读全部在 scalar pipe，一次性读完再算。
```python
# 关 auto-sync 后，标量读之间无需 PIPE_ALL（scalar pipe 内顺序执行）
# 只需在 scalar→vector 之间一次 set_flag("s","v")
for g_id in range(G):
    T.tile.mul(sm_2d_ub[g_id, :], p_reduced_tmp_ub, sm_max_ub[g_id])
# 手动只在循环末尾同步一次，而非每次 PIPE_ALL
```
关掉 auto-sync 后，这 320 次 PIPE_ALL **直接消失**，因为标量读+Muls 在同核顺序发射，只需在整个广播循环结束后做一次 `set_flag("v", <next consumer>)`。这是关 auto-sync 的**最大红利**。

**方案 B（真向量化，消除标量循环）**：用 `T.tile.broadcast` 或矩阵乘广播替代逐行 Muls。受限制#7（1D→2D broadcast 不支持），暂不可行。→ 用方案 A。

## 3. 逐 Scope 同步图谱（转写自 source_code.cpp）

### 3.1 V0 (AIV) — gather，行 123-145

现状 chunked gather（已改）。手动同步：
```
for s1_id:
  for chunk (double buffer, ping-pong id=chunk%2):
    [MTE2] 逐行 gather → key_chunk_ub[id]      (16 行随机读)
    set_flag("mte2","mte3", id)
    wait_flag("mte2","mte3", id)               # 首轮；后续靠 ping-pong 错开
    [MTE3] key_chunk_ub[id] → ws               (16KB burst)
    set_flag("mte3","mte2", id)                # buffer 可复用
set_cross_flag("V", 10)                        # 已有
```
标量 `idx` 读：关 auto-sync 后不再触发 PIPE_ALL，标量在 s pipe 顺序执行，MTE2 copy 依赖它由 `pipe_barrier` 或数据依赖隐式保证。

### 3.2 C1 (AIC) — Matmul1+Matmul2，行 146-185

```
wait_cross_flag(10)                            # 已有
# Matmul1: 手写 K 维 double buffer (替代 T.Pipelined)
preload Q_L1[0], GK_L1[0]  (d_idx=0)
set_flag("mte2","mte1",0)
for d_idx in range(d_num_total):
  id = d_idx % 2
  wait_flag("mte2","mte1", id)                 # L1 就绪
  # 预取下一块 (MTE2, 与当前 MMA 并发)
  if d_idx < d_num_total-1:
    copy Q_L1[1-id], GK_L1[1-id]  (d_idx+1)
    set_flag("mte2","mte1", 1-id)
  gemm_v0(Q_L1[id], GK_L1[id], C_L0C, init=(d_idx==0))   # MMA (m pipe)
set_flag("m","fix",7); wait_flag("m","fix",7)
copy C_L0C → bmm1Res                           # L0C→GM (fix pipe)
# Matmul2: 类似
copy QI_L1, GKI_L1; set_flag("mte2","m",0); wait_flag("mte2","m",0)
gemm_v0(QI_L1, GKI_L1, C2_L0C, init=1)
set_flag("m","fix",1); wait_flag("m","fix",1)
copy C2_L0C → bmm2Res (enable_relu)
set_cross_flag("FIX", 1)                        # 已有
```

### 3.3 V1 (AIV) — 核心计算，行 186-391（最复杂）

关键同步点（转写）：
```
wait_cross_flag(1)
copy sm_max,sm_sum→ub; set_flag("mte2","v",2); wait_flag("mte2","v",2)
abs,fill; pipe_barrier("v"); max
copy bmm1→ub; set_flag("mte2","v",3); wait_flag("mte2","v",3)
muls,fill
# ⭐ 广播循环 sm_max: 关 auto-sync 后无 PIPE_ALL, 循环末一次 pipe_barrier("v")
[sm_max 广播 128×]; pipe_barrier("v")
sub,exp
[sm_sum 广播 128×]; pipe_barrier("v")
div, reduce_sum, muls
set_flag("v","mte3",4); wait; copy p_reduced→GM
copy Weights→ub; set_flag("mte2","v",5); wait; cast
set_flag("mte3","v",6); wait                   # WAR: p_reduced_tmp 复用
[weight 广播 64×]; 
# ReduceSumY 循环 (16×, 可 double buffer bmm2_ub — 见 §4)
for n_rsy in range(16):
  set_flag("v","mte2",7); wait; copy bmm2→ub; set_flag("mte2","v",0); wait
  pipe_barrier("v"); mul; reduce_sum; copy ub→ub(rsy_all)
  pipe_barrier("mte3"); copy→ReduceSumY
  muls; reduce_max; max
# GlobalSoftmaxI: 标量 rsy_max — 关 auto-sync 后无 PIPE_ALL
[softmax_i 广播]
for n_idx_iter in range(16): ... (rsy_sum 累加)
# per-core SoftmaxI: 标量 rsy_sum
copy→SoftmaxI; fill ones/eps
set_flag("mte3","mte2",4); wait; copy SoftmaxI,PReduced→ub; set_flag("mte2","v",5); wait
sub(diff)
# dWeight/reluGrad 循环 (g_num×, double buffer — 见 §4)
for g_id in range(g_num):
  fill weight; set_flag("v","mte2",1); wait; copy bmm2→dw_chunk; set_flag("mte2","v",6); wait
  for i in range(vector_baseg): [标量 idx 逐行 — 关 auto-sync 后无 PIPE_ALL]
  reduce_sum; add
  cast rg_half; set_flag("v","mte3",7); wait; copy→reluGradRes    # 注意别名 rg_half/rsy_tmp
  cast dw_half; set_flag("v","mte3",0); wait; copy→dWeight_partial
# Loss 计算
... (loss_chunk 系列，MTE2_V/V_MTE2 交替)
set_flag("v","mte3",0); wait; copy→LossOut
set_cross_flag("V", 12)                         # 已有
```

### 3.4 C2 (AIC) — Matmul4，行 392-403

```
wait_cross_flag(12)
copy RG_L1, GKI2_L1; set_flag("mte2","m",1); wait_flag("mte2","m",1)
gemm_v0(RG_L1, GKI2_L1, DQ_L0C, transpose_B=False, init=1)
set_flag("m","fix",2); wait_flag("m","fix",2)
atomic_add DQ_L0C → dQueryIndexAccum
set_cross_flag("FIX", 3)                         # 已有
```

### 3.5 V3 (AIV) — cast，行 404-420

```
wait_cross_flag(3)
for g_id in range(v3_chunks):                    # double buffer 可选
  set_flag("v","mte2",5); wait; copy accum→dq_fp32; set_flag("mte2","v",3); wait
  set_flag("mte3","v",7); wait                   # WAR: dq_fp16 复用
  cast dq_fp16
  set_flag("v","mte3",4); wait; pipe_barrier("mte3"); copy→dQueryIndex
```

## 4. Double Buffer 叠加点（关 auto-sync 后才好做）

| 位置 | buffer | 收益 |
|------|--------|------|
| V0 gather | key/ki/kr_chunk ×2 | MTE2 fill ∥ MTE3 write |
| C1 Matmul1 | Q_L1/GK_L1 ×2 | MTE2 load ∥ MMA（替代 T.Pipelined） |
| V1 ReduceSumY | bmm2_ub ×2 | MTE2 load ∥ VEC compute |
| V1 dWeight/reluGrad | dw/rg_chunk ×2 | MTE2 load ∥ VEC ∥ MTE3 store |

## 5. 实施顺序（关键：不能增量验证，必须一次全部改对）

```
Step 1: pass_configs 关 TL_ASCEND_AUTO_SYNC
Step 2: 逐 scope 转写 §3 的手动 flag（先复刻，不加 double buffer）
        目标：手动同步结果 == auto-sync 结果（精度一致）
Step 3: 验证精度通过（远程 NPU）
Step 4: 消除 V1 广播循环 PIPE_ALL（§2.1 方案 A — 关 auto-sync 后自动消失）
Step 5: 叠加 double buffer（§4）
Step 6: 逐项验证性能
```

**风险控制**：Step 2 严格按 CPP 转写，一个 flag 都不省。先拿到"手动 == 自动"的正确基线，再优化。这样 Step 3 若 hang/错，问题一定在转写遗漏，可对照 CPP 逐行核对。

## 6. 待确认

- [x] 能否本地 JIT 编译（不执行）dump 出当前 0706 的 CPP 作对照？→ **已拿到**（source_code.cpp，block_n=64/n_num=32 配置），§7 为精确图谱。
- [ ] `T.pipe_barrier("s")` 是否支持 scalar pipe 局部屏障？
- [ ] 关 auto-sync 后，`enable_relu`（C1 bmm2 copy）、`atomic_add`（C2）是否需要额外同步？

## 7. ⭐ 精确同步图谱（转写自当前 0706 CPP source_code.cpp）

> 这是 Step 2 的逐行对照表。规则：**Set/Wait flag 与单 pipe barrier 全部保留（照抄）；PIPE_ALL 全部删除**（关 auto-sync 后标量读不再需要），个别真实依赖用最小单 pipe barrier 替代。

**PIPE_ALL 动态普查（S1=1）**：V0=192, C1=9, V1=327, C2=0, V3=0，合计 ≈528。根因统一：标量 `GetValue` 被 auto-sync 判 sliced → 全排空。关掉后 512 个直接消失。

### 7.1 V0（行 134-181）——当前竟无任何 event flag，纯 PIPE_ALL

现状：chunked ping-pong 结构已在（gather 写 `((chunk+1)%2)*8192`，drain 读 `(chunk%2)*8192`），但**没有 MTE2/MTE3 flag**，全靠 192 个 PIPE_ALL 兜底。

- **删除**：全部 192 个 PIPE_ALL（行 136/138/147/151/153/164）。
- **新增**（ping-pong 正确性所需，参照 sparse_flash_attn_pa.py）：
  - drain 前 `wait_flag("mte2","mte3", chunk%2)`（RAW：drain 等 fill 完）
  - drain 后 `set_flag("mte3","mte2", chunk%2)`（WAR：下轮同 buffer 的 fill 等 drain 完）
  - fill 前 `wait_flag("mte3","mte2", (chunk+1)%2)`
  - fill 后 `set_flag("mte2","mte3", (chunk+1)%2)`
- **保留**：行 173 `PipeBarrier<PIPE_MTE3>`（末尾 drain 前）、行 180 `CrossCoreSetFlag(10)`。
- 标量 `idx` 读：无需任何 barrier，硬件保证标量→MTE 地址操作数依赖。

### 7.2 C1（行 182-233）

- **保留**：186 `PIPE_MTE2`；212/215 `PIPE_M`；218/219 `M_FIX(1)`；224/225 `MTE2_M(2)`；228/229 `M_FIX(3)`；232 `CrossCoreSetFlag(1)`。
- **删除**：190/198/208 的 9 个 PIPE_ALL（Matmul1 K-loop 旋转 buffer）。
- **替代**：K-loop double buffer 用 `MTE2_M` flag——gemm 前 `wait_flag("mte2","m", id)`，预取后 `set_flag("mte2","m", 1-id/next)`。3-way 旋转可简化为 2-way ping-pong（见 §4）。

### 7.3 V1（行 234-443）——保留全部 Set/Wait，删全部 PIPE_ALL

**保留（照抄）所有 Set/Wait flag**：
MTE2_V(4) 238/239、MTE2_V(5) 245/246、V_MTE3(6) 279/280、MTE2_V(7) 283/284、MTE3_V(0) 286/287、V_MTE2(2) 299/300、MTE2_V(1) 302/303、V_MTE3(4) 356/357、MTE3_MTE2(5) 361/362、MTE2_V(6) 365/366、MTE3_MTE2(2) 371/372、MTE2_V(7) 376/377（→见下移出内层）、V_MTE3(0) 397/398、V_MTE3(1) 405/406、MTE2_V(0) 411/412、V_MTE2(1) 416/417、MTE2_V(2) 419/420、V_MTE2(3) 424/425、MTE2_V(4) 429/430、V_MTE3(5) 440/441、CrossCoreSetFlag(12) 443。

**保留全部单 pipe barrier**：所有 `PIPE_V`、`PIPE_MTE3`（311/399/473 等）照抄。

**删除全部 327 个 PIPE_ALL**：
- 254/266/292（广播循环 sm_max/sm_sum/weight，320 次）→ 直接删，循环内已有 252/264/290 的 `PIPE_V` 兜底 vector 顺序。
- 323/353/439（rsy_max/rsy_sum/loss 标量）→ 删。
- 396/404（dWeight cast 别名 `rg_half_ub`/`dw_half_ub` 与源 UB 重叠）→ 删，紧跟其后的 397/398 `V_MTE3(0)`、405/406 `V_MTE3(1)` 已提供 cast(v)→copy(mte3) 同步；若精度异常再补 `pipe_barrier("v")`。

**额外优化**：376/377 `MTE2_V(7)` 在 `i_2` 内层循环（32×）里，但 copy 在循环外（373）→ **移到循环前一次**，省 31 次同步。

### 7.4 C2（行 445-458）——已干净，照抄

保留：MTE2_M(6) 449/450、M_FIX(7) 453/454、CrossCoreSetFlag(3) 457。无 PIPE_ALL。

### 7.5 V3（行 459-476）——已干净，照抄

保留：V_MTE2(2) 462/463、MTE2_V(0) 466/467、MTE3_V(4) 468/469、V_MTE3(1) 471/472、473 `PIPE_MTE3`。无 PIPE_ALL。

### 7.6 转写等价性论证

关 auto-sync 后手动 flag 集 = CPP 现有 Set/Wait 全集 + V0/C1 新增 double-buffer flag。删除的仅是 PIPE_ALL（标量保护），其真实依赖(标量→MTE 地址)由硬件保证或被保留的 Set/Wait 覆盖。**理论上精度应与 auto-sync 完全一致**，性能提升来自 512 个全排空的消除。
