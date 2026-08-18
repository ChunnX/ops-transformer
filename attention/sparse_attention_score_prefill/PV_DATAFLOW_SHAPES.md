# PV Matmul 数据流每步 Shape 说明

> 以 `block_mmad_pv_prefill_a2.hpp` 为分析对象
>
> 示例参数: `test_bf16_minimal_debug` (q_heads=1, kv_heads=1, groupSize=1, q_tokens=1, kv=128, D=128)

---

## 1. 参数定义

| 参数 | 含义 | 示例值 | 来源 |
|---|---|---|---|
| `M` | Q 侧行数 = groupCount × groupRows | 1 | `operator()` 入参 `groupCount * groupRows` |
| `K` | KV block 有效 token 数 | 128 | `residentValidSize_`（LoadVResident 时保存） |
| `N` | head 维度 D | 128 | `operator()` 入参 `D` |
| `C0` | fractal 基本块大小 | 16 | `C0_NUM_PER_FRACTAL` |
| `L0_TILE_M` | L0 M 方向 tile 大小 | 64 | `BlockMmadPVPrefillA2::L0_TILE_M` |
| `L0_TILE_N` | L0 N 方向 tile 大小 | 128 | `BlockMmadPVPrefillA2::L0_TILE_N` |
| `L0_TILE_K` | L0 K 方向 tile 大小 | 128 | `BlockMmadPVPrefillA2::L0_TILE_K` |

### MMAD 三重循环切分

```
nL0Num = CeilDiv(N, L0_TILE_N) = CeilDiv(128, 128) = 1
mL0Num = CeilDiv(M, L0_TILE_M) = CeilDiv(1, 64) = 1
kL0Num = CeilDiv(K, L0_TILE_K) = CeilDiv(128, 128) = 1
```

每个 tile 的实际大小：
- `nAct` = min(L0_TILE_N, N - nL0Itr × L0_TILE_N) = 128
- `mAct` = min(L0_TILE_M, M - mL0Itr × L0_TILE_M) = 1
- `kAct` = min(L0_TILE_K, K - kL0Itr × L0_TILE_K) = 128

---

## 2. 完整数据流每步 Shape

### 2.1 QK 阶段（CUBE）

| 步骤 | 描述 | 位置 | 逻辑 shape | 对齐后 shape | 物理元素数 | dtype | layout |
|---|---|---|---|---|---|---|---|
| ① | Q tile (L1→L0A) | L0A | `[M, K]` = `[1, 128]` | `[16, 128]` | 2048 | bf16 | zZ |
| ② | K tile (L1→L0B) | L0B | `[K, N]` = `[128, 128]` | `[128, 128]` | 16384 | bf16 | nZ |
| ③ | S = Q×K^T (MMAD) | L0C | `[M, N]` = `[1, 128]` | `[16, 128]` | 2048 | fp32 | L0C NZ fractal |
| ④ | S fixpipe L0C→GM | GM | `[mFixAligned, nAct]` = `[16, 128]` | — | 2048 | bf16 | RowMajor, bi%2 双缓冲 |

### 2.2 Softmax 阶段（VEC）

| 步骤 | 描述 | 位置 | 逻辑 shape | 物理元素数 | dtype | 说明 |
|---|---|---|---|---|---|---|
| ⑤ | S GM→UB (DataCopyPad) | UB | `[m, nRound]` = `[1, 128]` | 128 | bf16 | m = subBlock 分到的行数; nRound = RoundUp(N, 16) |
| ⑥ | rowMax / rowSum | UB | `[m]` = `[1]` | 1 | fp32 | 每行一个 max/sum |
| ⑦ | P = exp(S - rowMax) | UB | `[m, nRound]` = `[1, 128]` | 128 | bf16 | mask 尾部 [cvl, N) 置 0 |
| ⑧ | P UB→GM (DataCopyPad) | GM | `[m, nRound]` = `[1, 128]` | 128 | bf16 | bi%3 三缓冲, subBlock 分区写入 |

### 2.3 PV 阶段（CUBE）

| 步骤 | 描述 | 位置 | 逻辑 shape | 对齐后 shape | 物理元素数 | dtype | layout |
|---|---|---|---|---|---|---|---|
| ⑨ | P GM→L1 (Nd2Nz) | L1 | `[M, K]` = `[1, 128]` | `[16, 128]` | 2048 | bf16 | zN |
| ⑩ | V GM→L1 (LoadVResident) | L1 | `[K, N]` = `[128, 128]` | `[128, 128]` | 16384 | bf16 | nZ |
| ⑪ | P L1→L0A | L0A | `[mAct, kAct]` = `[1, 128]` | `[16, 128]` | 2048 | bf16 | zZ |
| ⑫ | V L1→L0B (常驻) | L0B | `[kAct, nAct]` = `[128, 128]` | `[128, 128]` | 16384 | bf16 | nZ |
| ⑬ | O = P×V (MMAD) | L0C | `[M, nAct]` = `[1, 128]` | `[16, 128]` | 2048 | fp32 | L0C NZ fractal |
| ⑭ | O L0C→GM (NZ2ND scatter) | GM | `[groupRows, nAct]` = `[1, 128]` per slot | `[8, 128]` per slot | 1024 per slot | fp32/bf16 | RowMajor, 按 qToken 分散 |

---

## 3. L0C Fractal 布局详解

### 3.1 MakeLayoutL0C(M, N) 的物理结构

```
L0C layout (MakeLayoutL0C):
  shape:  ((C0, CeilDiv(M, C0)), (C0, CeilDiv(N, C0)))
        = ((16, CeilDiv(M,16)), (16, CeilDiv(N,16)))
  stride: ((C0, C0*C0), (1, RoundUp(M, C0) * C0))
        = ((16, 256), (1, RoundUp(M,16) * 16))
```

**物理内存排列**（以 M=1, N=128 为例）：

```
M_aligned = RoundUp(1, 16) = 16
N_aligned = RoundUp(128, 16) = 128
总元素 = 16 × 128 = 2048

fractal 结构: 1 个行 fractal × 8 个列 fractal = 8 个 fractal
每个 fractal: 16 行 × 16 列 = 256 元素

物理内存索引:
  fractal[0][0]: row 0-15, col 0-15    → indices [0, 255)
  fractal[0][1]: row 0-15, col 16-31   → indices [256, 511)
  fractal[0][2]: row 0-15, col 32-47   → indices [512, 767)
  ...
  fractal[0][7]: row 0-15, col 112-127 → indices [1792, 2048)

fractal 内部 (stride[0,0]=16, stride[1,0]=1):
  index = ir * 16 + ic  (ir=行内偏移, ic=列内偏移)
  即 fractal 内按列优先排列 (NZ)

有效数据 (M=1, 只有 row 0):
  row 0, col 0-15   → indices [0, 16)      ← 在 fractal[0][0] 的 ir=0 位置
  row 0, col 16-31  → indices [256, 272)   ← 在 fractal[0][1] 的 ir=0 位置
  row 0, col 32-47  → indices [512, 528)
  ...
  row 0, col 112-127 → indices [1792, 1808)
```

### 3.2 DumpTensor dump_size 的问题

```cpp
// 代码中:
AscendC::DumpTensor(l0CTensor_[l0CBufId], 980, M * nAct);
//                                    dump_size = 1 * 128 = 128
```

| dump_size | 覆盖范围 | 有效数据 | 缺失数据 |
|---|---|---|---|
| `M * nAct` = 128 | fractal[0][0] 前 128 元素 (rows 0-7, cols 0-15) | **row 0, cols 0-15** (16 个) | row 0, cols 16-127 (112 个) |
| `RoundUp(M,16) * nAct` = 2048 | 全部 8 个 fractal | **row 0, cols 0-127** (128 个) | 无 |

**结论**：dump_size 应改为 `RoundUp(M, 16) * RoundUp(N, 16)` = 2048 才能覆盖全部有效数据。

---

## 4. 各 DumpTensor 对应关系

| desc | 代码位置 | 位置 | dump_size 公式 | 示例 dump_size | 有效数据 |
|---|---|---|---|---|---|
| 900 | QK fixpipe 后的 GM S | GM | `mFixAligned × nAct` | `16 × 128 = 2048` | row 0 的 128 个 S 值 |
| 970 | PV GM→L1 后的 L1 P | L1 | `M × K` | `1 × 128 = 128` | 全部 128 个 P 值 |
| 980 | PV matmul 后的 L0C O | L0C | `M × nAct` | `1 × 128 = 128` | **仅 row 0 cols 0-15** (16 个) |
| 992 | VEC P UB→GM 后的 GM P | GM | `m × nRound` | `1 × 128 = 128` | 全部 128 个 P 值 |

---

## 5. 多头场景 (test_bf16_minimal_debug_2)

| 参数 | 值 |
|---|---|
| q_heads | 2 |
| kv_heads | 2 |
| groupSize | q_heads / kv_heads = 1 |
| numQTokens | 2 |
| topK | 1 |

每个 Phase 1 task (kvHeadIdx) 的参数：
- `groupCount` = 1 (1 个 Q token 选了该 KV block)
- `groupRows` = groupSize = 1
- `M` = groupCount × groupRows = 1
- `K` = validSize = 128
- `N` = D = 128

**每个 task 的 shape 与单头场景完全相同**。两个 task（kvHead 0, kvHead 1）在同一个核上顺序执行，每步 shape 不变，但共享 GM staging buffer。

### 多头时的 GM workspace 布局

```
GM S staging (per-core, 双缓冲):
  [task0 bi0 S] [task0 bi1 S] [task1 bi0 S] [task1 bi1 S]
   ↑ bi%2=0       ↑ bi%2=1      ↑ 覆盖 task0 bi0!

GM P staging (per-core, 三缓冲):
  [task0 bi0 P] [task0 bi1 P] [task0 bi2 P] [task1 bi0 P] ...
   ↑ bDe%3=0      ↑ bDe%3=1     ↑ bDe%3=2     ↑ 覆盖 task0 bi0!
```

当两个 task 顺序执行时，task1 的 bi=0 数据会覆盖 task0 的 bi=0 staging buffer。如果 task0 的 PV（滞后 PRE_LAUNCH=2 个 batch）还没消费完 bi=0 的数据，就会读到 task1 的数据 → NaN。

---

## 6. L0 Tile 常量

| 常量 | 值 | 说明 |
|---|---|---|
| `L0_STAGES` | 2 | L0A/L0B/L0C 双缓冲级数 |
| `L0_TILE_M` | 64 | L0 M 方向 tile 大小 (PV) |
| `L0_TILE_N` | 128 | L0 N 方向 tile 大小 |
| `L0_TILE_K` | 128 | L0 K 方向 tile 大小 |
| `L0C_HALF_SIZE` | L0C_SIZE / 2 | PV 用 L0C 上半区 (QK 用下半区) |
| `L0C_BUF_SIZE` | L0C_HALF_SIZE / 2 | L0C 单级 buffer 大小 |
| `C0_SIZE` | 16 | Cube 基本块 |
| `COPY_GRANULARITY` | 2 | Scatter 粒度: 每次写 2 个 group |
