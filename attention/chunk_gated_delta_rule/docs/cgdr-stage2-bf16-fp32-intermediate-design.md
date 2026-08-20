# CGDR Stage2 BF16-State FP32-Intermediate 精度优化设计

## 1. 背景与动机

ChunkGatedDeltaRule（CGDR）算子在 BF16 state 输入场景下，原实现中 state 的中间计算全程使用 BF16，导致以下精度问题：

1. **W@S 累加精度损失**：`CalVPrime` 中 `kCumdecay @ state^T` 的结果以 BF16 累加（`accum=1`）到 `vInner`，BF16 的累加精度不足，尤其在长序列场景下误差累积。
2. **State 衰减精度损失**：BF16 state 在 `CalGCumExpBf16` 中先 cast 到 FP32 做 g_cum decay，再 cast 回 BF16 输出到 GM，两次 cast 引入精度损失。
3. **(U-WS)@K 累加精度损失**：`CalStateNew` 中 `vInner^T @ kg` 的结果以 BF16 累加（`accum=1`）到 `stateOut`，同样存在 BF16 累加精度问题。
4. **atomicAdd 竞争**：多核并行时 BF16 的 `SetAtomicAdd` 存在精度风险和性能开销。

## 2. 修改方案

### 2.1 U-WS 计算改为 FP32 中间态（需求1）

**修改前**：
```
AIC: CalVPrime (BF16 matmul, accum=1) → vInner (BF16, GM)
```

**修改后**：
```
AIC: CalVPrime (BF16 output matmul, accum=0) → wsFp32 (BF16, GM, per-core)
AIV: 
  1. Read U (BF16) from vInner → cast to FP32 in UB
  2. Read W@S (BF16) from wsFp32 → cast to FP32 in UB
  3. Sub: U - W@S (FP32, in UB)
  4. Cast (U-WS) to BF16 → vInner (GM) for stage3
```

关键改动：
- `CalVPrime` 改用 `mm1Bf16`（BF16 output matmul），`accum=0`（不累加）
- 新增 `CalVPrimeAiv` 方法：将 BF16 结果 cast 到 FP32，在 UB 中按 FP32 计算 U-WS，再 cast 回 BF16 输出到 GM
- 新增 `wsFp32` workspace（per-core，`chunkSize × Dv` BF16）

### 2.2 State 中间计算改为 FP32（需求2）

**修改前**：
```
State 全程 BF16：initState (BF16) → CalGCumExpBf16 (BF16→FP32→BF16) → stateOut (BF16)
```

**修改后**：
```
Process() 开始时: CastStateBf16ToFp32 (initState BF16 → stateFp32Wk FP32, GM)
Stage2 循环中:
  AIV: Read stateFp32Wk (FP32) → stateFp32Ub_ (UB)
       CalGCumExpFp32 (FP32 in-place g_cum decay)
       Cast stateFp32Ub_ → curStateBf16 (BF16, GM) for AIC matmul input
  [Stage2 结束后 state 保持在 stateFp32Wk (FP32)]
Process() 结束时: CastStateFp32ToBf16 (stateFp32Wk FP32 → finalState BF16, GM)
```

关键改动：
- 新增 `stateFp32Wk_` workspace（`B × Nv × Dv × Dk` FP32）
- `ProcessAivBf16` 重写：从 FP32 state 读取，FP32 计算 g_cum decay，仅 cast 到 BF16 供 AIC matmul 使用
- `Process()` 中新增首尾 cast：开始时 BF16→FP32，结束时 FP32→BF16
- `CastStateBf16ToFp32` / `CastStateFp32ToBf16`：分布式 AIV cast，按 Nv 维度切分到各核

### 2.3 S+(U-WS)@K 取消 atomicAdd（需求3）

**修改前**：
```
AIC: CalStateNew (BF16 matmul, accum=1) → stateOut (BF16, GM, atomicAdd)
```

**修改后**：
```
AIC: CalStateNew (BF16 output matmul, accum=0) → wsStateFp32 (BF16, GM, per-core)
AIV:
  1. Read (U-WS)@K (BF16) from wsStateFp32 → UB, cast to FP32
  2. Add: stateFp32Ub_ + (U-WS)@K (FP32, in UB)
  3. Write stateFp32Ub_ → stateFp32Wk (FP32, GM)
```

关键改动：
- `CalStateNew` 改用 `mm1Bf16`（BF16 output matmul），`accum=0`（不累加）
- 新增 `CalStateAdd` 方法：将 BF16 结果 cast 到 FP32，在 UB 中按 FP32 做 Add，分 tile 处理大矩阵
- 新增 `wsStateFp32` workspace（per-core，`Dv × Dk` BF16）
- 完全移除 `SetAtomicAdd` 调用

## 3. 数据流图

```
                    ┌─────────────────────────────────────────────────┐
                    │                  Process()                       │
                    │                                                  │
  initState (BF16)──→ CastStateBf16ToFp32 ──→ stateFp32Wk (FP32, GM) │
                    │                                          │       │
                    │  ┌──────────────────────────────────┐    │       │
                    │  │         Stage2 (per chunk)        │    │       │
                    │  │                                   │    │       │
                    │  │  AIV:                             │    │       │
                    │  │   stateFp32Wk ──→ stateFp32Ub_    │    │       │
                    │  │   CalGCumExpFp32 (FP32 decay)     │    │       │
                    │  │   Cast → curStateBf16 (BF16, GM)  │    │       │
                    │  │         │                         │    │       │
                    │  │  AIC:   ↓                         │    │       │
                    │  │   CalVPrime (FP32 out, no accum)  │    │       │
                    │  │         │ → wsFp32 (FP32, GM)     │    │       │
                    │  │   CalAttnInter (BF16 → out)       │    │       │
                    │  │         │                         │    │       │
                    │  │  AIV:   ↓                         │    │       │
                    │  │   U(BF16) → FP32, Sub(U - W@S)    │    │       │
                    │  │   Cast → vInner (BF16, GM)        │    │       │
                    │  │         │                         │    │       │
                    │  │  AIC:   ↓                         │    │       │
                    │  │   CalStateNew (FP32 out, no accum)│    │       │
                    │  │         │ → wsStateFp32 (FP32, GM)│    │       │
                    │  │         │                         │    │       │
                    │  │  AIV:   ↓                         │    │       │
                    │  │   Add(S + (U-WS)@K) in UB (FP32)  │    │       │
                    │  │   → stateFp32Wk (FP32, GM)        │    │       │
                    │  └──────────────────────────────────┘    │       │
                    │                                          │       │
                    │  (next chunk reads stateFp32Wk) ←────────┘       │
                    │                                                  │
  stateFp32Wk (FP32)──→ CastStateFp32ToBf16 ──→ finalState (BF16, GM) │
                    └─────────────────────────────────────────────────┘
```

## 4. 同步信号设计

BF16-state 路径的 AIC-AIV 同步流程：

| 步骤 | AIV 操作 | 同步信号 | AIC 操作 |
|------|---------|---------|---------|
| 1 | Cast state → curStateBf16 (GM) | AIV→AIC (0x2) | Wait 0x2 |
| 2 | Wait 0x4 | AIC→AIV (0x4) | CalVPrime → wsFp32 (FP32) |
| 3 | CalVPrimeAiv: U-WS in UB → vInner (GM) | AIV→AIC (0x5) | Wait 0x5 |
| 4 | Wait 0x6 | AIC→AIV (0x6) | CalStateNew → wsStateFp32 (FP32) |
| 5 | CalStateAdd: S+(U-WS)@K in UB → stateFp32Wk (GM) | - | - |

## 5. Workspace 分配

### 5.1 interWorkspaceSz（新增）

| 变量 | 大小 | 条件 |
|------|------|------|
| `stateFp32Wk_` | `sizeof(float) × B × Nv × Dv × Dk` | BF16-state on ASCEND950 |
| `curStateBf16_` | `sizeof(bf16) × Nv × Dv × Dk` | BF16-state on ASCEND950 (原本仅 FP32-state 分配，现改为始终分配) |

### 5.2 stageWorkspaceSz（新增）

| 变量 | 大小 | 条件 |
|------|------|------|
| `wsFp32` | `sizeof(float) × chunkSize × Dv × aiCoreNum` | BF16-state on ASCEND950 |
| `wsStateFp32` | `sizeof(float) × Dv × Dk × aiCoreNum` | BF16-state on ASCEND950 |

### 5.3 matmulTilingFp32C

`matmulTilingFp32C`（BF16→BF16→FP32 matmul tiling）从仅 FP32-state 路径计算改为始终计算，供 BF16-state 路径的 `mm1Fp32` 使用。

## 6. 影响范围

### 6.1 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `op_host/chunk_gated_delta_rule_tiling.cpp` | 新增 BF16-state workspace；始终计算 matmulTilingFp32C |
| `op_kernel/arch35/chunk_gated_delta_rule_arch35.h` | 新增 stateFp32Wk/curStateBf16 分配；新增 Cast 方法；Process() 增加首尾 cast；RunStage2 传参 |
| `op_kernel/arch35/chunk_gated_delta_rule_stage2_arch35.h` | 重写 ProcessAivBf16/ProcessAic；新增 CalVPrimeAiv/CalStateAdd；CalVPrime/CalStateNew 改用 BF16 output (accum=0)，AIV cast 到 FP32 做 Sub/Add |

### 6.2 不受影响

- **arch22 实现**：不修改，arch22 仅支持 BF16 state，无 FP32-state 路径
- **FP32-state 路径**：保持原有逻辑不变，`ProcessAivFp32` / `ProcessAic`（kIsFp32 分支）不受影响
- **Stage1 / Stage3**：不修改，Stage1 的 vInner 输出类型由模板参数控制（BF16-state 路径仍输出 BF16），Stage3 读取 BF16 vInner

### 6.3 UB 空间评估

BF16-state 路径的 `tmpBuff_` 需要容纳：
- `stateFp32Ub_`: `Dv × curDk × sizeof(float)` (最大 128×128×4 = 64KB)
- `uFp32Ub_`: `chunkSize × dvAligned × sizeof(float)` (64×128×4 = 32KB)
- `wsFp32Ub_`: `chunkSize × dvAligned × sizeof(float)` (64×128×4 = 32KB)
- `lastGCum_`: 1 float

总计约 128KB，在 Ascend950PR 的 UB（192KB）范围内。

## 7. RDV 测试验证

在 Ascend950PR 设备上通过 C++ aclnn 测试验证：

| 测试用例 | B | T | Nk | Nv | Dk | Dv | has_g | state | 结果 |
|---------|---|---|----|----|----|----|-------|-------|------|
| BF16-state-g-001 | 1 | 2048 | 32 | 32 | 128 | 128 | True | BF16 | PASS |
| BF16-state-g-002 | 1 | 4096 | 32 | 32 | 128 | 128 | True | BF16 | PASS |
| BF16-state-g-003 | 1 | 2048 | 2 | 2 | 128 | 128 | True | BF16 | PASS |
| BF16-state-g-005 | 2 | 2048 | 32 | 32 | 128 | 128 | True | BF16 | PASS |
| BF16-state-g-006 | 1 | 128 | 32 | 32 | 128 | 64 | True | BF16 | PASS |
| BF16-state-g-007 | 1 | 128 | 2 | 2 | 16 | 16 | True | BF16 | PASS |
| BF16-state-nog-001 | 2 | 300 | 8 | 8 | 64 | 64 | False | BF16 | PASS |
| FP32-state-g-001 | 1 | 2048 | 32 | 32 | 128 | 128 | True | FP32 | PASS |
| FP32-state-g-002 | 1 | 128 | 32 | 32 | 128 | 64 | True | FP32 | PASS |
| FP32-state-g-003 | 1 | 128 | 2 | 2 | 16 | 16 | True | FP32 | PASS |

BF16-state 和 FP32-state 路径均通过验证，算子正确编译和执行。
