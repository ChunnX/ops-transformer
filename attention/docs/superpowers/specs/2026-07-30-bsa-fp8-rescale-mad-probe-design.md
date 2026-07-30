# BSA FP8 PV Rescale-Mad 性能穿刺设计

**日期：** 2026-07-30  
**状态：** 已批准（待实现）  
**范围：** Arch35 BSA `BlockMmadPV`，仅 FP8 全量化路径；纯 cycle 叠加穿刺，不要求数值正确

## 1. 目标与范围

在 Arch35 BSA **FP8 全量化**路径上，于 PV（P×V）mmad 完成之后、带 `deqScalar` 的 fixpipe 之前，插入一段 cube 侧探针：每 PV 一次 `InitConst` + MTE1 + 两次 fp32 mad，用于测量相对现网流水的额外开销。

**成功标准：**

1. 加 `-DBSA_PV_RESCALE_PROBE_FP8` 可编译通过
2. FP8 case 单测可跑通（数值允许错误）
3. msprof / 流水对比开关宏前后，可见 L1 InitConst + MTE1 + 2×Mmad
4. 非量化实例因 `if constexpr (FULL_QUANT_FP8)` **不执行**探针

**非目标：**

- 不替代、不旁路 `rescaleO`（含末 tile `×1/448`，照常执行）
- 不做精度对齐
- 不改 QK Softmax / `deqScalar` 语义
- 默认构建不打开该宏

## 2. 背景（FP8 管线）

Arch35 BSA full_quant 主路径：

```text
QK(fp8 + deq) → Softmax(P→fp8) → PV(fp8 mmad → L0C后半)
  → fixpipe(L0C→UB, deqScalar=vScale) → rescaleO(AIV)
```

插入点在 `block_mmad_pv_arch35_ABf16_C_to_UB.hpp` 的 `operator()` 中：第一个 N-chunk 的 m/k mmad 全部完成后、该 chunk 的 fixpipe 之前。

现网 PV 使用 L0C **后半区**（`L0C_HALF_BUF_SIZE + pingpong`）。探针使用 L0C **前半区**（`@0`），避免踩坏即将 fixpipe 的 PV 结果（含后续 deq）。

与非量化相对：P/V 为 `fp8_e4m3fn_t`，fixpipe 走 `deqScalar` 双 AIV 分半；rescaleO 仍在 fp32 UB 上做 `go*dm+lo`。

## 3. 方案选择

采用**方案 1：同文件 `#ifdef` 内联探针**。

| 方案 | 结论 |
|------|------|
| 1. 同文件 `#ifdef` + `if constexpr (FULL_QUANT_FP8)` | **采用**：改动集中、易开关、测量路径最短 |
| 2. fp8 `BlockMmadTla` 偏特化 | 否：大模板分叉，穿刺阶段过重 |
| 3. 独立 helper + full_quant 主循环调用 | 否：跨文件同步 / L0 生命周期更绕 |

决策记录：测量模式 = Additive；宏名 = `BSA_PV_RESCALE_PROBE_FP8`；每 PV 热路径保留 InitConst。

## 4. 插入点与时序

```text
for nL0Itr:
  for m,k: L1→L0(fp8) → Mmad → L0C(PV后半)     // 现有
  if nL0Itr == 0:
    #ifdef BSA_PV_RESCALE_PROBE_FP8
      if constexpr (FULL_QUANT_FP8):
        RunRescaleMadProbe()   // InitConst→MTE1→Mad×2 → L0C前半
    #endif
  WaitCrossCoreSync(mm2ToReFlag)                // n==0，现有
  fixpipe(..., deqScalar)                       // 现有，不动
SetCrossCoreSync mm2ToReFlag                    // 现有，不动
```

约束：

1. 仅 `nL0Itr == 0` 且 `FULL_QUANT_FP8` 执行探针（每个 FP8 PV `operator()` 一次）
2. 探针写 L0C 前半；fixpipe 仍读当前 `l0CBufId` 后半
3. 复用 L0A/L0B ping-pong stage0（`ReinterpretCast<float>`）；探针结束后恢复 `M_MTE1`，不断后续 N-chunk 事件链
4. 不改 `mm2ToReFlag` / rescaleO

## 5. Buffer 布局

### 5.1 L1（P/V 尾部）

```text
l1BufAddrStart
  ├─ P: l1A × l1ABufNum × sizeof(ElementA)   // FP8 为 1B
  ├─ V: l1B × l1BBufNum × sizeof(ElementB)
  └─ 探针区（尾部）
       ├─ probeL1A: 64×64  fp32 = 16 KB
       └─ probeL1B: 64×128 fp32 = 32 KB   (合计 48 KB)
```

地址：

```text
probeBase = l1BufAddrStart
          + l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum
          + l1BTileK * l1BTileN * sizeof(ElementB) * l1BBufNum

probeL1A @ probeBase
probeL1B @ probeBase + 64 * 64 * sizeof(float)
```

前提：`probeBase + 48 KB ≤ L1_SIZE`。若不够，实现阶段用断言暴露。

### 5.2 L0

| Buffer | 用途 |
|--------|------|
| L0A `[0]` `ReinterpretCast<float>` | 探针 A：64×64 |
| L0B `[0]` 同上 | 探针 B：64×128 |
| L0C `@0`（前半） | 探针累加输出 scratch |

### 5.3 类型

- L1 分配为 **fp32**（按需求尺寸）
- MTE1 / mad 走 fp32：`LoadData` + `Mmad`（不复用现有 bf16/fp8 `CopyL1ToL0A/B`）
- 若平台对 L0A/B fp32 有限制：以能出 cycle 为准，可在实现时调整 Load/Mmad 实际元素类型，但 L1 分配尺寸仍按上述 fp32 布局

## 6. 两次 Mad 与事件

| 步骤 | 动作 | 形状 |
|------|------|------|
| 1 | `InitConst` 填充 `probeL1A/B`（常量如 `1.0f`） | 64×64、64×128 fp32 |
| 2 | MTE1：`probeL1A→L0A[0]`，`probeL1B→L0B[0]` | 同上 |
| 3 | Mad1 → L0C_scratch | `(M=64,N=128,K=64)`，`cmatrixInitVal=true` |
| 4 | Mad2 → L0C_scratch | 同形状，再次 `cmatrixInitVal=true`（纯叠 cycle） |

事件：

```text
InitConst L1
Set/Wait MTE2_MTE1
Wait M_MTE1 (L0A=0, L0B=2)
  LoadData L1→L0A/L0B
Set/Wait MTE1_M
  Mmad × 2 → L0C_scratch(@0)
Set M_MTE1
→ 原有 fixpipe(deqScalar)
```

## 7. 改动文件

| 文件 | 改动 |
|------|------|
| `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp` | 全部探针代码在 `#ifdef BSA_PV_RESCALE_PROBE_FP8` 内；调用加 `if constexpr (FULL_QUANT_FP8)` |
| `attention/block_sparse_attention/op_host/CMakeLists.txt` | 实验构建可选 `-DBSA_PV_RESCALE_PROBE_FP8` |

不改：`block_sparse_attention_kernel_arch35_full_quant.h` 主循环、rescaleO、QK Softmax。

## 8. 验证

1. **无宏**：regular / full_quant 行为与现网一致
2. **有宏 + FP8 case**：跑通；profiler 可见 InitConst / MTE1 / 2×Mmad
3. **有宏 + 非量化 case**：探针不执行（无对应 Cube 段）
4. 数值可不对齐；rescaleO 仍在 timeline 中

## 9. 风险

| 风险 | 缓解 |
|------|------|
| L1 尾部不够 48KB | 断言；FP8 下 P/V 更小，通常足够 |
| L0 事件与 PV pingpong 冲突 | 仅用 stage0；结束后恢复 `M_MTE1`；仅 `nL0Itr==0` |
| InitConst 的 MTE2 与 SparseV MTE2 打架 | 探针在 PV k 循环之后；事件 id 固定 0/2 |
| 宏打开时 half 实例残留未调用符号 | 可接受；运行路径由 `FULL_QUANT_FP8` 门闩关闭 |
