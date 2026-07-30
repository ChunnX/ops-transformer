# BSA PV 后 Rescale Mad 性能穿刺设计

**日期：** 2026-07-29  
**状态：** 已批准（待实现）  
**范围：** Arch35 BSA `BlockMmadPV`，纯 cycle 穿刺，不要求数值正确

## 1. 目标

在 PV（P×V）完成之后、fixpipe 之前，插入一段 cube 侧探针，用两次 mad 模拟未来替代 `rescaleO` 的路径，测量额外开销。

成功标准：

1. 加 `-DBSA_PV_RESCALE_PROBE` 可编译通过
2. 单 case 可跑通（数值允许错误）
3. msprof / 流水对比开关宏前后，可见 L1 init + MTE1 + 2×Mmad

非目标：

- 不替代、不旁路 `rescaleO`（照常执行）
- 不做精度对齐
- 默认构建不打开该宏

## 2. 背景（现有管线）

Arch35 BSA 主路径：

```text
QK → Softmax(P@L1) → PV(mmad→L0C) → fixpipe(L0C→UB) → rescaleO(AIV)
```

插入点在 `block_mmad_pv_arch35_ABf16_C_to_UB.hpp` 的 `operator()` 中：第一个 N-chunk 的 m/k mmad 全部完成后、该 chunk 的 fixpipe 之前。

现网 PV 使用 L0C **后半区**（`L0C_HALF_BUF_SIZE + pingpong`）。探针使用 L0C **前半区**（`@0`），避免踩坏即将 fixpipe 的 PV 结果。

## 3. 方案选择

采用**方案 1：同文件 `#ifdef` 内联探针**。

| 方案 | 结论 |
|------|------|
| 1. 同文件 `#ifdef` 内联 | **采用**：改动集中、易开关、测量路径最短 |
| 2. 独立 `ProbeRescaleMad` helper | 否：穿刺阶段过重 |
| 3. 复用另一半 L0 ping-pong 作 scratch | 否：易与后续 N-chunk 事件打架 |

## 4. 插入点与时序

```text
for nL0Itr:
  for m,k: L1→L0 → Mmad → L0C(PV后半)     // 现有
  if nL0Itr == 0 && BSA_PV_RESCALE_PROBE:
    L1 InitConst → MTE1 → Mad×2 → L0C(前半) // 探针，每 PV 一次
  fixpipe                                       // 现有，不动
SetCrossCoreSync mm2ToReFlag                    // 现有，不动
```

约束：

1. 仅 `nL0Itr == 0` 执行探针（每个 PV `operator()` 一次）
2. 探针写 L0C 前半；fixpipe 仍读当前 `l0CBufId` 后半
3. 复用 L0A/L0B ping-pong；探针结束后恢复 `M_MTE1`，不断后续 N-chunk 事件链
4. 不改 `mm2ToReFlag` / rescaleO

## 5. Buffer 布局

### 5.1 L1（P/V 尾部硬编码）

```text
l1BufAddrStart
  ├─ P: l1A × l1ABufNum
  ├─ V: l1B × l1BBufNum
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

前提：MM2 L1 尾部至少还能再放 48 KB。若不够，实现阶段用断言暴露，或临时调整硬编码偏移。

### 5.2 L0

| Buffer | 用途 |
|--------|------|
| L0A `[0]` | 探针 A：64×64 |
| L0B `[0]` | 探针 B：64×128 |
| L0C `@0`（前半） | 探针累加输出 scratch |

### 5.3 类型

- L1 按要求为 **fp32**
- MTE1 / mad 走 fp32 路径（直接 `LoadData` + `Mmad`，不复用现有 bf16 `CopyL1ToL0A/B`）
- 若平台对 L0A/B fp32 有限制：以能出 cycle 为准，可在实现时调整 Load/Mmad 实际元素类型，但 L1 分配尺寸仍按上述 fp32 布局

## 6. 两次 Mad 与事件

| 步骤 | 动作 | 形状 |
|------|------|------|
| 1 | `InitConst` 填充 `probeL1A/B`（常量如 `1.0f`） | 64×64、64×128 fp32 |
| 2 | MTE1：`probeL1A→L0A[0]`，`probeL1B→L0B[0]` | 同上 |
| 3 | Mad1 → L0C_scratch | `(M=64,N=128,K=64)`，`cmatrix_init=true` |
| 4 | Mad2 → L0C_scratch | 同形状，再次 `cmatrix_init=true`（纯叠 cycle） |

事件顺序：

```text
Wait M_MTE1 (L0A/L0B)
  InitConst L1
  LoadData L1→L0A/L0B
Set MTE1_M
Wait MTE1_M
  Mmad × 2 → L0C_scratch(@0)
Set M_MTE1
→ 原有 fixpipe
```

刻意不做：探针结果不 fixpipe 到 UB；不校验数值。

## 7. 改动范围

**仅改 1 个文件：**

```
attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
```

- 探针常量与 L1 尾址计算
- `#ifdef BSA_PV_RESCALE_PROBE` 代码块
- 不改 host tiling、rescaleO、算子对外接口

编译开关：`-DBSA_PV_RESCALE_PROBE`

## 8. 验证

1. 开宏编译 BSA Arch35 路径通过
2. 单 case 跑通不挂
3. msprof 对比开关宏：确认多出 L1 init + MTE1 + 2×Mmad 段

## 9. 后续（本穿刺范围外）

若 cycle 可接受，再单独立项：用 cube mad 真正替代 AIV `rescaleO`（含 dm/gl 数据通路与数值对齐）。本设计不覆盖该项。