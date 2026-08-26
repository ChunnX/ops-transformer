# localMatmul 前置/后置重排优化 — 性能收益分析

> 日期: 2026-08-25
> 分析对象: `quant_matmul_reduce_scatter` 算子
> 核心问题: "通信在前就 localMatmul 前置，通信在后就把 localmatmul 后置。matmul 在前，要把本卡的 matmul 放在后面" 该优化是否能提升性能？

## 结论

**当前 R=2/R=4 场景下不会提升性能，甚至可能劣化。**

该优化模式（本卡 matmul 后置以与通信重叠）在当前算子中无效，原因是通信已被 AIC 计算完全掩盖，瓶颈在 AIC MTE2 内存带宽而非通信。

---

## 分析依据

### 1. 瓶颈在 AIC MTE2，不在通信

`perf_optimization_results.md:94-99` 明确指出：

- **11/12 用例为 MTE2 bound**（AIC 内存带宽瓶颈）
- **AIV 完全被 AIC 掩盖**（82% 时间在 WaitFlag）
- tiling 参数（baseM/baseN/baseK/stepK/dbL0c）已穷尽，无法突破 MTE2 瓶颈

当前 AIV 通信+reduceSum 已经被 AIC 计算完全掩盖：

```
AIC:  [mm0][mm1][mm2]...[mmT]   ← 瓶颈在这里（MTE2 带宽）
AIV:       [c0 ][c1 ][c2 ]...    ← 82% 时间在等 AIC（WaitFlag）
```

引用:
- AIC 侧顺序执行 matmul: `quant_matmul_reduce_scatter_impl.h:271` (`RunMatmul`)
- AIV 侧流水等待: `quant_matmul_reduce_scatter_impl.h:335` (`RunAllToAllAndReduceSum`)
- FragmentTensor 一次调用覆盖所有 rank 段: `qmm_mx_kernel_rs_frag.h:368` (`BuildFragmentTensors`)

### 2. 重排 AIC 内部计算顺序不减少 AIC 总工作量

"local matmul 后置"本质是**重排 AIC 上各 rank 段的计算顺序**（先算远端 rank 段，最后算本卡 rank 段）。但：

- AIC 总计算量 = 所有 rankSize 段的 matmul 之和，**与顺序无关**
- MTE2 总搬运字节数不变（A 矩阵全部要从 GM→L1）
- AIC 总时间不变 → AIV 掩盖关系不变 → 端到端时间不变

### 3. 通信提前启动无收益（AIV 已在等待）

该优化的理论收益是"远端段算完即启动 AllToAll，本卡段与通信重叠"。但：

- AIV 通信**已经**被完全掩盖（82% WaitFlag）
- 让通信更早启动 → AIV 等待时间更长，不是更短
- 唯一收益点是 **tile 0 的通信提前启动**，但 tile 0 之后所有 tile 的通信已被 `mm(t+1) ‖ comm(t)` 流水掩盖

当前多 tile 流水策略（`quant_matmul_reduce_scatter_impl.h:389-408`）：

```
AIC:  [tile0 frag mm] SetFlag [tile1 frag mm] SetFlag ...
AIV comm blocks:       WaitFlag [AllToAll0]   WaitFlag [AllToAll1]   ...
AIV compute blocks:    WaitFlag [ReduceSum0]  WaitFlag [ReduceSum1]  ...
```

AIC 计算 tile N 时，AIV 通信核在 PUT tile N，计算核在 ReduceSum tile N-1——通信和求和延迟已被计算掩盖。

### 4. 潜在劣化风险

| 风险 | 说明 |
|------|------|
| **FragmentTensor 拆分开销** | 当前一次 matmul 调用覆盖所有 rank 段（`qmm_mx_kernel_rs_frag.h:368`），拆分成本卡/远端两段调度 → 多次 matmul launch、scheduler 开销增加 |
| **L1 局部性破坏** | FragmentTensor 的 per-fragment L1 隔离设计假设连续调度，拆分后 B 矩阵可能重复加载 |
| **reduceSum 延迟** | reduceSum(t) 需要本卡段数据；本卡段延后计算 → reduceSum(t) 也被延后，可能拉长串行尾 |

FragmentTensor 的设计文档（`qmm_mx_kernel_rs_frag.h:15-28`）明确说明：ReduceScatter 场景下 A 数据全在本卡 GM 连续 `[m,k]`，各 rank 段天然连续，FragmentTensor 打包各 rank 段地址以保持 per-fragment L1 隔离。拆分本卡/远端会破坏这一连续性假设。

---

## 优化适用场景对比

"localMatmul 前置/后置"是经典的通算重叠模式，其有效性取决于**通信是否为瓶颈**：

| 场景 | 通信是否瓶颈 | 优化是否有效 | 说明 |
|------|-------------|-------------|------|
| **AllGather + MatMul**（通信在前） | 通信常为瓶颈 | ✅ 有效 | localMatmul 前置，与后续通信重叠 |
| **MatMul + ReduceScatter** 且 comm > compute | 通信瓶颈 | ✅ 有效 | localMatmul 后置，本卡计算与通信重叠 |
| **当前算子** R=2/R=4 | **通信被掩盖**，AIC MTE2 瓶颈 | ❌ 无效 | 通信非瓶颈，重排无收益 |

---

## 当前性能数据（参考）

来自 `perf_optimization_results.md:47-63`（msprof Task Duration 中位数，跨 rank max, 8/25 重新采集）：

| Case | 旧基线 (us) | 方案B (us) | 变化 |
|------|------------|-----------|------|
| 8192x1536x12288 R=2 | 2669 | 2435.5 | -8.7% |
| 8192x1536x12288 R=4 | 1696 | 1336.3 | -21.2% |
| 8192x3696x8192 R=2 | 1947 | 1772.0 | -9.0% |
| 8192x3696x8192 R=4 | 1328 | 1317.8 | -0.8% |
| 8192x1024x8192 R=2 | 1799 | 1622.5 | -9.8% |
| 8192x1024x8192 R=4 | 1207 | 918.7 | -23.9% |
| 4096x3584x4096 R=2 | 526 | 472.3 | -10.2% |
| 4096x3584x4096 R=4 | 415 | 307.5 | -25.9% |
| 2048x3584x4096 R=2 | 286 | 272.6 | -4.7% |
| 2048x3584x4096 R=4 | 222 | 175.6 | -20.9% |
| 4032x8192x4096 R=2 | 604 | 575.0 | -4.8% |
| 4032x8192x4096 R=4 | 609 | 615.4 | +1.1% |

- R=2 平均: -7.9%
- R=4 平均: -15.3%
- 总体平均: -11.6%

---

## 建议方向

若要真正提升性能，应针对 **AIC MTE2 瓶颈** 而非通信重叠：

| 方向 | 预期收益 | 难度 | 依据 |
|------|---------|------|------|
| **SplitK**（每卡算 K/R，AIC 计算量减半） | ~1.9x | 高（需 AllReduce 替代 ReduceScatter） | `perf_optimization_results.md:105`，直接减少 MTE2 工作量 |
| 框架修复 CrossDeviceExecute | 消除手动 CrossDevice 开销 | 中（框架侧修改） | 中等收益 |
| GET 模式 | 小（PUT 已稳定） | 低（4+rank 稳定性） | 小收益 |
| bias 支持 | 功能扩展 | 低 | 非性能优化 |
| **localMatmul 后置** | **~0%** | 中 | 通信已被掩盖，不解决瓶颈 |

---

## 何时 localMatmul 后置可能有效（需验证）

以下场景理论上可能从该优化受益，但当前无数据佐证：

- **大 rankSize（R≥16/32/64）**：通信量随 R 增长，可能从 compute-bound 翻转为 comm-bound。但当前 perf 数据仅有 R=2/R=4。
- **小 shape 场景**（M/K/N 小，matmul 快、通信占比高）：可能 comm 不再被掩盖。但 perf 报告显示小 shape（2048x3584x4096）仍是 MTE2 bound。

若需验证，建议：
1. 补充 R=8/16/32 的 msprof 采集，确认通信是否成为瓶颈
2. 若确认为 comm-bound，再评估 localMatmul 后置的收益

---

> 文档状态: 分析完成
> 适用版本: CANN 9.2.0（Apace 架构, dav-3510）
> 代码版本: quant_matmul_reduce_scatter_impl.h (方案B: 后R核通信+前28-R核计算)
> 参考文档: perf_optimization_results.md, DESIGN.md
