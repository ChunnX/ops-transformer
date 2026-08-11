# 设计串讲 — apace_quant_matmul_reduce_scatter

> **审查者**：Design Reviewer（独立审查）
> **审查范围**：DESIGN.md §1（算子设计）+ §2.4（伪代码），交叉验证 APACE MC2 skill references 与 asc-devkit API 文档
> **审查日期**：2026-08-11

---

## 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

**核心判定**：存在 1 个阻塞级问题（ProblemShape.K 取值错误）和 4 个需讨论级问题。阻塞问题修复后可进入开发。

---

## 质疑清单

### 问题 1：ProblemShape.K 应为 per-rank K（ka），而非全量 K

- **类别**：伪代码可实现性 / 多核策略
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §2.3（多核切分策略）、§2.7（tiling_data 结构体设计）、§2.8（Host 侧 tiling 推导）

**问题描述**：

设计在 §2.3、§2.7、§2.8 三处均将 ProblemShape.K 和 `QuantMatmulTilingData.k` 设为**全量 K**（`K = rankSize × ka`）：

```
§2.3: ProblemShape = `<M/rankSize, N, K, 1>` — K 为全量 K
§2.7: k | K (= rankSize × ka) | 完整 K（B 全量）
§2.8: tilingEngine.GetTilingData(mLocal, N, K, tilingData);  // k=全量K
```

但参考算子 `all_to_all_quant_matmul` 的 Host tiling 推导使用的是 **per-rank K（ka = K/rankNum）**，而非全量 K。

**审查者视角**：

根据 `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7 和 `references/apace/compute.md` §5.1，参考算子的 tiling 推导和 kernel 布局公式如下：

| 维度 | 参考算子（all_to_all_quant_matmul） | 本设计（ReduceScatter） |
|------|--------------------------------------|------------------------|
| GetTilingData k 参数 | `ka = K/rankNum`（per-rank K） | `K`（全量 K）❌ |
| ProblemShape.K | `ka` | `K = rankSize × ka` ❌ |
| layoutA | `(rankSize × originalM, ka)` → 匹配 A[M, ka] | `(rankSize × originalM, K)` → A 只有 ka 列，布局溢出 ❌ |
| layoutB | `(rankSize × ka, N) = (K_full, N)` → 匹配 B[K, N] | `(rankSize × K_full, N)` → 尺寸放大 rankSize 倍 ❌ |
| rank r 的 B 段 K 偏移 | `r × ka` → 正确访问 B[r×ka : (r+1)×ka] | `r × K_full` → 超出 B 实际数据范围 ❌ |

**影响分析**：

如果按设计当前方案（k = 全量 K）实现：

1. **layoutB 尺寸错误**：`layoutB = (rankSize × K_full, N)` 是实际 B 数据 `[K_full, N]` 的 rankSize 倍，kernel 访问 B 时地址越界
2. **B 段偏移错误**：rank r 的 B 段起点计算为 `r × K_full`（应为 `r × ka`），导致读取错误数据或 GM 越界访问
3. **layoutA 尺寸错误**：`layoutA = (M, K_full)` 但 A 实际只有 ka 列，MTE2 搬运会读取 A 越界数据
4. **SWAT tiling 参数错误**：baseK、stepK、scaleKL1 基于全量 K 推导，可能超出 L1 容量或与实际 per-rank 数据不匹配
5. **scaleKL1 错误**：scaleKL1 基于 `CeilDiv(K_full, 64) × 2` 推导，但 per-rank scale 数据只有 `CeilDiv(ka, 64) × 2` 个元素

**建议方案**：

将 ProblemShape.K、QuantMatmulTilingData.k、GetTilingData 的 k 参数全部改为 **ka = K / rankSize**（per-rank K）：

```cpp
// §2.3 修正:
ProblemShape = `<M/rankSize, N, ka, 1>`  // ka = K / rankSize（per-rank K）

// §2.7 修正:
| k | ka (= K / rankSize) | per-rank K（与参考算子一致） |

// §2.8 修正:
uint32_t ka = K / rankSize;
tilingEngine.GetTilingData(mLocal, N, ka, tilingData);  // k = per-rank K
```

**依据**：
- `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7：PUT 算子 GetTilingData k 参数 = per-rank K（ka = K/rankNum）
- `/ascendc-mc2-best-practice` skill `references/apace/compute.md` §5.1：layoutB = `(rankSize × K, N)` 中 K 为 ProblemShape.K = ka
- 参考算子 ST 代码 `apace/tests/st/all_to_all_quant_matmul/src/main.cpp`：`GetTilingData(M, N, ka, tilingData)`

---

### 问题 2：SetAtomicAdd/SetAtomicNone 的调用位置（Impl 还是 Kernel 内部）

- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.4（核心计算步骤）、§2.5（AIC/AIV 分工设计）

**问题描述**：

设计 §2.5 的伪代码在 Impl 的 `RunMatmul()` 中显式调用 `SetAtomicAdd<bfloat16_t>()` 和 `SetAtomicNone()`：

```
RunMatmul():
  SetAtomicAdd<bfloat16_t>()  ← 开启原子累加
  SetupParams(mode=REMOTE, splitKNum=rankSize-1)
  kernel_(params)
  SetAtomicNone()
```

但根据 `/ascendc-mc2-best-practice` skill `references/apace/compute.md` §4.1，`SetAtomicAdd`/`SetAtomicNone` 是由 **QuantMatmulMxKernel::Run() 内部** 调用的，不是由 Impl 调用：

```
kernel Run(params):
  ├── Init(params)  ← isAtomicAdd_ 在此设置（matmulMode==REMOTE && localMatmul==1）
  ├── if isAtomicAdd_: SetAtomicAdd<CType>()  ← kernel 内部调用
  ├── ProcessSingleBatch(...)
  └── if isAtomicAdd_: SetAtomicNone()  ← kernel 内部调用
```

**审查者视角**：

如果 Developer 按设计伪代码在 Impl 中显式调用 `SetAtomicAdd`，同时 kernel 内部也会调用，会导致：
1. **双重 SetAtomicAdd**：Impl 调一次 + kernel 内部调一次。虽然硬件上 SetAtomicAdd 可能是幂等的（设置原子模式），但语义混乱
2. **SetAtomicNone 时机错误**：kernel Run() 末尾会调 `SetAtomicNone()`，此时 Impl 的 `SetAtomicAdd()` 已被 kernel 内部关闭。如果 Impl 在 `kernel_(params)` 之后再调 `SetAtomicNone()`，是冗余的

**影响分析**：

可能导致 Developer 误解调用层次，在错误位置添加或遗漏 SetAtomicAdd 调用。参考算子 `all_to_all_mx_quant_matmul_udma_impl.h` 的 `RunMatmul()` 不显式调用 SetAtomicAdd（由 kernel 内部处理）。

**建议方案**：

明确伪代码中 `SetAtomicAdd`/`SetAtomicNone` 是 kernel 内部行为（由 `isAtomicAdd_` 自动触发），Impl 的 `RunMatmul()` 只需：

```cpp
RunMatmul():
  SetupParams(mode=REMOTE, splitKNum=rankSize-1)
  kernel_(params)  // kernel 内部根据 isAtomicAdd_ 自动调 SetAtomicAdd/SetAtomicNone
```

如果设计意图是让 Impl 显式控制（覆盖 kernel 内部行为），需说明理由并确认不会导致双重调用。

---

### 问题 3：测试用例 T4 的 headMSize=350 不符合推导公式

- **类别**：多核策略 / 伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：PLAN.md §3 测试用例表 T4、DESIGN.md §2.8 headMSize 推导

**问题描述**：

PLAN.md T4 指定 `headMSize=350`。但 DESIGN.md §2.8 的 headMSize 推导公式为：

```
headMSize = CeilDiv(usedCoreNum, nTile) * baseM
```

其中 `baseM` 由 SWAT 推导，典型值为 128/256/512（参考 `references/apace/compute.md` §6 关键常量 `BASIC_BLOCK_SIZE_16/64/128/256/512`）。`headMSize` 必须是 `baseM` 的整数倍。

350 不是 128、256、512 中任何一个的整数倍。

**审查者视角**：

- 如果 `headMSize` 在 cases.csv 中是手动指定并直接使用的：350 不是 baseM 的倍数，违反 headMSize 推导不变量，可能导致通信 tile 大小与 matmul tile 大小不对齐，破坏通算流水
- 如果 `headMSize` 在 cases.csv 中仅是信息性参数（实际由公式推导）：T4 的 350 不会被实际使用，但 cases.csv 误导 Developer

**影响分析**：

可能导致通信 tile 与 matmul tile 粒度不对齐，破坏通算重叠；或尾块处理异常。

**建议方案**：

1. 确认 cases.csv 中 `head_m_size` 列是否被 main.cpp 直接使用。如果直接使用，T4 的 headMSize 应改为 256（= baseM 256，M/rankSize=256 刚好一个 tile）或 128
2. 如果 headMSize 由公式推导（不使用 cases.csv 值），在 PLAN.md 中注明该列仅作参考

---

### 问题 4：测试用例 T8（K=4032）暴露 scale 通道 nonSplitAxisSize = ka/32 公式的精度风险

- **类别**：精度风险 / 多核策略
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.7 CommTilingData 填充、§2.8 输入校验

**问题描述**：

设计 §2.7 沿用参考算子的 scale 通道 `nonSplitAxisSize = ka / 32`（整数除法）。但 MXFP8 scale 的实际元素数为 `scaleKLen = CeilDiv(ka, 64) × 2`。

对于 T8（K=4032, rankSize=4, ka=1008）：
- `ka / 32 = 1008 / 32 = 31`（整数除法截断）
- `CeilDiv(ka, 64) × 2 = CeilDiv(1008, 64) × 2 = 16 × 2 = 32`
- **31 ≠ 32**：通信的 scale 数据量比实际少 1 个元素/行

设计 §2.8 的输入校验 `CeilDiv(ka, 64) % 2 == 0` 对于 ka=1008：`CeilDiv(1008, 64) = 16`，`16 % 2 == 0` ✅ 通过校验。但 ka=1008 不是 64 的倍数（1008 % 64 = 48 ≠ 0），导致 `ka/32` 截断。

**审查者视角**：

`ka / 32` 等于 `CeilDiv(ka, 64) × 2` **当且仅当** ka 是 64 的倍数。设计的校验 `CeilDiv(ka, 64) % 2 == 0` 只保证 scaleKLen 是偶数，不保证 ka 是 64 的倍数。

此问题继承自参考算子 `all_to_all_quant_matmul`（同样使用 `ka/32`），但参考算子的 ST 测试用例可能未覆盖 ka 非 64 倍数的场景。T8 是本设计新增的边界测试用例，会暴露此问题。

**影响分析**：

- scale 通信数据量少 1 个元素/行，导致最后一个 64-element group 的 scale 缺失
- AIC 读取 scale 时，最后一个 group 的 scale 为未初始化数据（GM 残留），可能导致反量化结果错误
- 影响范围：K 维最后 48 个元素（ka=1008 时，第 960-1007 列）的反量化 scale 错误

**建议方案**：

方案 A（推荐）：增加输入校验 `ka % 64 == 0`，拒绝 ka 非 64 倍数的输入：
```cpp
if (ka % 64 != 0) { reject; }
```

方案 B：将 `nonSplitAxisSize = ka / 32` 改为 `nonSplitAxisSize = CeilDiv(ka, 64) * 2`，正确处理非 64 倍数场景（需确认参考算子是否也需同步修复）。

方案 C：删除 T8 测试用例（不推荐，边界覆盖有价值）。

---

### 问题 5：flagId 14 与 SyncAll<true> 的冲突风险

- **类别**：多核策略
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.2 API 语义验证表、§2.5 同步点表、§2.6 flag 编排

**问题描述**：

设计使用 `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` 其中 `tid` 为通信 tile 索引（0~31，截断到 0~15）。同时使用 `SyncAll<true>()` 做 AIV 块间同步。

根据 asc-devkit API 文档 `SyncAll.md` 表3，对于 NPU 架构 3510（950PR），`SyncAll<true>`（isAIVOnly=true）占用 **flagId 14**（仅 AIV 侧）。

当 `tid % 16 == 14` 时，`CrossCoreSetFlag<0x2, PIPE_MTE3>(14)` 与 `SyncAll<true>` 内部使用的 flagId 14 冲突。

设计 §1.2 提到"避开 [11,14](SyncAll占用)"，但实际查 API 文档：
- `SyncAll<true>`（isAIVOnly=true）只占用 **flagId 14**（非 [11-14]）
- `SyncAll<false>`（isAIVOnly=false, __mix__(1,1)）占用 flagId 11、12、13

**审查者视角**：

本设计使用 `SyncAll<true>`（仅 AIV 同步），实际冲突点只有 flagId 14。设计描述的 [11,14] 范围过于保守（来自 skill `references/apace/fusion.md` §3.3 的笼统描述），但保守是安全的。

关键问题是：当通信 tile 总数 > 14 时，tid=14 的轮次会触发冲突。当前测试用例最大 tile 数为 2（T3），不会触发。但大 shape 场景可能触发。

**影响分析**：

- 当前测试用例（T1-T8）的 tile 数均 ≤ 2，不会触发冲突
- 大 shape 场景（如 M/rankSize > 14 × headMSize）可能触发，导致同步紊乱
- 参考算子（all_to_all_quant_matmul）同样存在此风险，是已知的理论风险

**建议方案**：

1. 在 DESIGN.md §2.11 分支场景覆盖表中增加说明：当 `splitAxisTileCnt + splitAxisTailCnt > 14` 时存在 flagId 14 冲突理论风险
2. 如需规避，可在 tiling 推导时限制 `headMSize` 使得 tile 总数 ≤ 14（牺牲通算流水深度），或接受此风险（与参考算子一致）
3. 修正 §1.2 的描述为"避开 flagId 14（SyncAll<true> 占用）"，更精确反映实际冲突范围

**依据**：
- asc-devkit `SyncAll.md` 表3：3510 架构 SyncAll<true> 占用 flagId 14（AIV 侧）
- asc-devkit `CrossCoreSetFlag_ISASI.md` 约束说明：需避免与 SyncAll 接口的 flagId 冲突

---

### 问题 6：PipeBarrier 补丁的来源标注不够清晰

- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.4 核心计算步骤、§2.5 AIC/AIV 分工设计

**问题描述**：

设计 §1.4 和 §2.5 在伪代码中标注了 `PipeBarrier<PIPE_ALL>() ★ 关键修复`，但未明确说明此 PipeBarrier **不在官网参考算子的 Run() 中**，是本算子新增的补丁。

根据 `/ascendc-mc2-best-practice` skill `references/apace/fusion.md` §5.4.3：
> ⚠️ **官网现状**：`all_to_all_mx_quant_matmul_udma_impl.h` `Run()` 当前为 `RunLocalMatmul()` 直接接 `RunMatmul()`，**两者之间无 PipeBarrier**。以下为推荐修复方案，Developer 在使用 localMatmul=1 时应添加。

**审查者视角**：

Developer 从参考算子复制 impl.h 后，需要在 `Run()` 中**新增** PipeBarrier 行。如果设计伪代码未明确标注这是新增代码（vs 从参考算子继承的代码），Developer 可能以为参考算子已有 PipeBarrier 而遗漏此改造点。

PLAN.md §3c 提到了"PipeBarrier 补丁"，但 DESIGN.md 的伪代码中只有 `★` 标记，未文字说明"官网无此行，需新增"。

**建议方案**：

在 DESIGN.md §2.5 的 Run() 伪代码中，对 PipeBarrier 行添加注释：
```cpp
PipeBarrier<PIPE_ALL>();  // ★ 新增补丁（官网参考算子无此行，localMatmul=1 必需）
```

---

### 问题 7：SyncAll<true> 在 MIX_AIC_1_1 kernel 中的语义需补充说明

- **类别**：多核策略
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.2 API 语义验证表

**问题描述**：

设计在 MIX_AIC_1_1 kernel 中使用 `SyncAll<true>()`（isAIVOnly=true）。根据 asc-devkit `SyncAll.md`：
> 对于Mix（包含Cube和Vector计算）算子，模板参数isAIVOnly设置为true，会导致接口实际只能完成Vector全核同步与预期同步行为不符。

但参考算子（all_to_all_quant_matmul）同样在 MIX_AIC_1_1 kernel 中使用 `SyncAll<true>()`，因为这里**只需要 AIV 全核同步**（保证 WriteNbi 对端可见性），AIC 不参与 SyncAll，AIC↔AIV 同步由 CrossCore flag 独立处理。

**审查者视角**：

API 文档的警告是针对"预期 SyncAll 完成 Mix 全核同步"的场景。本设计中 `SyncAll<true>` 的预期行为就是**仅 AIV 同步**，所以行为与预期一致。但设计未说明为什么在 MIX kernel 中用 isAIVOnly=true 是正确的，可能让 Developer 疑惑。

**建议方案**：

在 DESIGN.md §1.2 或 §2.5 的 SyncAll 行添加说明：
```
SyncAll<true>()  // isAIVOnly=true：仅 AIV 全核同步（保证 WriteNbi 对端可见性），
                 // AIC 不参与 SyncAll，AIC↔AIV 同步由 CrossCore flag 独立处理
```

---

### 问题 8：B 全量复制的内存开销量化缺失

- **类别**：内存规划
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.4 关键设计要点 3、§1.5 GM 输入输出表

**问题描述**：

设计 §1.4 提到"B 全量复制（rankSize 倍内存）"，但未给出具体内存开销估算。对于大 shape 测试用例 T3（K=8192, N=4096, FP8），每 rank 的 B = 8192 × 4096 × 1B = 32MB，4 卡总 B 内存 = 128MB。

**审查者视角**：

MC2 算子的 GM 内存规划应量化 B 的内存开销，确认不超过 GM 容量。虽然 950PR 的 GM 通常足够，但对于极大 shape（如 K=16384, N=8192），B 可能达到 128MB/rank，需要关注。

**建议方案**：

在 §1.5 GM 输入输出表中增加 B 内存估算行：
```
| B 内存 (per rank) | K × N × sizeof(AType) | T3: 8192×4096×1 = 32MB |
| B 总内存 (all ranks) | rankSize × K × N × sizeof(AType) | T3: 128MB |
```

---

## 审查维度汇总

| 序号 | 审查维度 | 结论 | 问题编号 |
|------|---------|------|---------|
| 1 | API 可行性 | ✅ 通过 | 所有 API 经 asc-devkit 文档验证存在且 950PR 支持 |
| 2 | 内存规划合理性 | ✅ 通过（建议补充 B 内存量化） | 问题 8 |
| 3 | 多核策略可行性 | ⚠️ 需讨论 | 问题 3（headMSize）、问题 5（flagId 冲突） |
| 4 | 伪代码可实现性 | 🔴 阻塞 | 问题 1（ProblemShape.K）、问题 2（SetAtomicAdd 位置）、问题 6（PipeBarrier 标注） |
| 5 | 精度风险 | ⚠️ 需讨论 | 问题 4（ka/32 精度风险） |

---

## API 验证依据汇总

| API | 验证源 | 文档路径 | 验证结论 |
|-----|--------|---------|---------|
| `CrossCoreSetFlag<0x2, PIPE_MTE3>` | asc-devkit 文档 | `docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md` | 950PR 支持 mode 2 ✅；PIPE_MTE3 支持 ✅；flagId 0-15（截断 4bit）✅ |
| `CrossCoreWaitFlag<0x2, PIPE_MTE2>` | asc-devkit 文档 | `docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md` | 与 SetFlag 配对 ✅ |
| `SetAtomicAdd<bfloat16_t>` | asc-devkit 文档 | `docs/zh/api/SIMD-API/c_api/simd_atomic/asc_set_atomic_add.md` | 950PR 支持 ✅；bfloat16_t 支持 ✅（`asc_set_atomic_add_bfloat`）；L1→GM 不支持（本设计用 L0C→GM，不受影响）✅ |
| `PipeBarrier<PIPE_ALL>` | asc-devkit 文档 | `docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.md` | 950PR 支持 ✅；PIPE_ALL 支持 ✅；性能影响提示 ✅ |
| `SyncAll<true>` | asc-devkit 文档 | `docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md` | 950PR 支持 ✅；isAIVOnly=true 占用 flagId 14（非 [11-14]）；MIX kernel 中仅 AIV 同步 ✅ |
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` | APACE skill | `references/apace/operator-anatomy.md` §5.2 | 核配比 1:1 ✅ |
| `Blaze::Gemm::Block::BlockMmad` | APACE skill + Blaze skill | `references/apace/compute.md` §2 | 组件签名正确 ✅ |
| `CollectiveComm<AllToAll, PUT, T, TeamBarrier>` | APACE skill | `references/apace/communication.md` §2 | 四段式 API ✅ |
| `UdmaCommWaitPolicy::WaitTile` | APACE skill | `references/apace/fusion.md` §4.2 | CrossCoreWaitFlag<0x2, PIPE_MTE2> ✅ |

---

## 总结

本设计在 APACE 路线约束遵守、通信模式选型、Blaze 组件选择、通算流水编排等方面与参考算子和 skill 文档高度一致，整体架构合理。主要问题集中在：

1. **阻塞级**：ProblemShape.K 取值错误（应为 per-rank ka 而非全量 K），直接影响 kernel 布局和 B 地址计算，必须修复后才能开发
2. **讨论级**：SetAtomicAdd 调用位置、headMSize 合规性、ka/32 精度风险、flagId 冲突 — 建议与 Architect 确认后逐项处理
3. **建议级**：PipeBarrier 标注、SyncAll 语义说明、B 内存量化 — 改善文档清晰度，不阻塞开发

建议 Architect 回应时优先处理问题 1（阻塞），其余问题可视情况采纳。

---

### Architect 回应

> **回应日期**：2026-08-11
> **回应模式**：串讲回应模式
> **DESIGN.md 是否更新**：是（已更新 §1.2、§1.3、§1.4、§1.5、§2.3、§2.5、§2.6、§2.8、§2.11；PLAN.md 同步更新 §3、§4、§5）

#### 问题 1：ProblemShape.K 应为 per-rank K（ka），而非全量 K

- **回应**：已修改（接受）
- **理由**：Design Reviewer 的质疑完全正确。经核实 `operator-anatomy.md` §3.7 Line 280 明确记载「PUT（all_to_all）的 `k` 参数 = per-rank K（ka = K/rankNum）」，Line 307 进一步确认「PUT 用 ka，AG 用 totalLogicalM」。`compute.md` §5.1 Line 286 的 `layoutB = (rankSize × K, N)` 中 K 即 ProblemShape.K，对 PUT 模式而言就是 ka，因此 layoutB 实际为 `(rankSize × ka, N) = (K_full, N)` 匹配 B 全量数据，rank r 的 B 段起点 `r × ka`。原设计误将 ProblemShape.K、QuantMatmulTilingData.k、GetTilingData 的 k 参数设为全量 K，会导致 layoutA/layoutB 尺寸放大 rankSize 倍、B 段偏移越界、SWAT tiling 参数错误。
- **文档依据**：
  - `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7 Line 280、307
  - `/ascendc-mc2-best-practice` skill `references/apace/compute.md` §5.1 Line 284-286、§5 Line 295
- **DESIGN.md 变更**：
  - §2.3：`ProblemShape = <M/rankSize, N, K, 1>` → `<M/rankSize, N, ka, 1>`（ka = K/rankSize），并补充 layoutB 匹配说明
  - §2.7 QuantMatmulTilingData 填充表 k 行：`K (= rankSize × ka)` → `ka (= K / rankSize)`
  - §2.8 GetTilingData 调用：增加 `uint32_t ka = K / rankSize;` 并改为 `GetTilingData(mLocal, N, ka, tilingData)`，新增"关键约束"段落说明 PUT 模式 k=ka 的硬性不变量
  - PLAN.md §4 改造点 #4：增加 ProblemShape.K = ka 说明；#6：修正为 `GetTilingData(M/rankSize, N, ka, tilingData)`

#### 问题 2：SetAtomicAdd/SetAtomicNone 的调用位置（Impl 还是 Kernel 内部）

- **回应**：已修改（接受）
- **理由**：Design Reviewer 的质疑正确。经核实 `compute.md` §4.1 Line 146-157 明确记载 `QuantMatmulMxKernel::Run` 内部流程：`Init(params)` → `if isAtomicAdd_: SetAtomicAdd<CType>()` → `ProcessSingleBatch` → `if isAtomicAdd_: SetAtomicNone()`。`isAtomicAdd_` 在 `matmulMode==REMOTE && localMatmul==1` 时由 kernel `Init()` 自动置位（compute.md §3 Line 137）。因此 SetAtomicAdd/SetAtomicNone 是 **kernel Run() 内部**调用，Impl 的 `RunMatmul()` 不应再显式调用，否则会导致双重 SetAtomicAdd（硬件幂等但语义混乱）+ 冗余 SetAtomicNone。`fusion.md` §5.4.3 的 PipeBarrier 修复方案伪代码（Line 297-316）也确认 Impl Run() 只调 `RunLocalMatmul()` → `PipeBarrier` → `RunMatmul()`，不显式调 SetAtomicAdd。
- **文档依据**：
  - `/ascendc-mc2-best-practice` skill `references/apace/compute.md` §4.1 Line 146-157、§3 Line 137
  - `/ascendc-mc2-best-practice` skill `references/apace/fusion.md` §5.4.3 Line 297-316
- **DESIGN.md 变更**：
  - §1.2 API 语义验证表 SetAtomicAdd 行：增加"由 kernel Run() 内部根据 isAtomicAdd_ 自动调用（见 compute.md §4.1）"
  - §1.3 数据流图：移除 Impl 层 `SetAtomicAdd<BF16>()` / `SetAtomicNone()` 独立行，改为 `RunMatmul()` 内标注"kernel 内部自动 SetAtomicAdd/SetAtomicNone"
  - §1.4 核心计算步骤 4：移除 `SetAtomicAdd<bfloat16_t>()` 和 `SetAtomicNone()`，改为"kernel 内部根据 isAtomicAdd_ 自动调"
  - §2.5 AIC 伪代码 RunMatmul()：移除 `SetAtomicAdd` / `SetAtomicNone` 两行，改为注释说明 kernel 内部自动调用
  - §2.6 localMatmul 编排图：移除 `SetAtomicAdd<BF16>()` / `SetAtomicNone()` 独立行，合并到 RunMatmul() 行

#### 问题 3：测试用例 T4 的 headMSize=350 不符合推导公式

- **回应**：部分修改
- **理由**：Design Reviewer 对 headMSize 必须为 baseM 整数倍的判断正确（`operator-anatomy.md` §3.7 Line 284-291 确认 headMSize 由公式 `CeilDiv(usedCoreNum, nTile) * baseM` 推导，baseM 候选值 128/256/512）。350 不是任何 baseM 候选值的倍数。但 Design Reviewer 提出的"cases.csv head_m_size 列是否被直接使用"的关键问题，经核实 `operator-anatomy.md` §3.7 Line 284 明确记载"headMSize 推导（PUT，内联于 ST main.cpp）：官网无 comm_tiling_base.h；通信切分与 matmul tile 粒度的对齐由 ST main.cpp 内联完成"——即 headMSize **由 main.cpp 公式推导，不从 cases.csv 读取**。因此 cases.csv 的 head_m_size 列是信息性参数，350 不会被实际使用，不会破坏通算流水。但保留 350 确实会误导 Developer，且 T4 的 M=1024（M/rankSize=256）是 2 的幂次，任何 baseM 倍数都整除它，无法触发尾块。
- **文档依据**：
  - `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7 Line 284-291（headMSize 公式推导，非 cases.csv 读取）
- **DESIGN.md 变更**：
  - §2.8：新增说明"cases.csv 的 head_m_size 列仅作信息性参考，不被 main.cpp 直接使用，实际 headMSize 由公式内联推导"
- **PLAN.md 变更**：
  - §3 T4：M=1024 → 1120（M/rankSize=280，baseM=128 时 280%128=24≠0 可触发尾块）；headMSize=350 → "由公式推导"
  - §6.1 ST 通路表 T4：M=1024 → 1120

#### 问题 4：测试用例 T8（K=4032）暴露 scale 通道 nonSplitAxisSize = ka/32 公式的精度风险

- **回应**：部分修改（采用方案 A）
- **理由**：Design Reviewer 的精度风险分析完全正确。T8（K=4032, rankSize=4, ka=1008）：`ka/32 = 31`（整数截断）≠ `CeilDiv(ka, 64)×2 = 32`（实际 scale 元素数），会导致 scale 通信数据量少 1 个元素/行，最后一个 64-element group 的 scale 缺失，反量化结果错误。此问题继承自参考算子（`operator-anatomy.md` §3.7 Line 145/147/309 确认参考算子也用 `ka/32`），但参考算子的 ST 用例隐含满足 ka%64==0。采用方案 A（增加 `ka % 64 == 0` 校验）而非方案 B（改 `nonSplitAxisSize = CeilDiv(ka, 64)×2`），理由：(1) 与参考算子隐含约束一致；(2) 不修改 `block/` `tiling/` 共享层逻辑（APACE 红线③）；(3) 方案 B 需修改 host 侧 CommTilingData 填充逻辑，引入与参考算子的偏差，增加维护成本。T8 的 K 值同步调整为 4352（ka=1088=64×17），既满足 ka%64==0 又保留"K 非 2 的幂次"的边界测试价值。
- **文档依据**：
  - `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7 Line 145、147、309（参考算子 `nonSplitAxisSize = ka/32`）
  - `/ascendc-mc2-best-practice` skill `references/apace/operator-anatomy.md` §3.7 Line 299（参考算子校验 `CeilDiv(ka, 64) % 2 == 0`）
- **DESIGN.md 变更**：
  - §2.8 输入校验：增加 `ka % 64 == 0`（确保 `ka/32` 等价于 `CeilDiv(ka, 64)×2`），保留原 `CeilDiv(ka, 64) % 2 == 0` 作为冗余校验
- **PLAN.md 变更**：
  - §3 T8：K=4032 → 4352（ka=1088=64×17）；需求追溯改为"ka=1088=64×17, ka%64=0, K 非 2 的幂次"
  - §6.1 ST 通路表 T8：K=4032 → 4352

#### 问题 5：flagId 14 与 SyncAll<true> 的冲突风险

- **回应**：部分修改
- **理由**：Design Reviewer 对冲突范围的修正正确。skill 文档 `fusion.md` §3.3 Line 112 和 `communication.md` §4.2 Line 379 说"SyncAll 硬同步内部占用 flagId [11-14]"是笼统描述（涵盖 isAIVOnly=true 和 false 两种情况）。经 asc-devkit `SyncAll.md` 表3 确认：`SyncAll<true>`（isAIVOnly=true）只占用 flagId 14（仅 AIV 侧），`SyncAll<false>`（isAIVOnly=false）占用 flagId 11/12/13。本设计使用 `SyncAll<true>`，实际冲突点只有 flagId 14。原设计 §1.2 的"避开 [11,14]"过于保守但安全；修正为"避开 flagId 14"更精确。当前测试用例 tile 数 ≤ 2 不会触发冲突，但大 shape 场景存在理论风险，需在 §2.11 显式记录。
- **文档依据**：
  - asc-devkit `docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md` 表3（3510 架构 SyncAll<true> 占用 flagId 14）
  - `/ascendc-mc2-best-practice` skill `references/apace/fusion.md` §3.3 Line 112（笼统描述 [11-14]）
- **DESIGN.md 变更**：
  - §1.2 API 语义验证表 CrossCoreSetFlag 行：`避开 [11,14](SyncAll占用)` → `避开 flagId 14（SyncAll<true> 占用，见 SyncAll.md 表3）`
  - §1.2 SyncAll 行：`占用 flagId [11-14]` → `isAIVOnly=true 占用 flagId 14（仅 AIV 侧，见 SyncAll.md 表3）`
  - §2.5 同步点表 AIV 块间同步行：`[11-14] 内部` → `14（isAIVOnly=true，见 SyncAll.md 表3）`
  - §2.11 分支场景覆盖表：新增"flagId 14 冲突理论风险"行，说明触发条件（tile 总数 > 14）和规避策略

#### 问题 6：PipeBarrier 补丁的来源标注不够清晰

- **回应**：已修改（接受）
- **理由**：Design Reviewer 的建议合理。`fusion.md` §5.4.3 Line 293 明确记载"官网现状：all_to_all_mx_quant_matmul_udma_impl.h Run() 当前为 RunLocalMatmul() 直接接 RunMatmul()，两者之间无 PipeBarrier。以下为推荐修复方案，Developer 在使用 localMatmul=1 时应添加。"原设计伪代码仅用 `★` 标记未文字说明，可能让 Developer 误以为参考算子已有 PipeBarrier 而遗漏此改造点。
- **文档依据**：
  - `/ascendc-mc2-best-practice` skill `references/apace/fusion.md` §5.4.3 Line 293
- **DESIGN.md 变更**：
  - §1.4 核心计算步骤 3：`PipeBarrier<PIPE_ALL>() ★ 关键修复` → `★ 关键修复（新增补丁，官网参考算子无此行）`
  - §2.5 AIC 伪代码 PipeBarrier 行：`★ MTE 修复` → `// ★ 新增补丁（官网参考算子 all_to_all_mx_quant_matmul_udma_impl.h Run() 无此行；localMatmul=1 必需，见 fusion.md §5.4）`
  - §2.6 localMatmul 编排图 PipeBarrier 行：`确保 LOCAL fixpipe 排空` → `新增补丁：确保 LOCAL fixpipe 排空`

#### 问题 7：SyncAll<true> 在 MIX_AIC_1_1 kernel 中的语义需补充说明

- **回应**：已修改（接受）
- **理由**：Design Reviewer 的建议合理。asc-devkit `SyncAll.md` 警告"对于 Mix 算子，模板参数 isAIVOnly 设置为 true，会导致接口实际只能完成 Vector 全核同步与预期同步行为不符"——这是针对"预期 SyncAll 完成 Mix 全核同步"的场景。本设计中 `SyncAll<true>` 的预期行为就是**仅 AIV 同步**（保证 WriteNbi 对端可见性），AIC 不参与 SyncAll，AIC↔AIV 同步由 CrossCore flag 独立处理，行为与预期一致。但原设计未说明这一点，可能让 Developer 疑惑。
- **文档依据**：
  - asc-devkit `SyncAll.md`（Mix 算子 isAIVOnly 警告）
  - `/ascendc-mc2-best-practice` skill `references/apace/communication.md` §4.3 Line 386（SyncAll<true> 是 AIV 块间硬同步）
- **DESIGN.md 变更**：
  - §1.2 API 语义验证表 SyncAll 行：增加"MIX_AIC_1_1 kernel 中仅 AIV 全核同步（AIC 不参与 SyncAll，AIC↔AIV 同步由 CrossCore flag 独立处理，行为与预期一致）"

#### 问题 8：B 全量复制的内存开销量化缺失

- **回应**：已修改（接受）
- **理由**：Design Reviewer 的建议合理。MC2 算子的 GM 内存规划应量化 B 的内存开销，确认不超过 GM 容量。
- **文档依据**：设计 §1.4 关键设计要点 3 已提到"B 全量复制（rankSize 倍内存）"，但未量化。
- **DESIGN.md 变更**：
  - §1.5 GM 输入输出表后：新增"B 内存开销估算"小表，含 per-rank 和 all-ranks 公式 + T3 示例（32MB/rank，128MB 总），并注明极大 shape 的 GM 容量关注点

#### 回应统计

- **接受**：5 项（问题 1、2、6、7、8）
- **部分修改**：3 项（问题 3、4、5）
- **保留原设计**：0 项
- **DESIGN.md 更新**：是（§1.2、§1.3、§1.4、§1.5、§2.3、§2.5、§2.6、§2.8、§2.11 共 9 个章节）
- **PLAN.md 更新**：是（§3 测试用例 T4/T8、§4 改造点 #4/#6、§5 已知问题记录新增 4 条）

#### 阻塞问题修复确认

| 阻塞问题 | 修复状态 | 验证依据 |
|---------|---------|---------|
| 问题 1：ProblemShape.K = ka | ✅ 已修复 | DESIGN.md §2.3/§2.7/§2.8 + operator-anatomy.md §3.7 Line 280/307 |

所有阻塞问题已修复，设计可进入开发阶段。
