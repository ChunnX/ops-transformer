# apace_quant_matmul_reduce_scatter_v2 设计串讲质疑清单

> **审查者**：Design Reviewer（独立审查）
> **审查对象**：DESIGN.md（§1 算子设计 + §2 架构设计）
> **审查依据**：asc-devkit API 文档、apace MC2 skill references（fusion.md/communication.md/architecture.md）、ops-nn `quant_matmul_reduce_sum` 参考实现、v1 分支 `feat/apace-quant-matmul-reduce-scatter` 完整代码

---

## 设计串讲

### 审查结论
- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

**总体评价**：v2 的 3 级流水架构（mm→aivComm→reduceSum + workspace 轮转）设计思路清晰，API 验证较为完备，与 ops-nn `quant_matmul_reduce_sum` 的流水线模式有良好的对应关系。但存在 4 个需讨论级问题（flagId 冲突风险、WaitFlag 模板参数、Blaze mm per-tile 输出地址、AIV reduceSum 循环配对）需 Architect 回应后方可进入开发。

---

### 质疑清单

#### 问题 1：commFlagId 映射函数未避开 SyncAll 硬同步占用的 flagId [11-14]

- **类别**：API 可行性 / 多核策略
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §2.6 flagId 分配策略、§2.8 输入校验
- **问题描述**：
  v2 的 commFlagId 映射函数 `(tid < 3) ? tid : (tid < 5) ? tid+1 : tid+2` 只避开了 flagId 3（回压）和 5（mm 完成），但未避开 SyncAll 硬同步内部占用的 flagId [11-14]。逐项展开映射结果：
  ```
  tid=0  → commFlagId=0   ✓
  tid=1  → commFlagId=1   ✓
  tid=2  → commFlagId=2   ✓
  tid=3  → commFlagId=4   ✓ (避开 3)
  tid=4  → commFlagId=6   ✓ (避开 5)
  tid=5  → commFlagId=7   ✓
  tid=6  → commFlagId=8   ✓
  tid=7  → commFlagId=9   ✓
  tid=8  → commFlagId=10  ✓
  tid=9  → commFlagId=11  ✗ 命中 SyncAll [11-14]
  tid=10 → commFlagId=12  ✗ 命中 SyncAll [11-14]
  tid=11 → commFlagId=13  ✗ 命中 SyncAll [11-14]
  tid=12 → commFlagId=14  ✗ 命中 SyncAll [11-14]
  tid=13 → commFlagId=15  ✓
  ```
  v2 的输入校验约束为 `headTileCnt + tailTileCnt <= 14`（§2.8），这仅保证 commFlagId 不绕回（不超过 15），但当 tile 总数 ≥ 10 时 commFlagId 会命中 [11-14]。

- **审查者视角**：
  依据 asc-devkit API 文档 `CrossCoreSetFlag_ISASI.md` 约束说明及 apace skill `fusion.md` §3.3 flagId 选择规则：**SyncAll 硬同步内部占用 flagId [11-14]**，官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步。v2 每轮通信后都调用 `SyncAll<true>()`（§2.5 Phase 1），同时使用 `CrossCoreSetFlag<0x2, PIPE_MTE3>(commFlagId)` 通信就绪 flag——当 `commFlagId ∈ [11,14]` 时存在与 SyncAll 内部 flag 冲突的理论风险。虽然官网两个 PUT 算子（`all_to_all_quant_matmul`、`all_gather_quant_matmul`）也组合使用了 SyncAll + CrossCoreSetFlag 且直接用 tid 作 flagId（不避开 [11-14]），但 skill 文档明确标注这是"理论风险，移植到新平台需重新确认"。v2 额外占用了 flagId 3 和 5（用于 mm 完成/回压计数信号量），映射函数已做避让，但遗漏了 [11-14] 区间。
- **建议方案**：
  方案 A（推荐）：将输入校验收紧为 `headTileCnt + tailTileCnt <= 10`（完全避开 3、5、[11-14]），可用 flagId = {0,1,2,4,6,7,8,9,10,15} 共 10 个。
  方案 B：保持 ≤ 14 约束，但在 DESIGN.md 中显式标注"tid 9-12 的 commFlagId 命中 SyncAll [11-14]，已在 950PR 上验证无冲突"（需 Developer 在实现时做实际验证）。
  方案 C：如果 tile 总数确实需要 > 10，考虑将 mm 完成/回压通道从固定 flagId（3/5）改为 per-tile flagId，释放固定 flagId 占用——但这会大幅增加 flag 编排复杂度，不推荐。

---

#### 问题 2：CrossCoreWaitFlag 调用未指定模板参数 modeId 和 pipe，950PR 上模板参数生效可能不工作

- **类别**：API 可行性
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §1.4 伪代码、§2.5 AIC/AIV 分工设计、§2.6 flag 编排模式图
- **问题描述**：
  v2 伪代码中 AIV 侧和 AIC 侧的计数信号量 WaitFlag 调用均未指定模板参数：
  ```cpp
  // AIV 侧（§1.4 伪代码）
  CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG=5)   // 等 AIC mm 完成
  // AIC 侧（§1.4 伪代码）
  CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG=3)   // 回压
  ```
  而 CrossCoreWaitFlag 的函数原型为 `template <uint8_t modeId = 0, pipe_t pipe = PIPE_S> void CrossCoreWaitFlag(uint16_t flagId)`，默认 modeId=0、pipe=PIPE_S。
- **审查者视角**：
  依据 asc-devkit API 文档 `CrossCoreWaitFlag_ISASI.md`：
  - 950PR 上模板参数 modeId 和 pipe **生效**（阻塞指定流水的后续指令）
  - 模式 0/1/2 **不支持** PIPE_S（文档原文："不支持的流水类型为 PIPE_ALL、PIPE_S"）
  - SetFlag 侧使用了 `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(5)`（modeId=2, pipe=PIPE_FIX）和 `CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(3)`（modeId=2, pipe=PIPE_MTE2）

  如果 WaitFlag 使用默认 modeId=0、pipe=PIPE_S，与 SetFlag 的 modeId=2 不匹配（Set/Wait 的 modeId 控制同步范围，不一致可能导致计数器不配对），且 PIPE_S 在模式 0/1/2 下不被支持。

  对比 ops-nn 参考实现（`quant_matmul_reduce_sum_mixcore.h`）：ops-nn 的 `CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG)` 同样不带模板参数，但 ops-nn 运行在 910b 上（文档明确："针对 A2/A3 产品，模板参数 modeId 和 pipe **不生效**，此时 CrossCoreWaitFlag 会阻塞全部流水的后续指令"），所以 ops-nn 在 910b 上无此问题。v2 运行在 950PR 上，模板参数生效，此问题需要正视。

  v2 DESIGN.md §1.2 API 语义验证详情表中已确认"950PR 模板参数生效"，但伪代码中 WaitFlag 却使用了默认参数，存在自相矛盾。
- **建议方案**：
  在伪代码和同步点表（§2.5）中明确 WaitFlag 的模板参数，与 SetFlag 配对：
  - AIV 等 AIC mm 完成：`CrossCoreWaitFlag<SYNC_MODE2, PIPE_S>(5)` 或选择合适的 pipe（AIV 侧阻塞后续 reduceSum 指令，可用 PIPE_V 或 PIPE_MTE3）
  - AIC 等 AIV 回压：`CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(3)` 或 `CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(3)`（AIC 侧阻塞后续 mm 指令）
  注意：pipe 参数应选择能阻塞后续相关指令的流水类型。Developer 需在 950PR 上验证 modeId=2 + 选择的 pipe 是否正常工作。

---

#### 问题 3：Blaze QuantMatmulMxKernel 的 mm 输出地址 per-tile 动态修改实现方式未明确

- **类别**：伪代码可实现性
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §1.4 伪代码 `mm.cGmAddr = workspace + workSpaceOffset`、§2.5 AIC Run() 流程
- **问题描述**：
  v2 伪代码在 AIC 的 mm 循环内动态设置 mm 输出地址：
  ```cpp
  workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum)
  mm.cGmAddr = workspace + workSpaceOffset
  mm(A[Win] × B → workspace[BF16])
  ```
  但审查 v1 的 `quant_matmul_mx_kernel.h` 实际实现发现：`cGmAddr_` 在 `ResetGmAddr()` 中设置一次，`ProcessSingleBatch()` 中 `gmC` Tensor 基于 `cGmAddr_` 创建一次，per-tile 循环内通过 `gmC.Slice(mPos, nPos)` 切片——**不存在 per-tile 动态修改 cGmAddr_ 的机制**。
- **审查者视角**：
  v1 的 `ProcessSingleBatch` while 循环结构：
  ```cpp
  auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);  // 循环外创建一次
  while (bs.GetTileIdx(blockIdx)) {
      auto gmBlockC = gmC.Slice(Te::MakeCoord(mPos, nPos), ...);  // per-tile 切片
      mmadOp_(..., gmBlockC, singleShape, remoteRankCnt);
  }
  ```
  v2 需要在 while 循环内、mmad 调用前，根据 `cubeNum` 动态更新 gmC 的基地址为 `workspace + workSpaceOffset`。这不是简单的成员变量赋值（`mm.cGmAddr = ...`），而是需要：
  1. 在 while 循环内重新创建 gmC Tensor（或修改 gmBlockC 的地址指针）
  2. 同时新增 cubeNum 计数、回压 flag Wait/Set、mm 完成 flag Set 等逻辑

  对比 ops-nn 参考实现（`quant_matmul_reduce_sum_mixcore.h` `MMCompute`）：ops-nn 使用 AscendC::Matmul 的 `mm.GetTensorC(mmOutGm[workSpaceOffset], 0, sequentialWrite)` 在 `mm.Iterate()` 循环内动态指定输出地址——AscendC::Matmul 原生支持 per-tile 输出地址。但 v2 使用 Blaze BlockMmad，其输出地址通过 Tensor Slice 控制，实现方式不同。

  v2 DESIGN.md §2.15 标注 `quant_matmul_mx_kernel.h` 为 [MODIFY] 文件，说明 Architect 预期会修改此文件。但伪代码 `mm.cGmAddr = workspace + workSpaceOffset` 的写法过于简化，Developer 可能误以为只需设置成员变量即可，实际需要改造 `ProcessSingleBatch` 的循环结构。
- **建议方案**：
  1. 在 DESIGN.md §1.4 或 §2.5 中补充 `ProcessSingleBatch` 改造的具体方式说明，例如：
     ```cpp
     // while (bs.GetTileIdx(blockIdx)) 循环内，mmad 调用前：
     if (cubeNum >= PIPELINE_NUM) {
         CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG);  // 回压
     }
     uint64_t workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum);
     auto gmCWorkspace = Te::MakeTensor(
         Te::MakeMemPtr<Te::Location::GM>(workspaceGmAddr_ + workSpaceOffset * sizeof(CType)), layoutCTile);
     // mmad 输出到 gmCWorkspace（不再 Slice，因为 workspace slot 就是 baseM×baseN 的整块）
     mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmCWorkspace, singleShape, remoteRankCnt);
     CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);  // mm 完成
     cubeNum++;
     ```
  2. 明确 workspace slot 的内存布局：每个 slot 是 baseM×baseN 的连续块（无 stride），gmCWorkspace 不需要 Slice。
  3. 注意 v1 的 REMOTE 模式会遍历所有 rank 的 A×B 累加到同一块 L0C，最后一次 fixpipe 写出。v2 的 mm 输出到 workspace 时，fixpipe 只写一次（splitKNum 个 rank 累加后），workSpaceOffset 应在 fixpipe 时生效——需确认 Blaze BlockMmad 的 fixpipe 输出地址是否就是传入的 gmBlockC 地址。

---

#### 问题 4：AIV reduceSum 循环次数需与 AIC mm tile 总数严格配对，但伪代码未明确循环次数推导

- **类别**：多核策略 / 伪代码可实现性
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §1.4 伪代码 AIV Phase 2 `for each tile`、§2.5 AIV Run() Phase 2
- **问题描述**：
  v2 的 flagId 3（回压）和 5（mm 完成）作为计数信号量使用——AIC 每完成一个 mm tile SetFlag(5) 一次，AIV 每做一次 reduceSum WaitFlag(5) 一次；AIV 每做一次 reduceSum SetFlag(3) 一次，AIC 当 cubeNum ≥ 4 时 WaitFlag(3) 一次。Set/Wait 必须严格配对，否则计数器不归零会导致后续 WaitFlag 永久阻塞（死锁）。

  v2 伪代码 AIV Phase 2 写的是 `for each tile:`，但未明确这个 "each tile" 的循环次数如何推导。AIC 侧的 mm tile 总数由 `BlockScheduler::GetTileIdx()` 调度推导，AIV 侧无法直接知道 AIC 的 tile 总数。
- **审查者视角**：
  对比 ops-nn 参考实现（`quant_matmul_reduce_sum.h` `Process()`）：ops-nn 的 AIC 和 AIV 在同一个 `Process()` 循环中——`computeOp.MMCompute()` 和 `computeOp.VectorCompute()` 交替调用，两者的 tile 数由相同的 `blockDimM × blockDimN` 推导，天然配对。

  v2 的 AIV 侧不在 BlockScheduler 调度循环内（AIV 做 reduceSum 时 AIC 还在循环内做 mm）。AIV 的 reduceSum 循环次数需要 = AIC 的 mm tile 总数 × rankSize（因为每个 rank 的 mm 输出都需要 reduceSum）？还是 = AIC 的 mm tile 总数（rankSize 个 rank 的部分和已在 workspace 中，reduceSum 只需做一次 AtomicAdd 累加）？

  从 v2 的数学公式看：`C_r = Σ_{i=0}^{rankSize-1} mm(A_i × B_i)`。每个 rank 的 AIC 做 mm 写自己的 workspace，然后所有 rank 的 workspace 需要累加到 yGm。但 AllToAll PUT 是把 A 发送到各 rank，每个 rank 的 AIC 从 Win 区读所有 rank 的 A 做 mm——所以每个 rank 的 AIC 已经在 mm 中累加了所有 rank 的部分和（L0C FP32 累加 + fixpipe 写 workspace）。reduceSum 只需要把 workspace 的 BF16 结果 AtomicAdd 到 yGm。

  但 ReduceScatter 语义是"输出按 M 轴切分"——rank r 只需要 C 的第 r 个 M/rankSize 段。所以：
  - 每个 rank 的 AIC 做 mm 输出到 workspace（全量 M × N 的部分和？还是只有 my_M_seg？）

  从 v2 §1.1 数学公式：`C_r = Σ_i dequant(A_i[r*M/R:(r+1)*M/R]) × dequant(B_i)`。每个 rank 的 AIC 计算 `A_i[my_M_seg] × B_i` 的部分和，然后跨 rank 累加。

  但 v2 的 AllToAll PUT 是发送 A 的 M/rankSize 段给各 rank——每个 rank 的 Win 区收到所有 rank 的 A 的 my_M_seg 段。AIC 从 Win 读所有 rank 的 A[my_M_seg] × B → workspace。所以 workspace 中已经是累加后的结果（L0C FP32 累加所有 rank，fixpipe 写 workspace BF16）。

  那 reduceSum 做什么？从 workspace 读 BF16 → AtomicAdd 到 yGm。但 workspace 中已经是累加后的结果，为什么还需要 AtomicAdd？

  **关键疑问**：如果每个 rank 的 AIC 已经在 L0C 中累加了所有 rank 的部分和并 fixpipe 写 workspace，那 workspace 中的结果就是最终结果，reduceSum 只需要 DataCopyPad(workspace → yGm)，不需要 AtomicAdd。

  但 v2 设计用了 AtomicAdd——这意味着 workspace 中的结果不是最终结果，还需要跨 rank 累加？或者多个 AIC 核的 workspace 结果需要累加到 yGm 的同一位置？

  从 v2 §1.3 数据流看：`输出: rank r 获得 C_r[M/rankSize × N] = Σ_i reduceSum(workspace_i[my_M_seg])`。这里的 `Σ_i` 表示跨 rank 累加？但 AllToAll PUT 已经把 A 发送到各 rank，每个 rank 的 AIC 从 Win 读所有 rank 的 A 做 mm——workspace 中的结果已经是 Σ_i mm(A_i × B_i)。

  除非：v2 的 AIC mm 没有在 L0C 中累加所有 rank 的部分和，而是每个 rank 的 A 分别做 mm 写不同的 workspace slot，然后 reduceSum 跨 slot 累加。但这与 v1 的 REMOTE 模式（L0C 累加所有 rank）不同。

  或者：v2 的 AtomicAdd 是为了处理多核场景——不同 AIC 核可能计算 yGm 同一位置的部分结果，需要 AtomicAdd 累加。但 v2 的多核切分是按 M×N 轴切分，不同核计算不同的 M×N tile，不应有位置重叠。

  **这是一个设计逻辑不清的问题**——reduceSum 的 AtomicAdd 累加对象不明确。如果 workspace 中已经是最终结果，reduceSum 只需 DataCopyPad 不需要 AtomicAdd；如果需要跨核/跨 slot 累加，需要明确累加来源。
- **建议方案**：
  1. Architect 需在 DESIGN.md 中明确：reduceSum 的 AtomicAdd 累加对象是什么？（跨 rank？跨核？跨 workspace slot？）
  2. 如果 workspace 中已经是 L0C 累加后的最终结果（v1 REMOTE 模式语义），reduceSum 可简化为 `DataCopyPad(workspace → yGm)`，无需 AtomicAdd，yGm 也无需清零——这能简化设计并消除精度风险。
  3. 如果确实需要 AtomicAdd（例如 AIC mm 没有跨 rank L0C 累加，而是每个 rank 的 mm 写不同 workspace slot），需在 §1.3 数据流和 §1.4 伪代码中补充 slot 间的累加逻辑。
  4. AIV reduceSum 循环次数需在伪代码中明确推导公式（如 `= AIC mm tile 总数`），确保 flag Set/Wait 严格配对。

---

#### 问题 5：AIV 时分复用（先通信后 reduceSum）在通信轮次较多时可能导致 AIC 长时间阻塞，降低流水线效率

- **类别**：多核策略
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §2.5 AIV 时分复用设计、§2.11 核分工总结
- **问题描述**：
  v2 的 AIV 时分复用设计：每个 AIV 核先完成所有通信轮次（Phase 1，`for tid in 0..commTurn-1`），然后切换到 reduceSum（Phase 2）。AIC 在 AIV 通信期间持续做 mm 写 workspace，但当 `cubeNum >= PIPELINE_NUM=4` 时，AIC 会阻塞在 `CrossCoreWaitFlag(3)`（回压 flag），等待 AIV 完成 Phase 1 并切换到 Phase 2 执行 reduceSum 释放回压。
- **审查者视角**：
  如果 `commTurn`（通信轮次）较大（例如 10+），AIC 在完成 4 个 mm tile 后就会阻塞，等待 AIV 完成全部通信并切换到 reduceSum。这意味着 AIC 的 mm 计算能力在 AIV 通信期间被浪费（只有前 4 个 tile 能做）。

  v2 的 3 级流水架构目标是"通过 workspace 多 buffer 轮转提升通算并行度"（PLAN.md §7），但如果 AIC 在 AIV 通信期间长时间阻塞，实际通算并行度可能不如 v1 的串行模式（v1 的 AIC localMatmul 与 AIV PUT 并行）。

  这不是死锁（AIV 完成通信后会切换到 reduceSum 释放回压），但可能影响性能达标。
- **建议方案**：
  1. 在 DESIGN.md §2.5 或 §2.11 中补充流水线效率分析：评估 commTurn 与 PIPELINE_NUM 的关系对 AIC 阻塞时间的影响。
  2. 考虑替代方案：AIV 通信和 reduceSum 交替执行（每轮通信后做 1 个 tile 的 reduceSum），而不是先完成所有通信再切换。但这需要更复杂的 flag 编排（通信就绪和 mm 完成通道交替使用）。
  3. 如果性能不达标，可回退到 v1 的串行模式（localMatmul=1 + AtomicAdd 直写 yGm），v1 的开发历史数据为 143us。

---

#### 问题 6：reduceSum 的 DataCopyPad 2D 搬运参数设置未在伪代码中说明

- **类别**：伪代码可实现性
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §1.4 伪代码 `DataCopyPad(workspace→UB)` / `DataCopyPad(UB→yGm)`、§1.2 API 映射表
- **问题描述**：
  v2 伪代码 reduceSum 部分仅写了 `DataCopyPad(workspace[workSpaceOffset] → UB)` 和 `DataCopyPad(UB → yGm[outOffset])`，未说明 DataCopyExtParams 参数（blockCount、blockLen、srcStride、dstStride）的设置。
- **审查者视角**：
  reduceSum 涉及 2 次数据搬运：
  1. **workspace → UB**：workspace 中每个 slot 是 baseM×baseN 的连续块（无 stride）。如果 baseN × sizeof(bfloat16_t) 是 32 字节对齐的（baseN=128 → 256B ✓），可用 `blockCount=1, blockLen=baseM*baseN*sizeof(BF16)`。但如果 baseN 不对齐，需要按行搬运（blockCount=baseM, blockLen=baseN*sizeof(BF16), srcStride=0, dstStride=...）。
  2. **UB → yGm (AtomicAdd)**：yGm 是 C[M/rankSize × N] 的二维矩阵，每个 tile 写入 yGm 的 [mPos:mPos+baseM, nPos:nPos+baseN] 区域。需要 2D 搬运：`blockCount=baseM, blockLen=baseN*sizeof(BF16), srcStride=对齐填充, dstStride=(N-baseN)*sizeof(BF16)`。

  对比 ops-nn 参考实现（`quant_matmul_reduce_sum.h` `DataCopyPad2D` 和 `quant_matmul_reduce_sum_mixcore.h` `DataCopyOut`）：ops-nn 明确设置了 2D 搬运参数，包括 srcStride（UB 侧对齐填充）和 dstStride（GM 侧行间距）。v2 的伪代码缺少这些细节，Developer 需要自行推导。

  另外，DataCopyPad UB→GM 的 dstStride 在 950PR 上数据类型为 int64_t，取值范围 [0, 2^40-1]（依据 `DataCopyPad_UBToGM.md`），与 910b 的 uint32_t 不同，Developer 需注意类型匹配。
- **建议方案**：
  在 DESIGN.md §1.4 或 §2.5 中补充 reduceSum 的 DataCopyPad 参数设置伪代码，参考 ops-nn 的 `DataCopyPad2D` 和 `DataCopyOut` 实现：
  ```cpp
  // workspace → UB（连续块，1D 搬运）
  DataCopyExtParams cpIn{1, baseM * baseN * sizeof(BF16), 0, 0, 0};
  DataCopyPadExtParams<BF16> padIn{false, 0, 0, 0};
  DataCopyPad(reduceSumBuf, workspaceGm[workSpaceOffset], cpIn, padIn);

  // UB → yGm（2D 搬运，AtomicAdd）
  DataCopyExtParams cpOut{baseM, baseN * sizeof(BF16), 
      (AlignUp(baseN, 16) - baseN) * sizeof(BF16) / 32,  // srcStride (UB 侧)
      (N - baseN) * sizeof(BF16), 0};                      // dstStride (GM 侧)
  SetAtomicAdd<bfloat16_t>();
  DataCopyPad(yGm[outOffset], reduceSumBuf, cpOut);
  SetAtomicNone();
  ```

---

#### 问题 7：host 侧 aclrtMemset 清零 yGm 后的同步保证未说明

- **类别**：多核策略
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §1.4 要点 6、§2.8 Host 侧 yGm 清零、§2.13 精度策略
- **问题描述**：
  v2 设计用 host 侧 `aclrtMemset(deviceOutput, 0, mLocal * n * sizeof(bfloat16_t))` 清零 yGm（AtomicAdd 前提条件），但未说明 aclrtMemset 与 kernel launch 之间的同步保证。
- **审查者视角**：
  aclrtMemset 是异步 API（在 device 侧异步执行）。如果在 aclrtMemset 完成前 launch kernel，kernel 中的 AtomicAdd 可能读到未清零的 yGm 旧值，导致结果错误。

  通常情况下，aclrtMemset 和 kernel launch 在同一个 stream 上时，stream 内操作顺序执行，aclrtMemset 完成后才会执行 kernel。但 v2 是多卡场景（4 rank），每个 rank 的 host 侧独立执行 aclrtMemset 和 kernel launch。如果不同 rank 的 stream 不同步，可能出现 rank A 的 kernel 已开始执行而 rank B 的 aclrtMemset 还未完成的情况。

  对比 ops-nn 参考实现（`quant_matmul_reduce_sum_init_output.h`）：ops-nn 使用 kernel 侧 InitOutput（一个独立的 kernel task），通过 `WaitPreTaskEnd()` / `SetNextTaskStart()` 串行化 InitOutput kernel 和主计算 kernel。v2 的 PLAN.md §5 决策记录说明"直调场景用 host 侧 aclrtMemset，不用 ops-nn 的 kernel-side InitOutput（WaitPreTaskEnd/SetNextTaskStart 仅适用 SuperKernel）"——这个决策是合理的，但需补充同步保证说明。
- **建议方案**：
  在 DESIGN.md §2.8 中补充：`aclrtMemset` 与 kernel launch 必须在同一个 stream 上执行（默认行为），或显式调用 `aclrtSynchronizeStream()` 确保清零完成后再 launch kernel。多卡场景下，各 rank 的 host 侧应在 `CreateDeviceContext` 后的跨 rank barrier 之前完成 aclrtMemset（因为 barrier 之后才 launch kernel）。

---

## 审查依据汇总

| # | 依据来源 | 验证内容 |
|---|---------|---------|
| 1 | asc-devkit `CrossCoreSetFlag_ISASI.md` | 950PR 支持模式 0/1/2/4；模式 2 支持 PIPE_V/M/MTE1/MTE2/MTE3/FIX；flagId 0-15 截断低 4bit；SyncAll 占用 [11-14] |
| 2 | asc-devkit `CrossCoreWaitFlag_ISASI.md` | 950PR 模板参数生效；默认 modeId=0, pipe=PIPE_S；模式 0/1/2 不支持 PIPE_S |
| 3 | asc-devkit `SetAtomicAdd.md` | 950PR 支持 bfloat16_t；数据通路 UB/L0C→GM；执行前不自动清零 GM |
| 4 | asc-devkit `DataCopyPad_GMToUB.md` | 950PR 支持 bfloat16_t（含 mode 模板参数原型）；dstStride 950PR 为 int64_t |
| 5 | asc-devkit `DataCopyPad_UBToGM.md` | 950PR 支持 bfloat16_t（含 mode 模板参数原型）；dstStride 950PR 为 int64_t [0, 2^40-1] |
| 6 | apace skill `fusion.md` §3.3 | SyncAll 硬同步占用 flagId [11-14]；PUT 模式组合使用 SyncAll + CrossCoreSetFlag 有理论风险 |
| 7 | apace skill `fusion.md` §5.4 | localMatmul=1 需 PipeBarrier<PIPE_ALL>() 修复 MTE 异常（v2 默认 0，不涉及） |
| 8 | apace skill `communication.md` §4.2 | flagId 选择规则：Matmul 高阶 API 占用 [0,2N-1]；SyncAll 占用 [11-14] |
| 9 | ops-nn `quant_matmul_reduce_sum_mixcore.h` | PIPELINE_NUM=4；workSpaceOffset 计算；flag 编排（AIC SetFlag<SYNC_MODE2, PIPE_FIX>(5) / AIV SetFlag<SYNC_MODE2, PIPE_MTE2>(3)）；PostCompute drain |
| 10 | ops-nn `quant_matmul_reduce_sum_utils.h` | flag 常量定义：SYNC_AIV_AIC_FLAG=3, SYNC_AIC_AIV_FLAG=5, SYNC_MODE2=2 |
| 11 | ops-nn `quant_matmul_reduce_sum_init_output.h` | kernel 侧 InitOutput 实现（WaitPreTaskEnd/SetNextTaskStart 串行化） |
| 12 | v1 `apace_quant_matmul_reduce_scatter_udma_impl.h` | v1 通信编排（RunAllToAll）；localMatmul=1 + PipeBarrier；mm 输出直写 cGm |
| 13 | v1 `quant_matmul_mx_kernel.h` | v1 ProcessSingleBatch 实现；cGmAddr 循环外设置一次；REMOTE 模式遍历所有 rank L0C 累加；无 per-tile 动态输出地址机制 |

---

## Architect 回应

> **回应模式**：串讲回应
> **回应原则**：接受=更新 DESIGN.md；保留原设计=附文档依据；部分修改=更新受影响章节

### 回应概要

| 问题 | 严重程度 | 回应类别 | 是否更新 DESIGN.md |
|------|---------|---------|-------------------|
| 问题 1（flagId 映射未避开 [11-14]） | 需讨论 | **接受** | ✅ 更新 §2.6/§2.8 |
| 问题 2（WaitFlag 模板参数未指定） | 需讨论 | **接受** | ✅ 更新 §1.4/§2.5/§2.6 |
| 问题 3（Blaze mm per-tile 动态输出地址） | 需讨论 | **接受** | ✅ 更新 §1.4/§2.5 |
| 问题 4（reduceSum AtomicAdd 累加对象） | 需讨论 | **接受**（重大修正） | ✅ 更新 §1.1/§1.2/§1.3/§1.4/§2.5/§2.8/§2.13/§2.14 |
| 问题 5（AIV 时分复用效率） | 建议 | **部分修改** | ✅ 补充 §2.11 流水线效率分析 |
| 问题 6（DataCopyPad 2D 参数未说明） | 建议 | **接受** | ✅ 更新 §1.4 |
| 问题 7（aclrtMemset 同步保证） | 建议 | **自动消解**（问题 4 修正后 yGm 无需清零） | ✅ 删除 §2.8 清零逻辑 |

**统计**：接受 5 个，部分修改 1 个，自动消解 1 个，保留原设计 0 个。DESIGN.md 有更新。

---

### 问题 1 回应：接受

**回应类别**：接受（采纳方案 A）

**理由**：
apace skill `fusion.md` §3.3 和 `communication.md` §4.2 均明确说明："SyncAll 硬同步内部占用 flagId [11-14]，官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步——PUT 模式组合使用两者，当 `tid % 16 ∈ [11,14]` 时存在与 SyncAll 内部 flag 冲突的理论风险，移植到新平台需重新确认。"

v2 运行在 950PR（DAV_3510），每轮通信后都调用 `SyncAll<true>()`，同时使用 `CrossCoreSetFlag<0x2, PIPE_MTE3>(commFlagId)`。虽然官网两个 PUT 算子也组合使用 SyncAll + CrossCoreSetFlag 且不避开 [11-14]，但 skill 文档明确标注这是"理论风险，移植到新平台需重新确认"。v2 作为新算子开发，应采取保守策略。

**采纳方案 A**：将输入校验收紧为 `headTileCnt + tailTileCnt <= 10`。可用 flagId = {0,1,2,4,6,7,8,9,10,15} 共 10 个（避开 3、5、[11-14]）。

**DESIGN.md 更新**：§2.6 flagId 分配策略和 §2.8 输入校验已更新。

---

### 问题 2 回应：接受

**回应类别**：接受

**理由**：
`api-crosscore-sync.md` §4 平台生效性差异表明确说明："950（DAV_3510）上 modeId/pipe 模板参数生效；模式 0/1/2 **不支持 `PIPE_S`/`PIPE_ALL`**；`PIPE_S` 仅模式 4 支持。"

v2 运行在 950PR 上，CrossCoreWaitFlag 的默认模板参数（modeId=0, pipe=PIPE_S）违反官方约束：
1. modeId=0 与 SetFlag 的 modeId=2（SYNC_MODE2）不匹配
2. pipe=PIPE_S 在模式 0/1/2 下不被支持

ops-nn 的 `CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG)` 不带模板参数能工作，是因为 ops-nn 运行在 910b（DAV_2201）上，模板参数不生效（§4："A3/910b 上不生效——CrossCoreWaitFlag 阻塞全部流水"）。v2 运行在 950PR 上，必须显式指定模板参数。

**采纳建议方案**：
- AIV 等 AIC mm 完成：`CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(5)`（阻塞 reduceSum 的 DataCopyPad workspace→UB，MTE2 流水）
- AIC 等 AIV 回压：`CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(3)`（阻塞 mm 计算，M 流水）

参考 `api-crosscore-sync.md` §5.1 示例："AIC 侧：复用前等待回压 `CrossCoreWaitFlag<0x2, PIPE_M>(flagId); // PIPE_M = MAD/Cube 主流水"。

**DESIGN.md 更新**：§1.4 伪代码、§2.5 同步点表、§2.6 flag 编排模式图已更新。

---

### 问题 3 回应：接受

**回应类别**：接受

**理由**：
审查 v1 `quant_matmul_mx_kernel.h` 的 `ProcessSingleBatch` 实现确认：
- `cGmAddr_` 在 `ResetGmAddr()` 中设置一次（基于 `params.mmadParams.cGmAddr`）
- `gmC` Tensor 在 `ProcessSingleBatch()` 循环外创建一次：`auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);`
- per-tile 通过 `gmC.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(...))` 切片
- **不存在 `mm.cGmAddr = ...` 这样的成员变量赋值 API**

伪代码 `mm.cGmAddr = workspace + workSpaceOffset` 的写法过于简化，Developer 可能误以为只需设置成员变量。实际需要改造 `ProcessSingleBatch` 的循环结构。

ops-nn 的 `AscendC::Matmul` 用 `mm.GetTensorC(mmOutGm[workSpaceOffset], 0, sequentialWrite)` 在 `mm.Iterate()` 循环内动态指定输出地址——这是 AscendC::Matmul 原生支持的。但 v2 使用 Blaze BlockMmad，其输出地址通过 Tensor 句柄控制，实现方式不同。

**改造方式**（已写入 DESIGN.md §1.4/§2.5）：

```cpp
// ProcessSingleBatch while 循环内，mmad 调用前：
if (cubeNum >= PIPELINE_NUM) {
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(SYNC_AIV_AIC_FLAG);  // 回压
}
// ... 通信等待逻辑（与 v1 相同）...

// ★ v2 改造点：per-tile 动态创建 workspace Tensor（不再用循环外的 gmC.Slice）
uint64_t workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum);
auto layoutCTile = MakeLayoutC{}(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
auto gmCWorkspace = Te::MakeTensor(
    Te::MakeMemPtr<Te::Location::GM>(
        workspaceGmAddr_ + workSpaceOffset * sizeof(CType)),
    layoutCTile);

// mmad 输出到 gmCWorkspace（不再 Slice，workspace slot 就是 baseM×baseN 的整块）
mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias,
        gmCWorkspace, singleShape, remoteRankCnt);

CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);  // mm 完成
cubeNum++;
```

**关键改造点**：
1. `workspaceGmAddr_` 作为新成员变量，在 `ResetGmAddr()` 中从 `params.mmadParams.cGmAddr` 绑定（host 侧传入 workspace 地址）
2. while 循环内 per-tile 创建 `gmCWorkspace` Tensor（基于 workspace 基址 + workSpaceOffset）
3. `gmCWorkspace` 不需要 Slice（workspace slot 就是 baseM×baseN 的连续块，无 stride）
4. A/B 的切片逻辑不变（仍基于 mPos/nPos 从 gmA/gmB Slice）
5. Blaze `Te::MakeTensor` 是常规函数，可在循环内任意位置调用，无 API 约束

**workspace slot 内存布局**：每个 slot 是 `baseM × baseN × sizeof(bfloat16_t)` 的连续块，无行间距 stride。`workSpaceOffset` 以元素个数为单位（非字节），`Te::MakeMemPtr` 构造时乘以 `sizeof(CType)` 转为字节偏移。

**DESIGN.md 更新**：§1.4 伪代码和 §2.5 AIC Run() 流程已补充改造说明。

---

### 问题 4 回应：接受（重大设计修正）

**回应类别**：接受

**这是本次串讲最重要的发现。** 经详细分析 v1 代码和 ops-nn 参考实现，确认 v2 原设计的 AtomicAdd 是冗余的。

**分析过程**：

1. **v1 REMOTE 模式的 L0C 累加语义**（v1 `quant_matmul_mx_kernel.h` `ProcessSingleBatch` REMOTE 分支）：
   ```cpp
   for (uint64_t rank = 0; rank < rankSize; rank++) {
       // 从 Win 区读各 rank 的 A，切片
       auto gmBlockA = gmA.Slice(Te::MakeCoord(actualMPos, kPos), ...);
       // ...
       mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC,
           singleShape, remoteRankCnt);
       remoteRankCnt++;
   }
   ```
   `mmadOp_` 的最后一个参数 `remoteRankCnt` 控制 L0C 累加：`0` = reset，递增 = 累加，计满 `splitKNum` 触发单次 fixpipe。v1 `SetupParams` 中 `localMatmul==0` 时 `splitKNum = rankSize`（=4），所以 REMOTE 模式遍历 4 个 rank，L0C FP32 累加 4 次，最后一次触发 fixpipe 写 gmBlockC。

2. **v2 的 AIC mm 组件与 v1 完全相同**（DESIGN.md §2.2："Blaze 组件不变，仅 mm 输出地址从 yGm 改为 workspace"）。v2 默认 `localMatmul=0`，所以 v2 的 AIC mm 同样在 L0C 中跨 rank 累加所有 4 个 rank 的 A×B 部分和，最后一次 fixpipe 写 workspace。

3. **workspace 中已经是最终结果**：`workspace[slot] = Σ_{i=0}^{rankSize-1} mm(A_i[my_M_seg] × B_i)`，这是 L0C FP32 全精度累加后截断为 BF16 的结果。

4. **v2 多核切分无位置重叠**：不同 AIC 核计算不同的 M×N tile，写到 workspace 的不同 slot（`workSpaceOffset = baseM * baseN * (coreIdx + (cubeNum % PIPELINE_NUM) * coreNum)`）。reduceSum 从 workspace 读 → 写 yGm 的 `[mPos:mPos+baseM, nPos:nPos+baseN]` 区域，不同 AIV 核写 yGm 不同位置，无重叠。

**结论**：v2 的 reduceSum **不需要 AtomicAdd**，只需 `DataCopyPad(workspace → yGm)`。原设计中的 AtomicAdd 是基于错误假设——误以为 workspace 中是 per-rank 的部分和，需要 reduceSum 跨 rank 累加。实际上 AIC mm 已经在 L0C 中完成了跨 rank 累加。

**设计修正**：
1. reduceSum 简化为 `DataCopyPad(workspace→UB) + DataCopyPad(UB→yGm)`，去掉 `SetAtomicAdd/SetAtomicNone`
2. yGm 不需要 host 侧 `aclrtMemset` 清零（AtomicAdd 前提条件不再适用）→ **问题 7 自动消解**
3. 消除 AtomicAdd 的 BF16 read-modify-write 精度风险
4. 简化 flag 编排（mm 完成 + 回压通道仍需保留，但 yGm 清零同步保证不再需要）
5. AIV reduceSum 循环次数 = AIC mm tile 总数（通过 tilingData 传入），Set/Wait 严格配对

**localMatmul 模式调整**：
- `localMatmul=0`（默认）：REMOTE-only，L0C 跨 rank 累加 + 单次 fixpipe → workspace，reduceSum = DataCopyPad
- `localMatmul=1`（可选）：改为映射 `MatmulMode::DEFERRED_SYNC`（per-tile 内 local A×B → L0C reset → wait_flag → remote A×B → L0C 累加 → 单次 fixpipe → workspace），reduceSum 仍 = DataCopyPad
- **不再支持 v1 的 localMatmul=1（LOCAL+REMOTE+AtomicAdd to yGm）**，因为 v2 的 mm 输出到 workspace 而非 yGm，AtomicAdd to yGm 的语义不适用

**文档依据**：
- apace skill `fusion.md` §5.1 三模式定义表：mode 0（REMOTE-only）"L0C 累加 → 单次 fixpipe → C"，不需要 AtomicAdd
- apace skill `fusion.md` §5.3 L0C 容量约束："所有模式下，单个 tile 内各 rank 的部分和都累加进同一块 L0C FP32 累加器"
- v1 `quant_matmul_mx_kernel.h` REMOTE 分支：for rank 循环 + mmadOp_(..., remoteRankCnt) L0C 累加
- v1 `apace_quant_matmul_reduce_scatter_udma_impl.h` `SetupParams`：`localMatmul==0` → `splitKNum = rankSize`

**DESIGN.md 更新**：§1.1 数学公式、§1.2 API 映射（去掉 SetAtomicAdd）、§1.3 数据流、§1.4 伪代码、§2.5 同步点表、§2.8 Host 侧（去掉 yGm 清零）、§2.13 精度策略、§2.14 差异对比表均已更新。

---

### 问题 5 回应：部分修改

**回应类别**：部分修改（保留时分复用设计 + 补充效率分析）

**理由**：
时分复用设计（先通信后 reduceSum）是 v2 3 级流水架构的核心特征，简化了 flag 编排（避免了通信和 reduceSum 交替执行时的 flag 通道复用复杂性）。AIC 在 AIV 通信期间可完成 PIPELINE_NUM=4 个 mm tile，之后阻塞等待 AIV 切换到 reduceSum 释放回压。

**流水线效率分析**（已补充到 DESIGN.md §2.11）：
- AIC 阻塞条件：`cubeNum >= PIPELINE_NUM=4` 时 WaitFlag(3) 回压
- AIC 阻塞持续时间 = AIV 完成剩余通信轮次的时间 + AIV 切换到 reduceSum 的时间
- 最坏情况：commTurn 较大时，AIC 在前 4 个 tile 后阻塞，直到 AIV 完成全部 commTurn 轮通信
- 缓解因素：AIV 通信每轮耗时较短（UDMA PUT + SyncAll），且 AIC mm 单 tile 耗时通常 ≥ 通信单轮耗时
- 回退路径：如果性能不达标，可回退到 v1 串行模式（localMatmul=1 + AtomicAdd 直写 yGm，开发历史数据 143us）

**保留时分复用设计的理由**：
1. 简化 flag 编排——通信就绪通道用 per-tile flagId，mm 完成/回压通道用计数信号量，三通道独立
2. 如果改为交替执行（每轮通信后做 1 个 tile reduceSum），需要通信就绪和 mm 完成通道交替使用同一 flagId，大幅增加编排复杂度
3. v2 的目标是验证 3 级流水架构的可行性，性能不达标可回退 v1

**DESIGN.md 更新**：§2.11 已补充流水线效率分析。

---

### 问题 6 回应：接受

**回应类别**：接受

**理由**：
reduceSum 的 DataCopyPad 参数确实需要在伪代码中明确，Developer 不应自行推导。结合问题 4 的修正（reduceSum 不需要 AtomicAdd），DataCopyPad 参数设置如下（已写入 DESIGN.md §1.4）：

```cpp
// workspace → UB（连续块，1D 搬运）
// workspace slot 是 baseM×baseN 的连续块，无 stride
DataCopyExtParams cpIn{1, baseM * baseN * sizeof(BF16), 0, 0, 0};
DataCopyPadExtParams<bfloat16_t> padIn{false, 0, 0, 0};
DataCopyPad(reduceSumBuf, workspaceGm[workSpaceOffset], cpIn, padIn);

// UB → yGm（2D 搬运，无 AtomicAdd）
// yGm 是 C[M/rankSize × N] 的二维矩阵，每个 tile 写入 [mPos:mPos+baseM, nPos:nPos+baseN]
DataCopyExtParams cpOut{
    baseM,                                          // blockCount = baseM（按行搬运）
    baseN * sizeof(bfloat16_t),                     // blockLen = 每行字节数
    (AlignUp(baseN, 16) - baseN) * sizeof(bfloat16_t) / 32,  // srcStride (UB 侧对齐填充)
    (N - baseN) * sizeof(bfloat16_t),               // dstStride (GM 侧行间距)
    0
};
DataCopyPad(yGm[outOffset], reduceSumBuf, cpOut);
// 无 SetAtomicAdd/SetAtomicNone（问题 4 修正后不需要）
```

参考 ops-nn `quant_matmul_reduce_sum.h` `DataCopyPad2D` 和 `quant_matmul_reduce_sum_mixcore.h` `DataCopyOut` 的参数设置。注意 950PR 上 `dstStride` 类型为 `int64_t`（依据 `DataCopyPad_UBToGM.md`）。

**DESIGN.md 更新**：§1.4 伪代码已补充 DataCopyPad 参数。

---

### 问题 7 回应：自动消解

**回应类别**：自动消解（问题 4 修正后 yGm 无需清零）

**理由**：
问题 4 的修正（reduceSum 不需要 AtomicAdd）直接消解了本问题。yGm 不再需要 `aclrtMemset` 清零，因为：
1. reduceSum 使用普通 `DataCopyPad(UB → yGm)`（非 AtomicAdd），直接覆盖 yGm 旧值
2. 多核切分无位置重叠，不同 AIV 核写 yGm 不同区域
3. 无需 `aclrtMemset` 与 kernel launch 的同步保证

**DESIGN.md 更新**：§2.8 Host 侧已删除 yGm 清零逻辑，§2.13 精度策略已更新（去掉 yGm 清零前提条件）。

---

### 回应总结

本次串讲共回应 7 个问题：
- **接受 5 个**（问题 1、2、3、4、6）：均已更新 DESIGN.md 对应章节
- **部分修改 1 个**（问题 5）：补充流水线效率分析，保留时分复用设计
- **自动消解 1 个**（问题 7）：问题 4 修正后 yGm 无需清零
- **保留原设计 0 个**

**最重要的修正**：问题 4 的 AtomicAdd 冗余发现。v2 的 AIC mm（Blaze QuantMatmul REMOTE 模式）已在 L0C 中跨 rank FP32 累加，workspace 中是最终结果，reduceSum 只需 DataCopyPad 不需 AtomicAdd。这一修正：
1. 简化了 reduceSum 实现（去掉 SetAtomicAdd/SetAtomicNone）
2. 消除了 yGm 清零需求（去掉 aclrtMemset 及其同步保证）
3. 消除了 AtomicAdd 的 BF16 read-modify-write 精度风险
4. 简化了 flag 编排（mm 完成 + 回压通道不变，但 yGm 清零同步不再需要）

DESIGN.md 已根据以上回应更新。
