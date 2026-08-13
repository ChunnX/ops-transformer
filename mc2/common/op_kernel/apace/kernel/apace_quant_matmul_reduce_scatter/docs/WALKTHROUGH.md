# apace_quant_matmul_reduce_scatter_v3 设计串讲质疑清单

> **审查者**：Design Reviewer（独立审查）
> **审查对象**：`docs/DESIGN.md`（v3 = v2 + 取消 workspace 轮转回压）
> **审查方法**：API 文档原文核实（`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/`）、Blaze 源码逐行核对（`v2/third_party/ops-tensor/include/blaze/gemm/block/block_scheduler_qbmm.h`）、v2 kernel 代码行号比对（`v2/kernel/apace_quant_matmul_reduce_scatter_v2/`）

---

## 设计串讲

### 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）—— **2 项需讨论，6 项建议，无阻塞**
- [ ] 设计存在严重问题，无法开发

**总体评价**：v3 的核心论证链（§2 上界公式、§4.3 flagId=5 计数器风险、§4.5/4.6 无死锁/无越界）经独立复核**全部成立**：官方文档确认计数器 0-15 上限属实且每 flagId 独立计数；Blaze 源码四项证据行号与语义属实；v2 变更点行号引用（C1-C5/C9）全部吻合；槽位寻址去取模后 AIC/AIV 两侧公式一致；删除回压后无新增 WAR 竞争（槽位独占 + 通道 2 per-tile 先写后读保证充分）。2 项需讨论问题均不影响架构正确性，但影响测试计划可达性与用户约束符合性，需 Architect 回应。

---

### 质疑清单

#### 问题 1：T11/T12 重点用例的可行性依赖未经验证的 usedCoreNum=32 假设

- **类别**：多核策略 / 内存规划
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §3.1-3.2（估算表注「coreNum = min(32, blockNums)」）、§4.3（覆盖性评估）
- **问题描述**：§3.2 估算表和 §4.3「覆盖全部目标 shape」的结论都假设 SWAT tiling 对 M=16384 大 shape 会选择 `usedCoreNum=32`。但 `usedCoreNum` 由 QuantMatmulTilingSwat 运行时推导，无设计级保证。反例：T12（M=16384, N=4096, blockNums=256）若 SWAT 选择 `usedCoreNum ≤ 17`，则 `T = CeilDiv(256, 17) = 16 > 15`，host 校验（C8）将**拒绝 launch 这个用于验证回压消除收益的核心用例**，v3 的价值验证落空。注：950PR PCIE 28 核场景 `CeilDiv(256, 28) = 10` 仍安全，风险仅在 usedCoreNum ≤ 17 时触发，概率不高但无证据排除。
- **审查者视角**：host 校验本身是正确设计（宁可拒绝不可出错），但设计的测试计划把「T12 精度+性能双验证」作为核心验收项，其可达性建立在一个未验证的 tiling 行为假设上。开发到 Step 6 才发现用例被拒将造成返工。
- **建议方案**：开发早期（框架搭建阶段）先用 v2 工程对 T11/T12 shape 打印 SWAT tiling 日志确认 `usedCoreNum` 实际值；若 < 32，在 PLAN 中预设备选 shape 或调整 baseM/baseN 候选使 T ≤ 15 成立。同时建议在 §3.2 明确标注「表中 coreNum=32 为假设值，以运行时 tiling 日志为准」的验证责任归属。

#### 问题 2：大 shape workspace 37.7MB 超出用户「~18MB 可接受」口径

- **类别**：内存规划
- **严重程度**：需讨论
- **设计文档位置**：DESIGN.md §3.2（估算表末两行 18.9MB / 37.7MB，结论「符合用户『~18MB 可接受』约束」）
- **问题描述**：用户约束原文为「大 shape workspace 增大到 ~18MB 可接受」。估算表中 128×256 tiling 配置为 18.9MB（吻合），但 256×256 配置下 M=16384,N=2048 为 21.0MB、N=4096 为 **37.7MB**——后者已达用户口径的 2 倍。DESIGN §3.2 结论将 19~38MB 整体判定为「符合约束」，属于对用户约束的扩大解释。
- **审查者视角**：用户约束同时声明「不考虑 buffer 复用（workspace 足够大）」，可能 38MB 确实可接受，但这是用户的决定而非 Architect 的推断。审查者不能代替用户放行 2 倍偏差。
- **建议方案**：向用户确认 38MB 量级是否可接受；或在 DESIGN 中补充约束派生规则（如「workspace ≤ 40MB 硬上限，超出时 host 报错」），使用户约束可执行、可测试。

#### 问题 3：maxTilesPerCore 上界公式可证明更紧（+1 非必需），且 host 校验边界过保守 1 档

- **类别**：内存规划 / API 可行性
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §2.1-2.2（公式 `CeilDiv(blockNums, coreNum) + 1`）、§4.3（校验 `maxTilesPerCore ≤ 15`）
- **问题描述**：独立推导 Blaze 源码可得**严格上界为 `T ≤ CeilDiv(blockNums_, coreNums_)`，不需要 +1**：
  - 初始 `roundNums_ = CeilDiv(blockNums_, coreNums_)`（L63），`blockIdx_ > endBlockIdx_` 的核 `-1`（L64-66）；
  - `UpdateTailTile` 的 `+1` 条件为 `blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx`（L92）——**而 `roundNums_ = CeilDiv` 的核恰是 `blockIdx_ ≤ endBlockIdx_` 的核，永远不满足 +1 条件**；`blockIdx_ > endBlockIdx_` 的核从 `CeilDiv-1` 起跳，+1 后至多回到 `CeilDiv`；
  - 因此 max(roundNums_) = CeilDiv，与是否尾块拆分无关。
  
  推论：(a) 设计公式 `CeilDiv+1` 宽松 1 槽，每核多分配 `baseM×baseN×2B`（32 核 256×256 时约 4MB），方向安全但非 tight；(b) host 校验 `maxTilesPerCore ≤ 15` 实际门控的是 `CeilDiv ≤ 14`，即合法计数器峰值 15（T=15）的 shape 会被误拒——校验边界比硬件能力保守 1 档。
- **审查者视角**：宽松方向不引入正确性风险，且 §2.2 可选加固段落表明 Architect 有意保留余量对抗 Blaze 未来行为变化，工程上可接受。但 §2.2 自称「可证明安全」的论证链（证据 3「影响至多为 +1」→ 结论 `CeilDiv+1`）掩盖了更紧的真实上界，论证精度有提升空间；校验边界差 1 档会使 T=15 的边界 shape 失去本可合法运行的机会。
- **建议方案**：二选一——(a) 保持公式与校验不变，在 §2.2 补注「真实上界为 CeilDiv，+1 为主动保留的防御余量」；(b) 公式收紧为 `CeilDiv`、校验收紧为 `CeilDiv ≤ 15`（等价 `maxTilesPerCore ≤ 16`），同时在 T13 增补 T=15 恰好放行的边界用例。倾向 (a)，改动最小。

#### 问题 4：SyncAll flagId 占用引用不精确（3510 下 `SyncAll<true>` 实际仅占 AIV flagId 14）

- **类别**：API 可行性
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §4.1（「SyncAll [11-14] 硬同步内部占用」）、§4.4（C9「只需避开 5 和 [11-14]」）
- **问题描述**：经核实官方文档 `SyncAll.md` 表 3（NPU 架构版本 3510）：`isAIVOnly=true` 时占用**仅 AIV flagId 14**；`isAIVOnly=false + __mix__(1,1)` 才占 AIC 11/12/13 + AIV 12/13。v2/v3 代码实际调用为 `SyncAll<true>()`（v2 udma_impl.h L299，AIV-only 路径），故真实占用只有 flagId 14，DESIGN 的「[11-14]」沿用了更保守的历史口径。
- **审查者视角**：保守方向零风险——v2 的 CommFlagIdV2 映射和 C9 的可选简化映射都避开 [11-14]，功能完全正确。但 DESIGN 将「避开 [11-14]」作为事实陈述而非保守选择，且 C9 的「可用集 11 个」推理若基于真实占用（仅 14 + 5）还可再多 3 个。文档准确性问题，不影响实现。
- **建议方案**：在 §4.1/4.4 将表述修正为「SyncAll<true> 在 3510 架构下实际占用 AIV flagId 14（SyncAll.md 表 3）；本设计保守避开 [11-14]」，并注明若未来改用 `SyncAll<false>` 需重新评估 AIC 侧 11/12/13 冲突。

#### 问题 5：C6 字段替换的 plumbing 清理范围未列全，存在半迁移风险

- **类别**：伪代码可实现性
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §1.1 C6（仅列 `tiling_data.h`）、§7（文件清单）
- **问题描述**：`pipelineNum` 在 v2 中的完整 plumbing 链为：`tiling_data.h` 字段 → `main.cpp` 赋值（L201）→ `InitBaseParams`（udma_impl.h L421）→ `baseParams_.pipelineNum`（L197）→ `SetupParams` → `LocalParams.pipelineNum`（kernel L136）→ kernel 成员 `pipelineNum_`（L193）→ 消费点（kernel L419、udma_impl L363，即 C2/C4）。C6 只声明替换 `tiling_data.h` 字段，未声明中间各环节（`LocalParams.pipelineNum`、`baseParams_.pipelineNum`、kernel 成员 `pipelineNum_`、两处赋值代码）是删除还是保留为死字段。C2/C4 去取模后这些字段已无消费者。
- **审查者视角**：保留死字段可编译通过，但「kernel 槽位寻址不消费该字段」（C6 语）与字段仍存在于 LocalParams 之间会产生误导——后续维护者可能误以为寻址仍受 pipelineNum 约束。v3 定位是「外科手术式优化」，变更清单的完整性是其核心价值。
- **建议方案**：在 C6 或 §7 明确处置决策：建议全链删除（tiling_data 字段、main.cpp 赋值、baseParams_、LocalParams、kernel 成员、SetupParams 赋值），或显式标注「保留字段但标记 deprecated/未使用」。

#### 问题 6：AIC 零形状早退跳过通道 1 兜底 drain，残留计数跨 launch 生命周期未论证

- **类别**：多核策略
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §1.2（通道 1「含 readyTileIdx 兜底 drain，完全不变」）、§4.5（无死锁论证）
- **问题描述**：v2 kernel L397-398：尾块拆分轮中 `GetBlockShape` 返回零形状时 AIC 直接 `return`，**跳过 L501-504 的通道 1 兜底 drain 循环**。此时 AIV Phase 1 已无条件 Set 全部 commFlagId，未被 Wait 的 fid 计数器在 kernel 退出时残留为 1。v3 保持该行为不变（§1.2）。若计数器状态跨 kernel launch 持久，则下一次 launch 同 fid 的 Wait 会被残留计数提前解除，破坏通道 1 的通信就绪时序（AIC 可能在本次通信完成前读 Win 区）。
- **审查者视角**：此为 **v2 已有行为**（非 v3 回归），且 v2 12/12 精度 PASS 表明在实际 ST 场景（每用例独立进程/launch）未触发问题——大概率硬件在每次 kernel launch 时初始化调度模块计数器。但 DESIGN §4.5 的无死锁/配平论证以「每个 fid Set 1 次 / Wait ≤1 次（去重 + 兜底 drain 配平）」为前提，未覆盖早退跳过 drain 的分支，论证存在小缺口。T14（尾块拆分用例）恰好会走这条路径。
- **建议方案**：在 §4.5 补充说明：早退分支下通道 1 残留计数 ≤ commTurn ≤ 10（不超 0-15 上限，无溢出风险）；残留计数是否跨 launch 持久以官方行为为准——v2 实证未暴露问题，v3 不引入变化，记录为已知行为即可。无需代码改动。

#### 问题 7：C8 校验的执行顺序与取值来源未明确规定

- **类别**：伪代码可实现性
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §1.1 C7/C8、§8 T13
- **问题描述**：两处实现细节未规定：
  (a) **校验顺序**：T13 构造的 shape（M=32768, N=8192）`maxTilesPerCore=33`，若 host 先按公式 `aclrtMalloc` workspace（33×32×256×256×2B ≈ 138MB）再校验，大 shape 下分配本身可能失败或浪费，报错信息将偏离「maxTilesPerCore 超限」的本意；
  (b) **取值来源**：`maxTilesPerCore` 必须基于 SWAT tiling **最终输出**（`tilingData.tileQbmmTilingData.{m, n, baseM, baseN, usedCoreNum}`，v2 main.cpp L196-198 已有此先例），而非 raw `M/4` 或 baseM 候选默认值 256——SWAT 可能对 m/n 做对齐 padding 或细化 baseM/baseN，用错来源会导致校验与 kernel 实际行为脱节。
- **审查者视角**：v2 main.cpp 已从 tilingData 取数计算 workspace，顺延实现 C7/C8 没有障碍，属于「设计未写明但显然如此」的缺口。但 T13 是验收用例，校验顺序直接决定其报错行为是否符合预期（「不 launch、报错信息清晰」）。
- **建议方案**：在 C8 补充两句：校验在 SWAT tiling 完成之后、workspace `aclrtMalloc` 之前执行；`maxTilesPerCore` 的四个输入全部取自 `tileQbmmTilingData` 最终值。

#### 问题 8：CANN 路径在 environment.md 与 DESIGN/PLAN 之间不一致

- **类别**：伪代码可实现性（工程一致性）
- **严重程度**：建议
- **设计文档位置**：DESIGN.md §0.1（`/home/l00947526/Ascend806/cann`）、PLAN.md §6.1（同左） vs environment.md（`/home/l00947526/Ascend806/cann-9.2.0`）
- **问题描述**：environment.md 记录的 ASCEND_HOME_PATH 为 `cann-9.2.0`，DESIGN/PLAN 要求 source 的 `set_env.sh` 路径为 `cann`。若两者非同一目录（软链接），编译/运行环境可能与环境检查结论不符。
- **审查者视角**：大概率是软链接或笔误，但环境路径是 Step 1 门禁的产物，应与设计文档严格一致。
- **建议方案**：Developer 框架搭建阶段核实 `cann` 是否为 `cann-9.2.0` 的软链接；若是，统一文档写法；若不是，以 environment.md 为准修正 DESIGN/PLAN。

---

### 已核实通过项（无需修改，供串讲确认）

| # | 核实项 | 证据 |
|---|--------|------|
| V1 | flagId 计数器 0-15 上限**属实** | `CrossCoreWaitFlag_ISASI.md` 约束说明原文：「flagId对应的计数器计数范围为0-15。如果计数器的值超出该范围，则会异常报错，中断流程」；「每个flagId都对应一个计数器」（独立计数，§4.2 前提成立） |
| V2 | 模式 2 在 950PR 生效且支持 `__mix__(1,1)` / KERNEL_TYPE_MIX_AIC_1_1 | `CrossCoreSetFlag_ISASI.md` 表 3 +「950PR 模板参数 modeId/pipe 生效」（WaitFlag 文档） |
| V3 | Blaze 源码四项证据行号与语义全部属实 | 构造函数 L51-66（roundNums_ ≤ CeilDiv）、UpdateTailTile L85-101（+1 唯一增量）、GetTileIdx L190-235（迭代数 = roundNums_）、零形状早退 L168-170；另核实 `UpdateNextBatchBlockRoundParams`（L176-188）本算子不触及、kernel 侧 `needUpdateTail_` 保护单次调用（v2 kernel L375-380） |
| V4 | v2 变更点行号引用全部吻合 | C1: kernel L404-406（设计写 403-405，含注释行，实质一致）；C2: L418-419；C3: 调用 L217 + 本体 L507-519；C4: udma_impl L362-363；C5: L399-401；C9: CommFlagIdV2 L59-66 |
| V5 | 通道 2 配对平衡（含早退分支） | AIC `return` 早退在 Set(5) 之前（L397-398 vs L496-497），AIV `break` 在 Wait(5) 之前（L350-352 vs L359），两侧同一确定性调度器序列 → Set/Wait 次数严格相等，kernel 退出时 flagId=5 计数器归 0 |
| V6 | 删除 PostCompute 语义完整 | v2 PostCompute 仅 drain 残留 flag 3（L507-519）；v3 删除全部 Set(3) 后无残留可 drain；其附带的「AIC 等 AIV 消费完」收尾屏障非正确性必需（kernel 完成以全部核退出为准，host 侧同步等待） |
| V7 | 槽位寻址去取模后两侧一致 + 无越界 | AIC/AIV 公式逐字一致（baseM×baseN×(coreIdx + cubeNum×coreNum)），coreIdx=GetBlockIdx() 两侧配对（MIX_AIC_1_1）；最大写偏移 coreNum×T×baseM×baseN×2B < workspaceSize 成立（即使按问题 3 的更紧上界） |
| V8 | 无新增 WAR/精度风险 | 槽位独占 → 每槽位写 1 次读 1 次；通道 2（Set PIPE_FIX → Wait PIPE_MTE2）保证先写后读且 fixpipe 数据对 AIV MTE2 可见；AIC 写 slot k 与 AIV 读 slot j (j≠k) 地址不相交；数值路径零变更 |
| V9 | Blaze 组件不占 CrossCore flagId | `block_mmad_qbmm_mx.h` 仅用核内 `SetFlag<HardEvent::*>`；third_party 中使用 CrossCore 的 kernel（qbmm_streamk/qgmm_swiglu_mx）非本算子组件；亦无 Matmul 高阶 API 占用 [0, 2N-1] 的问题 |
| V10 | host 侧 maxTilesPerCore 可实现 | v2 main.cpp L196-198 已有 `usedCoreNum/baseM/baseN` 取值先例，`m/n` 同在 tileQbmmTilingData；blockDim=usedCoreNum（<<<>>> 启动参数）与 scheduler `GetBlockNum()` 一致 |
| V11 | flagId=5 峰值上界推导独立复核通过 | 模式 2 AIC→AIV 方向每次 Set(5) 使 AIV 侧计数器 +1；AIC 每 tile Set 恰好 1 次，全程至多 T 次；峰值 ≤ T ≤ CeilDiv ≤ 14（在校验 maxTilesPerCore ≤ 15 下），距 15 上限有 ≥1 余量，**校验充分** |
| V12 | 无新死锁环 | AIC 唯一等待源=通道 1，AIV Phase 1 不依赖 AIC 任何状态且全核执行 SyncAll<true>；AIV Phase 2 唯一等待源=通道 2 且配对平衡；跨卡 TeamBarrier/AllToAll 逻辑未动 |

### 审查方法附录

- **API 文档**：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/{CrossCoreSetFlag_ISASI,CrossCoreWaitFlag_ISASI,SyncAll}.md`（全文通读，含约束说明与 flagId 占用表）
- **Blaze 源码**：`operators/apace_quant_matmul_reduce_scatter_v2/third_party/ops-tensor/include/blaze/gemm/block/block_scheduler_qbmm.h`（335 行全文）、`block_mmad_qbmm_mx.h`（flag 使用检索）
- **v2 代码**：`quant_matmul_mx_kernel.h`（522 行全文）、`apace_quant_matmul_reduce_scatter_v2_udma_impl.h`（495 行全文）、`st/.../main.cpp`（workspace/tiling 取值检索）

---

### Architect 回应

> **回应者**：Architect（串讲回应模式）
> **总体处理**：2 项需讨论问题中，问题 1 接受（补充验证责任与备选预案）、问题 2 接受收紧重算但**数值仍超用户口径，提交用户确认**；6 项建议全部接受。DESIGN.md 已同步更新（§0.1/§1.1 C6·C8/§2.1-2.2/§3.1-3.2/§4.1/§4.2/§4.3/§4.4/§4.5/§5/§6/§7/§8/§9）。

#### 问题 1（usedCoreNum=32 假设）— 接受

- 认可质疑：T11/T12 的可达性确实建立在未验证的 tiling 行为假设上，且审查者指出的反例（T12 若 `usedCoreNum ≤ 17` 则 `CeilDiv(256,17)=16 > 15` 被 C8 拒绝）成立。
- 处理（已落入 DESIGN.md）：
  1. §3.2 表头与表下加注「**coreNum=32 为假设值，以运行时 SWAT tiling 日志为准**」；
  2. §8 新增 **T0 前置用例**：框架搭建阶段先用 v2 工程对 T11/T12 shape 打印 SWAT tiling 日志确认实际 `usedCoreNum`；
  3. 预设备选方案：若实测 `usedCoreNum ≤ 17` 导致 T12 被拒，调整 shape 或 baseM/baseN 候选使 `CeilDiv ≤ 15` 成立后再执行 T12，避免开发到 Step 6 才返工。
- 补充说明：采纳问题 3 的收紧上界后，T12 的放行条件从「`CeilDiv ≤ 14`」放宽为「`CeilDiv ≤ 15`」（即 usedCoreNum ≥ 18 即可），被误拒的窗口收窄了 1 档。

#### 问题 2（workspace 37.7MB 超用户 ~18MB 口径）— 接受收紧重算，数值提交用户确认

- 已按问题 3 证明的严格上界 `T = CeilDiv(blockNums, coreNum)`（无 +1）重算 §3.2 全部 shape：

  | shape (M,K,N) | tiling 配置 | T=CeilDiv | 收紧前（+1） | **收紧后** |
  |---|---|---|---|---|
  | 512,1024,512 | 128×256 | 1 | 0.26 MB | 0.13 MB |
  | 1120,2048,1024 | 256×256 | 1 | 2.1 MB | 1.0 MB |
  | 2048,4096,2048 | 256×256 | 1 | 4.2 MB | 2.1 MB |
  | 16384,4096,2048 | 256×256 | 4 | 21.0 MB | **16.8 MB** ✅ |
  | 16384,4096,2048 | 128×256 | 8 | 18.9 MB | **16.8 MB** ✅ |
  | 16384,4096,4096 | 256×256 | 8 | 37.7 MB | **32.0 MB** ⚠️ |

- 结论：收紧后 M=16384,N=2048 两种配置均降至 16.8MB（符合 ~18MB 口径）；但 **M=16384,N=4096 最大配置仍为 32.0MB，超出用户口径约 1.8 倍（>25MB）**，Architect 不代替用户放行。
- **提交 CANNBot 向用户确认**：M=16384,N=4096 shape 的 32.0MB workspace 是否可接受；若不可接受，备选方案为对该 shape 启用 §2.2 之外的约束（如限定该 shape 走 128×256 之外的更低 blockNums 配置，或回退 v2 轮转方案）。
- DESIGN.md §3.2 结论、§8 T12 备注已更新为上述数值，并显式标注「32.0MB 待用户确认」，不再做「整体符合约束」的扩大解释。

#### 问题 3（上界公式可证明更紧）— 接受（采纳方案 (b) 的公式收紧 + 方案 (a) 的论证补注）

- 认可审查者的独立推导：`roundNums_ = CeilDiv` 的核（`blockIdx_ ≤ endBlockIdx_`）永不满足 `UpdateTailTile` 的 +1 条件，满足 +1 条件的核从 `CeilDiv-1` 起跳至多回到 CeilDiv，**严格上界为 CeilDiv，无需 +1**。
- 处理（已落入 DESIGN.md）：
  1. §2.1 公式收紧为 `maxTilesPerCore = CeilDiv(blockNums, coreNum)`；
  2. §2.2 证据 3 补充「+1 核与 CeilDiv 核互斥」的关键点 1，「综上」改为严格上界表述；保留「可选加固：host 按 +1 分配」的防御性注释（默认不启用）；
  3. §4.3 校验语义同步收紧：`maxTilesPerCore ≤ 15` 现在门控 `CeilDiv ≤ 15`，**T=15 边界 shape 可合法放行**，消除原校验保守 1 档的误拒；
  4. §8 T13 增补「`CeilDiv = 15` 恰好放行」的边界用例；
  5. §3.2 按收紧公式重算（见问题 2 回应表），每核节省 1 槽（32 核 256×256 时约 4MB）。
- 选择 (b) 而非审查者倾向的 (a) 的理由：公式收紧直接服务问题 2 的用户约束符合性（37.7→32.0MB），且严格上界已由双方独立推导交叉验证，无正确性风险；防御余量以注释形式保留。

#### 问题 4（SyncAll flagId 占用引用不精确）— 接受

- 认可核实结果：`SyncAll.md` 表 3（3510 架构）`isAIVOnly=true` 仅占 AIV flagId 14；v2/v3 实际调用 `SyncAll<true>()`（v2 udma_impl.h L299），真实占用只有 14。
- 处理（已落入 DESIGN.md）：§4.1 表格、§4.2 flagId 空间独立条目、§4.4 开头均修正为「实际占用 AIV flagId 14（SyncAll.md 表 3 已核实）；本设计**保守避开 [11-14]**」，并注明「若未来改用 `SyncAll<false>`（`__mix__(1,1)`）需重新评估 AIC 侧 11/12/13 冲突」。保守方向维持不变，仅修正事实陈述。

#### 问题 5（C6 plumbing 清理范围未列全）— 接受

- 认可质疑：变更清单完整性是 v3「外科手术式优化」的核心价值，半迁移会误导后续维护者。
- 处理（已落入 DESIGN.md）：§1.1 C6 明确决策为**全链删除**（非保留死字段），完整列出链路：`tiling_data.h` 字段 → `main.cpp` 赋值（v2 L201）→ `InitBaseParams`（udma_impl.h L421）→ `baseParams_.pipelineNum`（L197）→ `SetupParams` → `LocalParams.pipelineNum`（kernel L136）→ kernel 成员 `pipelineNum_`（L193）及两处赋值代码；§7 文件清单 tiling_data.h 条目同步标注全链清理。

#### 问题 6（AIC 零形状早退跳过通道 1 兜底 drain）— 接受（补充论证，无代码改动）

- 认可质疑：§4.5 的配平论证前提未覆盖早退分支，存在论证缺口；同时确认此为 **v2 已有行为、非 v3 回归**。
- 处理（已落入 DESIGN.md §4.5 新增条目）：
  1. 早退分支下通道 1 残留计数 ≤ commTurn ≤ 10，不超 0-15 上限，**无溢出风险**；
  2. 残留计数是否跨 launch 持久以硬件行为为准——v2 12/12 精度 PASS（每用例独立进程/launch）实证未暴露问题，大概率硬件每次 kernel launch 初始化调度模块计数器；
  3. v3 保持行为完全不变，记录为**已知行为**，无需代码改动；§8 T14 标注其覆盖该早退分支。

#### 问题 7（C8 校验顺序与取值来源）— 接受

- 认可质疑：两点均属「设计未写明但显然如此」的缺口，且 T13 是验收用例，校验顺序直接决定其报错行为。
- 处理（已落入 DESIGN.md §1.1 C8）：
  1. **执行顺序**：校验在 SWAT tiling 完成之后、workspace `aclrtMalloc` **之前**执行（避免大 shape 先分配 ~138MB 再拒绝，报错偏离本意）；
  2. **取值来源**：`maxTilesPerCore` 的四个输入（m、n、baseM、baseN、usedCoreNum）全部取自 SWAT tiling 最终输出 `tilingData.tileQbmmTilingData`（v2 main.cpp L196-198 已有取数先例），不使用 raw `M/4` 或 baseM 候选默认值 256。

#### 问题 8（CANN 路径不一致）— 接受，已核实为软链接

- 已实际核实：`/home/l00947526/Ascend806/cann` 是指向 `/home/l00947526/Ascend806/cann-9.2.0` 的**软链接**（`readlink -f` 解析一致），二者等价，无环境不一致风险。
- 处理（已落入 DESIGN.md §0.1）：CANN 路径统一写为 `cann-9.2.0`，并注明 `cann` 为已核实软链接；建议 PLAN.md 同步统一（Developer 框架搭建阶段执行）。

#### 回应统计

| 类别 | 数量 | 问题编号 |
|------|------|----------|
| 接受（文档/设计已更新） | 7 | 1、3、4、5、6、7、8 |
| 接受重算 + 提交用户确认 | 1 | 2（32.0MB > ~18MB 口径，待用户裁决） |
| 保留原设计 | 0 | — |

**遗留事项**：
1. 【用户确认】M=16384,N=4096 workspace 32.0MB 是否可接受（问题 2）；
2. 【开发期验证】T0 前置用例确认 T11/T12 实际 usedCoreNum（问题 1）；
3. 【PLAN 同步】CANN 路径统一写法（问题 8，软链接已核实，仅文档一致性）。

## 设计串讲仲裁

**仲裁日期**：2026-08-12
**仲裁人**：CANNBot（提交用户裁决）

### 遗留事项 1：workspace 32.0MB 超出口径（问题 2）

- **裁决**：✅ **用户接受 32.0MB，但标记为遗留风险**
- **裁决依据**：用户明确选择「接受但标记遗留」——M=16384,N=4096 大 shape workspace 32.0MB 可用，后续版本再优化 workspace 复用
- **执行要求**：PLAN.md 遗留事项中记录该风险；后续版本（local 后置/fragmentTensor 优化版）需重新评估 workspace 复用策略

### 遗留事项 2/3
- T0 前置用例验证 usedCoreNum：由 Developer 在开发早期执行
- CANN 路径写法同步：由 Developer 在实现时统一
