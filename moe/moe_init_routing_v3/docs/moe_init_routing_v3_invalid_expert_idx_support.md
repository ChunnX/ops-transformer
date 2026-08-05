# MoeInitRoutingV3 支持 expertIdx = -1（无效专家）特性说明

## 1. 背景

### 1.1 需求来源

MoeInitRoutingV3 算子的 `expertIdx` 输入中可能出现 -1，表示无效专家（即该位置不参与路由）。算子需要正确处理含 -1 的输入，确保 -1 被过滤，不影响有效专家的排序、gather 和统计结果。

### 1.2 -1 的语义

-1 表示无效专家，语义上等同于"该位置不参与路由"。根据 README 公式：

$$availableIdxNum = |\{x \in expertIdx \mid expert\_start \le x < expert\_end\}|$$

-1 不在 `[expertStart, expertEnd)` 范围内，应被过滤。对算子各输出的影响：

- **expandedRowIdxOut**：无效位置（`i >= availableIdxNum`）在 rowIdxType=0（GATHER）时由 -1 填充，rowIdxType=1（SCATTER）时未初始化。
- **expandedXOut**：无效位置不参与 gather，对应行数据不写入（保持初始值）。
- **expertTokensCountOrCumsumOut**：仅统计有效专家的 token 数量，-1 不计入。
- **expandedScaleOut**：与 expandedXOut 一致，无效位置不写入。

### 1.3 适用范围

本特性仅涉及 arch35（Ascend 950PR/Ascend 950DT）实现，不影响 A2/A3 路径。

## 2. 需要适配的点

### 2.1 排序阶段（已具备，无需修改）

arch35 的 `FilterExpertIdx`（`moe_v3_full_load_base.h`、`moe_v3_sort_one_core.h`、`moe_v3_sort_multi_core.h`）**无条件**将 `< expertStart` 的元素 mask 为 `MIN_FP32`，经 `Sort<float, true>` 降序排序后排到队列末尾，不依赖 `ep_` 标志。排序阶段已天然支持 -1。

### 2.2 SIMT 函数缺少下界守卫（需修复）

4 个 SIMT 函数以 `n*k`（含无效元素）为上界遍历排序后的 expertIdx，但仅有 `>= expertEnd → break` 守卫，缺少 `< expertStart` 守卫。排序后末尾的无效元素经 `Cast(float → int32)` 溢出，结果不可假设为固定值，若溢出为负值会触发越界写。

**修复**：在 `>= expertEnd` break 之前加 `< expertStart → continue`。

| 文件 | 函数 | 修复内容 |
|------|------|---------|
| `arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertFirstIndexSimt` | 加 `if (currExpertId < expertStart) { continue; }` |
| `arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertCountOutSimt` | 同上 |
| `arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertFirstIndexSimt` | 同上 |
| `arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertCountOutSimt` | 同上 |

注：`FullLoadComputeActualIdxNumSimt` 已有双向守卫，无需修改。

### 2.3 CopyOutRowIdx 无效位置未填充 -1（需修复）

`MoeV3FullLoadBase::CopyOutRowIdx` 的非 epFullload、非 SCATTER 分支直接拷贝 `expandedRowIdx`，但 `ComputeGatherIdx` 的第二次 sort 产生的 `expandedRowIdx` 对无效位置（`i >= actualExpertIdxNum_`）可能包含脏数据。

**修复**：在 DataCopyPad 之前，遍历 `[actualExpertIdxNum_, totalLength_)` 将对应位置显式置 -1，确保输出符合 README 约定的"rowIdxType 为 0 时无效数据由 -1 填充"。

### 2.4 Gather 阶段（已具备，无需修改）

- **ScatterCopy 路径**：遍历上界为 `expertTotalCount`（有效 token 数），不触及无效元素。
- **GatherCopy 路径**：遍历 `n*k`，但有 `outIndex < 0` / `hasValidOut` 守卫跳过无效位置。
- **FullLoad GatherOutX**：有 `outIndex >= 0 && outIndex < outputRows` 守卫。
- **DropPad 路径**：`MoeV3RowIdxGatherDropPad` 有 `expertIdx < 0` 守卫，`MoeV3GatherOutDropPad` 有 `srcRow < 0` / `outPosition < 0` 守卫。

## 3. 对性能分支判断的影响

arch35 的 `ComputeUseGatherCopy` 根据 `(expertEnd - expertStart) * k > expertNum * 3` 在 tiling 侧决定 GatherCopy 或 ScatterCopy。该条件等价于 `ratio × k > 3`（`ratio` 为有效专家占比），即**期望每行有效 token 数 > 3 时选 GatherCopy**。其前提假设是 top-k 专家在专家区间内均匀分布，此时每行有效 token 数的期望为 `k × ratio`。

GatherCopy 相比 ScatterCopy 的核心收益是**避免同一源行 x 的重复搬入**：ScatterCopy 每个有效 token 独立搬入一次 x，同一源行的多个 token 会重复搬入；GatherCopy 每个源行只搬入一次 x，输出给该行的所有有效 token。当每行有效 token 数 > 3 时，GatherCopy 每行至少节省 2 次重复搬入，收益足以覆盖其固定开销（额外的 RowIdxGather + SyncAll）。全局节省量为 `n × (每行节省次数) × cols`，n 越大绝对收益越大。

引入 -1 后，该判断条件是否仍然准确，分两种 -1 分布场景分析：

### 3.1 -1 均匀分散分布

每个 token 的 k 个专家中 -1 与其他专家等概率出现。此时实际每行有效 token 数的期望为 `k × ratio × (expertNum / (expertNum + 1))`，与 tiling 侧估计的 `k × ratio` 偏差很小（expertNum 较大时约 0.4%）。该偏差不会导致判断阈值 3 附近的误判，**判断条件基本准确，无需调整。**

### 3.2 -1 整行出现（某些 token 的 k 个专家全是 -1）

等价于有效行数减少（n 变小），但每行的有效 token 数不变（仍为 `k × ratio`），判断条件 `ratio × k > 3` 不含 n，不受影响。且 n 变小时 GatherCopy 的全局节省量 `n × (每行节省次数) × cols` 相应减小，但每行收益不变，阈值判断仍然合理。同时 GatherCopy 的 `hasValidOut` 机制天然跳过无效行不搬 x，自动适应有效行减少。**判断条件仍然正确。**

### 3.3 结论

当前 tiling 侧的 `useGatherCopy` 判断逻辑对 -1 场景是鲁棒的，**无需迁移到 kernel 侧**。

## 4. 性能影响与验收标准

### 4.1 性能影响

- **不含 -1 的场景**：4 个 SIMT 函数仅增加一次标量比较（continue 分支不执行），`CopyOutRowIdx` 新增循环不执行（`actualExpertIdxNum_ == totalLength_`），**性能无感知影响**。
- **含 -1 的场景**：各阶段处理量随有效 token 比例下降而减少，整体耗时不会比不含 -1 的相同 shape 更长。

### 4.2 性能验收标准

| 维度 | 用例设计 | 验收标准 |
|------|---------|---------|
| **回归无劣化** | 不含 -1 的原有用例（FullLoad unquantized/dynamic/static + 非 FullLoad） | 与修改前基线对比，偏差 < 3% |
| **-1 递进测试** | 固定 shape，invalid_ratio 从 0%→10%→30%→50%→90% 递进 | 耗时随有效 token 比例下降而下降，无突变 |
| **GatherCopy/ScatterCopy 切换点** | `ratio × k ≈ 3` 附近的 shape | 切换后性能无突变 |
| **全 -1 边界** | 所有 expertIdx = -1 | 不崩溃，输出全 -1，耗时最短 |

## 5. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `op_kernel/arch35/moe_v3_full_load_base.h` | 2 个 SIMT 函数加 `< expertStart` continue + `CopyOutRowIdx` else 分支无效位置置 -1 |
| `op_kernel/arch35/moe_v3_expert_tokens_count.h` | 2 个 SIMT 函数加 `< expertStart` continue |
| `tests/ut/op_kernel/moe_init_routing_v3_data/gen_data.py` | 支持可选第 15 参数 `invalid_ratio` 生成 -1 expertIdx |
| `tests/ut/op_kernel/moe_init_routing_v3_data/gen_tiling.py` | 添加 `case_invalid_1/2/3` tiling case |
| `tests/ut/op_kernel/test_moe_init_routing_v3.cpp` | 添加 3 个含 -1 expertIdx 的 UT 用例 |

## 6. UT 用例说明

| 用例名 | 模板 | activeExpertRange | expertTokensNumType | invalid_ratio | 覆盖点 |
|--------|------|-------------------|---------------------|---------------|--------|
| `test_case_invalid_expert_idx_1` | FullLoad Unquantized | `[873,889]`（子集） | KEY_VALUE | 0.3 | epFullload=1 路径 + SIMT 函数 |
| `test_case_invalid_expert_idx_2` | FullLoad Unquantized | `[0,734]`（全范围） | KEY_VALUE | 0.3 | epFullload=0 路径 + CopyOutRowIdx else 分支 |
| `test_case_invalid_expert_idx_3` | FullLoad Unquantized | `[873,889]`（子集） | COUNT | 0.3 | COUNT 模式 + ExpertTokensCount SIMT 函数 |

golden 计算（`gen_data.py`）已正确处理 -1：将 `expertIdx < expert_start` 的元素设为 `INT32_MAX`，排序后排到末尾，不参与有效输出。
