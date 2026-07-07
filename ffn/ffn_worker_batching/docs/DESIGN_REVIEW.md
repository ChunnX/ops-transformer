# FfnWorkerBatching — 方案评审报告（1.3R，新增 Ascend950/A5 支持）

**状态**: ✅通过

**条款总数**: 12 | 通过: 12 | 发现问题(HIGH): 0 | 需关注(MED): 4 | 提示(LOW): 2

**评审对象**: `ffn/ffn_worker_batching/docs/DESIGN.md`（441 行，v1.0）
**真值源**: `docs/spec.yaml`（9-stage PASS）+ `docs/SPEC_REVIEW.md`（✅通过，spec 冲突数=0）
**评审范围**: 为已有 A2/A3 算子新增 Ascend950(arch35/DAV_3510) 全新代次路径，对外支持范围对齐 A2
**方法论**: ascendc-ops-architect 场景四（C1-C6 强制规则 + 7 维度条款级评审 + API 逐参数文档演练）

**环境说明（对 DESIGN §6 前提的修正）**: DESIGN §6 称「`ASC_DEVKIT_DIR` 未设置、在线文档检索不可用」。实测 `ASC_DEVKIT_DIR` 确为空，但 devkit API 文档树在 `/root/.cannbot/repo/.../asc-devkit/docs/api/` 落盘可用，样板 `moe_init_routing_v3/op_kernel/arch35/` 亦存在。故本评审**未接受** DESIGN §6 的降级前提，对全部关键 API 做了官方文档逐参数演练（见下表）。这不构成设计缺陷（开发期 devkit 就绪后仍需按 §6 checklist 复核 tmpBuffer/DCCI），仅澄清评审证据强于设计文档所述。

---

## API 演练记录（C2：每处关键 API 官方文档佐证 + 逐参数推导）

| API | 文档路径 | 已读配图 | 关键参数推导 | 结论 |
|-----|----------|----------|--------------|------|
| `Sort<float,true>` | `SIMD-API/高阶API/排序操作/Sort.md` | 方式一 float/half proposal 布局图（8B/元素，index 高 4B/score 低 4B）| 原型 `(dst, concat, index, tmp, repeatTime)`；950 每迭代排 32 元素，repeatTime∈[0,255]；index 固定 uint32；起始地址 32B 对齐；相同 score 按输入序稳定。DESIGN §3.4 `(dstProposal,srcValue,srcIndex,tmpBuffer,repeatTimes=sortNum/32)` ✔ | ✅ 950 支持，签名/稳定性一致 |
| `Concat` | `SIMD-API/高阶API/排序操作/Concat.md` | 无内嵌图（复用 Sort 布局） | `(concat,src,tmp,repeatTime)`，每迭代 16 元素，tmp 由 GetConcatTmpSize 求。DESIGN §3.4 `(dst,srcValue,tmpBuffer,repeatTimes)` ✔ | ✅ 950 支持 |
| `Extract` | `SIMD-API/高阶API/排序操作/Extract.md` | 无内嵌图 | `(dstValue,dstIndex,sorted,repeatTime)`，950 每迭代 64 float/128 half；dstIndex 固定 uint32。DESIGN §3.4 `(dstValue,dstIndex,srcProposal,repeatTimes)` ✔ | ✅ 950 支持 |
| `MrgSort<float,true>` | `SIMD-API/高阶API/排序操作/MrgSort-93.md` | 方式一 float/half 8B 队列布局图（2 张）| 原型 `MrgSort<T,isExhaustedSuspension>(dst, sortList, elementCountList[4], sortedNum[4], validBit, repeatTime)`；validBit 0b11/0b111/0b1111；elementCount∈[0,4095]。禁 MrgSort4 与 DESIGN 一致 | ✅ 950 支持；⚠ **签名标注漂移见问题清单 DESIGN-API-3** |
| `GetSortOffset`/`GetSortLen` | `SIMD-API/高阶API/排序操作/GetSortOffset.md` | 无 | `GetSortOffset<T>(elemOffset)` 返回**字节**偏移。DESIGN §3.4 写「`(n)→n×2`（float 数）」——语义等价（float proposal 8B=2 float/元素）但单位口径为字节 | ✅ 950 通用；LOW 单位口径提示 |
| `DataCopyPad`+`DataCopyExtParams`/`DataCopyPadExtParams<T>` | `SIMD-API/.../GMToUB非对齐数据搬运(DataCopyPad).md` | datacopypad1/2/3.png + paddingMode.png（4 张，含负 stride 重复搬运 + Compact 模式）| blockCount∈[0,4095]、blockLen∈[0,2097151]B（须 sizeof(T) 整数倍）；**950 专属**：srcStride int64∈[-blockLen,2^40-1]（=-blockLen 重复搬同块）、dstStride int64∈[0,65535]、Compact 模式尾部统一补齐。支持 int8/half/bf16/int32/float。DESIGN §3.4 blockLen 单位 byte + 非对齐必用 ✔ | ✅ 950 支持；直接支撑 P2 int8+32B scale 连续 gather |
| `Duplicate` | `SIMD-API/.../数据填充/Duplicate.md` | 无 | `(dst,scalar,count)`；int32 支持；950 上 count=0 不保证 NOP（需注意）。DESIGN §3.4 group_list 清零 [0,0] ✔ | ✅ 950 支持 |
| `SetAtomicAdd<int32_t>` | `SIMD-API/基础API/原子操作/SetAtomicAdd.md` | 无 | 950 支持 int8/int16/half/bf16/int32/float；通路 UB/L0C→GM；**执行前不自动清零 GM**（须先 Duplicate 清零，DESIGN §3.4 已含）；建议 DisableDmaAtomic 收尾。DESIGN §3.4 整数精确累加保确定性 ✔ | ✅ 950 支持，确定性成立 |
| `GetSortTmpSize`/`GetConcatTmpSize` | `SIMD-API/高阶API/排序操作/GetSortTmpSize.md` | 无 | `(platform, elemCount, dataTypeSize)`。DESIGN §3.4 `(platform,tileSize,4)` ✔（dataTypeSize=4=float）| ✅ 签名一致；DESIGN 已诚实标 ⚠ 开发期复核，合理 |
| `GetCoreMemSize`/`GetCoreNumAiv`/`GetLibApiWorkSpaceSize` | `Utils-API/平台信息获取/PlatformAscendC/*.md` | 无 | `GetCoreMemSize(CoreMemType::UB, size)` 返回**运行时字节数**；示例即 tiling 期取 UB。支撑 DESIGN §3.6.3「禁写死 UB/核数」✔ | ✅ 支撑无 arch 常量硬编码 |
| `ArithProgression<int32_t>` | 未在本 devkit 快照定位到 .md | — | 引用自样板 `moe_v3_sort_one_core.h:48`（已部署 950）；DESIGN §3.4 `(dst,start,stride,count)` | ⚠ 文档变体未定位；样板实证 950 可用，非阻塞（LOW） |
| `SyncAll` | 通用同步原语 | — | 本算子 arch22 + 样板通用；step 间栅栏 | ✅ 通用 |

> C2/C3 判定：全部主路线 API 均以官方文档逐参数演练佐证，且已 Read 关键内嵌配图（Sort proposal 布局、MrgSort 队列布局、DataCopyPad 4 图含负 stride/Compact）。仅 `ArithProgression` 文档变体在本快照未定位（LOW，样板实证）。

---

## 12 条 DESIGN-* 条款评审

| 条款 ID | 维度 | 状态 | 证据（DESIGN 位置）| 结论 |
|---------|------|------|------|------|
| DESIGN-ALGO-1 | 算法 | ✅ | §1.3 三步语义 / §3.5 数据流 | 与 spec `math_semantics` 三步（sort→gather→group_listing）逐条一致；y[out_i,:]=token_data[gather_idx[out_i],:] 逐字节；无浮点算术 |
| DESIGN-ALGO-2 | 算法边界 | ✅ | §3.5（mask token 置极大值排末尾）/ §3.6.3 硬校验 | k=0 全 mask、same_bin、越界报错均显式承接；确定性稳定排序 + 固定核间分工 |
| DESIGN-TIL-1 | Tiling 切分 | ✅ | §3.6.1 方案 A / §3.6.3 平台运行时取值 | 双代隔离方案 A sound（见「符号隔离专项」）；核数/UB 运行时取值。**MED**：`MAX_EXPERT_NUM=5120`/`GROUP_LISTING_*` 常量分类未彻底（问题清单 #3）|
| DESIGN-TIL-2 | UB 预算 | ✅ | §3.7.2 UB 分配表 + §3.6.3 阈值表 + UB 容量验证公式 | 预算表存在且以运行时 `≤GetCoreMemSize(UB)` 断言（C4 未触发：非缺失、非超限）。H 无上界 → 符号化 + loop 切分是设计期正确形态。**MED**：缺代表性 shape 数值实例 + 缺 sort 步/RECV(101) 分 TilingKey 显式列表（问题清单 #2）|
| DESIGN-TIL-3 | TilingKey 覆盖 | ✅ | §3.6.3 SetBatchingTilingKey / §3.5 | 100(NORM)/101(RECV) 与 process 分支一一对应，无重叠；token_dtype 正交 kernel 内处理；102 有意不分发（与 spec 一致，见专项 #5）|
| DESIGN-API-1 | API 平台可用 | ✅ | §3.4 / §6 API 表 | 全部主路线 API 官方文档确认 950PR/DT 支持（见 API 演练记录）|
| DESIGN-API-2 | API dtype/约束 | ✅ | §3.4 各行约束列 | int8/half/bf16/int32/float 字节搬运 + float proposal 排序均文档确认；32B 对齐/blockLen∈sizeof(T)倍/禁 MrgSort4 均已标 |
| DESIGN-API-3 | API 参数签名 | ✅ | §3.4 MrgSort 行 / GetSortOffset 行 | 核心签名一致。**MED**：MrgSort 第 6 运行时参数标为 `exhaustion`（实为 `repeatTime`；exhaustion 是模板参 `isExhaustedSuspension`），`validBitTail` 实为 `validBit`（问题清单 #1）；**LOW**：GetSortOffset 单位口径（问题清单 #5）|
| DESIGN-BRANCH-1 | 分支覆盖 | ✅ | §2.3 运行视图 / §3.5 NORM+RECV | NORM/RECV 双路径 + 单/多核 sort + RECV BsK 变体全覆盖；核间 gather/group_listing 分工描述完整 |
| DESIGN-REQ-1 | 需求承接 | ✅ | DESIGN §4 映射 / REQUIREMENTS §11 追溯 | C5 未触发：REQUIREMENTS §4.2-4.6 每条 shape/dtype/维度/精度均有承接路径；无缺项 |
| DESIGN-SPEC-1 | spec 一致性 | ✅ | §4「spec.yaml 一致性映射」章节 | 含专章 + 5 易误用字段全 1:1（见下「DESIGN-SPEC-1 专项」）；无从 REQUIREMENTS 重解释覆盖 spec |
| DESIGN-PERF-1 | 性能 | ✅ | §7.1 | 双缓冲 TQueBind MTE2/V/MTE3 流水 + 多核 + arch35 大 UB 放大批量有论证；性能为可选验收（REQUIREMENTS §7 无量化指标）|

---

## DESIGN-SPEC-1 专项：spec.yaml 一致性映射 + 5 易误用字段核查

**「spec.yaml 一致性映射」章节存在性**：✅ DESIGN §4 为强制专章，逐字段列出 spec 取值 → 设计落点，并声明「以 spec.yaml 为唯一真值源，禁止从 REQUIREMENTS 正文重新解释」。

**5 最易误用字段 1:1 核查**（spec.yaml 原文 ↔ DESIGN 承接）：

| # | spec-owned 字段 | spec.yaml 取值 | DESIGN 承接 | 判定 |
|---|---|---|---|---|
| 1 | `dtype_policy.supported_combinations` | P0 `token_payload:float16→y:float16`；P1 `bfloat16→bfloat16`；P2 `int8→int8`；其余输出恒定（group_list int64 / 4×id int32 / dynamic_scale float32 / actual_token_num int64）| §3.3 三 profile 表逐行 + §4 映射表行 1 逐字复述；y dtype 由 infershape 依 token_dtype 设置；A5 不增删 | ✅ 一一对应 |
| 2 | `outputs[].shape_rule` | `y=(A*BS*topK+1, H)`；`group_list=(expert_num,2)`；4×id + dynamic_scale=`(Y,)`；`actual_token_num=(1,)`（numpy_expr）| §4 映射表行 2 + §1.3/§3.7；Y=A*BS*K；静态 infershape 共用不改 | ✅ 一一对应 |
| 3 | `numerical_tolerance.per_dtype` | fp16/bf16/int8/int32/int64/fp32 全 `{rtol:0,atol:0,bitwise_equal}`| §4 映射表行 3 + §1.3 精度口径「二进制一致」+ §3.1「原样字节 gather 无算术直接满足」| ✅ 一一对应 |
| 4 | `boundary_conditions[]` | k=0/k=N(A=1024,topK+1=64)/same_bin(8192)/expert_num>8192 报错/topK+1>64 报错/A>1024 报错/长度≠4 报错/dtype≠int8 报错/非 1 维报错/need_schedule 0→100/1→101 | §4 映射表行 4 + §3.6.3 硬校验阈值(EXPERT_IDX_MAX/MAX_SESSION_NUM/MAX_K_NUM)+ §3.5 两路径 TilingKey | ✅ 一一对应（10+ 条边界全覆盖）|
| 5 | `extreme_inputs[]` | all_same_index(8192)/token_dtype=2(尾部 32B FP32 scale 连续 gather)/normal_sparse(~50% mask,need_schedule=1)| §4 映射表行 5 + §3.3 P2(BLOCK_BYTES=32) + §3.5 mask 排末尾；穿刺见 PLAN 迭代一/三 | ✅ 一一对应 |

**其余 spec-owned 字段**：`op.category`/`paradigms`(§1.1/§3.5)、`platform_constraints.supported_chips`(§1.1)、`math_semantics`(§1.3)、`determinism`(§3.4/§3.5)、`broadcast:none`(§3.1)、`numerical_stability:false`(§3.1) 均在 §4 映射表承接，无缺项。

**冲突检查结论（流程终止信号判定）**：
- SPEC_REVIEW.md 已判 spec ↔ REQUIREMENTS 在 spec-owned 字段冲突数 = **0**。
- 本评审复核 DESIGN §4 未从 REQUIREMENTS 正文重新解释覆盖任何 spec 字段（REQUIREMENTS §4.4/§4.5/§4.6 与 spec dtype_policy/boundary/shape_rule 一致）。
- **未发现 REQUIREMENTS/spec 在 spec-owned 字段冲突 → 非流程终止信号。DESIGN-SPEC-1 = ✅通过。**

---

## 符号隔离专项（DESIGN §5 + 主 Agent 5 设计要点核实）

四层隔离逐层评审，对照主 Agent 已读码确认的 5 要点：

**要点 1 — tiling 双代隔离方案 A（sound 性 + 链接期符号冲突 + 最小改动）**：✅ 站得住
- 事实前提**经设计诚实修正**：现有 arch22 tiling 是 monolithic `IMPL_OP_OPTILING(FfnWorkerBatching).Tiling(Tiling4FfnWorkerBatching)`（§3.6.1），非 2000 档模板注册，仓内无现成模板可并列。此修正与题述「与 2000 档隔离」的假设偏差被 DESIGN 明确记录，属正确的读码校正。
- **单 IMPL_OP_OPTILING 约束下不产生链接期符号冲突**：方案 A 不新增第二个 `IMPL_OP_OPTILING`（那会导致重复符号），而是在唯一分派器 `Tiling4FfnWorkerBatching` 顶部加 `IsRegbaseSocVersion` 守卫 → ascend950 转 `TilingRegistry::DoTilingImpl`；arch35 类经 `REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, ...RegbaseTiling, 1000)` 追加进注册表（与 IMPL_OP_OPTILING 正交的机制），`IsCapable()=IsRegbaseSocVersion`。全程 IMPL_OP_OPTILING 符号唯一，注册表为增量登记 → **无链接期冲突，论证成立**。
- **最小改动**：单一 tiling 入口约束下，唯一触碰 arch22 的点是分派器 3 行守卫，arch22 tiling 类体零改。对比方案 B（arch22 全模板化，须重构继承 TilingBaseClass、违反「A2/A3 逻辑不动」）——方案 A 确为最小改动，取舍论证充分。样板 moe_init_routing_v3_tiling_arch35.cpp:181-184 的 IsCapable=IsRegbaseSocVersion 佐证有效。

**要点 2 — arch35 主路线走经典 SIMD/MemBase（非 RegBase）**：✅ 决策站得住
- 依据架构师方法论「路线决策」：DAV_3510 上仅 **vector 类**算子默认 RegBase；FfnWorkerBatching 为 index_gather/搬移重排类（非 vector 算术），按规则默认走通用 SIMD/MemBase。
- 实证链闭合：样板同族原语(sort/gather/count)在 DAV_3510 主体为经典 SIMD（已 Read moe arch35 目录确认 moe_v3_sort_*/mrgsort/gather_out/expert_tokens_count 存在）；RegBase 仅现于 quant/cast 密集变体（moe_v3_gather_dynamic_quant/*_fp8_*/*_mxfp* 等，目录实证存在）；gather 为 bitwise_equal 原样字节搬运无寄存器算术，经典 DataCopyPad 即满足。RegBase 作 P2 可选优化且「不改对外语义」——决策 sound。

**要点 3 — _apt.cpp 同名 extern "C" 符号唯一性论证**：✅ 正确
- 机制：`def.cpp` `ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")` 只对 ascend950 config 生效 → ascend950 编译单元只取 `_apt.cpp`、A2/A3 只取 `ffn_worker_batching.cpp`。同名 `extern "C" ffn_worker_batching(11 参数)` 在**各 soc 各自的编译产物**中唯一，不同时进入同一链接。对齐 moe_gating_top_k_apt/moe_init_routing_v3_apt 实证。论证正确。

**要点 4 — UB/核数运行时取值彻底性**：⚠ 基本彻底，一处未闭合（见问题清单 #3）
- §3.6.3 平台接口取值 + §3.7 workspace/UB 全符号化推导，`GetCoreMemSize(UB)`/`GetCoreNumAiv`/`GetLibApiWorkSpaceSize` 文档确认返回运行时值，248/192KB 仅作设计推演不入代码——主体彻底。
- 未彻底点：kernel 侧 `MAX_EXPERT_NUM=5120`（REQUIREMENTS §4.6 明示「片上暂存约束」）与 `GROUP_LISTING_MULTI_CORE_LENGTH`/`GROUP_LISTING_*_AIV_NUM` 常量，DESIGN §3.6.3 表格自己也指出「若含 UB 字节假设须改运行时推导」但**未给出该判定的结论**（是纯并行度阈值可保留，还是含 UB 假设须改）。属需在开发期闭环的分类缺口，非阻塞。

**要点 5 — TilingKey 102(RECV_GATHER) 不移植 + SortMaskBase 伴随件取舍**：✅ 与 spec/A2 一致
- spec/REQUIREMENTS §4.5 明示 102 为「已实现未启用保留分支，tiling 从不 SetTilingKey(102)」。DESIGN §1.4/§3.2/§3.6.3 一致：不产生 102 分发；`ffn_wb_scan_get_valid_experts.h`(#4, `KernelScanGetValidExperts:SortMaskBase`) 作为 SortMaskBase 继承族伴随件随 #4 头一起移植（保持类层次完整，因 #6-#12 sort kernel 均继承 SortMaskBase），但不生成 102 路径。取舍与 spec「102 不作对齐必需项」及 A2 行为一致，论证清晰。

---

## 问题清单

| # | 条款 ID | 严重度 | 证据(DESIGN位置) | 文档依据 | 修复建议 |
|---|---------|--------|------------------|----------|----------|
| 1 | DESIGN-API-3 | MED | §3.4 MrgSort 行「`(dst,...,sortedNums,validBitTail,exhaustion)`」；§6 API 表同 | `MrgSort-93.md` 原型 `MrgSort<T,isExhaustedSuspension>(dst,sortList,elementCountList[4],sortedNum[4],validBit,repeatTime)` | 将运行时第 6 参数标注由 `exhaustion` 改为 `repeatTime`；`exhaustion` 归位为模板参 `isExhaustedSuspension`；`validBitTail`→`validBit`。仅文档标注订正，不影响方案可行性 |
| 2 | DESIGN-TIL-2 | MED | §3.7.2 UB 分配表（符号公式）| 场景四核心原则 §5「每 TilingKey 输入+输出+中间 UB 显式列表」| 补一组代表性 shape（如 P0 fp16 单核 / P2 int8 / 大 shape 多核）的 UB 占用**数值实例**，并对 sort 步 tmpBuffer(GetSortTmpSize) 与 RECV(101) gather 分别给占用行，佐证 ≤ 运行时 UB。当前 H 无上界故符号化合理，此为证据补强非阻塞 |
| 3 | DESIGN-TIL-1 / 要点4 | MED | §3.6.3 阈值表「若含 UB 字节假设须改运行时推导」（未给结论）；REQUIREMENTS §4.6 `MAX_EXPERT_NUM=5120` | REQUIREMENTS §4.6 片上暂存约束 | 明确 `MAX_EXPERT_NUM=5120`/`GROUP_LISTING_MULTI_CORE_LENGTH`/`GROUP_LISTING_*_AIV_NUM` 各自是「纯并行度阈值(可原样保留)」还是「隐含 arch22 UB 字节假设(须改 GetCoreMemSize 推导)」，给出逐项判定结论。避免把 arch22 的隐性 UB 假设带入 arch35 |
| 4 | DESIGN-API-1 | LOW | §3.4 索引初始化行 `ArithProgression<int32_t>` | 本 devkit 快照未定位 ArithProgression.md 变体 | 开发期 devkit 就绪后按 §6 checklist 补 ArithProgression 官方文档变体核验（当前样板 moe_v3_sort_one_core.h:48 实证 950 可用，非阻塞）|
| 5 | DESIGN-API-3 | LOW | §3.4 `GetSortLen/GetSortOffset` 行「`(n)→n×2`(float 数)」| `GetSortOffset.md`：返回值单位为**字节** | 标注补充为「返回字节偏移」，避免误读为元素数；语义等价（float proposal 8B），非功能问题 |

---

## 汇总与判定

- **DESIGN-SPEC-1 判定**：✅通过。「spec.yaml 一致性映射」专章存在；5 易误用字段（dtype_policy.supported_combinations / outputs[].shape_rule / numerical_tolerance.per_dtype / boundary_conditions[] / extreme_inputs[]）与 spec.yaml 逐字 1:1；无从 REQUIREMENTS 重解释覆盖 spec 的情况。
- **spec 冲突判定（流程终止信号）**：**无** REQUIREMENTS/spec 在 spec-owned 字段冲突。非终止信号。
- **强制规则**：C1（仅评审 Markdown，未评审 .cpp/.h）✔；C2（每处关键 API 官方文档演练 + 配图 Read）✔；C3（状态字段在顶部）✔；C4（UB 预算表存在、未缺失/未超限）**未触发**；C5（需求承接无缺项）**未触发**；C6（只评审未改 DESIGN.md）✔。
- **主 Agent 5 设计要点**：要点 1/2/3/5 论证成立（✅）；要点 4 主体彻底、一处常量分类未闭合（MED #3，开发期闭环）。
- **发现问题**：HIGH=0，MED=3，LOW=2。无 HIGH 阻塞项，无 spec 冲突。MED/LOW 均为证据补强 / 文档标注订正 / 开发期复核项，不改变方案可行性与对外语义。

**最终状态**: ✅通过 —— 方案 sound，可进入 CP2 用户确认。建议在开发前顺带吸收 MED #1（MrgSort 标注）、MED #2（UB 数值实例）、MED #3（片上常量分类结论），并在开发期按 §6 checklist 闭环 GetSortTmpSize/DCCI/ArithProgression 文档变体复核（LOW #4）。

