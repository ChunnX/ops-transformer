# FfnWorkerBatching 迭代执行计划（新增 Ascend950/A5 支持）

> 目标：为已有 A2/A3 算子新增 Ascend950(arch35/A5) 全新代次路径，功能/精度对齐 A2；A2/A3 分支保留不动。三步算法（sort→gather→group_listing）照搬 A2，仅做 arch35 指令/资源模型适配。
>
> **穿刺 = NPU 实跑（非 Mock）**：所有穿刺/整合结果判定必须在实测 Ascend950DT_9584（arch35/DAV_3510，CANN 9.1.0）真实 NPU 上运行，与 CPU golden 二进制一致（`np.array_equal`）为通过标准；不接受 CPU/Mock 仿真替代最终判定。
>
> **对外规格锁定**（spec.yaml）：3 profile（P0 fp16 / P1 bf16 / P2 int8）× 2 路径（NORM=TilingKey 100 / RECV=TilingKey 101）= 6 条有效组合。TilingKey 只由 need_schedule 决定（0→100/1→101），token_dtype 正交（kernel 内处理）。102(RECV_GATHER) 保留分支不移植。

## 迭代一穿刺列表

> 主线单 dtype = **P0 fp16**（算子主 dtype）；单 TilingKey = **100 NORM**。目标：打通 arch35 全新代次骨架（`_apt.cpp` 入口 + `arch35/` sort→gather→group_listing 主链 + 1000 档 Regbase tiling + `config/ascend950/`），验证 arch35 最短闭环。

| 任务类型 | TilingKey | 路径 | Dtype(profile) | 排序模式 | Memory Strategy | 穿刺验证目标 |
|---------|-----------|------|-------|---------|-----------------|-------------|
| **主线** | 100 | NORM | fp16 (P0) | 单核 sort（小 shape） | UB 运行时取值，单 loop | arch35 骨架端到端跑通：apt 入口分发→sort→gather→group_listing→8 路输出，NPU 对拍 A2 二进制一致 |
| 穿刺1 | 101 | RECV | fp16 (P0) | 单核 sort | 同主线 + scan_token_info 前置 | RECV 路径（need_schedule=1）扫描+batching 骨架可行性 |
| 穿刺2 | 100 | NORM | fp16 (P0) | **多核 sort + MrgSort 归并** | 大 shape 触发多核切分 | 多核 sort→MrgSort 4 路归并 + 核间 gather/group_listing 分工可行性 |
| 穿刺3 | 100 | NORM | **int8 (P2)** | 单核 sort | +尾部 32B FP32 scale 连续 gather | P2 dynamic_quant int8：BLOCK_BYTES=32 scale 连续搬运 + dynamic_scale 输出可行性 |
| 穿刺4 | 100 | NORM | **bf16 (P1)** | 单核 sort | 同主线（tokenDtypeSize 同 fp16） | P1 bf16 字节 gather 可行性（确认 y=bf16 二进制一致） |

**并行要求**：主线 + 穿刺1 + 穿刺2 + 穿刺3 + 穿刺4 **必须在同一次响应发起**（NPU 实跑并行验证各分支，快速暴露 arch35 移植风险点）。

**迭代一穿刺聚焦风险**（对应 DESIGN §7.2）：
- ScheduleContext 1024B GM 逐字节解析 + DCCI 语义（主线首验）
- Sort/MrgSort 元素上限 + `GetSortTmpSize` arch35 平台参数（穿刺2）
- P2 int8 + 32B scale 连续 gather arch35 搬运语义（穿刺3）
- 1000 档 Regbase tiling + IsRegbaseSocVersion 守卫与 arch22 隔离（全部任务共用，验证 A2/A3 零回归）

## 迭代二整合目标

**整合范围**：NORM(100) + RECV(101) 双 TilingKey 完整整合，单/多核 sort 全路径打通。

- 合并迭代一主线 + 穿刺1（NORM+RECV 双路径）+ 穿刺2（多核 sort）为统一 arch35 kernel。
- 完成 RECV 路径三种 sort 分支：`KernelScanSortMaskOneCore` / `KernelScanSortMaskMultiCore` / `KernelScanSortMaskMultiCoreBsK`（isNormalBsK 分流）。
- tiling 完成 `SetBatchingTilingKey`(100/101) + UB 派生阈值（`sortLoopMaxElement` 等）全部运行时推导，RECV 限核阈值（TH_RECV_CORE_NUM/TH_RECV_Y_NUM）arch35 标定。
- 单 dtype 保持 P0 fp16（profile 全覆盖留迭代三）。

## 迭代二穿刺列表

> 验证迭代三全覆盖的难点（P2 量化、边界 shape、DCCI 一致性），为迭代三降风险。

| 任务类型 | 验证目标 | TilingKey×路径 | Profile | 判定 |
|---------|---------|--------|--------|------|
| 穿刺1 | P2 int8 + 32B scale 在 RECV(101) 路径 gather | 101 RECV | int8 (P2) | NPU 实跑 dynamic_scale 二进制一致 |
| 穿刺2 | 边界 shape：A=1024 上界 + topK+1=64 上界（k=N 全有效） | 100 NORM | fp16 (P0) | 上界 shape 不溢出、二进制一致 |
| 穿刺3 | 边界：全 mask(k=0) + same_bin(expert_num=8192 全同专家) | 100 NORM | fp16 (P0) | actual_token_num=0 / group_list 单专家非零，二进制一致 |
| 穿刺4 | 稀疏 ~50% mask（无效 token 排末尾）在 RECV | 101 RECV | fp16 (P0) | 排序稳定性 + actual_token_num<Y，二进制一致 |

## 迭代三全覆盖目标

> 整合迭代二穿刺结果 → 3 profile × 2 路径全功能覆盖 + 全边界。

| 覆盖维度 | 内容 | 说明 |
|---------|------|------|
| 全 profile | P0 fp16 / P1 bf16 / P2 int8（token_dtype 0/1/2） | y dtype 随 token_dtype；P2 输出有效 dynamic_scale |
| 全路径 | NORM(100) + RECV(101) × 3 profile = 6 组合 | TilingKey 由 need_schedule 决定；102 不移植 |
| 排序模式 | 单核 / 多核(MrgSort) / RECV scan 三变体(One/Multi/MultiBsK) | 由 shape 规模 + isNormalBsK 分流 |
| 边界 case（spec.boundary_conditions） | k=0(全 mask)、k=N(A=1024/topK+1=64)、same_bin(expert_num=8192)、expert_num>8192 报错、topK+1>64 报错、A>1024 报错、max_out_shape 长度≠4 报错、dtype≠int8 报错、非 1 维报错 | 硬校验对齐 A2；报错类 UT 覆盖 |
| 极端输入（spec.extreme_inputs） | all_same_index(全同专家)、P2 32B scale 连续、normal_sparse(~50% mask) | NPU 实跑对拍 |
| 精度口径 | 全输出二进制一致（`np.array_equal`，rtol=atol=0） | 对拍 A2 / CPU golden |
| A2/A3 回归 | Ascend910B/910_93 tiling+kernel 行为零变化 | 验证 IsRegbaseSocVersion 守卫隔离正确 |

## 穿刺结果判定规则

> **判定环境**：真实 Ascend950DT_9584（arch35/DAV_3510），CANN 9.1.0。**NPU 实跑，非 Mock/CPU 仿真。**

| 状态 | 判定标准（NPU 实跑） | 处理 |
|------|---------|------|
| ✅ 成功 | arch35 NPU 实跑通过，8 路输出与 CPU golden **二进制一致**（`np.array_equal`），且该分支 A2 同输入对拍一致 | 复用代码，纳入下一迭代整合 |
| ⚠️ 部分成功 | NPU 跑通但部分输出/边界不一致，或需绕行实现 | 参考逻辑，问题归档 issue，下一迭代重做该点 |
| ❌ 失败 | NPU 无法跑通（编译/运行错）或结果系统性不一致 | 下一迭代重做；若涉 arch35 API 约束，回 DESIGN 修订 API 映射 |

**通过底线**：
- 精度：全输出 `np.array_equal` 二进制一致（非计算类，无浮点容差，对齐 spec.numerical_tolerance bitwise_equal）。
- 确定性：相同 schedule_context 多次运行输出完全一致（spec.determinism.bitwise_reproducible）。
- 隔离：A2/A3（Ascend910B/910_93）功能与精度零回归（IsRegbaseSocVersion 守卫 + opFile 隔离生效）。
- 资源：无写死 arch UB 字节/核数常量（代码审查 + UB 占用 ≤ 运行时 `GetCoreMemSize(UB)` 返回值）。
