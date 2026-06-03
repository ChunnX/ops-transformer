# 实现子任务拆解

## 1. 目标

本文基于 `05_kernel_design.md`、`06_inter_core_sync_design.md`、`07_intra_core_sync_design.md`，将首期 kernel 实现拆解为可落地、可检查的子任务。

拆解原则：

1. 子任务按实现边界组织，而不是按文档章节机械拆分。
2. 每个子任务必须给出需求、实现要点和检查项。
3. 核间同步、核内同步和业务计算分别检查，避免职责混淆。
4. tail、空任务、pingpong buffer、event id 上限必须作为每个相关任务的验收条件。
5. 每个子任务必须能作为独立提交切片完成编译验证；Host 侧改动必须有 UT 覆盖。

## 2. 子任务总览

独立执行的含义：

1. 子任务可以按依赖顺序单独开发、单独提交。
2. 每个提交后仓库必须能编译通过；不能留下破坏编译的半成品。
3. Kernel 子任务若数值逻辑未完整接入，需要通过 stub、feature guard 或空实现保持 build 通过。
4. Host 子任务必须同步补 UT；仅编译通过不算完成。
5. 同步相关任务归 kernel 任务，当前阶段可先不做测试，但必须维持 kernel 编译通过并完成静态检查。
6. 后续任务可以依赖前序任务的接口，但不能要求多个任务同时落地才可编译。

| ID | 子任务 | 主要产物 | 依赖 | 独立验证 |
|----|--------|----------|------|----------|
| T00 | Host checker/tiling contract + UT | checker、tiling 字段、host UT | 无 | Host UT + host 编译 |
| T01 | Controller 与 basic block 骨架 | controller tile range 划分、`Init/ComputeRange/End` 分派框架 | T00 | kernel 编译 |
| T02 | AIV local slice 与 tail row range | AIV0/AIV1 token slice、tail row 范围 | T01 | kernel 编译 + host UT 扩展 |
| T03 | 静态 buffer 与 tensor 绑定 | L1/UB tensor、bufferId、reserve 子区 | T01 | kernel 编译 |
| T04 | CrossCore 封装与 flag id | `SetAivToAic/WaitAivToAic/SetAicToAiv/WaitAicToAiv` | T01 | kernel 编译 |
| T05 | 核内 event 常量与 Init/End | AIV/AIC event id、初始化 Set、结束 Wait | T03 | kernel 编译 + event id 静态检查 |
| T06 | AIV Q/K/V 前处理 | input load、RMSNorm、RoPE、AL1 写入、`A_READY` | T02/T03/T04/T05 | kernel 编译 + preprocess UT |
| T07 | AIC rotation 常驻与 Cube matmul | `rotation_l1_nz`、AL1 读取、matmul | T03/T04/T05/T06 | kernel 编译 + Cube UT |
| T08 | AIC Fixpipe 分发与 `C_READY` | C 输出按 M range 分发到两个 AIV UB | T02/T04/T07 | kernel 编译 |
| T09 | V 分支提前后处理 | `V * v_scale`、FP8 cast、`v_cache` 写回 | T03/T05/T06 | kernel 编译 + V path UT |
| T10 | AIV Q/K 后处理 | `qk_after_cube_ub` 原地 staging、dynamic quant、cache/out 写回、`C_CONSUMED` | T04/T05/T08 | kernel 编译 + Q/K postprocess UT |
| T11 | Rolling pipeline 与 drain | tile i/i-1 流水、最后一个 tile drain | T06/T07/T08/T09/T10 | kernel 编译 |
| T12 | tail 与空任务同步 | 无空洞 row 切分、空 AIV 固定参与同步 | T02/T04/T11 | kernel 编译 + Host UT 扩展 |
| T13 | 同步协议审计 | Set/Wait 次数、pipe、event id、死锁检查 | T04/T05/T11/T12 | kernel 编译 + 静态审计 |
| T14 | Kernel UT/ST | 单元、同步、tail、端到端验证 | T06-T13 | UT/ST 全量通过 |

## 3. 子任务明细

### T00. Host checker/tiling contract + UT

需求：

1. Host 侧必须定义 kernel 可稳定依赖的 fast path contract。
2. checker/tiling 必须拒绝当前 kernel 不支持的输入。
3. Host 侧改动必须有 UT 覆盖。
4. Tiling data 字段要支持后续 kernel 子任务逐步接入，且字段默认值不能破坏编译或运行。

实现要点：

1. checker 强制当前 fast path 约束：`D == 128`、`Nv == Nk`、`rotation` 非空、`v_scale` 非空、`head_nums=[Nq,Nk,Nv]` 合法。
2. checker 校验 dtype、shape、`cu_seq_len` rank/长度、`slot_mapping` 长度、cache/v_scale head 维一致性。
3. tiling 写入 kernel 必需字段：全局 token/tile 信息、head 数、stride、block 参数、fast path 标志。
4. tiling 字段必须足够 controller 计算全局 tile 数和每个 `cubeIndex` 的 `QkvRrcTileRange{begin,end}`。
5. tiling 不把 UB/L1 静态 buffer 数和 reserve 子区大小写入 tiling data；这些由 kernel 静态常量管理。
6. Host UT 覆盖 normal case、拒绝 case、边界 case 和典型 tiling 结果。

检查项：

1. Host UT 覆盖 `Nv != Nk`、`D != 128`、缺失 `rotation/v_scale`、非法 `head_nums`、非法 shape/dtype。
2. Host UT 覆盖 tail 相关 tiling：`T < tokenTile`、`T % tokenTile != 0`。
3. Host UT 覆盖典型 `Nq/Nk/Nv` 组合，tiling 结果与设计表一致。
4. Host UT 覆盖 controller 所需的全局 tile 数、tile range 边界和空 range 场景。
5. 新增 tiling data 字段有默认初始化，不导致 kernel 编译失败。
6. Host 子任务提交时必须能单独通过 host UT，不能只依赖后续 kernel 任务验证。

### T01. Controller 与 basic block 骨架

需求：

1. 建立 controller 到 basic block 的执行入口。
2. controller 层负责按逻辑 cube group 划分 tile range。
3. 同一 cube group 内的 AIC、AIV0、AIV1 必须获得相同 tile range。
4. basic block 内按 AIC/AIV 身份分派到不同 rolling 分支。
5. 统一组织 `Init`、`ComputeRange`、`End` 生命周期。

实现要点：

1. controller 根据当前核身份计算 `cubeIndex`：
   - AIC: `cubeIndex = GetBlockIdx()`
   - AIV: `cubeIndex = GetBlockIdx() / 2`
2. controller 使用 `cubeIndex` 对全局 tile 数做核组级 range 切分，生成 `QkvRrcTileRange{begin,end}`。
3. controller 不按 AIV0/AIV1 再切 tile range；AIV0/AIV1 的 token 子区间由 basic block 内 `MakeAivSlice(tile, GetSubBlockIdx())` 计算。
4. `Process()` 调用 `basicBlock.Init(...)`、`basicBlock.ComputeRange(range)`、`basicBlock.End()`。
5. `ComputeRange` 内部按当前核身份分派到 `ComputeAivRolling` 或 `ComputeAicRolling`。
6. `Init` 内完成 buffer 绑定、CrossCore flag 常量准备、核内 event 初始化。
7. `End` 内完成核内 event 结束配平。

检查项：

1. 同一个 `cubeIndex` 下，AIC、AIV0、AIV1 的 `tileRange` 完全一致。
2. 不同 `cubeIndex` 的 tile range 覆盖全局 tile 空间，且不重叠、不遗漏。
3. AIV local id 不参与 controller 级 tile range 切分，只参与 tile 内 token slice 切分。
4. tail token 切分不在 controller 层做，必须留给 basic block/AIV slice 处理。
5. AIC 分支不会执行 AIV vec 逻辑。
6. AIV 分支不会执行 AIC cube 逻辑。
7. `InitIntraCoreEvents` 和 `EndIntraCoreEvents` 成对出现。
8. `ComputeRange` 对空 range 行为明确，不破坏同步协议。

### T02. AIV local slice 与 tail row range

需求：

1. 固定支持 `KERNEL_TYPE_MIX_AIC_1_2` 下的两个 AIV local slice。
2. AIV0/AIV1 对同一个 tile 做连续 token 切分。
3. tail 不允许在 A 矩阵 M 方向留下空洞。
4. 空 slice 也必须保留同步协议参与能力。

实现要点：

1. AIV 使用 `GetSubBlockIdx()` 区分 AIV0/AIV1。
2. `MakeAivSlice(tile, aivLocalId)` 只处理 tile 内切分，不改变 controller 传入的 tile range。
3. tail 切分使用实际 token 数：

```text
actualAiv0 = ceil(actualTokenTile / 2)
actualAiv1 = actualTokenTile - actualAiv0

AIV0 rows = [0, actualAiv0 * Hqk)
AIV1 rows = [actualAiv0 * Hqk, actualTokenTile * Hqk)
```

检查项：

1. AIV1 row 起点不能使用固定 `tokenTilePerAiv * Hqk`。
2. tail 中 `actualAiv1 == 0` 时，AIV1 不写数据但仍参与同步。
3. AIC 使用的 `M_tile = actualTokenTile * Hqk`。
4. AIC Fixpipe 分发 C 输出时使用同一 row 边界。
5. AIV local slice 不影响核组级 tile range。

### T03. 静态 buffer 与 tensor 绑定

需求：

1. 建立 L1/UB tensor 与 double buffer 的固定绑定。
2. 明确哪些 buffer 参与核间同步，哪些只参与核内同步。
3. `qk_after_cube_ub` 首期固定原地复用为 Q/K FP8 staging。

实现要点：

1. L1 tensor 包括 `rotation_l1_nz`、`a_rot_l1_nz`。
2. UB tensor 包括 `qk_after_cube_ub`、`v_ub`、`gamma_ub`、`cos_sin_ub[2]`、`slot_mapping_ub[2]`、`v_scale_ub`、`v_out_ub[2]`。
3. `bufferId = tileId & 1` 只选择数据 buffer，不参与 CrossCore flag id 分配。
4. Q/K dynamic quant 的 FP8 staging 不额外开独立区域，复用 `qk_after_cube_ub` 所在 output buffer。

检查项：

1. `qk_after_cube_ub` 原地覆盖前，AIV 已完成对 FP32 C 的必要读取。
2. `v_out_ub` 在 reserve 内 double buffer，不占 input/output DB pool。
3. `rotation_l1_nz` 只由 AIC 使用，不需要 AIV 等待。
4. `a_rot_l1_nz` / AL1 属于 Cube 侧资源，不占 AIV 核内 event id。

### T04. CrossCore 封装与 flag id

需求：

1. 只使用固定语义 flag id。
2. 不按 `bufferId`、`generation` 或数组分配 CrossCore flag。
3. AIC 侧封装负责同时处理 AIV0/AIV1。

实现要点：

```cpp
constexpr uint64_t SYNC_A_READY = 0;
constexpr uint64_t SYNC_C_READY = 1;
constexpr uint64_t SYNC_C_CONSUMED = 2;
```

1. AIV0 使用基础 id，AIV1 由框架通过 `flag + FLAG_ID_MAX` 推导。
2. `WaitAivToAic` 在 AIC 侧 wait 两次：`syncFlag + FLAG_ID_MAX`、`syncFlag`。
3. `SetAicToAiv` 在 AIC 侧 set 两次：`syncFlag + FLAG_ID_MAX`、`syncFlag`。
4. AIV 侧 `WaitAicToAiv` 只 wait 基础 id。

检查项：

1. `SYNC_A_READY/C_READY/C_CONSUMED` 的 Set/Wait 次数每 tile 配平。
2. AIV 侧代码不手动加 `FLAG_ID_MAX`。
3. `A_READY` 使用 `AIV Set<PIPE_MTE3>`、`AIC Wait<PIPE_MTE1>`。
4. `C_READY` 使用 `AIC Set<PIPE_FIX>`、`AIV Wait<PIPE_V>`。
5. `C_CONSUMED` 使用 `AIV Set<PIPE_MTE3>`、`AIC Wait<PIPE_FIX>`。

### T05. 核内 event 常量与 Init/End

需求：

1. 为 AIV/AIC 核内 buffer 复用定义 event id 常量。
2. 所有 event id 落在 `0..5`。
3. 每个反向同步在 init/end 中配平。

实现要点：

1. AIV `<PIPE_V, MTE3>` 计划：
   - `v_out_ub[2]`: `0/1`
   - `qk_after_cube_ub[2]`: `2/3`
   - 可选 A temporary UB staging `[2]`: `4/5`
2. AIV `<MTE3, PIPE_V>` 计划：
   - `v_out_ub[2]` reuse: `0/1`
   - `qk_after_cube_ub[2]` reuse: `2/3`
   - 可选 A temporary UB staging `[2]` reuse: `4/5`
3. AIC event 参考 `MTE1/MTE2/MMAD/FIXP` pipe pair，最终按实际 buffer 数压缩到 `0..5`。
4. Init 对所有可复用 buffer 先 `Set<consumer, producer>`。
5. End 对所有可复用 buffer 最后 `Wait<consumer, producer>`。

检查项：

1. 同一个 `<src_pipe, dst_pipe>` 内没有两个 live buffer 共用同一个 event id。
2. 不出现连续 `Set` 同一 event id。
3. 不为 4 个 AL1 物理 buffer 分配 AIV 核内 event id。
4. 可选 A temporary UB staging 若不存在，对应 event 不初始化、不等待。
5. 所有 Init `Set` 都有 End `Wait` 配平。

### T06. AIV Q/K/V 前处理

需求：

1. AIV 读取本地 slice 的 Q/K/V、gamma、cos/sin、slot_mapping。
2. 对 Q/K 做 RMSNorm、RoPE、BF16/NZ 转换。
3. MTE3 写 `a_rot_l1_nz` / AL1 完成后发送 `A_READY`。
4. V 分支数据暂存到 `v_ub`，供提前后处理使用。

实现要点：

1. `MakeAivSlice(tile, GetSubBlockIdx())` 只覆盖本 AIV 的连续 token 范围。
2. input UB 的 `MTE2 -> PIPE_V` 正向同步保护输入可读。
3. input UB 的 `PIPE_V -> MTE2` 反向同步保护复用。
4. AL1 写入完成后调用 `SetAivToAic<PIPE_MTE3>(SYNC_A_READY)`。
5. 如果 A 写 AL1 前存在 temporary UB staging，仅对该 staging 加 AIV 核内 event。

检查项：

1. `A_READY` 不能早于 MTE3 写 AL1 完成。
2. AL1 到 AIC 读取的可见性由 `A_READY` 硬同步保证，不再额外分配 AIV AL1 event。
3. tail 中当前 AIV token 数为 0 时，仍执行固定同步流程。
4. V 分支不参与 RMSNorm/RoPE/rotation，只做后续 `V * v_scale`。

### T07. AIC rotation 常驻与 Cube matmul

需求：

1. AIC 在 tile 循环前通过 MTE2 搬入 `rotation_l1_nz` 并常驻。
2. AIC 等两个 AIV 的 `A_READY` 后读取 AL1。
3. 执行 BF16 x BF16 -> FP32 Cube matmul。

实现要点：

1. `rotation` 是外部 dense BF16 `[D,D]`，kernel 不假设 Hadamard 结构。
2. `PrepareRotationBeforeLoop()` 完成 ND2NZ 搬入和 AIC 核内同步。
3. `WaitAivToAic<PIPE_MTE1>(SYNC_A_READY)` 通过后再启动 MTE1/Cube 读取 `a_rot_l1_nz`。
4. AIC 内部 L1/L0/L0C 同步属于 AIC 核内流水管理。

检查项：

1. `rotation_l1_nz` 不需要核间 flag。
2. AIC 对 `SYNC_A_READY` 每 tile wait 两次。
3. AIC 不在两个 AIV ready 前启动完整 `M_tile` matmul。
4. AIC 内部 event id 不把 AL1 物理 buffer 当成 AIV event。

### T08. AIC Fixpipe 分发与 C_READY

需求：

1. AIC 将 L0C/C 输出通过 Fixpipe 分发到两个 AIV 的 UB。
2. 按 M 方向 row range 分发，边界与 AIV token 切分一致。
3. Fixpipe 输出完成后通知两个 AIV `C_READY`。

实现要点：

1. AIV0 C rows 对应 `[0, actualAiv0 * Hqk)`。
2. AIV1 C rows 对应 `[actualAiv0 * Hqk, actualTokenTile * Hqk)`。
3. `SetAicToAiv<PIPE_FIX>(SYNC_C_READY)` 已足够表达 Fixpipe 输出完成后再通知 AIV。
4. AIC 内部 `MMAD -> FIXP`、`FIXP -> MMAD` event 只属于 AIC 核内同步。

检查项：

1. C 输出分发不能使用固定 half-tile 边界处理 tail。
2. `C_READY` 每 tile AIC set 两次。
3. AIV 收到 `C_READY` 前不能读 `qk_after_cube_ub`。
4. AIC 在 `C_CONSUMED` 前不能复用对应 C 输出路径或 buffer。

### T09. V 分支提前后处理

需求：

1. AIV 在发送当前 tile `A_READY` 后即可执行 V 分支后处理。
2. V 分支不依赖 Cube 输出。
3. V 分支写 `v_cache`。

实现要点：

1. 读取本 AIV slice 的 `v_ub` 和 `slot_mapping_ub`。
2. 执行 `v * v_scale[nv]`。
3. 按 RINT/SAT cast 到 FP8 E4M3FN。
4. 使用 `v_out_ub[bufferId]` 作为 MTE3 写 GM/cache 前的 double buffer。
5. 使用 `PIPE_V -> MTE3` 和 `MTE3 -> PIPE_V` event 保护 `v_out_ub`。

检查项：

1. V 分支不等待 `C_READY`。
2. `v_out_ub` event id 使用 `0/1`，不与 Q/K output event 冲突。
3. `slot_mapping` 与 Q/K cache 写回使用同一 token slot 语义。
4. `v_scale == 0` 行为与 FP8 cast 路径一致，不新增额外语义分支。

### T10. AIV Q/K 后处理

需求：

1. AIV 等 `C_READY` 后消费上一 tile 的 `qk_after_cube_ub`。
2. 对 Q/K row 做 dynamic quant。
3. `qk_after_cube_ub` 原地覆盖复用为 FP8 staging。
4. Q/K 后处理相关 MTE3 写出完成后发送 `C_CONSUMED`。

实现要点：

1. `WaitAicToAiv<PIPE_V>(SYNC_C_READY)` 通过后，`PIPE_V` 读取 `qk_after_cube_ub`。
2. 对每个 row 计算 `max_abs`、`scale = max_abs / 448`。
3. 使用 `Div -> RINT/SAT cast FP8 E4M3FN`。
4. Q 分支写 `q_out` 和 `q_scale`。
5. K 分支写 `k_cache` 和 `k_scale_cache`。
6. MTE3 写出完成后 `SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED)`。

检查项：

1. `qk_after_cube_ub` 原地覆盖前，FP32 C 已完成当前 row 的 scale/quant 读取。
2. Q/K output event id 使用 `2/3`，不与 V output event 冲突。
3. `C_CONSUMED` 不能早于 Q/K 后处理 MTE3 写出完成。
4. `max_abs == 0` 不新增零行保护假设，行为对齐 dynamic_quant 路径。

### T11. Rolling pipeline 与 drain

需求：

1. 支持 AIV 前处理当前 tile、后处理上一 tile 的 rolling pipeline。
2. 支持最后一个 tile 的 drain。
3. 保证 pingpong buffer 的同步事件按固定顺序消费。

实现要点：

1. AIV 每轮先 preprocess tile i 并发送 `A_READY`。
2. AIV 随后可执行 tile i 的 V 分支提前后处理。
3. 当 tile i - 1 存在时，等待上一 tile `C_READY` 并执行 Q/K 后处理。
4. 循环结束后 drain 最后一个 tile 的 Q/K 后处理和 `C_CONSUMED`。
5. AIC 每轮顺序为 `A_READY -> Matmul/Fixpipe -> C_READY -> C_CONSUMED`。

检查项：

1. tile 0 不等待不存在的上一 tile `C_READY`。
2. 最后一个 tile 必须 drain，否则 AIC 会等待 `C_CONSUMED`。
3. `bufferId` 复用只依赖固定顺序和 flag 计数，不派生 flag id。
4. 单 buffer 退化时同步语义仍成立。

### T12. tail 与空任务同步

需求：

1. tail 只改变数据范围，不改变同步次数。
2. 某个 AIV 实际 token 数为 0 时，仍参与 `A_READY/C_READY/C_CONSUMED`。

实现要点：

1. `MakeAivSlice` 允许返回空 slice。
2. 空 slice 跳过实际数据计算，但不跳过 CrossCore 协议。
3. AIC 仍等待两个 AIV 的 `A_READY`。
4. AIC 仍 set 两个 AIV 的 `C_READY`。
5. AIC 仍等待两个 AIV 的 `C_CONSUMED`。

检查项：

1. tail row range 连续无空洞。
2. 空 slice 不访问非法 GM/UB/L1 地址。
3. 空 slice 不导致 AIC 永久等待。
4. Set/Wait 计数与非 tail tile 完全一致。

### T13. 同步协议审计

需求：

1. 在实现完成后独立审计核间和核内同步。
2. 输出可人工核对的同步边与计数。

实现要点：

1. 对每个 tile 统计：
   - `A_READY`: AIV set 2 次，AIC wait 2 次。
   - `C_READY`: AIC set 2 次，AIV wait 合计 2 次。
   - `C_CONSUMED`: AIV set 2 次，AIC wait 2 次。
2. 对每个核内 event 统计 Init/End 配平。
3. 检查所有 event id 是否在 `0..5`。
4. 检查同 pipe pair 内 event id 是否唯一覆盖 live buffer。

检查项：

1. 没有 `flag = base + bufferId` 或 `flag = base + generation`。
2. 没有连续 `Set` 同一个核内 event id。
3. 没有遗漏 tail/empty-work 的 Set/Wait。
4. CrossCore pipe pair 与设计一致。
5. AL1 不占 AIV 核内 event id。

### T14. Kernel UT/ST

需求：

1. 覆盖数值正确性、tail 行为和同步协议。
2. 覆盖容易死锁或错位消费的边界。

实现要点：

1. 单 tile UT：preprocess、Cube、postprocess 各自可验证。
2. 多 tile UT：覆盖 buffer 0/1 pingpong 和 tile 2 复用 buffer 0。
3. tail UT：覆盖奇数 tail、`actualAiv1 == 0`、`actualAiv0 != actualAiv1`。
4. Sync UT：在可测试封装下记录 Set/Wait 次数和顺序。
5. ST：端到端对比 golden，覆盖 random dense `rotation`。

检查项：

1. 每个输出 tensor 与 golden 对齐。
2. `slot_mapping` 连续、非连续、跨 block 均覆盖。
3. 多 batch 且 tile 跨 `cu_seq_len` 边界时 RoPE 位置正确。
4. `v_scale == 0`、`max_abs == 0` 均覆盖。
5. 长序列多 tile 不死锁，不错消费上一轮 flag。

## 4. 交付检查清单

实现提交前逐项确认：

1. `05_kernel_design.md` 中 rolling pipeline 与代码一致。
2. `06_inter_core_sync_design.md` 中三条 CrossCore 边与代码一致。
3. `07_intra_core_sync_design.md` 中 event id 常量、Init/End、pipe pair 与代码一致。
4. 所有 CrossCore flag id 仍为 `0/1/2`。
5. 所有核内 event id 仍在 `0..5`。
6. AIV0/AIV1 tail row range 连续无空洞。
7. `qk_after_cube_ub` 原地复用没有引入生命周期重叠。
8. `rotation_l1_nz` 只由 AIC 内部管理，不引入 rotation 核间 flag。
9. AL1 到 AIC 读取由 `A_READY` 硬同步保证，不分配 AIV 核内 event id。
10. UT/ST 覆盖 tail、empty slice、pingpong 复用和多 tile drain。

## 5. 子任务提交门禁

每个子任务完成后必须满足对应门禁：

1. T00 必须通过 Host UT；checker/tiling 变更没有 UT 不允许合入。
2. T01-T13 每个 kernel 子任务提交后都必须保持 kernel 编译通过。
3. T02、T12 涉及 tiling/tail contract 的改动，必须同步扩展 Host UT。
4. T04、T08、T11、T13 涉及同步协议的改动，当前阶段不要求测试，但必须维持 kernel 编译通过，并在文档或代码审计记录中说明 Set/Wait 次数。
5. T05 涉及核内 event 的改动，必须静态检查 event id 范围和同 pipe pair 冲突。
6. T06-T10 涉及数值路径的改动，必须至少有对应模块级 UT 或可复用 golden 对比。
7. T14 完成前必须通过端到端 ST；在此之前，各子任务不能依赖“最后一起修”来维持编译。
