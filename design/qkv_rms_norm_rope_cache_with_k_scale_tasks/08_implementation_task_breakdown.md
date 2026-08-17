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
5. 同步相关任务归 kernel 任务，必须维持 kernel 编译通过，并静态审计实际 event id、pipe pair、live buffer 冲突和 Set/Wait 生命周期配平。
6. 后续任务可以依赖前序任务的接口，但不能要求多个任务同时落地才可编译。

| ID | 子任务 | 主要产物 | 依赖 | 独立验证 |
|----|--------|----------|------|----------|
| T00 | Host checker/tiling contract + UT | checker、tiling 字段、host UT | 无 | Host UT + host 编译 |
| T01 | Controller 与 basic block 骨架 | controller token range 划分、tile param 填充、`Init/PrepareBeforeLoop/ComputeTile/End` 分派框架 | T00 | kernel 编译 |
| T02 | AIV local slice 与 tail row range | AIV0/AIV1 token slice、tail row 范围 | T01 | kernel 编译 + host UT 扩展 |
| T03 | 静态 buffer 与 tensor 绑定 | L1/UB tensor、owner use-id、其他固定 buffers | T01 | kernel 编译 |
| T04 | CrossCore 语义函数与 flag id | `SetAivMte3ToAicMte1AReady/WaitAivMte3ToAicMte1AReady` 等 basic_block 内语义函数 | T01 | kernel 编译 |
| T05 | 核内 event 常量与 Init/End | AIV/AIC event id、初始化 Set、结束 Wait | T03 | kernel 编译 + event/pipe/lifecycle 审计 |
| T06 | AIV Q/K/V 前处理 | input load、RMSNorm、RoPE、AL1 写入、`A_READY` | T02/T03/T04/T05 | kernel 编译 + DiscussionForum golden |
| T07 | AIC rotation 常驻与 Cube matmul | `rotation_l1_nz`、AL1 读取、matmul | T03/T04/T05/T06 | kernel 编译 + DiscussionForum dense rotation golden |
| T08 | AIC Fixpipe 分发与 `FIX_OUTPUT_READY` | Fixpipe 输出按 M range 分发到两个 AIV UB | T02/T04/T07 | kernel 编译 |
| T09 | V 分支提前后处理 | `V * v_scale`、FP8 cast、`v_cache` 写回 | T03/T05/T06 | kernel 编译 + DiscussionForum V/cache golden |
| T10 | AIV Q/K 后处理 | `qk_after_cube_ub` 原地 staging、dynamic quant、cache/out 写回、`FIX_OUTPUT_CONSUMED` | T04/T05/T08 | kernel 编译 + DiscussionForum Q/K golden |
| T11 | Rolling pipeline 与 drain | tile i/i-1 流水、最后一个 tile drain | T06/T07/T08/T09/T10 | kernel 编译 |
| T12 | tail 与空任务同步 | 无空洞 row 切分、空 AIV 固定参与同步 | T02/T04/T11 | kernel 编译 + Host UT 扩展 |
| T13 | 同步协议审计 | Set/Wait 次数、pipe、event id、死锁检查 | T04/T05/T11/T12 | kernel 编译 + 静态审计 |
| T14 | DiscussionForum Kernel 验证 | NPU golden、TTK、tail、多 tile、端到端验证 | T06-T13 | NPU 用例通过且同步静态审计完成 |

## 3. 子任务明细

### T00. Host checker/tiling contract + UT

需求：

1. Host 侧必须定义 kernel 可稳定依赖的 当前实现约束。
2. checker/tiling 必须拒绝当前 kernel 不支持的输入。
3. Host 侧改动必须有 UT 覆盖。
4. Tiling data 字段要支持后续 kernel 子任务逐步接入，且字段默认值不能破坏编译或运行。

实现要点：

1. checker 强制当前实现约束：`D == 128`、`0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`、
   `rotation` 非空、`v_scale` 非空、`head_nums=[Nq,Nk,Nv]` 合法、`layout_qkv in {NTD,TND}`、
   `layout_q_out in {NTD,TND}`、支持 `NTD->NTD`、`TND->TND`、`TND->NTD`，拒绝 `NTD->TND`，
   `qkv` shape 合法。
2. checker 校验 dtype、shape、`query_start_loc` rank/长度、`seq_lens` rank/shape/dtype、`slot_mapping` 长度、cache/v_scale head 维一致性，并在 scene resolver 后要求 RoPE `v_scale=[Nv]`、M-RoPE `v_scale=[Nv,D]`；按 `layout_qkv` 校验 `qkv` shape、按 `layout_q_out` 校验 `q_out/q_scale` shape。
3. tiling 写入 kernel 必需字段：全局 token/tile 信息、head 数、可推导 GM stride 所需的 `T/D/qkvN`、cache stride、block 参数；`layout_qkv` 和 `layout_q_out` 生成 tiling key 常量，不写入 tiling data。
4. tiling 字段必须足够 controller 计算每个 `cubeIndex` 的 `TokenRange{begin,end}`，并生成本核组内的 token-size tile。
5. tiling 不把 UB/L1 静态 buffer 数和各固定 buffer 大小写入 tiling data；这些由 kernel 静态常量管理。
6. Host UT 覆盖 normal case、拒绝 case、边界 case 和典型 tiling 结果。

检查项：

1. Host UT 覆盖 `Nq > 64`、`Nq != 8 * Nk`、`Nk != Nv`、`D != 128`、缺失 `rotation/v_scale`、
   非法 `head_nums`、非法 `layout_qkv`、非法 `layout_q_out`、`layout_qkv=NTD,layout_q_out=TND`、
   非法 qkv shape/dtype。
2. Host UT 覆盖 tail 相关 tiling：`T < tokenTile`、`T % tokenTile != 0`。
3. Host UT 覆盖 `Nq=8*Nk`、`Nk=Nv` 且 `Nq<=64` 的完整 8 组合法 head tuple，tiling 结果与设计表
   一致，并覆盖 `NTD->NTD`、`TND->TND`、`TND->NTD` 下生成对应 `QKV_LAYOUT/Q_OUT_LAYOUT` tiling key。
4. Host UT 覆盖 controller 所需的 token range 边界、tile 生成边界和空 range 场景。
5. 新增 tiling key 模板参数有合法取值校验，非法 `layout_qkv` 不生成 kernel key。
6. Host 子任务提交时必须能单独通过 host UT，不能只依赖后续 kernel 任务验证。

### T01. Controller 与 basic block 骨架

需求：

1. 建立 controller 到 basic block 的执行入口。
2. controller 层负责按逻辑 cube group 划分 token range。
3. 同一 cube group 内的 AIC、AIV0、AIV1 必须获得相同 token range。
4. controller 入口集中完成 AIC/AIV 分派，basic block 暴露明确的 AIC/AIV 入口。
5. 统一组织 `Init`、`ComputeTile`、`End` 生命周期。

实现要点：

1. controller 根据当前核身份集中计算 `cubeIndex`：
   - AIC: `cubeIndex = GetBlockIdx()`
   - AIV: `cubeIndex = GetBlockIdx() / 2`
2. controller 使用 `cubeIndex` 对全局 token 做核组级 range 切分，生成 `TokenRange{begin,end}`。
3. controller 不按 AIV0/AIV1 再切核组 token range；AIV0/AIV1 的 token 子区间在 `FillTileParam` 中前置到 `TileParam::aivBlockTokenOffset/aivTokenOffset/aivTokenSize`。
4. controller 在 `TokenRange{begin,end}` 内展开 token-size tile 循环，并填充当前 tile 参数。
5. AIC 路径通过 `basicBlock.Init(tensors, tilingData)`、`PrepareBeforeLoop()`、循环内
   `ComputeTile(tile, lastTile)`、`End(lastTile)` 进入 basic block，再由 basic block 内部按
   `ASCEND_IS_AIC` 分派到 Cube。
6. AIV 路径使用同一组 basic block 公开入口，再由 basic block 内部按 `ASCEND_IS_AIC` 分派到 Vec。
7. lower layers 不直接使用 `g_coreType` 判断核类型。
8. `End` 内完成核内 event 结束配平。
9. `InitAic/InitAiv` 直接接收 tiling data 指针，并分别将同一指针传给 `Cube::Init/Vec::Init`；lower
   layer 复制自己需要的字段，不保存指针，也不在 BasicBlock 展开长标量参数列表。

检查项：

1. 同一个 `cubeIndex` 下，AIC、AIV0、AIV1 的 token range 完全一致。
2. 不同 `cubeIndex` 的 token range 覆盖全局 token 空间，且不重叠、不遗漏。
3. AIV local id 不参与 controller 级 token range 切分，只参与 tile 内 token slice 切分。
4. tail token tile 由 controller 填充 tile param 时处理，AIV0/AIV1 slice 仍在 tile 内处理。
5. AIC 分支不会执行 AIV vec 逻辑。
6. AIV 分支不会执行 AIC cube 逻辑。
7. `InitIntraCoreEvents` 和 `EndIntraCoreEvents` 成对出现。
8. controller 对空 range 行为明确，不调用单 tile compute，且不破坏同步协议。
9. basic block、cube、vec 层没有 `g_coreType` 判断。

### T02. AIV local slice 与 tail row range

需求：

1. 固定支持 `KERNEL_TYPE_MIX_AIC_1_2` 下的两个 AIV local slice。
2. AIV0/AIV1 对同一个 tile 做连续 token 切分。
3. tail 不允许在 A 矩阵 M 方向留下空洞。
4. 空 slice 也必须保留同步协议参与能力。

实现要点：

1. AIV 使用 `GetSubBlockIdx()` 区分 AIV0/AIV1。
2. `FillAivTileParam(tile, aivLocalId)` 只处理 tile 内切分，不改变 controller 传入的 token range。
3. tail 切分使用实际 token 数：

```text
actualAiv0 = ceil(actualTokenTile / 2)
actualAiv1 = actualTokenTile - actualAiv0

AIV0 rows = [0, actualAiv0 * qkN)
AIV1 rows = [actualAiv0 * qkN, actualTokenTile * qkN)
```

检查项：

1. AIV1 row 起点不能使用固定 `tokenTilePerAiv * qkN`。
2. tail 中 `actualAiv1 == 0` 时，AIV1 不写数据但仍参与同步。
3. Vec 有效后处理行数使用 `actualTokenTile * qkN`，但 AIC Cube/Fixpipe 物理行数使用 `cubeTokenSize * qkN`，pad 行参与 Cube/Fixpipe。
4. AIC Fixpipe 分发 C 输出时使用 `cubeHalfTokenSize` capacity 边界；Vec 只按 `aivTokenSize` 消费有效行。
5. AIV local slice 不影响核组级 token range。

### T03. 静态 buffer 与 tensor 绑定

需求：

1. 建立 L1/UB tensor 与 double buffer 的固定绑定。
2. 明确哪些 buffer 参与核间同步，哪些只参与核内同步。
3. `qk_after_cube_ub` 首期固定原地复用为 Q/K FP8 staging。

实现要点：

1. L1 tensor 包括 `rotation_l1_nz`、`a_rot_l1_nz`。
2. UB tensor 包括 `qk_after_cube_ub`、`qkv_ub`、`gamma_ub`、`cos_sin_ub[2]`、`slot_mapping_ub[2]`、`v_scale_ub`、`v_out_ub[2]`。
3. 数据 buffer 不再统一使用 `tileId & 1`。TileParam、A_rot、RoPE input/output/V output、M-RoPE
   input/raw cos-sin/output/K rotation/V output/K quant 和 L0C 分别由 owner 的 use counter 通过
   `% BUFFER_NUM` 选择，不使用要求槽数为 2 次幂的位掩码；
   `hasPreviousTile_` 只表示 rolling pipeline 是否存在上一拍。
4. Q/K dynamic quant 的 FP8 staging 不额外开独立区域，复用 `qk_after_cube_ub` 所在 output buffer；Q/K scale staging 复用 `v_out_ub` 对应的 float view。
5. typed GM tensor 绑定下沉到具体计算模块：Cube 只绑定 `rotation`/workspace 等 Cube 侧 GM，Vec 绑定 `qkv`、gamma、`cos_sin`、`slot_mapping`、cache、输出、`query_start_loc`、`seq_lens`、`v_scale` 等 AIV 侧 GM。
6. basic block 不保存统一的 `BoundGlobalTensors` 或共享 GM 绑定状态。
7. 删除通用 `QKV_K_SCALE_DOUBLE_BUFFER_NUM`，也不为每类物理资源在 common 新增一组全局
   `*_BUFFER_NUM`。common 只保留 pool/slot 容量；Controller/BasicBlock/Vec owner 在类内局部定义
   TileParam ring 深度或通过 `POOL_CAPACITY / ONE_SLOT_CAPACITY` 推导槽数。各资源 counter 仍独立，修改
   K quant 深度不得隐式改变 A_rot、output 或 TileParam ring。
8. M-RoPE 前处理 pool 与 RoPE 共用成员名 `outputDbPoolUb_`；不保留容易被理解成独立物理池的
   `qkPreprocessDbPoolUb_`。
9. Vec 的 input tensor 及其公共 pool 使用中性命名 `inputPoolUb_`、`QKV_K_SCALE_INPUT_POOL_*`；只有
   RoPE 单槽大小保留 `QKV_K_SCALE_ROPE_INPUT_ONE_BUFFER_*`。metadata/raw cos-sin 由独立
   `metadataBufferUseId_` 选择，M-RoPE raw cos/sin 的 4 槽由其 pool/slot 容量在 Vec 内局部推导。
10. `StageMropeK` 仅在 non-empty slice 实际占用 K quant slot 时推进 `kQuantBufferUseId_`；empty slice
    不推进该资源计数器，但仍完成固定 CrossCore 握手。

检查项：

1. `qk_after_cube_ub` 原地覆盖前，AIV 已完成对 FP32 C 的必要读取。
2. `v_out_ub` 独立 double buffer，不占 input/output DB pool。
3. `rotation_l1_nz` 只由 AIC 使用，不需要 AIV 等待。
4. `a_rot_l1_nz` / AL1 属于 Cube 侧资源，不占 AIV 核内 event id。
5. Cube/Vec 各自只绑定本模块实际使用的 GM tensor。
6. controller/basic block 不保留 typed bound-GM bundle。

### T04. CrossCore 封装与 flag id

需求：

1. 只使用固定语义 flag id。
2. 不按 `bufferId`、`generation` 或数组分配 CrossCore flag。
3. AIC 侧封装负责同时处理 AIV0/AIV1。

实现要点：

```cpp
constexpr uint64_t SYNC_A_READY = 0;
constexpr uint64_t SYNC_FIX_OUTPUT_READY = 1;
constexpr uint64_t SYNC_FIX_OUTPUT_CONSUMED = 2;
constexpr uint64_t SYNC_A_CONSUMED = 3;
constexpr uint64_t SYNC_MROPE_K_CONSUMED = 4;
```

1. AIV0 使用基础 id，AIV1 对应 `flag + QKV_K_SCALE_AIV1_FLAG_OFFSET`，当前 offset 为 16。
2. AIC 侧 AIV->AIC wait 语义函数 wait 两次：`syncFlag + QKV_K_SCALE_AIV1_FLAG_OFFSET`、`syncFlag`。
3. AIC 侧 AIC->AIV set 语义函数 set 两次：`syncFlag + QKV_K_SCALE_AIV1_FLAG_OFFSET`、`syncFlag`。
4. AIV 侧 AIC->AIV wait 语义函数只 wait 基础 id。

检查项：

1. RoPE 的 `SYNC_A_READY/FIX_OUTPUT_READY/FIX_OUTPUT_CONSUMED/A_CONSUMED`，以及 M-RoPE 的
   `SYNC_A_READY/FIX_OUTPUT_READY/MROPE_K_CONSUMED/A_CONSUMED`，其 Set/Wait 次数分别在 tile loop、
   initial credit 和 End drain 中配平；两个场景共享正向 Fixpipe-ready flag。
2. AIV 侧 set/wait 代码不手动加 `QKV_K_SCALE_AIV1_FLAG_OFFSET`。
3. `A_READY` 使用 `AIV Set<PIPE_MTE3>`、`AIC Wait<PIPE_MTE1>`。
4. `FIX_OUTPUT_READY` 在 RoPE/M-RoPE 均使用 `AIC Set<PIPE_FIX>`、`AIV Wait<PIPE_V>`；不再定义独立
   `MROPE_K_READY` flag 或封装函数。
5. RoPE 的 `FIX_OUTPUT_CONSUMED` 使用 `AIV Set<PIPE_MTE3>`、`AIC Wait<PIPE_FIX>`；M-RoPE compact
   使用 `MROPE_K_CONSUMED` 的 `V -> FIX`。

### T05. 核内 event 常量与 Init/End

需求：

1. 为 AIV/AIC 核内 buffer 复用定义 event id 常量。
2. 所有 event id 落在 `0..5`。
3. 每个反向同步在 init/end 中配平。

实现要点：

1. AIV `<PIPE_V, MTE3>` 正向 ready event：
   - V cache source ready: `EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0`
   - Q/K AL1 或 output source ready: `EVT_PIPE_V_TO_MTE3_QK_READY = 1`
2. AIV `<MTE3, PIPE_V>` 计划：
   - `v_out_ub[2]` reuse: `0/1`
   - `qk_after_cube_ub[2]` reuse: `2/3`
3. AIC event 参考 `MTE1/MTE2/MMAD/FIXP` pipe pair，最终按实际 buffer 数压缩到 `0..5`。
4. Init 对所有可复用 buffer 先 `Set<consumer, producer>`。
5. End 对所有可复用 buffer 最后 `Wait<consumer, producer>`。
6. 核内 event id 应封装在 AIV/AIC 计算类内部，Basic block 只分发生命周期调用，不直接维护具体 event id。
7. 当前 `PIPE_V -> MTE3` 使用两个固定语义 ready event：V cache 使用 id `0`，Q/K AL1 和 output 使用 id `1`；两者均不按 `bufferId` 派生。MTE3 完成后的 reusable 状态由 `<MTE3, PIPE_V>` 的 V output `0/1` 和 QK output `2/3` 表达。
8. 当前 AIC Init/End 只覆盖 L0 pingpong (`M_MTE1`，ids `3/4`) 的反向复用配平；`rotation_l1_nz` 固定常驻，不做 L1 反向复用初始化，L0C 当前未单独接入 `FIX_M` reusable event。

检查项：

1. 同一个 `<src_pipe, dst_pipe>` 内没有两个 live buffer 共用同一个 event id。
2. 不出现连续 `Set` 同一 event id。
3. 不为 4 个 AL1 物理 buffer 分配 AIV 核内 event id。
4. `EVT_PIPE_V_TO_MTE3_V_CACHE_READY` 和 `EVT_PIPE_V_TO_MTE3_QK_READY` 只在 Vec 内部对应写出前成对 Set/Wait，不进入生命周期 Init/End。
5. 所有 Init `Set` 都有 End `Wait` 配平。
6. 审计所有实际 active event id 均在 `0..5`，同一 pipe pair 内没有 live buffer 冲突，并确认 Init/End 及循环内 Set/Wait 配平；event id 范围 `static_assert` 不作为要求。

### T06. AIV Q/K/V 前处理

需求：

1. AIV 读取本地 slice 的 Q/K/V、gamma、cos/sin、slot_mapping。
2. 对 Q/K 做 RMSNorm、RoPE、BF16/NZ 转换。
3. MTE3 写 `a_rot_l1_nz` / AL1 完成后发送 `A_READY`。
4. V 分支从 input UB 读取，供提前后处理使用；input UB 的物理 staging 随 `layout_qkv` 变化，但 VF 使用统一的 `qkvUbStrideN/qkvUbStrideT` 逻辑 stride 访问。

实现要点：

1. AIV 本地 slice 已在 controller 填入 `TileParam`：`aivBlockTokenOffset/aivTokenOffset/aivTokenSize` 只覆盖本 AIV 的连续 token 范围。
2. input UB 的 `MTE2 -> PIPE_V` 正向同步保护输入可读，按 pingpong buffer 使用 `BASE + bufferId`。
3. input UB 的 `PIPE_V -> MTE2` 反向同步保护复用。
4. GM 到 UB 搬运使用封装的 input-layout aware copy helper，按 tiling key 的 `QKV_LAYOUT` 模板常量选择路径：`NTD` 按 N 轴分段搬入当前 token slice，`TND` 按 token 连续搬入；`cos_sin` 按 `query_start_loc` local 前缀和和 `seq_lens` 历史偏移解析出的连续 position run 分段搬运。
5. 调用 Q/K/V VF 前按 `QKV_LAYOUT` 模板常量计算 input UB stride：`NTD` 下 `qkvUbStrideN = sliceTokenSize * D`、`qkvUbStrideT = D`；`TND` 下 `qkvUbStrideT = qkvN * D`、`qkvUbStrideN = D`。
6. D=128 Q/K 数值路径通过 VF 完成 FP32 RMSNorm + half-RoPE + BF16 NZ staging。Q/K VF 使用 `n_qkv * qkvUbStrideN + localToken * qkvUbStrideT` 读取 input UB，Q 分支 `n_qkv = qHead`，K 分支 `n_qkv = Nq + kHead`。
7. RoPE 后写 `qRopeNz/kRopeNz` 时，Vec 按 `QKV_LAYOUT/Q_OUT_LAYOUT` 模板常量计算 input/output token/head stride，再传给 `QkRmsNormRopeD128SegmentNzVfImpl`；VF 内按 stride 直接寻址，不保留 runtime layout 分支。`NTD` 为 `localRow = headLocal * tile.cubeHalfTokenSize + localToken`，`TND` 为 `localRow = localToken * headSize + headLocal`，`ubNzOffset = dBlock * QK_PREPROCESS_UB_NZ_ROW_STRIDE * C0 + localRow * C0 + dInner`。`layout_qkv=TND,layout_q_out=NTD` 时，Q/K 使用 TND source stride 读取输入，但按 NTD row order 写 `qRopeNz/kRopeNz`。
8. MTE3 从 UB NZ 写 `A_rot_l1_nz` 时使用固定 Q/K segment：`copyRows = headSize * tile.cubeHalfTokenSize`，`dstRowOffset = tile.aivBlockTokenOffset * headSize`，`srcStride = QK_PREPROCESS_UB_NZ_ROW_STRIDE - copyRows`，`dstStride = AlignUp(tile.cubeTokenSize * headSize, C0) - copyRows`。
9. V VF 使用同一个 input UB 读取 `n_qkv = Nq + Nk + nv`，不能只改 V 分支而遗漏 Q/K 分支。
10. AL1 写入完成后调用 `SetAivMte3ToAicMte1AReady()`。
11. A 写 AL1 前用 `EVT_PIPE_V_TO_MTE3_QK_READY = 1` 表达源数据 ready；写入使用的 output DB slot 复用由 `EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE + outputBufferId` 保护。

检查项：

1. `A_READY` 不能早于 MTE3 写 AL1 完成。
2. AL1 到 AIC 读取的可见性由 `A_READY` 硬同步保证，不再额外分配 AIV AL1 event。
3. tail 中当前 AIV token 数为 0 时，仍执行固定同步流程。
4. V 分支不参与 RMSNorm/RoPE/rotation matmul，只做后续 `V * v_scale`。
5. 当前实现阶段已接通 QKV/gamma/slot_mapping/cos-sin staging、D=128 RMSNorm/RoPE VF、BF16 ND2NZ 写 AL1、AL1 handoff 和 `A_READY`；preprocess 数值验证统一通过 DiscussionForum NPU golden/TTK 用例执行。

### T07. AIC rotation 常驻与 Cube matmul

需求：

1. AIC 在 tile 循环前通过 MTE2 搬入 `rotation_l1_nz` 并常驻。
2. AIC 等两个 AIV 的 `A_READY` 后读取 AL1。
3. 执行 BF16 x BF16 -> FP32 Cube matmul。

实现要点：

1. `rotation` 是外部 dense BF16 `[D,D]`，Q/K segment 共享同一个右矩阵；kernel 不假设 Hadamard 结构。
2. `PrepareRotationBeforeLoop()` 完成一次 ND2NZ 搬入、一次 MTE1 加载和 AIC 核内同步。
3. `WaitAivMte3ToAicMte1AReady()` 通过后再启动 MTE1/Cube 读取 `a_rot_l1_nz`。
4. AIC 内部 L1/L0/L0C 同步属于 AIC 核内流水管理。

检查项：

1. `rotation_l1_nz` 不需要核间 flag。
2. AIC 对 `SYNC_A_READY` 每 tile wait 两次。
3. AIC 不在两个 AIV ready 前启动完整 `M_tile` matmul。
4. AIC 内部 event id 不把 AL1 物理 buffer 当成 AIV event。

### T08. AIC Fixpipe 分发与 FIX_OUTPUT_READY

需求：

1. AIC 将 L0C/C 输出通过 Fixpipe 分发到两个 AIV 的 UB。
2. 按 M 方向 row range 分发，边界使用 `cubeHalfTokenSize` capacity；odd tail 的 pad 行也随完整 segment 搬到 AIV UB。
3. Fixpipe 输出完成后通知两个 AIV `FIX_OUTPUT_READY`；RoPE 的 Q/K output 与 M-RoPE 的 K rotation
   复用同一正向语义。

实现要点：

1. AIV0 C capacity rows 对应 `[0, cubeHalfTokenSize * qkN)`。
2. AIV1 C capacity rows 对应 `[cubeHalfTokenSize * qkN, cubeTokenSize * qkN)`；其中超出 `actualTokenTile` 的部分是 pad rows。
3. `SetAicFixToAivOutputReady()` 已足够表达 Fixpipe 输出完成后再通知 AIV；不增加独立
   `SetAicFixToAivMropeKReady()`。
4. AIC 内部 `MMAD -> FIXP`、`FIXP -> MMAD` event 只属于 AIC 核内同步。

检查项：

1. C 输出分发可以使用 `cubeHalfTokenSize` capacity 边界，但 Vec 后处理不能把 pad rows 当有效行。
2. `FIX_OUTPUT_READY` 每 tile AIC set 两次。
3. AIV 收到 `FIX_OUTPUT_READY` 前不能读场景对应的 Fixpipe output。
4. RoPE AIC 在 `FIX_OUTPUT_CONSUMED` 前不能复用对应 output buffer；M-RoPE AIC 在
   `MROPE_K_CONSUMED` 前不能复用对应 K rotation buffer。

### T09. V 分支提前后处理

需求：

1. 当前实现采用 V-first pipeline：AIV 在 `Vec::ComputeTile` 内先执行 V quant，再执行 Q/K BF16 NZ staging，随后写 `v_cache`、搬运 Q/K AL1，并由 BasicBlock 发送当前 tile `A_READY`。V-first 表示 V vector quant 先发起，不表示 `v_cache` 写回先于 Q/K preprocess。
2. V 分支不依赖 Cube 输出。
3. V 分支写 `v_cache`。

实现要点：

1. 从本 AIV slice 对应的 input UB 读取 V 行，并读取 `slot_mapping_ub`；V 行 offset 使用 `n_qkv * qkvUbStrideN + localToken * qkvUbStrideT`。
2. 按编译期场景执行缩放：RoPE 为 `v[t,nv,d] * v_scale[nv]`，M-RoPE 为
   `v[t,nv,d] * v_scale[nv,d]`。
3. 按 RINT/SAT cast 到 FP8 E4M3FN。
4. 使用 `v_out_ub[bufferId]` 作为 MTE3 写 GM/cache 前的 double buffer。
5. 使用 `PIPE_V -> MTE3` 和 `MTE3 -> PIPE_V` event 保护 `v_out_ub`。

检查项：

1. V 分支不等待 `FIX_OUTPUT_READY`。
2. `v_out_ub` event id 使用 `0/1`，不与 Q/K output event 冲突。
3. `slot_mapping` 与 Q/K cache 写回使用同一 token slot 语义。
4. `v_scale == 0` 行为与 FP8 cast 路径一致，不新增额外语义分支。
5. M-RoPE 使用 channel-wise 非均匀 `[Nv,D]` 输入，确认 VF 没有把第 0 通道或单个 head scale 广播到整行。

### T10. AIV Q/K 后处理

需求：

1. AIV 等 `FIX_OUTPUT_READY` 后消费上一 tile 的 `qk_after_cube_ub`。
2. 对 Q/K row 做 dynamic quant。
3. `qk_after_cube_ub` 原地覆盖复用为 FP8 staging。
4. Q/K 后处理相关 MTE3 写出完成后发送 `FIX_OUTPUT_CONSUMED`。

实现要点：

1. `WaitAicFixToAivOutputReady()` 通过后，`PIPE_V` 读取 `qk_after_cube_ub`。
2. 对每个 row 计算 `max_abs`、`scale = max_abs / FP8_E4M3FN_MAX`。
3. 使用 `Div -> RINT/SAT cast FP8 E4M3FN`。
4. Q 分支语义上按 `layout_q_out` 写 `q_out` 和 `q_scale`：`NTD` 写 `[nq,t,d]`/`[nq,t]`，`TND` 写 `[t,nq,d]`/`[t,nq]`。kernel 按 `Q_OUT_LAYOUT` 模板常量选择写回。Q FP8/scale staging 也应按目标 layout 组织，不能固定 `qScaleNtd` 后再统一写回。
5. Q dynamic quant 需要两个 VF 入口：`QDynamicQuantD128NtdVfImpl`、`QDynamicQuantD128TndVfImpl`，由 `Q_OUT_LAYOUT` 模板常量选择。`layout_qkv=TND,layout_q_out=NTD` 的 Q/K row order 转换已在 RmsNormRopeNz 阶段完成，dynamic quant 直接按 NTD source row order 读取，不新增 TND-to-NTD VF 入口。
6. K dynamic quant 同样由 `Q_OUT_LAYOUT` 模板常量选择 NTD/TND source row order。K cache 写回仍按 cache slot/head/token 目标布局 scatter，`layout_q_out` 不改变 `k_cache/k_scale_cache` 的对外 shape 或 stride。
7. MTE3 写出完成后 `SetAivMte3ToAicFixOutputConsumed()`。

检查项：

1. `qk_after_cube_ub` 原地覆盖前，FP32 C 已完成当前 row 的 scale/quant 读取；`TND->NTD` 路径必须确认 RmsNormRopeNz row order 和 Q/K 后处理 dispatch 均按 `layout_q_out=NTD`，不在 dynamic quant 阶段做原地转置。
2. Q/K output event id 使用 `2/3`，不与 V output event 冲突。
3. `FIX_OUTPUT_CONSUMED` 不能早于 Q/K 后处理 MTE3 写出完成。

### T11. Rolling pipeline 与 drain

需求：

1. 支持 AIV 前处理当前 tile、后处理上一 tile 的 rolling pipeline。
2. 支持最后一个 tile 的 drain。
3. controller 使用两槽 tile param ring 传 current/last slot，basic block 不额外保存跨控制迭代的 K tile
   状态。
4. 保证 pingpong buffer 的同步事件按固定顺序消费。

实现要点：

1. AIV 每轮在 `Vec::ComputeTile` 内按 `QuantVToFp8 -> BuildRopeBf16NzUb -> ScatterVCache -> Q/K AL1 copy` 处理 tile i。
2. `Vec::ComputeTile` 返回后 BasicBlock 发送 `A_READY(i)`。
3. basic block 使用 `hasPreviousTile_` 判断上一 tile 是否存在；它不是 use counter，不参与任何 physical
   buffer id 计算。上一 tile 存在时，用 output resource 自己的 counter 定位 slot，等待上一 tile
   `FIX_OUTPUT_READY` 并执行 Q/K 后处理。
4. `EndAiv(lastTile)` drain 最后一个 tile 的 Q/K 后处理和 `FIX_OUTPUT_CONSUMED`。
5. RoPE output pool 有两个物理槽。AIV 在 loop 前提供两个初始 `FIX_OUTPUT_CONSUMED` credit，AIC 每个
   tile 开始前等待一个；AIV 在 tile 内或 End 完成 Q/K 后处理后返还一个，`EndAic()` 再 drain 两个。
   不保留 `isLastTile` 特判。
6. M-RoPE 同样保留一拍 current/last rolling。上一拍存在时，顺序固定为
   `StageMropeK(lastTile) -> Set MROPE_K_CONSUMED -> ScatterMropeK(lastTile)`，三个动作位于同一控制
   迭代。K quant 仍为双缓冲，Stage 选择 slot 前等待其 `MTE3 -> V` reusable event，Scatter 写完 K
   cache/K scale 后返还该 event。

检查项：

1. tile 0 不等待不存在的上一 tile `FIX_OUTPUT_READY`。
2. 最后一个 tile 必须 drain，否则 AIC 会等待 `FIX_OUTPUT_CONSUMED`。
3. `bufferId` 复用只依赖固定顺序和 flag 计数，不派生 flag id。
4. AIC 不能在对应输出 DB buffer 的 `FIX_OUTPUT_CONSUMED` 到达前复用该 buffer。
5. 单 buffer 退化时同步语义仍成立。

### T12. tail 与空任务同步

需求：

1. tail 只改变数据范围，不改变同步次数。
2. 某个 AIV 实际 token 数为 0 时，仍参与场景对应的固定握手：RoPE 为
   `A_READY/FIX_OUTPUT_READY/FIX_OUTPUT_CONSUMED`，M-RoPE 为
   `A_READY/FIX_OUTPUT_READY/MROPE_K_CONSUMED`。

实现要点：

1. `FillAivTileParam` 允许填出 `aivTokenSize == 0` 的空 slice。
2. 空 slice 跳过实际数据计算，但不跳过 CrossCore 协议。
3. AIC 仍等待两个 AIV 的 `A_READY`。
4. 两个场景下 AIC 都仍 set 两个 AIV 的 `FIX_OUTPUT_READY`。
5. RoPE 下 AIC 仍等待两个 AIV 的 `FIX_OUTPUT_CONSUMED`；M-RoPE compact 下按固定协议等待
   `MROPE_K_CONSUMED` credit。

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
   - `FIX_OUTPUT_READY`: AIC set 2 次，AIV wait 合计 2 次。
   - 当前场景 consumed：两个 AIV 各 set 1 次，AIC 对两个 AIV 各 wait 1 次。
2. 对每个核内 event 统计 Init/End 配平。
3. 检查所有 event id 是否在 `0..5`。
4. 检查同 pipe pair 内 event id 是否唯一覆盖 live buffer。

检查项：

1. 没有 `flag = base + bufferId` 或 `flag = base + generation`。
2. 没有连续 `Set` 同一个核内 event id。
3. 没有遗漏 tail/empty-work 的 Set/Wait。
4. CrossCore pipe pair 与设计一致。
5. AL1 不占 AIV 核内 event id。

### T14. DiscussionForum Kernel 验证

需求：

1. 使用 DiscussionForum NPU golden/TTK 覆盖数值正确性、layout、tail、多 tile 和 cache stride。
2. 覆盖容易死锁或错位消费的边界。
3. `ops-transformer` 内不新增 `tests/ut/op_kernel` 或 ST target。

实现要点：

1. NPU golden 入口：`designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py` 和 `assets/qkv_rms_norm_rope_cache_with_k_scale_precision.py`。
2. TTK case 定义：`GEN_CASE/qkv_rms_norm_rope_cache_with_k_scale/test_cases/*.csv`，通过外部 TTK harness 执行。
3. 单 tile 覆盖 preprocess、Cube、postprocess；多 tile 覆盖 buffer 0/1 pingpong 和 tile 2 复用 buffer 0。
4. tail 覆盖奇数 tail、`actualAiv1 == 0`、`actualAiv0 != actualAiv1`。
5. 端到端覆盖 random dense `rotation`；内部 Set/Wait 数量、顺序和 event 生命周期由 T13 静态设计审计保证。

检查项：

1. 每个输出 tensor 与 golden 对齐。
2. `slot_mapping` 连续、非连续、跨 block 均覆盖。
3. 多 batch 且 tile 跨 `query_start_loc` 边界时 RoPE 位置正确，并覆盖 `seq_lens` 使跨 batch position 不连续的场景。
4. `v_scale == 0` 覆盖；M-RoPE 另覆盖每个 head/channel 均不同的 `[Nv,D]` scale。
5. 长序列多 tile 不死锁，不错消费上一轮 flag。
6. NPU 入口依赖已安装算子、`torch_npu` 和可用 NPU；验证入口存在不等于当前 worktree 已执行通过。

## 4. 交付检查清单

实现提交前逐项确认：

1. `05_kernel_design.md` 中 rolling pipeline 与代码一致。
2. `06_inter_core_sync_design.md` 中两个场景共享的 `FIX_OUTPUT_READY`、RoPE 的
   `FIX_OUTPUT_CONSUMED` 以及 M-RoPE 的 `MROPE_K_CONSUMED` 与代码一致。
3. `07_intra_core_sync_design.md` 中 event id 常量、Init/End、pipe pair 与代码一致。
4. 所有 CrossCore 基础 flag id 为 `0..4`：两个场景共享 `0/1/3`，RoPE 使用 `2` 的
   `FIX_OUTPUT_CONSUMED`，M-RoPE 使用 `4` 的 `MROPE_K_CONSUMED`。
5. 所有核内 event id 仍在 `0..5`。
6. AIV0/AIV1 tail row range 连续无空洞。
7. `qk_after_cube_ub` 原地复用没有引入生命周期重叠。
8. `rotation_l1_nz` 只由 AIC 内部管理，不引入 rotation 核间 flag。
9. AL1 到 AIC 读取由 `A_READY` 硬同步保证，不分配 AIV 核内 event id。
10. DiscussionForum NPU golden/TTK 覆盖 tail、empty slice、pingpong 复用和多 tile drain。

## 5. 子任务提交门禁

每个子任务完成后必须满足对应门禁：

1. T00 必须通过 Host UT；checker/tiling 变更没有 UT 不允许合入。
2. T01-T13 每个 kernel 子任务提交后都必须保持 kernel 编译通过。
3. T02、T12 涉及 tiling/tail contract 的改动，必须同步扩展 Host UT。
4. T04、T08、T11、T13 涉及同步协议的改动，当前阶段不要求测试，但必须维持 kernel 编译通过，并在文档或代码审计记录中说明 Set/Wait 次数。
5. T05 涉及核内 event 的改动，必须审计实际 event id 范围、同 pipe pair 冲突以及 Init/End 和循环内 Set/Wait 配平；不要求范围 `static_assert`。
6. T06-T10 涉及数值路径的改动，必须有可复用 golden 对比，并统一通过 DiscussionForum NPU golden/TTK 入口执行。
7. T14 完成前必须通过 DiscussionForum 端到端 NPU 用例；在此之前，各子任务不能依赖“最后一起修”来维持编译。

## 6. 当前实现状态

| 范围 | 状态 | 说明 |
|------|------|------|
| T00-T04 | 已落地 | Host contract、controller、AIV slice、静态 buffer、RoPE 基础边和 M-RoPE K 边均有代码对应。 |
| T05 | 已落地 | active event id 当前均在 `0..5`；同 pipe pair 分配无 live buffer 冲突，Init/End 与循环内 Set/Wait 已配平；范围 `static_assert` 不作为门禁。 |
| T06-T13 | 已落地主路径 | V-first、Q/K preprocess、Cube/Fixpipe、Q/K postprocess、rolling pipeline、tail/empty slice 和同步协议已有实现；数值回归归 DiscussionForum NPU golden/TTK 用例。 |
| T14 | 代码路径已落地；关键回归为历史记录 | 历史 DiscussionForum NPU golden 覆盖 RoPE decode、prefill、显式空 `mrope_section=[]` 和 `TND->NTD`；本 worktree 未在本次文档同步中重新执行 NPU。`ops-transformer` 不要求 `op_kernel` UT/ST target。 |

该表区分“代码路径已接通”和“当前 worktree 已实跑验证”。历史结果及验证入口存在都不能替代当前 worktree 重新构建并执行 DiscussionForum NPU golden/TTK。

## 7. M-RoPE 组合模式子任务

以下子任务共同落实 M-RoPE 的最终功能合同。

| ID | 子任务 | 关键门禁 |
|----|--------|----------|
| T15 | OpDef/ACLNN/Torch schema | input 8/9 改为 OPTIONAL，五个输出均为 REQUIRED；OpDef 新增 Int 属性 `q_out_dtype`，Torch public/schema 使用 `torch.dtype`/`ScalarType?`；ACLNN 无独立 `qOutDtype`，从 qOut tensor 读取；三类相关向量等长；Torch 两套 schema 的第 2 项均声明为 `Tensor` |
| T16 | Checker/InferShape/InferDtype | 由 `Q/S/P/M` presence 匹配候选场景后校验完整合同；ACLNN 校验 qOut dtype 与 q_quant_mode，graph direct 额外校验属性 6 与 qOut descriptor 一致；InferShape 按固定索引推导五个必选输出，InferDtype 固定设置 q_scale 为 FP32 |
| T17 | Tiling key/data/UB | resolver 先于 shape/batch 派生；host presence -> `ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE`；M-RoPE 只走 compact，资源不满足时缩小 token tile，最小 tile 仍不可行才拒绝；RoPE 3 key + M-RoPE 1 key，共 4 个注册选择；M-RoPE `batch=0` 且编译期不加载；变更前 compact 为 222 KiB |
| T18 | M-RoPE Vec/VF | 三个 mode 编译期隔离位置、K 量化和 Q 量化分支；T/H/W 全行搬运；`mropeSectionH/W` 分控；RegBase Gather 元素索引 |
| T19 | BF16 Q/INT8 K 后处理 | `NoQuant` Q 由 Fixpipe F32->BF16 直写 GM，删除 Vec QCast；K `/127`；Stage 后同迭代 Scatter；cache scatter/layout/tail 不回退 |
| T20 | DiscussionForum golden/TTK | 单一 compact 模板覆盖常规与 `T=45000/65536/131072` 大输出；两个合法 presence signature、所有 M-RoPE case 默认使用可复现随机 `mrope_position[T,3]`、`T=5` token-major 行布局、`mrope_section[0]` 不变量、H/W 配置不等、量化模式/BF16/INT8、RoPE 场景、多 tile 无死锁 |
| T21 | 用户 API 文档与最终同步 | 保持稳定 API 文档名，更新 optional input、required output、presence 判定和组合限制，实跑结果与历史记录分离 |
| T22 | M-RoPE `[Nv,D]` V scale | scene-aware checker/base tiling shape；compact reserve 40 KiB 和 4 KiB scale UB；逐通道 VF；Golden/API/Host/NPU 正反例；RoPE `[Nv]` 回归 |

提交门禁：

1. T15-T17 需要 Host/Tiling/API UT；不新增 InferShape 完整合同校验。
2. T15-T17 必须覆盖全部 16 种 `query_start_loc/seq_lens/mrope_position/mrope_section` effective presence
   mask；其中 `M` 表示 `mrope_section` 非空，只允许 `1100` 和 `0011` 进入后续合同校验。空 tensor、
   undefined tensor 或 shape 缺失不能伪装成 absent；空 `mrope_section=[]` 等价于未传 section，必须覆盖
   其配合合法 RoPE 位置输入的成功路径。
3. T15 必须验证 inplace/functional schema 的 nullable 输入与 `Tensor` 类型第 2 项返回；所有 tensor 的
   DataType/Format/UnknownShapeFormat 相关向量长度一致，新增 input 导致的 flat GM 序号顺延已同步入口。
   Torch public API 必须以 `torch.dtype` 接收 q_out dtype；RoPE 运行时返回 FP32 q_scale，M-RoPE 运行时
   返回 `None`。Torch 不校验量化模式、dtype、shape 或 presence 的完整场景合同，这些门禁由 ACLNN
   checker 负责。
4. T18-T19 每步保持 kernel 编译通过，数值只通过 DiscussionForum NPU golden/TTK 验证。
5. T18 不得把 `mrope_section[0]` 写入 Gather id，也不得复用 op-plugin golden 中“W 误用 H 项 `mrope_section[1]`”的历史错误。
6. T19 的 BF16 模板遵循 T15 定义的 M-RoPE q_scale 接口合同。
7. host 必须根据 optional effective presence 选择三个 tiling key mode 并校验
   `q_quant_mode/q_out_dtype`；其中 ACLNN 从 qOut dtype 派生内部属性，graph direct 使用属性 6。T17-T19
   不得在 kernel 中根据量化模式、dtype、optional GM 空地址、mrope_section 或 tiling data flag 动态选择
   分支，位置、K 量化和 Q 量化只能来自对应模板常量。
8. T22 不增加 runtime scene flag；已有 `ROPE_MODE` 编译期分支同时决定 V scale 的 shape 合同和 VF 读取方式。
   M-RoPE 必须精确读取 `[head * D + d]`，RoPE 必须保持 `[head]` 广播行为。

当前 `6ca8b3400` 已重新构建并通过 Host/Tiling UT `120/120`、ACLNN UT `128/128`；DiscussionForum
NPU 全量 `185/185`，section-0 不变量比较 `3/3`。大输出 E2E 为
`Nq/Nk/Nv=16/2/2,T=45000`，Q+K+V+K-scale
实际写入约 198.10 MiB，四路输出均 100% 对齐。16 MiB workspace 属于框架强制预留。

T22 已完整落地：M-RoPE 使用 `[Nv,D]` 逐通道 scale，RoPE 保持 `[Nv]` 广播；Host/API shape
反例、正式 package 构建和 NPU 实跑均已完成。单轮 M-RoPE `T=5379` task duration 相对 T22 前基线
增加 `0.883 us`（`+2.21%`），7 个 Rope 对照在 `-0.93%..+0.16%`。
