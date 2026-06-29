# 2026-06-23 收敛为 W4-only 基线

- 按最新讨论将当前实现和 spec 从 `W=2/3/4` 收敛为仅支持 `W=4`。
- 修改 `op_host/causal_conv1d_decode_tiling.cpp`：
  - `weight` 形状校验改为固定 `[4, dim]`。
  - 普通 decode `state_len >= 3`；存在 `num_accepted_tokens` 时要求 `state_len >= seqLen + 2`。
- 修改 `op_kernel/arch35/causal_conv1d_decode_cut_bh.h`：
  - `CopyWeightBias` 固定搬 4 行 weight。
  - `conv_states` 固定搬 3 行历史窗口。
  - single-S 只调用 `Conv1dDecodeW4`，multi-S 只调用 `Conv1dDecodeW4MultiS`。
- 修改 `op_kernel/arch35/vf/causal_conv1d_decode_width4_vf.h`：
  - 删除 W2/W3 VF 死代码和 wrapper 分发，保留 W4 single-S 与 W4 multi-S。
- 修改 `tests/reference_causal_conv1d_decode.py`：
  - reference 只接受 `weight.shape[0] == 4`，smoke 只覆盖 W4。
- 删除早期未接入 CMake target 且引用过期 helper 的 `tests/ut/op_host/test_causal_conv1d_decode_tiling.cpp`；当前本地轻量验证以 Python reference smoke 为准，CANN/op-host 编译仍需目标环境。
- 同步更新需求、spec、dataflow、API 和 vLLM 对齐报告中的当前范围描述；历史记录保留旧 W2/W3 扩展尝试，作为回溯信息。

# 2026-06-22 spec/流程图数据流表述收敛

- 更新 `reports/baseline_alignment_and_flow.zh.md` 第 5 节：
  - 将原 Mermaid 总流程图替换为纯文本数据流，按 `GM tensors -> UB buffers -> Reg/VF -> GM outputs/side effects` 表达。
  - 明确 `[T,D]` 只是变长输入的存储形态，核间不把 `B*S` 或 `B*T` 作为分核轴；当前主路径仍为 `batch x dim`。
  - 明确 `state_ub` 只搬 `W-1` 行历史窗口，`weight_ub/bias_ub` 每个 dimLoop 搬一次并跨 batch 复用。
  - 明确 VF 按 64 个 `d_local` 一组计算，在每个 window tap `k` 上做 64-dim 向量乘加。
- 新增 PNG：
  - `reports/images/causal_conv1d_decode_overall_flow_text.png`
- 同步更新：
  - `reports/spec.zh.md` / `.md`
  - `specs/ascendc_operator_dataflow_spec.zh.md` / `.md`
  - `work/causal_conv1d_decode_bf16/specs/causal_conv1d_decode_bf16.md`
- 清理部分容易误导的 `flatten` 表述，将 batch-group x 搬运描述为 row copy/storage rows，而不是核间分核轴。

# 2026-06-20 vLLM Ascend PR 8215 decode 语义审计

- 已 clone `vllm-project/vllm-ascend` 并 checkout `pull/8215/head` 到本地 `external/vllm-ascend-pr8215`。
- 只审计 `csrc/causal_conv1d_v310` 和 `_310p/ops/fla/gdn_310.py` 中的 decode/update 路径。
- 新增 `reports/vllm_ascend_pr8215_decode_audit.zh.md` / `.md`，记录原实现三种 decode 输入模式：
  - `[batch, dim]`：非 MTP 单 token decode。
  - `[batch, seqlen, dim]`：定长 MTP/speculative decode。
  - `[cuSeqlen, dim] + queryStartLoc`：变长 flatten decode。
- 修正 `reports/spec.zh.md` / `.md` 和 `docs/REQUIREMENTS.zh.md` / `.en.md`：
  - 将输入模式从旧的 `[B,S,D]` / `[T,D]` 二分法改成 `[B,D]`、`[B,S,D]`、`[T,D]+query_start_loc` 三类。
  - 将 flatten token 范围从旧口径 `0..4` 改为 decode 业务口径 `1..4`，并保留 host 无法静态检查运行时差分的说明。
  - 当时标注 `[B,D]` 非 MTP decode 尚未完整支持，后续已在 2026-06-21 修正。
  - 当时标注 `num_accepted_tokens` 尚未影响 state 历史读取窗口，后续已在 2026-06-21 修正；state writeback 仍有差距。
- 明确当前实现的 batch-group flatten rows 一次搬运方案可行但风险更高：它依赖 `query_start_loc` 单调、batch group 内 row 连续、token 数合法，以及 compute 正确还原 batch 在 UB 中的偏移；vLLM 原实现每个 task 只处理一个 batch segment，边界更直接。

# 2026-06-21 decode 对齐代码修正

- 修改 `op_host/causal_conv1d_decode_tiling.cpp`：
  - optional input 判定改为识别空 shape，避免把空 optional tensor 误认为存在。
  - 2D `x` 无有效 `query_start_loc` 时走 `[B,D]` 非 MTP decode，`seqLen=1`。
  - 2D `x` 有有效 `query_start_loc` 时继续走 flatten `[T,D]` decode。
- 修改 `op_kernel/arch35/causal_conv1d_decode_cut_bh.h`：
  - 增加 `GetHistoryBase(accepted)`。
  - `num_accepted_tokens` 存在时，用 `accepted - 1` clamp 后作为 state history window 起点，影响 `y` 的卷积历史输入。
  - `num_accepted_tokens` 不存在时，普通 decode 使用 `conv_states[slot, 0:W-1]` 作为持久历史窗口。
- 修改 `tests/reference_causal_conv1d_decode.py`：
  - 新增 `[B,D]` non-MTP decode reference。
  - 新增 accepted count 影响 history window 的 reference 语义。
  - Python smoke 已通过 width 2/3/4 的 `[B,D]`、`[B,S,D]`、`[T,D]+query_start_loc`。
- 后续补充：
  - 已继续补齐 vLLM Ascend PR 8215 width=4 speculative state writeback：保留 `accepted-1` 选择出的两行历史，从 state row 2 开始写入当前 token。
  - CANN 编译仍未验证。
- 分核路线确认：
  - 当前功能对齐主路径继续采用 BH：`batch x dim` 分核，token/S 核内循环。
  - BSH 不进入当前主路径，只保留为未来性能实验候选，主要面向长 token、prefill/mixed 或 `B x dimBlocks` 不足以喂饱 core 的形状。

# 2026-06-21 multi-S VF 优化候选

- 新增并随后拆分为 `Conv1dDecodeW2/W3/W4MultiSVF` / `Conv1dDecodeMultiS`：
  - `tokenCount == 1` 继续走原 single-S VF。
  - `tokenCount >= 2` 时一次 VF 最多维护 4 个 FP32 accumulator，同算同一 batch 的最多 4 个 token row。
  - multi-S VF 复用同一组 weight 和重叠的 state/x timeline，减少重复 weight cast 和多次 VF 调用开销。
- 修改 `causal_conv1d_decode_cut_bh.h`：在 batch 内构造 `[history..., x0, x1, x2, x3]` timeline，传入 multi-S VF。
- 已将 VF 乘加路径按 W 拆成 `Conv1dDecodeW2/W3/W4` 和 `Conv1dDecodeW2/W3/W4MultiS`：
  - kernel/AICore wrapper 层根据 `kernelWidth` 做一次分发。
  - `__simd_vf__` 内不再用 `kernelWidth >= 3/4` 控制有效 tap，避免在 VF 内循环里保留 W 维度分支。
  - `tokenCount >= 2/3/4` 的 S 维度分支仍保留在 multi-S VF 内，后续是否拆 S 专用函数需要 CANN profiling 决定。
- CANN 编译和 profiling 仍未验证。

# 2026-06-21 decode state W-1 搬运修正

- 修改 `causal_conv1d_decode_cut_bh.h`：
  - `stateQueueIn` 的 UB buffer 从按 `stateLen` 预留改为按最大 `W-1=3` 行历史预留。
  - `Compute` 中每个 batch 只从 GM 搬 `W-1` 行历史窗口到 UB：
    - 无 `num_accepted_tokens` 时从 `conv_states[slot, 0:W-1]` 开始搬。
    - 有 `num_accepted_tokens` 时从 `clamp(accepted - 1, 0, stateLen - (W - 1))` 开始搬。
  - UB 内 `stateLocal[0..W-2]` 现在表示紧凑历史窗口，计算路径不再用完整 state 偏移索引。
  - 普通 decode state update 改为写回最新 `W-1` 行历史，而不是完整 `state_len` 滑窗。
  - W=4 speculative writeback 保持 vLLM-style layout：`state[0:2]` 写紧凑历史窗口的后两行，`state[2:2+token_count]` 写本次 token chunk。
- 修改 `causal_conv1d_decode_tiling.cpp`：
  - UB 估算按最多 3 行 state 历史计算。
  - 普通 decode 的 `state_len` 最小要求放宽为 `W-1`；W=4 speculative layout 仍要求 `state_len >= 2 + seqLen`。
- 修改 CPU reference：普通路径只更新 `conv_states[slot, 0:W-1]` 为最新 `W-1` 行历史。
- CANN 编译仍未验证。

# 2026-06-21 dimLoop 外层 weight/bias 复用

- 修改 `causal_conv1d_decode_cut_bh.h`：
  - 核内循环从 `batchLoop -> dimLoop` 改为 `dimLoop -> batchLoop`。
  - 拆分原 `CopyIn(x + weight + bias)`：
    - `CopyWeightBias()`：每个 dimLoop 只搬一次 `weight[W, dimTile]` 和可选 `bias[dimTile]` 到 `weightBiasBuf`。
    - `CopyX()`：每个 batchLoop 搬当前 batch tile 的 x 到 `xQueue`。
  - `Compute()` 接收当前 dimLoop 的 `weightBiasLocal`，在所有 batchLoop 内复用同一份 weight/bias。
  - state 搬入、y 写回、conv_states 更新语义不变。
- 修改 `docs/REQUIREMENTS.zh.md` / `.en.md`、`reports/spec.zh.md` / `.md`、`reports/tiling_strategy.zh.md` / `.md`、`specs/ascendc_operator_dataflow_spec.zh.md` / `.md`：
  - 同步记录 `dimLoop -> batchLoop` 循环顺序和 `weightBiasBuf` 生命周期。
- Python reference smoke 仍通过；CANN 编译和 profiling 仍未验证。

# 2026-06-21 基线对齐报告

- 新增 `reports/baseline_alignment_and_flow.zh.md`：
  - 明确功能基线对齐 vLLM Ascend PR 8215 decode/update，接口和 A5 数据流对齐 ops-transformer。
  - 记录有意差异：BF16、INT32 metadata、decode-only、不支持 `initialStateMode/runMode`、v1 只支持 `activation_mode=0`。
  - 补充总体流程图、核间/核内分核、GM->UB->VF->GM 数据流。
  - 补充变长 `[T,D]+query_start_loc`、`W=4`、`state_len=6` 的完整 case。
- 修正早期 `reports/summary.zh.md` 和 vLLM audit 中会误导当前基线的旧口径。

# 2026-06-17 文档图示补充

- 在 `docs/REQUIREMENTS.zh.wiki` / `docs/REQUIREMENTS.en.wiki` 中补充分核、GM/UB 搬运连续性、`conv_states` slot 搬运、状态写回、VF 64-lane + tail mask 的图示。
- 新增标准 Markdown 版本 `docs/REQUIREMENTS.zh.md` / `docs/REQUIREMENTS.en.md`，避免 wiki 格式无法正确渲染 Mermaid、Markdown 表格和代码图。
- 新增按 `ascendc-docs-gen` 结构组织的需求说明书 `docs/REQUIREMENTS.ascendc-docs-gen.zh.md` / `.en.md`，并把详细计算语义、分核和内存排布放入附录。
- 重写 `specs/ascendc_operator_dataflow_spec.zh.md` / `.md`，用 Mermaid 和 ASCII 内存排布图表达 GM -> UB -> VF/register -> UB -> GM 的完整路径。
- 在 `reports/tiling_strategy.zh.md` / `.md` 中补充二维 core 分核图、核内 UB tile 图、GM 行内连续/行间 stride 图、VF 子切分图和状态更新图。
- 在 `reports/spec.zh.md` / `.md` 的分核摘要里加入缩略图，避免 workflow 只读摘要时漏掉 VF 内循环。
- 修正 dataflow/tiling 文档旧口径：当前 width 为 2..4，`conv_states` 是原地副作用，SiLU 不是 v1 当前调优项。

# 2026-06-15 远程迁移与 BF16 vf 更新

- 已将 `attention/causal_conv1d_decode` 迁移到新服务器路径 `/home/ops-transformer-master`。
- 已废弃早期 scalar GM-loop kernel，改成 A5 风格 BH 结构：`causal_conv1d_decode_apt.cpp`、`arch35/causal_conv1d_decode_cut_bh.h` 和 `arch35/vf/causal_conv1d_decode_width4_vf.h`。
- vf 入口已明确改成 BF16 专用。输入和输出是 `bfloat16_t`，FP32 只作为 vf 内部累加寄存器类型。
- 已新增 AscendC GM / 核间分核 / UB / vf 数据流规范，中英文各一份。
- 远端 reference 脚本已通过。CANN 构建未完成，因为当前 GPU 镜像没有配置 Ascend toolkit 路径。

# 2026-06-16 VF/SIMT API 与数据搬运补充

- 已新增本机 `ascendc-vf-simt-api` skill，中英文 reference 覆盖 VF、Reg、SIMT、DataCopyPad、TQue、Duplicate 等 API 调查。
- 已明确 kernel 数据流模型：`TQue::EnQue/DeQue` 是 queue 管理的 UB tensor 阶段同步边界；`TBuf` 常驻 metadata 仍通过显式 `SetFlag/WaitFlag` 表达跨 pipeline 顺序。
- 当时曾将合并的 x/weight UB queue、`stateQueueIn`、`yQueue` 改成 single buffer；后续 2026-06-21 已进一步拆成 `xQueue + weightBiasBuf`，以支持 dimLoop 外层复用 weight/bias。
- 已为 strided `DataCopyPad`、`Duplicate` 默认广播 optional input、bias 逻辑广播、state 搬入、y 写回和 state window 更新补充简短代码注释。
- reference 脚本仍通过。当前 GPU 镜像没有 Ascend toolkit 路径，CANN 编译仍不可用。

# 2026-06-17 vLLM update/decode 功能对齐

- 本轮按用户确认的范围对齐 vLLM Ascend `CausalConv1d` update/decode 功能：
  - 不加入 `initialStateMode` 和 `runMode`，它们被视为 prefill/模式切换相关，不属于本算子 decode-only 范围。
  - 暂不改为 INT64，继续使用当前框架内的 INT32 metadata；这意味着 ABI 不是 vLLM 原型级直接替换，但功能语义可通过 wrapper/cast 对齐。
  - 新增 optional `query_start_loc`，支持 flatten `x: [total_tokens, dim]` 的 update 输入。
  - `cache_indices` 改为 optional；缺省时 `slot = batch_id`。
  - `num_accepted_tokens` 保持 optional；缺省时 `accepted = token_count`。
  - `conv_states` 改为输入原地更新，公开输出只保留 `y`。
- width 从固定 4 扩展为 `weight.shape[0] in [2, 4]`：
  - host tiling 写入 `kernelWidth`。
  - kernel 根据 `kernelWidth` 选择 causal window。
  - 当前 VF 已进一步拆成 W=2/3/4 固定宽度函数，width=2/3 不读取不存在的 weight 行，也不在 VF 内执行 W 分支。
- CPU reference 已更新并验证通过：
  - 覆盖 `[B,S,D]` 固定 batch 输入。
  - 覆盖 `[T,D] + query_start_loc` flatten update 输入。
  - 覆盖 width 2、3、4。
- 未完成项：
  - 当前远程服务器无 Ascend toolkit，仍无法完成 CANN 编译验证。
  - `activation_mode=1` 的 SiLU 路径仍未在 VF 中实现；host 仍限制 `activation_mode=0`。
