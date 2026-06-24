# S2 Split FD 公共化工作进展

## 当前目标

阶段 1 的代码抽取和静态检查已完成。设备编译及精度回归仍待具备 Ascend 环境后执行；在该验证完成前不进入阶段 2。

## 工作原则

- 以已验证的 QSMLA 代码剪切为主，仅将不可见成员替换为参数或公共 layout。
- 不在抽取过程中重排 DMA、VF、event 或 S2 loop 控制流。
- `RunInfo`、`ConstInfo`、metadata、slot/M-offset 计算、输出 copy 与 `SyncAll()` 留在各算子侧。
- 当前环境不具备算子执行条件；本阶段仅做静态检查、编译可行性检查和 diff 审查。设备精度/稳定性测试由接手人补做。

## 阶段状态

| 阶段 | 内容 | 状态 | 验证 |
| --- | --- | --- | --- |
| 0 | ABI/边界梳理 | 已完成 | 文档审查 |
| 1 | 公共 staging layout 与地址计算 | 代码完成，设备验证待补 | `git diff --check`、公共 header 语法检查、公式与调用点审查通过 |
| 2 | Vec1 max/sum staging 剪切 | 未开始 | 参数集定向回归 |
| 3 | Vec2 partial O staging 剪切 | 未开始 | 参数集定向回归 |
| 4 | FD chunk reducer 剪切 | 未开始 | 参数集定向回归与重复执行 |
| 5 | 清理与全量回归 | 未开始 | 全量参数集 |

## 阶段 1 完成记录（2026-06-24）

- 新增 `attention/common/op_kernel/arch35/flash_decode_s2_split.h`，提供无 QSMLA 依赖的 `AttentionCommon::S2SplitFdStagingLayout`。该 layout 定义 `stagingM`、`dAlign`、`slotCount`、`broadcastElems`，以及 partial O、max、sum 三段的 byte/region helper。
- QSMLA 的 `StagingMSize()`、`GetStagingSlotNum()`、`GetFaStagingWorkspaceIdx()`、`GetFaStagingMOffset()` 未改动。原 `StagingAttenOutBytes()` 和 `StagingMaxSumBytes()` 已改为同名薄包装，转调 common layout。
- Vec1 的 max/sum staging、Vec2 的 partial O staging、FD reducer 的 partial O/max/sum 起始地址均改为使用同一个 common layout；DMA、VF、event、slot 编号、M offset 和 Split-G 控制流未改动。
- 源码树相对 include 路径已加入，并保留现有打包路径回退，避免 common header 在直接源码构建时不可见。

## 阶段 1 静态检查（2026-06-24）

- `git diff --check` 通过。
- 使用 `g++ -fsyntax-only`（以空宏模拟 `__aicore__`/`__gm__`）检查新的 common header，通过。该检查只覆盖 C++ 语法，不替代 Ascend 目标编译。
- 公式审查：设物理 staging slot 数为 `P`、最大 split 数为 `K`、行数为 `M`。host 分配为 `P * (K * M * D_SIZE * 4 + 2 * K * M * 8 * 4)`；kernel 的 `slotCount = K * P`，最后访问上界加一为 `K * P * M * dTemplateAlign64 * 4 + 2 * K * P * M * 8 * 4`。当前 `D_SIZE == dTemplateAlign64 == 512`，两式相等。该结论同时覆盖非 Split-G 的 `P=aicNum` 与 Split-G 的 `P=aicNum/2`。
- 调用点审查确认所有现有 staging region 地址均使用 `S2SplitFdStagingLayout`；没有修改 host workspace 公式或 metadata 语义。

## 已解除的阻塞（2026-06-24）

- 当前工作区允许写入 `attention/common/op_kernel/arch35`，已实际新增公共 header 并同时完成 QSMLA 调用点替换。

## 仍待设备环境执行的验证

当前环境不能运行算子。具备设备环境后，按以下顺序执行：

1. 编译 MixedQuantSparseFlashMla 的 arch35 kernel，确认公共头文件 include 与 device 编译通过。
2. 运行 `single_batch_2_cores`、`single_batch_3_cores`、`N32_single_batch_3_cores`、`N128`。
3. 对照迁移前后 metadata trace 和输出；阶段 1 不应改变任何 slot、M-offset、workspace 字节数或数值结果。
4. 阶段 1 的设备验证通过后，才进入 Vec1/Vec2/reducer 的代码剪切。

## 已知限制与风险

- host workspace 当前以固定 D=512 计算，而 kernel staging 使用 `dTemplateAlign64`；本阶段已证明两者在当前 D=512 模板中相等，但仍保留后续单独收敛的风险记录。
- `slotCount` 已包含最大 split 数；不得误传“物理 core slot 数”。
- `max/sum` 必须维持每行 8-float 广播存储格式。
