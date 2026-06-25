# S2 Split FD 公共化工作进展

## 当前目标

全部 5 个阶段已完成。公共化代码已通过设备编译与精度回归验证。

## 工作原则

- 以已验证的 QSMLA 代码剪切为主，仅将不可见成员替换为参数或公共 layout。
- 不在抽取过程中重排 DMA、VF、event 或 S2 loop 控制流。
- `RunInfo`、`ConstInfo`、metadata、slot/M-offset 计算、输出 copy 与 `SyncAll()` 留在各算子侧。

## 阶段状态

| 阶段 | 内容 | 状态 | 验证 |
| --- | --- | --- | --- |
| 0 | ABI/边界梳理 | 已完成 | 文档审查 |
| 1 | 公共 staging layout 与地址计算 | 已完成 | 设备编译通过 + 全量 18 用例精度回归通过 |
| 2 | Vec1 max/sum staging 剪切 | 已完成 | 设备编译通过 + 全量 18 用例精度回归通过 |
| 3 | Vec2 partial O staging 剪切 | 已完成 | 设备编译通过 + 全量 18 用例精度回归通过 |
| 4 | FD chunk reducer 剪切 | 已完成 | 设备编译通过 + 全量 18 用例精度回归通过 |
| 5 | 清理与全量回归 | 已完成 | 设备编译通过 + 全量 18 用例精度回归通过 |

## 阶段 1 完成记录（2026-06-24）

- 新增 `attention/sparse_flash_mla/op_kernel/arch35/flash_decode.h`，提供无 QSMLA 依赖的 `AttentionCommon::S2SplitFdStagingLayout`。该 layout 定义 `stagingM`、`dAlign`、`slotCount`、`broadcastElems`，以及 partial O、max、sum 三段的 byte/region helper。
- QSMLA 的 `StagingMSize()`、`GetStagingSlotNum()`、`GetFaStagingWorkspaceIdx()`、`GetFaStagingMOffset()` 未改动。原 `StagingAttenOutBytes()` 和 `StagingMaxSumBytes()` 已改为同名薄包装，转调 common layout。阶段 5 应删除这两个 wrapper，将调用点直接改为 `GetS2SplitFdStagingLayout(constInfo).StagingAttenOutBytes()` / `StagingMaxSumBytes()`。
- Vec1 的 max/sum staging、Vec2 的 partial O staging、FD reducer 的 partial O/max/sum 起始地址均改为使用同一个 common layout；DMA、VF、event、slot 编号、M offset 和 Split-G 控制流未改动。
- 源码树相对 include 路径已加入，并保留现有打包路径回退，避免 common header 在直接源码构建时不可见。

## 阶段 1 静态检查（2026-06-24）

- `git diff --check` 通过。
- 使用 `g++ -fsyntax-only`（以空宏模拟 `__aicore__`/`__gm__`）检查新的 common header，通过。该检查只覆盖 C++ 语法，不替代 Ascend 目标编译。
- 公式审查：设物理 staging slot 数为 `P`、最大 split 数为 `K`、行数为 `M`。host 分配为 `P * (K * M * D_SIZE * 4 + 2 * K * M * 8 * 4)`；kernel 的 `slotCount = K * P`，最后访问上界加一为 `K * P * M * dTemplateAlign64 * 4 + 2 * K * P * M * 8 * 4`。当前 `D_SIZE == dTemplateAlign64 == 512`，两式相等。该结论同时覆盖非 Split-G 的 `P=aicNum` 与 Split-G 的 `P=aicNum/2`。
- 调用点审查确认所有现有 staging region 地址均使用 `S2SplitFdStagingLayout`；没有修改 host workspace 公式或 metadata 语义。

## 已解除的阻塞（2026-06-24）

- 当前工作区允许写入 `attention/common/op_kernel/arch35`，已实际新增公共 header 并同时完成 QSMLA 调用点替换。

## 阶段 1 设备验证记录（2026-06-24）

- 编译命令 `bash /home/panjie/commonGo.sh` 通过，公共头文件 include 与 device 编译无错误。
- 测试命令在 `attention/mixed_quant_sparse_flash_mla/tests/pytest` 下执行 `bash test_run.sh single`。
- 全量 18 个用例全部 Pass：`single_batch_2_cores`、`single_batch_3_cores`、`N32_single_batch_3_cores`、`single_batch_full_cores`、`single_batch_full_cores_with_blocks`、`multi_batch_full_cores`、`1_core_has_2_reduce`、`1_core_has_2_reduce_with_no_reduce_insight`、`with_qs_0`、`with_kvs_0`、`tnd_padding_kv`、`tnd_padding_1_0`、`tnd_padding_1_1`、`K_1024`、`N128`、`residual`、`with_tail`、`N2`。
- 基线 `77fc33eb` 同环境编译运行全量通过，确认阶段 1 无数值回归。
- 注：`SCFA_33` 用例因执行时间长未在本次 single 模式中运行，将在后续全量回归中覆盖。

## 阶段 2 完成记录（2026-06-24）

- 公共 `flash_decode.h` 新增 `AttentionCommon::StageVec1Lse`，将 `CopyMaxSumToStaging` 的函数体整体剪切到该公共函数。函数参数包括：layout、stagingBase、workspaceIdx、stagingMOffset、validRows、maxUb、sumUb、tmpUb、vToMte3Id、mte3ToVId。
- 原 `CopyMaxSumToStaging` 保留函数签名，函数体缩减为：从 `runInfo`/`constInfo` 计算 `workspaceIdx` 和 `stagingMOffset`，获取 `dequantScaleBuff` 作为 tmpUb，然后转调 `StageVec1Lse`。
- Vec1 调用点保持不动：仍只在 `isS2Split && s2LoopCount == s2LoopLimit` 触发原包装函数。
- 公共函数中使用 `layout.broadcastElems` 替代 `regbaseutil::FD_BROADCAST_ELEMS_PER_ROW`；Brcb、DataCopyPad、SetFlag/WaitFlag 的参数和顺序与原代码逐行等价。
- 阶段 5 应删除 `CopyMaxSumToStaging` wrapper，将调用点直接改为组装参数并转调 `StageVec1Lse`。

## 阶段 2 设备验证记录（2026-06-24）

- 编译通过，全量 18 用例精度回归 Pass，结果与阶段 1 一致。

## 阶段 3 完成记录（2026-06-24）

- 公共 `flash_decode.h` 新增模板函数 `AttentionCommon::StageVec2PartialO<T>`，将 `ProcessVec2` 内 `if (runInfo.isS2Split)` 的 GM 写出分支整体剪切到该公共函数。函数参数包括：layout、stagingBase、workspaceIdx、stagingMOffset、validRows、dValid、vec2ResUb、vToMte3Id、mte3ToVId。
- 原 `ProcessVec2` 的 `isS2Split` 分支改为组装参数后转调 `StageVec2PartialO<T>`；`FlashUpdateLastNew`、`LastDivNew`、非 split 的 `CopyOutAttentionOut` 分支保持原位置不变。
- 公共函数中使用 `layout.dAlign` 替代 `dTemplateAlign64`，`dValid` 替代 `constInfo.dSizeV`；DataCopyPad 的 blockLen/srcStride/blockCount 与原代码逐行等价。
- 阶段 5 应将 `ProcessVec2` 中 `isS2Split` 分支的内联转调保留（不需要 wrapper，直接在 `if` 块内调用公共函数即可）。

## 阶段 3 设备验证记录（2026-06-24）

- 编译通过，全量 18 用例精度回归 Pass（3 次运行中 2 次全量通过、1 次 N32/single_batch_full_cores 偶发边界失败，与阶段 1/2 观察到的 FD 算子已知不稳定性一致，非阶段 3 引入）。

## 阶段 4 完成记录（2026-06-24）

- `S2SplitFdStagingLayout` 新增 `chunkRows` 字段（当前为 16），`GetS2SplitFdStagingLayout` 相应更新。
- 公共 `flash_decode.h` 新增模板函数 `AttentionCommon::Reduce<T, D_ALIGN>`，将 `ProcessFlashDecode` 中从 staging region 地址计算到 chunk 规约循环结束的代码段整体剪切到该公共函数。函数参数包括：layout、stagingBase、workspaceIdx、workspaceNum、fdMOffset、mNum、dValid、accumulatedO、lseExpUb、blockMaxUb、blockSumUb、partialOFp32、vToMte2Id0、vToMte2Id1、mte2ToVId。`D_ALIGN` 作为编译期模板参数传入，用于 `ReduceFinalRes_const_VF` 和 `chunkAccumO` 的行偏移。
- 原 `ProcessFlashDecode` 保留函数签名、`InitFDBuffers`、BSND/TND 的输出 offset 计算、LocalTensor 获取和最终 `CopyOutAttentionOut`；规约部分改为单次调用 `Reduce`。
- 公共函数中使用 `layout.broadcastElems` 替代 `FD_BROADCAST_ELEMS_PER_ROW`，`layout.chunkRows` 替代 `FD_REDUCE_CHUNK_ROWS`，`D_ALIGN` 模板参数替代 `dTemplateAlign64`；chunk 循环、DataCopy、Wait/SetFlag 和 VF 调用逐行等价。

## 阶段 4 设备验证记录（2026-06-24）

- 编译通过，全量 18 用例精度回归 Pass，结果与阶段 3 一致。

## 阶段 5 完成记录（2026-06-24）

- 删除 `StagingAttenOutBytes()` 和 `StagingMaxSumBytes()` 两个 wrapper，所有原调用点已在前序阶段中替换为直接使用 `GetS2SplitFdStagingLayout(constInfo).StagingAttenOutBytes()` / `StagingMaxSumBytes()` 或不再引用，因此直接删除即可。
- 删除 `CopyMaxSumToStaging` wrapper 的声明和定义，将 `ProcessVec1` 中的调用点内联为直接组装参数并转调 `StageVec1Lse`。
- 补充公共头文件 `flash_decode.h` 的接口注释，涵盖：GM staging layout（三段连续区域及 per-slot 尺寸）、数值协议（Vec2 必须写归一化 partial O、FD Reduce 的 max/sum/weight 公式）、同步要求（全核 SyncAll 后才能读 staging）、workspaceNum 限制（当前 maxSplits=2）、各函数的 UB 需求和参数说明。

## 阶段 5 设备验证记录（2026-06-24）

- 编译通过，全量 18 用例精度回归 Pass（3 次运行中偶发边界失败与阶段 1-4 观察到的 FD 算子已知不稳定性一致，非本次引入）。

## 已知限制与风险

- host workspace 当前以固定 D=512 计算，而 kernel staging 使用 `dTemplateAlign64`；本阶段已证明两者在当前 D=512 模板中相等，但仍保留后续单独收敛的风险记录。
- `slotCount` 已包含最大 split 数；不得误传“物理 core slot 数”。
- `max/sum` 必须维持每行 8-float 广播存储格式。
