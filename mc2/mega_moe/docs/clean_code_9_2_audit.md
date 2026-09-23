# MegaMoe arch22 清理记录（9.2.0）

## 基线与验收口径

- 本地 `9.2.0` 已 rebase 到 `cann/ops-transformer` 的 `9.2.0`，基线提交为 `e66cb2f8e9`；原本地分支没有独有提交，实际为快进。在此基础上创建 `mega_clean_code_9_2`。
- 问题清单为 `/root/workspace/Q3_mega/mega_q.xlsx`：Sheet1 有 657 条重复片段记录（656 条在 `op_kernel`），Sheet2 有 54 条超长函数记录（51 条在 `op_kernel`）。表格按旧版本行号记录，条目之间有重叠。
- 用 `python mc2/mega_moe/docs/audit_sheet1_remaining.py /root/workspace/Q3_mega/mega_q.xlsx` 复算 Sheet1。脚本从基线的每条记录提取连续的 10 行代码窗口，忽略注释和空白；若窗口仍出现在原记录文件及当前 `mc2/mega_moe` 的另一位置，就算该条仍未解决。三条在未修改基线中也无法在扫描范围内找到同伴的记录按“未解决”保守计入。用未修改基线自检为 `657/657`。
- 当前结果为 **126/657（19.18%）**，低于 20%；其中 1 条属于 `op_host`。这是对原表条目的静态复核口径，不是对全仓库新生重复片段的通用检测，也不能证明设备运行时等价。

## 已做的合并

| 范围 | 处理方式 | 保留的差异 |
| --- | --- | --- |
| A2 BF16、INT8 | 合为 `mega_moe_kernel_a2.hpp` 的模板内核；原两个头仅保留类型别名。 | 类型、量化分支、AIV 核数、epilogue 策略通过编译期条件选定。 |
| A2 W4A8 与 A2 公共路径 | 提取 cumsum、六区 GM 清零、三路径跨 rank allgather、GMM 坐标偏移计算。 | W4A8 的输入步长、权重量化、矩阵计算和 epilogue 仍由原路径处理。 |
| A2 W4A8 与 A3 公共路径 | 提取双缓冲 `CopyGMToGM` 和 INT8→INT4 预处理；GMM 坐标计算也共用。 | INT4 输出步长仍分别使用 A2 的 `aStride`、A3 的 `hiddenSize`。 |
| `utils`、`utilsA2` | 合并 BlockMmad、SwiGLU、V2 的同构实现；提取任务分配、tile 索引、scale 预读和写回、路由交集算术、SiLU gate 计算；删除未引用的两个 A2 row epilogue。 | A2/A3 的 FixPipe、事件、GM 偏移宽度、跨服务器输出分支与各自的策略仍按编译期条件保留。 |
| 入口、prologue、unpermute | 将 A2/A3 入口类合并；prologue 与 unpermute 的重复步骤提取为小函数。 | A2 的 INT32 权重禁用条件、INT8 半数 AIV、W4A8 行步长、A3 mask 与多轮接收参数分别保留。 |
| A3 接收轮与各架构计数 | 提取反量化行数累计、路由计数和同步任务判定。 | 原调用点及分支条件保留。 |

原清理版 `574e64ec12` 的这段同步等价性结论不完整：后续审查确认，统一 A2 内核把 INT8 专有的一个 SEND 后 `SyncAll` 和四个 GMM `PIPE_ALL` 带入了 BF16/FP16。下述修复为这五处补回编译期 dtype 条件；通用搬运及其他同步路径本次未修改。机器码逐字节一致不是验收条件。

## 审查后的 F1/F2 同步修复

严格基线为 `e66cb2f8e9f5f8cc5c851e857ef7d7b439e3272d`，发现问题的版本为 `574e64ec12a75c6212042d374e4e1effc5c7ee43`。本次保留现有提取与合并结构，仅在 `mega_moe_kernel_a2.hpp` 的五处调用外增加 `if constexpr (IS_INT8)`：GMM1 的初始 cumsum 通知等待后及每 expert 设置 B/scale 地址后各一处 `PIPE_ALL`；GMM2 入口两处 `PIPE_ALL`；AIV 每 expert 的 SEND→RECV 之间一处 `SyncAll`。旧 BF16/FP16 没有这些调用，旧 INT8 有；A2 W4 使用独立内核，A3 不使用该 A2 模板。

BF16/FP16 走 `MegaMoeKernelA2BF16`（`IS_INT8=false`），INT8 走 `MegaMoeKernelA2Int8`（`IS_INT8=true`）。令每 chunk 的专家数为 `E=expertPerRank`、chunk 数为 `C=ceil(bs/1024)`：**相对有回归的 `574e64ec12`**，修复后每个浮点 AIV 少 `C×E` 次 SEND 后全核同步，每个浮点 AIC 少 `C×(E+3)` 次 GMM 全流水屏障；**相对严格基线 `e66cb2f8e9`，这五处浮点路径的调用次数差为 0**，因为基线本来就没有这些调用。INT8 在三个版本中的这五处调用次数和位置一致。即使某个 expert 的 token 数为 0，外层 expert 循环仍执行，因此计数成立。RECV 后 `SyncAll`、其后的 `CrossCoreSetFlag`，以及 GMM 的等待和通知次序没有改动。

接收 TPE 的对齐容量与 stride、preSum 紧凑布局、ExpertTokenNums 的 `DataCopyPad` 均未改动。A2 INT8 V2 的 32 位 GM offset 是基线已有风险，本次未处理。修复后的独立核查与编译记录保存于 `/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-sync-fix-20260923/verification.md`；原审查报告及证据保持不变。

## 剩余条目与暂缓原因

| 原记录范围 | Sheet1 剩余 | 原因 |
| --- | ---: | --- |
| A2 W4A8 kernel | 54 | 与 A3 或统一 A2 的 GMM、发送/接收、workspace、类参数仍有相似片段；整体合并会交织不同的同步 flag、RDMA/IPC 与矩阵调度。 |
| A3 kernel | 30 | 与 W4A8/A2 的结构和局部计算相似，但 A3 的多轮接收、route cache、mask 与同步协议不同。 |
| `utils`、`utilsA2` | 39 | SwiGLU、BlockMmad、V2、HCCL 仍有类型、UB 布局或事件路径专有的片段；当前公共代码已覆盖核对过的同构部分。 |
| unpermute 源码与 tiling 头 | 2 | 其中一条是运行态字段与 tiling ABI 字段列表的重复；直接继承或嵌套 tiling 结构会改变对象布局和读取方式。 |
| `op_host` | 1 | 本轮已纳入 host 拆分；该重复记录仍按脚本规则保守计入。 |

Sheet2 的较长函数没有全部变成 50 行以内。GMM1/GMM2、Dispatch/Combine 等函数承载不同的通信和计算路径；此次只拆出经逐段核对的共同逻辑。可继续按已核对的阶段边界做结构拆分；数值、死锁及跨卡时序的最终验证仍需要 A2/A3 设备及 BF16、INT8、W4A8 的 golden 用例。当前没有设备，无法完成这些实机验证。

### Sheet2 追加拆分与 Sheet1 新增长函数复核

追加拆分保留各调用点和原执行顺序：W4A8 与 A3 的 `initBuffer` 分离缓存/掩码初始化，A3 `BuildRoundTables` 分离轮表填充，prologue `PermuteTokens` 分离单 token 步骤。按函数首尾的物理行数，这四个原 Sheet2 函数现分别为 46、48、36、37 行；新提取的对应函数均不超过 50 行。按原表 54 条逐项映射当前入口/共享实现，这使不超过 50 物理行的条目从 7 条增至 11 条，其余 43 条仍对应超长实现。A2 的 `CrossRankAllGatherShared` 原是 Sheet1 去重后新增的超长公共函数，现按 self、同 server、跨 server 发送和接收阶段拆分；该文件所有新增函数均不超过 50 个物理行，发送、flag 等待、preSum 计算及末尾 `SyncAll` 的源码次序保留。

复核也发现 `a2_a3_copy_helpers.hpp` 的 `FetchAndPreprocessInt8ToInt4Shared` 仍有 140 个物理行。它是 Sheet1 合并 A2 W4A8/A3 时产生的公共实现，两个原入口虽分别缩为 7、8 行，但主体只是转移到了这里，不能计作整体超长问题已解决。统一 A2 内核中的 GMM、SEND 和 Dispatch/Combine，以及合并入口的 `MegaMoe::Process` 仍超过 50 行；这部分不以短包装或换文件名充当解决。上述行数为源码物理行统计，不等同原表扫描器的非空非注释行口径，也不是当前全仓所有超长函数的数量。

Sheet2 当前剩余 38 条的原表编号、当前对应函数、行数及暂缓原因逐项列于 [Sheet2 剩余清单](sheet2_remaining_9_2.md)。其中多条 BF16/INT8 或 `utilsA2` 原记录映射到同一公共函数，清单按原记录计数，并另列 Sheet1 合并产生的超长公共实现。

## 编译与验证边界

F1/F2 修复后重新执行 `/root/workspace/Q3_mega/编译.md` 对应的 `build.sh --pkg --ops=mega_moe` 命令，分别指定 A2 `--soc=ascend910b` 与 A3 `--soc=ascend910_93`，使用 9.3.0 weekly 工具链、本地 third-party 缓存及可写的 `CCACHE_DIR`。两机型各自的 18 个 kernel 实例（编号 0–17）均重新编译成功，完整构建及 `.run` 打包均成功。算子定义覆盖 FP16/BF16、INT8 和 W4 的 dtype 组合；没有 A2/A3 算子编译错误。

两个默认打包过程均在 `es_math` Python wheel 的隔离构建阶段因离线环境无法取得 `setuptools>=40.8.0` 中断。这是打包依赖问题；用环境已安装的 setuptools 执行 `pip wheel . --no-deps --no-build-isolation --wheel-dir dist`，生成 wheel 标志后重跑 CMake 构建和 package 均通过。实际命令、18 个实例日志与本次代码补丁保存在上述独立验证记录的目录。

静态审查和编译只能验证源码分支、实例化与生成。当前无 A2/A3 设备，尚无数值、不均匀路由、同 server/跨 server 通信、死锁或性能的设备侧对比结果；这些项目不能标记为通过。

Sheet2 追加拆分后，再次按上述 A2/A3 命令分别编译了各 18 个 kernel 实例；两次均只在离线 `es_math` wheel 隔离构建阶段中断，使用本地 wheel 续构建与 `.run` 打包成功。新一轮源码长度、Sheet1 复核和编译日志见 `/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-sheet2-followup-20260923/verification.md`。该编译结果仍不能代替设备侧数值、通信或性能验证。

### Sheet2 第二轮：统一 A2、共用 V2、通信及 host

本轮基于剩余清单拆分 9 个函数：统一 A2 `SendTokensV3` 144→64 行、`DispatchAndCombine` 337→153 行；V2 带 scale/无 scale 重载分别 114→54、66→28 行；RDMA 投递 88→39 行、带 ctrBuffer 的跨 rank 同步 75→33 行；host 属性、shape、主 tiling 分别 164→50、150→52、158→60 行。新增 34 个 helper 均不超过 50 行（最长 45）；Sheet2 剩余由 43 条降为 38 条，对应 27 个当前函数。`FetchAndPreprocessInt8ToInt4Shared` 140 行仍保留，不把旧入口变短计作主体解决。

结构展开对照以本轮修改前的工作树为基准，9 个函数全部通过。仅归一化 prologue 类型别名，以及 host 输出标量初始化/archCode 声明的位置；其余分支、表达式、日志与调用 token 保持。host 首个失败返回与显式日志消息/次序不变（日志宏附带的函数名和行号随拆分变化），A2/A3 workspace 原公式及整数类型保持；通信继续先投递全部目标再轮询；F1/F2 五处 INT8 条件、空 expert 和 chunk 的同步调用关系保持。未修改 W4/A3 内核、BlockMmad、路由布局、DMA 参数或地址宽度；共用 V2/HCCL 的结构变化覆盖引用它们的路径，需由两机型编译检查。

新证据独立保存于 `/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-sheet2-round2-20260923/verification.md`，原审计证据不变。Sheet1 复算为 126/657（19.18%），其中额外减少的 5 条原 W4 文件记录源于其同伴片段的拆分；W4 内核本轮没有修改。

本轮最终源码（含 helper 张量参数按引用传递的修订）已重新完成 A2/A3 各 18 个实例的编译，覆盖 FP16/BF16、INT8 和 W4；默认 `--pkg` 均仅在离线 es_math wheel 依赖下载处退出 2，使用本地依赖生成 wheel 后 CMake 续构建和 `.run` 打包均退出 0。最终源码哈希、原始命令、dtype 参数和完整日志均保存于本轮独立证据目录。仍未进行设备数值、死锁或性能验证。

### A2 scale 实参的编译期分支

按用户指定范围，A2 GMM1/GMM2 的 `blockMmad` 调用改用 `if constexpr (IS_INT8)`：INT8 保留原 `gmS[gmOffsetS]` / `gmS2[gmOffsetS]` 及偏移表达式；BF16/FP16 直接传入 `gmS` / `gmS2`，不再求值 scale 下标视图。原 `GetTensorAddr(...ptrScale1/2)`、`SetGlobalBuffer`、BlockMmad 接口、屏障及 flag 顺序保留。这不构成完整的空 scale 输入修复，原地址解析仍要求有效的列表描述符。

同类检索确认 A3 已按 INT8/W4/浮点编译期分支处理，浮点直接传原张量；W4 内核的 scale 下标用于实际量化。无需修改这些位置。纯数值 padding/stride 的常量三目表达式维持原写法。当前工作树已有的 RecvTokensV3 声明作用域调整保留。

Sheet1 当前为 127/657（19.33%），仍低于 20%；较上一轮多匹配 1 条原 W4 记录，其同伴片段因 A2 分支展开重新满足窗口规则，W4 文件本次未改。Sheet2 GMM1/GMM2 为 118/110 物理行，剩余记录数量不变。独立验证记录：`/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-scale-constexpr-20260923/verification.md`。

上述 scale 实参修订完成后，A2/A3 各 18 个实例、默认完整构建及 `.run` 打包均退出 0；最终两处差异及事件序列静态核查通过。未执行设备数值、死锁或性能回归。

### 移植 PR 12815 的 A2 浮点清零同步修复

来源：<https://gitcode.com/cann/ops-transformer/pull/12815/diffs>，固定 HEAD `5f74c639cb5e98026448590d2ccd282d84069e10`。用户指定文件 `mega_moe_kernel_a2_bf16.hpp` 在 PR 中仅有一处改动：初始清零 ExpertTokenNums 时，先 Set/Wait V_MTE3，再 DataCopyPad，最后 Set/Wait MTE3_S；移除原清零位置的 PIPE_V。

当前 BF16 头是统一 A2 内核的类型别名，实际移植位置为 `mega_moe_kernel_a2.hpp::ResetExpertTokenNums`。使用 `if constexpr (IS_INT8)` 保留 INT8 原分支，在 else 中应用 PR 的 BF16/FP16 修复。该 helper 仍在 chunk 循环前调用一次、仅 coreIdx==0 写回；DMA 参数不变。此处是用户明确要求移植的同步修复，浮点初始清零的事件顺序有意变化，不再套用此前“本轮同步序列不变”的历史结论；F1/F2 的 GMM 与 SEND/RECV 条件保持。

静态对照确认 INT8 分支与移植前一致，浮点分支与固定 PR 文件一致；其他源码文件哈希保持。`ResetExpertTokenNums` 为 28 个物理行，没有引入超长 helper。Sheet1 为 127/657（19.33%），Sheet2 剩余条数不变。独立补丁、PR 原文和验证结果见 `/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-pr12815-20260923/verification.md`。

PR 12815 单文件移植后，A2/A3 各 18 个实例、完整构建和 `.run` 打包均成功（退出 0），覆盖浮点、INT8、W4 编译组合。最终源码与固定 PR 分支对照及其他源码哈希核查再次通过；未执行设备侧回归。

### WorkspaceInfo 编译期合并

统一 A2 内核的 WorkspaceInfoInt8/WorkspaceInfoBf16 合并为 WorkspaceInfo。两段 per-token scale 指针赋值及 workspaceOffset 累加置于 `if constexpr (IS_INT8)` 内；浮点的两个未使用 scale 字段为 nullptr，不分配对应 GM workspace 空间。该结构是内核私有地址描述符，不是 host tiling ABI；浮点描述符增加两个指针字段，但实际 GM 区域偏移、容量、顺序保持。移除了未使用的 n2 局部变量，构造函数为 50 个物理行。

逐分支展开后，偏移算术与类型和两个原构造函数一致（仅忽略注释/空白及未使用的 n2 声明），统一 A2 文件其余部分保持。PeermemInfo 两种布局继续独立。Sheet1 当前 126/657（19.18%），Sheet2 剩余数量不变。独立验证见 `/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-workspace-constexpr-20260923/verification.md`。

WorkspaceInfo 合并后构建验证：A2（ascend910b）与 A3（ascend910_93）各 18/18 个内核实例编译完成，build.sh 编译及打包退出码均为 0；参数与 Done 日志核对通过。证据见 `refactor_notes/audit_evidence/trans-clean-code-workspace-constexpr-20260923/build_verification.log`（工作区根目录下）。无设备，未进行数值、死锁和性能实机验证。
