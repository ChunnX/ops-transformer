# MegaMoe Sheet2 剩余超长函数（9.2.0）

## 统计口径

`/root/workspace/Q3_mega/mega_q.xlsx` 的 Sheet2 有 54 条原记录。按原表编号逐项映射当前实现，以函数声明行至结束花括号的**物理行数**核对，16 条所指入口不超过 50 行，剩余 **38 条**列于下文。38 条映射到 27 个当前函数：A2 BF16/INT8 的统一内核、A2/A3 的共用 `utils` 实现会让多条原记录指向同一函数。此处不把原文件删除或短包装视为主体已拆分；物理行数也不等于原表的非空非注释行统计。表中“待拆”说明当前边界和下一步可拆方向，不表示已证明其他拆法不安全。

下表路径均相对 `mc2/mega_moe/`，`#` 为原 Sheet2 的行序号（不含表头）。

## A2 BF16/FP16 与 INT8 统一内核：12 条，6 个当前函数

| 原编号 | 当前实现（物理行） | 待拆原因与边界 |
| --- | --- | --- |
| #4、#8 | `op_kernel/arch22/mega_moe_kernel_a2.hpp::GMM1`（118） | BF16/FP16 与 INT8 的矩阵、scale 和等待逻辑合并在模板中；可继续提取坐标/地址准备，须保留 F2 的 INT8 专用屏障及 tile 次序。 |
| #15、#18 | 同文件 `GMM2`（110） | epilogue、专家循环和等待相邻；拆分要保持 GMM2 的两处 INT8 专用 `PIPE_ALL` 条件。 |
| #22、#49 | 同文件 `SendTokensV3`（64） | 本轮已拆成 self、同 server、跨 server 三种发送 helper（28/31/43 行）；入口保留 UB 和 copy 对象生命周期、零行返回及路径选择，仍超过物理行阈值。 |
| #31、#52 | 同文件 `DispatchAndCombine`（153） | 本轮拆出计数清零/累加、prologue、expert SEND/RECV、两段 SwiGLU 和 unpermute。剩余 chunk 调度、epilogue 对象生命周期及阶段同步仍在入口，F1 条件与通知/等待顺序保留；未把整个长主体移到另一大函数。 |
| #6、#43 | 同文件 `CombineV2`（114） | 路由结果、远端写入和 epilogue 输出相邻；需要核对空 expert 与多 chunk 的偏移和事件。 |
| #27、#44 | `op_kernel/arch22/mega_moe_arch22.h::MegaMoe::Process`（193） | 原 A2/A3 入口现共用模板，长段主要是类型策略、布局及参数装配；可优先提取类型/参数装配，不应将短转发函数算作解决。 |

## A2 W4A8 内核：5 条，5 个当前函数

| 原编号 | 当前实现（物理行） | 待拆原因与边界 |
| --- | --- | --- |
| #25 | `op_kernel/arch22/mega_moe_kernel_a2_bf16_w4a8.hpp::GMM1`（102） | INT4 输入、scale 与 tile 计算相邻；可继续提取纯地址/布局准备，保留独立的 W4 计算路径。 |
| #42 | 同文件 `GMM2`（94） | 矩阵计算与 W4 epilogue 相邻；拆分时保持原 tile 和事件顺序。 |
| #26 | 同文件 `SendTokensV3`（105） | token 与 scale 的发送、接收顺序交织；不宜仅因与普通 A2 同名而合并。 |
| #47 | 同文件 `DispatchAndCombine`（291） | allgather、INT4 预处理、通信和 combine 连续执行；应按实际阶段边界拆分并核对跨 server 路径。 |
| #19 | 同文件 `CombineV2`（119） | W4 的量化布局和远端写回专有；保留其行步长和 epilogue 次序。 |

## A3 内核：6 条，6 个当前函数

| 原编号 | 当前实现（物理行） | 待拆原因与边界 |
| --- | --- | --- |
| #29 | `op_kernel/arch22/mega_moe_kernel_a3.hpp::GMM1`（137） | 多轮接收后的矩阵坐标、mask 和 epilogue 条件交织；可先提取纯算术准备。 |
| #1 | 同文件 `GMM2`（130） | 每轮 GMM2 与同步 flag 相邻；拆分需核对 round/chunk 状态。 |
| #7 | 同文件 `AllGatherTokenPerExpert`（84） | token 计数、flag 和 DMA 搬运交织；须保留生产/消费顺序。 |
| #40 | 同文件 `ProcessRecvRound`（202） | 多轮接收、预处理、GMM 和 mask 的阶段函数；需用不均匀路由与跨 server 用例验证下一轮切分。 |
| #10 | 同文件 `DispatchAndCombine`（213） | chunk/round 调度及各阶段同步仍在同一函数；可拆纯参数准备，保留通知与等待配对。 |
| #51 | 同文件 `CombineV2`（97） | 路由输出与远端写回相邻；要核对每轮偏移及 epilogue。 |

## utils、utilsA2 与通信：13 条，8 个当前函数

| 原编号 | 当前实现（物理行） | 待拆原因与边界 |
| --- | --- | --- |
| #5 | `op_kernel/arch22/utils/block_mmad_w4a4.hpp::operator()`（82） | W4A4 tile 调度与事件配对仍集中；可拆纯 tile 索引。 |
| #23 | 同文件 `L1TileMmad`（124） | L1/L0 数据搬运、MMAD 与量化步骤相邻；需保持流水事件。 |
| #28、#35 | `op_kernel/arch22/utils/block_mmad_preload_async_fixpipe_quant.hpp::operator()`（72） | 原 `utilsA2` 实现已映射到共用 BlockMmad；算子调用仍超过阈值，不能按旧文件变短计为解决。 |
| #34、#45 | 同文件 `L1TileMmad`（128） | A2/A3 FixPipe 和 tile 事件条件共用模板；拆分需逐 dtype 核对等待与通知。 |
| #14、#48 | `op_kernel/arch22/utils/block_epilogue_pertoken_v2.hpp::operator()`（带 scale，54） | 本轮拆出 token scale 处理、路由选择和每段写回；入口仍保留双缓冲、搬运/转换和事件生命周期。A2/A3 GM 偏移宽度、CAST_RINT 及编译期分支保持。 |
| #38、#41、#46 | `op_kernel/arch22/utils/block_epilogue_pertoken_swiglu.hpp::operator()`（带 scale，94） | 原 A2 BF16/INT8 和 A3 入口共用动态量化主体；双缓冲、scale 写回和激活次序仍集中。 |
| #32 | `op_kernel/arch22/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp::operator()`（110） | W4 的解包、激活、量化与 UB 复用相邻；应单独核对 W4 事件路径。 |
| #39 | `op_kernel/arch22/utils/block_epilogue_w4a8post_pertoken_v2.hpp::operator()`（131） | W4 反量化/路由写回连续执行；拆分需保留 stride 与事件配对。 |

## op_host tiling：2 条，2 个当前函数

| 原编号 | 当前实现（物理行） | 待拆原因与边界 |
| --- | --- | --- |
| #24 | 同文件 `MegaMoeA2A3CheckShapeAndSetTiling`（52） | 本轮提取 routing 输入、weight 输入、scale、bias 四阶段；主函数保留阶段返回及最终 tiling 字段/日志，首个错误与 getter 次序不变。 |
| #36 | 同文件 `MegaMoeA2A3TilingFuncImpl`（60） | 本轮提取公共检查、tiling key、common 字段写入、A2/A3 workspace 计算。入口仍管理平台/permute 对象及 workspace 设置；原失败分支和算术类型保留。 |

## Sheet1 合并后新增、未计入上述 38 条的超长公共实现

`op_kernel/arch22/a2_a3_copy_helpers.hpp::FetchAndPreprocessInt8ToInt4Shared` 为 **140 个物理行**。原 Sheet2 的 #11、#33 入口现只有 8、7 行，双缓冲 UB、INT8→INT4 高/低位处理和事件流水仍位于公共实现中；因此这两个原记录只是入口缩短，整体风险保留。本轮已将 `a2_routing_helpers.hpp` 的 `CrossRankAllGatherShared` 拆为不超过 50 行的阶段/路径函数。无 A2/A3 设备，以上剩余计算与通信路径尚无数值、死锁或性能实机对比。

## 本轮降至不超过 50 行的记录

| 原编号 | 当前实现 | 修改前 → 修改后（物理行） |
| --- | --- | ---: |
| #12、#17 | `block_epilogue_pertoken_v2.hpp::operator()`（无 scale） | 66 → 28 |
| #13 | `hccl_shmem.hpp::AIVRDMAPostSend` | 88 → 39 |
| #20 | `hccl_shmem.hpp::CrossRankSync(ctrBuffer)` | 75 → 33 |
| #2 | `MegaMoeA2A3CheckAttrAndSetTiling` | 164 → 50 |

本轮共修改 9 个原函数（合计 13 条原记录），新增 34 个阶段/路径 helper，最长 45 行。以上 5 条原记录对应的主体与新 helper 均不超过 50 行；其他已缩短但仍超过阈值的记录继续列在剩余表。`utilsA2` 为共用实现的 include/别名入口，本轮修改共用 V2 实现即可覆盖其相应重载，没有额外复制 A2 专用实现。

本轮源码展开对照、每个新 helper 的行数与独立编译结果见：
`/root/workspace/Q3_mega/refactor_notes/audit_evidence/trans-clean-code-sheet2-round2-20260923/verification.md`。
展开对照使用本轮开始前的工作树快照，不以修复前旧审计脚本的结论代替验证。仍无设备数值、死锁或性能验证。

A2 scale 实参追加修订：GMM1/GMM2 以 `if constexpr (IS_INT8)` 区分偏移张量与浮点原张量，物理行数更新为 118/110；原初始化和同步顺序保留，剩余 38 条的计数不变。
