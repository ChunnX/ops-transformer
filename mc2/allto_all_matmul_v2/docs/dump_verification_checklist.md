# AlltoAllMatmulV2 异常 Dump 验证清单

基于 `dump_design.md` 方案与代码实现整理（代码已确认全部落地，链路闭环）。

## 一、编译与开关

| 验证项 | 说明 | 代码位置 |
|---|---|---|
| `MC2_DFX_ENABLE` 开/关均编译通过 | 所有打点代码受宏保护 | urma_impl L158、apt.cpp L50、tiling.cpp L62 |
| 关闭 DFX 时零性能回归 | stub 全部为空内联，跑基线性能对比 | mc2_runtime_info.h L152-161 |
| 开启 DFX 时性能影响可接受 | 打点含 GM 写 + DataCacheCleanAndInvalid 刷盘，AIV 每轮通信 4 次刷写 + AIC 每次计算前 1 次刷写，开/关差值即打点开销；记录典型 shape 下的 e2e 时延/带宽/利用率差值（阈值如 <5%），超阈值需优化打点频率 | urma_impl L242-273、L359-384 |

### 性能测试方法

- 编译两版（`MC2_DFX_ENABLE=0/1`），相同环境跑相同 case 集（覆盖 M/N/K 典型 shape、不同 rank 数、localMatmul 0/1）
- 指标：e2e 时延（多轮取均值）、单 kernel 耗时、多卡带宽利用率
- 对比项：开/关差值（打点开销）、开启版相对基线的回归比例
- 注意 DataCacheCleanAndInvalid 会刷脏 cacheline，对同 UB 复用的 kernel 有额外影响，不只看打点次数

## 二、tiling 与 workspace 布局（方案根基）

| 验证项 | 说明 | 代码位置 |
|---|---|---|
| `workspaceSize = libApiSize + 64KB` 框架实际分配 | GetWorkspaceSize | tiling_base.h L735-750 |
| 段表填充正确 | seg0={0, libApiSize, LIB_API}，seg1={libApiSize, 64KB, RUNTIME_INFO} | tiling_base.h L709-714 |
| tiling 结构体大小 | 新增 `workspaceLayout` + `peermemDataSize` 末尾字段后，总大小 ≤ 2048B dump 上限，且不影响二进制兼容（结构体 host/device 共用） | all_to_all_matmul_tiling_data.h L30-31 |

### tilingdata 验证方法（日志比对）

- 在 DoOpTiling 末尾加日志，字段级打印完整 tiling 结构体（含 sizeof），或使用现有 tiling debug 日志开关
- 触发异常 dump 后，用与 device 相同头文件（all_to_all_matmul_tiling_data.h）编译的解析脚本按 `allToAllMatmulTilingData` 布局解析 `tiling_data_*`（2048B 二进制），与日志逐字段比对
- 单 case 隔离运行，避免并发 op 的日志行与 dump 文件错配
- 比对覆盖全链路：host 计算值 → 序列化 → 框架搬运 tilingGM → 异常 D2H dump
- 逐字段不一致时先排查 padding/对齐（日志打的是 host 结构体，解析按同头文件应一致；若不同说明序列化有差异）
- 对账项：
  - workspaceLayout：segCount=2、seg0={0, libApiSize, LIB_API}、seg1={libApiSize, 64KB, RUNTIME_INFO}、totalSize = workspaceSize_，与 GetWorkspaceSize 分配值对账
  - peermemDataSize：与实际通信数据量（各 rank A 分块 + ScaleA 总量）对账
- 有效内容长度 = sizeof(allToAllMatmulTilingData)，2048B 尾部为 padding，不做比较

## 三、RUNTIME_INFO 写入正确性（workspace 指针两面性——最易出错处）

| 验证项 | 说明 | 代码位置 |
|---|---|---|
| 正常跑完后 dump `workspace_seg1_type12_*`，各核 slot 有 `magicNum=0x5A5A5A5A` | 验证 `userOffset = libApiSize - libApiSize = 0` 定位正确，不越界不崩溃 | mc2_runtime_info.h L51 |
| AIC/AIV slot 不重叠越界 | AIC=[0, aicCoreNum)，AIV 排后；核总数 ≤128（64KB/512B） | mc2_runtime_info.h L62-65 |
| host dump 出的 RUNTIME_INFO 与 kernel 写入一致 | 验证"kernel 用跳过 LIB_API 的指针、dump 用原始指针+段表 offset"两条路径指向同一物理地址 | mc2_runtime_info.h L41-56 |
| 正常结束核的 phase = `RT_PHASE_COMM_FINALIZE` | 验证 finalize 打点生效 | urma_impl L269-273 |

## 四、打点语义

| 验证项 | 说明 | 代码位置 |
|---|---|---|
| RunAllToAll：commit 前 POS_COMM_BEFORE → IncCommitCount → Wait 后 IncWaitCount | 通信全流程打点 | urma_impl L242-258 |
| RunMatmul → POS_COMP_CUBE_MATMUL_BEFORE；RunLocalMatmul → POS_COMP_LOCAL_BEFORE | 计算前打点 | urma_impl L380/L360 |
| localMatmul=1 与 =0 两种模式打点均可达 | 两条路径 | urma_impl |

## 五、异常 dump 产出完整性（触发一次真实异常）

对照设计文档产出表逐项检查 6 个文件：

| 文件 | 内容 | 大小 | 来源 |
|---|---|---|---|
| `args_info_*` | kernel args 全部 GM 指针 | 256B | Mc2DumpTilingData |
| `tiling_data_*` | tiling 结构体（含 workspaceLayout + peermemDataSize） | 2048B | Mc2DumpTilingData |
| `workspace_seg0_type0_*` | LIB_API 段原始数据 | ~16MB | Mc2DumpTilingData |
| `workspace_seg1_type12_*` | Mc2RuntimeInfo 每核执行进度 | 64KB | Mc2DumpTilingData |
| `comm_context_*` | CommContext 通信拓扑 | ~1KB | Mc2DumpUrmaContext |
| `peermem_window_*` | peermem 窗口数据（AllToAll 后的 A+ScaleA） | 1MB | Mc2DumpUrmaContext |

## 六、Mc2DumpUrmaContext 专项

| 验证项 | 说明 | 代码位置 |
|---|---|---|
| `UrmaCommContextForDump` 与 device 上真实 `CommUdmaContext` 运行时布局一致 | 静态看字段一致（u32+u32+u64[64]+u64[64]），但需确认 device 侧实际写入的 CommContext 无额外字段/padding | moe_distribute_comm_ctx.h L48 vs collective_comm_context.h L28 |

### comm_context 验证方法

CommContext 是 device 运行时数据，host 日志拿不到，用以下锚点验证：

| 验证项 | 方法 |
|---|---|
| 布局解析正确 | 脚本按 UrmaCommContextForDump 解析 `comm_context_*` 出值正常，rankId < rankSize ≤ 64；落盘大小 = sizeof(UrmaCommContextForDump) = 1032B，可对账 |
| 已知值锚点 | rankId/rankSize 与启动脚本的 rank 配置（HCCL/环境变量）一致；channelHandles 非零个数与实际建链 rank 数一致 |
| 拓扑自洽 | commBufferAddrs 非零项对齐且互不相同（各 rank 窗口地址）；commBufferAddrs[rankId] 非零——即 peermem_window 的 dump 起始地址 |
| 跨文件对账 | peermem_window_* 成功落盘本身证明 CommContext 解析链路（contextAddr → D2H → rankId 取 selfWinAddr）正确 |
| 反向注入 | 伪造非法 rankId（≥64）/空 selfWinAddr，确认走校验分支不崩溃 |
| rankId<64、selfWinAddr≠0 校验路径 | 坏数据不崩溃 | mc2_exception_dump.h(utils) L1028-1035 |
| peermemDataSize 逻辑（实现比设计多了此能力） | tiling 读到有效值(0<size≤64MB)用之；否则回退 1MB WIN_SIZE | mc2_exception_dump.h(utils) L1043-1064 |
| peermem_window 内容可读且是 AllToAll 后的 A+ScaleA | 与正常输出比对确认 dump 的是有效数据 | |
| dump 大小覆盖全部有效数据 | tiling 填的 peermemDataSize ≥ 实际各 rank A 分块 + ScaleA 总大小，否则 dump 截断，尾部缺失会被误判为通信错误 | tiling_base.h |
| 窗口内数据布局可解析 | 窗口按 rank 顺序排列 A 分块 + ScaleA，各块 offset/大小需与 commTilingData 对应；建议写解析脚本按 rank 提取分块 | |

### peermem 数据正确性验证方法

| 验证项 | 说明 |
|---|---|
| golden 比对 | AllToAll 是置换：窗口中第 i 个 rank 分块应等于 rank i 本地 x1/x1_scale 的对应切片；构造已知输入（如按 rank 编号填充），脚本逐块比对 |
| 比对时序前提 | 仅在 commit==wait（通信已完成）时窗口才是终态，golden 比对须在此前提下做；commit>wait 时数据可能 in-flight/torn，差异≠通信错误（需结合 RUNTIME_INFO 判断） |
| 最佳注入点 | 在 POS_COMP_CUBE_MATMUL_BEFORE（RunMatmul 打点处）注入崩溃：此时通信已完成、数据未被计算消费，窗口为终态，可完整 golden 比对 |
| 单 rank 视角局限 | 回调只 dump 本 rank 窗口（commBufferAddrs[rankId]）；完整通信正确性验证需各 rank 均触发 dump 后交叉比对，或用源端输入推断 |

## 七、故障注入（验证"崩溃定位"核心价值）

| 场景 | 预期现象 |
|---|---|
| hang 在通信 Wait（如断点/只 commit 不 wait） | commit > wait |
| 在 RunMatmul 中注入崩溃 | position = POS_COMP_CUBE_MATMUL_BEFORE |
| 通信数据错误 | peermem_window 数据比对可发现 |
| 拓扑配置错误 | comm_context 中 rankId/rankSize/bufferAddrs 异常 |

## 八、回归与边界

- 非 Ascend950（strstr 不匹配型号）回调直接 return，无 dump 也无副作用（tiling.cpp L66-68）
- hcomm（GET）路径同文件共存打点，确认互不影响（all_to_all_mx_quant_matmul_hcomm_impl.h）
- workspace 多占 64KB 后内存对齐、多并发 stream 场景

## 崩溃定位信息对照（复核产出用）

| 文件 | 回答的问题 |
|---|---|
| tiling_data_* | 算子参数对不对（M/N/K/tiling/coreNum） |
| workspace_seg1_type12_* | 卡在哪一步（execTurn/execPosition/commPhase），commit≠wait 说明卡在通信 Wait |
| comm_context_* | 通信环境对不对（rankId/rankSize/channel/bufferAddrs 拓扑） |
| peermem_window_* | 通信后的数据对不对（A 矩阵 + ScaleA 内容） |
| args_info_* | 所有 GM 地址对不对 |

## 优先级建议

三（指针两面性）> 五/七（dump 产出与定位价值）> 六（布局镜像）> 其余回归项。

其中"tiling 结构体变大后的兼容性"和"CommContext 运行时布局镜像"是设计文档未展开、但最容易翻车的两个点。
