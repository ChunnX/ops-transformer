# FfnWorkerBatching arch35(Ascend950/A5)代码检视报告

> 范围：本次 A5 新增/改动的 arch35 delta（`op_kernel/arch35/`、`ffn_worker_batching_apt.cpp`、`op_host/ffn_worker_batching_tiling_arch35.cpp`、`ffn_worker_batching_tiling.cpp` 分派守卫）。A2/A3 分支（ascend910b/ascend910_93 config、非 arch35 顶层 kernel 头、A2/A3 tiling）与 arch35 头中与 A2 逐字节一致的部分不在深审范围（A2 已在产）。
> 方法：假设检验 + 逐项取证（file:line）。检视人：内联（主 Agent），阶段0 概要/条例路由由子 Agent 产出。
> 结论：**delta 无阻塞级代码逻辑缺陷；1 个 HIGH 需在泛化阶段实测坐实（UB 大 shape），其余为设计内/已验证/待覆盖。**

## HIGH — 需泛化/精度阶段实测坐实

### H1. UB 口径大 shape 溢出风险未验证（文档结论自相矛盾）
- **位置**：`ffn_worker_batching_tiling_arch35.cpp:220-221`
- **现状**：`sortLoopMaxElement = ubSize_ / (4B*2*4) / 32 * 32`，其中 `ubSize_` = host `GetCoreMemSize(UB)` = **248KB(253952)**，算得 `sortLoopMaxElement=7936`。
- **矛盾**：
  - `INTEGRATION_REPORT.md §4` 结论：以 `GetCoreMemSize(UB)`=248KB 为 host/kernel **唯一真值源**（即 248KB 可用，代码按此走）。
  - `PROBE_SUMMARY`(迭代二穿刺2) 结论：kernel 侧 `UBUF_PER_VECTOR_CORE`=**216KB**，host 派生 SLM=7936 → 多核 sort 每核 4-buffer footprint `4*7936*8=253952=248KB` **超物理 216KB 达 32KB → OVERFLOW**，主线应改按 `min(host,kernel)` 派生。
- **判定**：**SUSPICIOUS-HIGH**。代码走了 248KB 路径；两份文档对"248KB 是否安全"结论相反；大 shape（A=1024、topK+1=64、单核每 loop tile 打到 SLM）NPU 实跑**未做过**（小 shape probe 已过，未触发）。
- **动作**：泛化/精度阶段**必须**跑大 shape 用例（A=1024×BS×K=64、Y 使单核元素≥7936）观察是否 507035/踩内存。若溢出→按穿刺2 结论改 `sortLoopMaxElement` 分母用 `min(host_ub, kernel_ub)` 或实测每核可用 UB（仍禁写死 arch 常量）。

## MEDIUM / LOW — 确认 OK 或设计内

### M1. RECV `while(true)` 忙等无超时（设计内，A2 一致）
- **位置**：`ffn_wb_scan_token_info.h:100-122`，仅 block0 轮询 `readyNum==A` 后 break。
- **判定**：分离部署 RECV 协议固有设计（FFN worker 等 attention worker 置位），与 A2 byte 一致移植，依赖上游契约。**非新增缺陷**。逻辑正确（DataCopyPad 每 session 1 flag padding 到 32B，ReduceSum 求和，全 ready 退出）。低。建议：长期可加最大轮询次数兜底（非本次范围）。

### M2. k=0 守卫充分性（单核已修，多核待 NPU 覆盖）
- 单核 NORM 路径（`group_listing.h` `groupListingDealFlag==0`）：已加 `validGatherIdxLength>0` 守卫，输出 `group_list[0]=[0,0]` 正确。
- `firstExpertId` 在 k=0 被赋脏值，但**全文件无读取（dead write）**，不外泄。安全。
- 多核路径（`dealFlag==1`）：靠 `expertCountCompute` 中 `count>0` 过滤 + 末尾 `(actualExpertTotalNum-1)*2+2` 索引在 actualExpertTotalNum=0 时恰好落 0 → 产 `[0,0]`（正确但索引算术略脆）。**多核 k=0 未 NPU 实测**（probe 只覆盖单核）→ 泛化补一条多核 k=0 用例。

### 确认 PASS 项
- **P1 Tiling 校验完整性 ✅**：spec `boundary_conditions` 全部错误契约都有 `OP_CHECK→GRAPH_FAILED`：int8 dtype(:124)、x 1维(:131)、expert_num∈(0,8192](:143)、max_out_shape len=4(:150)、A∈(0,1024](:158)、BS>0(:161)、K∈(0,64](:164)、H>0(:167)、token_dtype∈[0,2](:175)、need_schedule∈[0,1](:183)、layer_num∈[0,expert_num](:191)。与新增 6 条 arch35 负路径 UT(E1-E6)对应。
- **P2 分派隔离零回归 ✅**：`REGISTER_OPS_TILING_TEMPLATE(...,1000)` + `IsCapable()=IsRegbaseSocVersion(context_)`，仅 Regbase SoC(ascend950) 激活；A2/A3 走 monolithic 模板，类体零改动、无 `IMPL_OP_OPTILING` 符号冲突。
- **P3 TilingKey 语义 ✅**：`GetTilingKey()` 仅由 need_schedule 决定 NORM(100)/RECV(101)，token_dtype 正交、kernel 内处理；保留分支 102 从不产出。
- **P4 UB/核数运行时取值 ✅**：aivNum=`GetCoreNumAiv`、ubSize=`GetCoreMemSize`、sysWorkspace=`GetLibApiWorkSpaceSize`，无硬编码 arch 常量（符合项目约束）。
- **P5 int8 dynamic_scale 路径**：probe3(int8单核)+迭代二穿刺1(int8 RECV) 已 NPU 二进制一致，本次静态未见新问题；全 dtype×路径矩阵留泛化。
- **P6 死代码**：`ffn_wb_scan_get_valid_experts.h`（102 RECV_GATHER 保留分支）apt.cpp 不 include/不分发/不编译，按 DESIGN 有意保留，非缺口。

## 待验证阶段承接的清单
1. **【H1】大 shape UB 溢出**（头号，泛化必跑）。
2. **【M2】多核 group_listing k=0** 用例（泛化补）。
3. 全 dtype(fp16/bf16/int8)×2 路径(NORM/RECV)×边界(k=0/k=N/same_bin/max-shape) NPU 全量对拍 A2（本次仅 probe 抽样验证过）。
