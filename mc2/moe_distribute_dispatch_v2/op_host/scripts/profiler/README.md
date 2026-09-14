# A5 FullMesh Dispatch 分段打点

这次只在 `zxy_pr` 的 FullMesh 内核添加打点，不修改分核、量化、路由、搬运或原有同步。采集与分析合在根目录 `moe_operator_test_v3v4_dispatch_prof_dump.py` 中，源自 `3fee851400a4bacf1f0e6b3d37dd08016ea86437` 的测试脚本，保留顶部参数配置、golden 生成、运行流程和 CSV/SVG/trace 格式。

设备侧参照 `fullmesh_core_debug`：各核暂存事件与累计指标，Process 结束后，排空并全核同步，再统一 Flush 到各自记录区。每个 rank 只保留最后一次调用；循环内不写 profiler GM，也不生成逐 token 记录。

## 使用限制

- 调试开关 `MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_PROFILING` 默认 1，在 `op_kernel/arch35/moe_distribute_dispatch_v2_full_mesh_profiler.h` 中设为 0 可关闭。
- 与参考分支一致，记录覆盖 **expand_x 起始 115264 字节**。覆盖后的张量不能作为 combine 的业务输入，也不能用于 dispatch 精度验收。脚本默认运行 `is_continue=1`，要求保留 `is_continue_golden_expand_x=True`：图模式与非图模式均保留原始 dispatch 输出用于导出打点，combine 使用 golden expand_x 和实际 dispatch 返回的计数、索引等参数，继续执行原有 combine 结果比对。这种模式用于级联调试，不验证 dispatch payload 的端到端正确性。也可切换为仅 `is_dispatch=1`，导出并分析后跳过 dispatch 输出比对；两种模式择一，不支持单独 combine 或 CPU-only 采集。
- expand_x 容量不足时内核不写 profiler，脚本会报容量或 magic 错误；不扩大业务输出、不越界写入。实际 AIV 核数支持至 72。
- 所有测量使用原有流水与等待；没有为了分段计时在测量区域增加 barrier。因此搬运、量化事件是**代码下发区间及其原有等待**，不是独立硬件引擎耗时。量化尾部等待可能体现在后面的 metadata/V_S 区间，异步广播等待可能体现在发布 flag 或后续 drain。
- `KERNEL` 是 Process 中标记的区间，不含 Init、末尾诊断用全核同步和 Flush；不是 profiler 中的完整算子时延。最后的输出排空、全核同步属于导出保护，置于 KERNEL_END 后。
- 每次 `GetSystemCycle`、累计量写入以及暂存事件本身都有扰动。前后必须使用同一套打点；不能把带打点结果直接与无打点端到端耗时相减。

## 编译和采集

在有目标 CANN SDK 的环境，沿用原有算子包构建、安装流程，例如仓库支持的子包命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend950 --ops=moe_distribute_dispatch_v2 -j16
```

安装生成的包、加载其环境，再将根目录脚本放到原测试目录（或把该目录加入 PYTHONPATH）。脚本依赖与参考 commit 相同，包括 `toolss.py`、`quantize.py`、torch_npu、torchair 和 cann_ops_transformer；这里没有替换运行框架。`OPS_TRANSFORMER_TORCH_EXTENSION` 可指向实际 torch_extension。

保持顶部编辑参数的格式：

```python
is_dispatch = 0
is_combine = 0
is_continue = 1
is_continue_golden_expand_x = True  # combine 使用 golden，原始 dispatch 输出留给打点导出
comm_alg = 'fullmesh_v2'
rank_per_dev = 8           # 按真实进程/设备映射设置；内部启动 rank，不再外套 torchrun
profile_cycles_per_us = 1000.0
```

原参考脚本的其他 shape/quant/seed 默认值没有替换成用户用例；运行前需要填写原始用例。ID020 对照需对齐 BS=192、H=7168、K=6、local_moe_expert_num=48、quant_mode=2、bf16→int8、seed=423、无零计算专家、原 scales/graph 设置。表格中的 commAlg=fullmesh_v1 与此参考采集入口 fullmesh_v2 不能静默视为相同路径；应确认所测是命中本 A5 FullMesh header 的相同实现。前后两次使用完全相同配置。

```bash
python3 moe_operator_test_v3v4_dispatch_prof_dump.py
```

每个 rank 输出 `fullmesh_v2_prof_rank<N>.bin`，主进程自动生成 `dispatch_profiler_result/`。采集前只清理本轮对应的旧 rank dump；若有 rank 没有生成 dump，分析会失败，防止误读旧数据。多机每个进程节点先分析本机 dump，需要全部 rank 时集中 bin 再离线分析。

微秒换算沿用参考 commit 的 1000 ticks/us 默认值。必须按实际 GetSystemCycle 时钟确认，不能把 AI Core 动态频率直接当作系统计数器频率；未确认时以原始 cycles 比较。

## 前后对比：同一个脚本离线分析

分别保存相同打点、相同用例的前后 bin。离线入口仅依赖 Python 3.10+ 标准库，不导入 NPU 测试依赖。

```bash
python3 moe_operator_test_v3v4_dispatch_prof_dump.py --parse-only before_bin/fullmesh_v2_prof_rank*.bin --output-dir before
python3 moe_operator_test_v3v4_dispatch_prof_dump.py --parse-only after_bin/fullmesh_v2_prof_rank*.bin --output-dir after --compare-baseline before
```

可用 `--cycles-per-us` 指定换算。旧格式 2/3 的 bin 可以解析，但新增细分段在旧 bin 中显示缺失，不能从旧粗粒度事件推导其耗时。要完整比较所有新阶段，需要把同样的打点移植到基线内核；移植时适配函数签名，不引入新分支上的发送算法。统计比较按角色而不是固定 coreId 分组，避免分核变化后把发送核与 cumsum 核混比。

重要输出：

| 文件 | 用途 |
|---|---|
| fullmesh_v2_events.csv | 原始事件、payload、cycle，沿用参考格式 |
| fullmesh_v2_stages.csv | 所有配对阶段的 cycle 区间 |
| fullmesh_v2_core_summary.csv | 每核角色、等待、发布和输出时刻 |
| fullmesh_v2_send_summary.csv | 最慢 TRANSFER_TOKENS 核的原有摘要 |
| fullmesh_v2_dispatch_detail.csv | 每 rank/core/stage 的 cycle/us、实际发送/cumsum 核数 |
| fullmesh_v2_dispatch_metrics.csv | 所有核的循环内累计量及发送/搬入/量化次数 |
| fullmesh_v2_comparison.csv | 前后各角色、各阶段/指标 mean/max、delta=after-before |
| fullmesh_v2_timeline.svg / fullmesh_v2_trace.json | 按 rank/core 的嵌套时间线 |

累计表中 `send_tasks`、`send_input_count`、`send_quant_count` 是次数，其余 `*_cycles` 为累计区间。格式 4 无调用的指标为 0；旧格式缺失字段保持空值。payload 的累计 cycle 达到 uint32 上限会显式报错，不能作为正常结果比较。事件超容量、缺 rank、空核、配对不完整、无 KERNEL 区间也会报错。

比较表的 max 变化只描述这一轮采样；不同阶段的 max 可能属于不同核，不能相加，也不能把嵌套父子区间相加。跨 rank 的绝对 cycle 不直接相减，时间线各 rank 按自己的最早事件归零。判断是否优化仍需重复采集排除波动。

## 分段含义

| 部分 | 事件/指标 | 定位问题 |
|---|---|---|
| 发送整体 | SEND_TOKENS / TRANSFER_TOKENS | 准备与完整发送路径是否变慢 |
| 输入准备 | PREPARE_TOKEN_INPUT | expertIds、mask、权重准备 |
| 发送累计 | SEND_POSITION_CYCLES | CalTokenSendExpertCnt 目标位置计算 |
| 发送累计 | SEND_INPUT_CYCLES / SEND_INPUT_COUNT | 输入搬入及现有队列等待；有附带 scales 时包含其搬入 |
| 发送累计 | SEND_QUANT_CYCLES / SEND_QUANT_COUNT | 量化或转换调用区间，不新增完成等待 |
| 发送累计 | SEND_METADATA_CYCLES | 三元组/权重填写及原 MTE2_S/V_S/S_V 等待 |
| 发送累计 | SEND_BUFFER_WAIT_CYCLES | 复用发送槽前的 MTE3_V 等待 |
| 发送累计 | SEND_PACK_CYCLES | Copy 打包及原 V_MTE3 等待 |
| 发送累计 | SEND_MTE3_ISSUE_CYCLES | payload DataCopy 与后续 SetFlag 下发 |
| 发送排空 | DRAIN_SEND_BUFFERS | 发送循环后原有全部槽位等待 |
| cumsum 整体 | CAL_CUMSUM | 完整 CalCumSum，不含 Process 后续 drain |
| 计数准备 | INIT_COUNT_BUFFERS / PREPARE_COUNT_INPUT | UB 配置与计数输入准备 |
| 专家统计 | INIT_COUNT_STATUS / COMPUTE_EXPERT_COUNTS / TRANSFER_COUNTS | 初始化、逐专家计数、计数发送；elastic 路径计算/发送交错，见累计指标 |
| 计数累计 | COUNT_COMPUTE_CYCLES / COUNT_ISSUE_CYCLES | 每次 CalExpertCnt；elastic 的交错发送下发累计 |
| 远端计数 | POLL_RECV_COUNTS / CLEAR_RECV_STATUS | 仅等待与清理，poll 的 end_payload 是轮询次数 |
| 局部和 | REDUCE_LOCAL_COUNTS / PUBLISH_LOCAL_SUM | 本核归约与 C 份局部和发布 |
| 核间前缀 | POLL_CORE_SUMS / REDUCE_CORE_PREFIX / CLEAR_CORE_SUMS | C 核就绪等待、前驱和归约、列清理 |
| 本核前缀 | BUILD_RECV_CUMSUM | 当前负责状态段的标量前缀循环 |
| 前缀发布 | WRITE_COUNTS_OUTPUT / BROADCAST_PREFIX / PUBLISH_PREFIX_FLAGS | 输出、A 份 workspace 复制、完成标记发布 |
| 接收侧 | WAIT_CUMSUM / READ_PREFIX / FORMAT_OUTPUT | 前缀等待、本卡前缀读取、窗口数据等待与整理 |
| 收尾 | CLEAR_LOCAL_FLAGS / DRAIN_PIPELINES | 输出标记清理及 Process 原有流水排空 |

WAIT_RECV_COUNTS、GATHER_RECV_COUNTS、GET_CUMSUM、PUBLISH_RECV_COUNTS 等旧事件保留为外层区间，内部新事件进一步拆解。共享专家核没有 MoE 目标位置计数，SEND_POSITION_CYCLES 为 0。未分配任务/无活跃 token/无有效接收专家有对应空路径事件。

## 本地验证边界

已执行 Python 语法检查、模拟 bin 的解析/前后比较/旧格式/异常输入检查，以及 profiler header 的 C++ host stub 测试（格式、逐核槽位、累计值、复位和 overflow）。去除新增 profiler 语句后，原内核 token 序列保持一致。

continue 路径已用 mock 验证非图执行函数和图模型函数体：combine 接收 golden，导出保留最后一次原始 dispatch 输出，覆盖性能循环、elastic 和 performance_info 分支。此检查不涉及真实图编译或设备执行。

本机 `/usr/local/Ascend/ascend-toolkit/latest` 指向的 CANN 目录不可用，未执行真实 AscendC 编译、8p 采集或精度验证。host stub 测试不证明设备代码编译与缓存可见性；设备侧仍需按上述流程验证。
