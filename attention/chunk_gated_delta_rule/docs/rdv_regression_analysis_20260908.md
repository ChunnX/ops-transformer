# CGDR RDV 性能分析：新版 Streaming 验证（2026-09-08）

**最新结论：64-token批处理和PackedRows已消除之前ASC-024的性能退化。新版约1.64～1.66ms，比旧Streaming快11.60～11.73×，也比同环境复测的强制融合Cube快约6%～8%。因此撤回旧版对ASC-024禁用Streaming的建议。ASC-026虽明显改善，但Cube仍更快，应单独校准分流。**

本轮完成288例性能、8例RDV精度及24例补充边界精度测试，并采集新版msopprof分核指标。以下先记录最新版；文末完整保留旧版本分析作为历史对照，旧建议不代表当前结论。

关键结果已按版本归档到 [docs/rdv_perf](rdv_perf/README.md)：[旧Streaming与CANN全量对比](rdv_perf/old_streaming/comparison.csv)、[强制Cube与CANN对比（8例）](rdv_perf/forced_cube/comparison.csv)、[新Streaming与CANN全量对比](rdv_perf/new_streaming/comparison.csv)。跨版本重点用例与msopprof摘要保存在 `rdv_perf/cross_version/`。归档对比表已合并独立精度结果：强制Cube和新版各8例标记为 `True`；没有对应通过记录的行保留 `False`，不代表精度失败。

## 新版源码与 rebase

原工作区处于以 `121832027ced531c2b057713b96bf5d0bc600b72` 为基底的rebase，`76bcecade` 与新版Streaming发生冲突。已保留基底中的新版实现，原优化提交的重复内容不再重复应用；原中文注释提交合并为 **`6f599eb3a`（update comments）**，分支为 `improved_cgdr`，rebase已完成。

6个源码文件去除注释后的代码标记与 `121832027` 完全一致，合并没有退回旧逐token实现；中文注释已适配批处理、PackedRows及通用BatchVF。旧性能报告作为独立工作区文档保留，未混入历史注释提交。[合并核对记录](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/rebase_token_validation.json)。

构建测试源码为rebase后的 `6f599eb3a`，独立目录 `/home/s00940320/cgdr_stream_update_121832027`；构建命令：

```bash
bash build.sh --pkg --soc=ascend950 --ops=chunk_gated_delta_rule --vendor_name=cgdr_stream_update
```

测试包安装到本次数据目录 `opp/vendors/cgdr_stream_update_transformer`，测试进程通过环境变量选择。系统默认custom包仍是旧版，包哈希单独审计；本文数据必须用所列实验vendor复现。测试采用device0、eager、seed=20260908，与此前输入生成流程相同；新增测试代码差异仅为未启用的graph配置，不影响eager测量。

## 最新性能结果

### 受影响的8例

新Streaming与同环境强制Cube各预热4次、测量20次；旧Streaming取前次同口径20次测量，CANN原版取原始5次测量。单位 **µs**，均为msprof `ChunkGatedDeltaRule` 的设备 `Task Duration(us)`，不包含输入生成、外部Cast或host耗时。强制Cube使用前轮独立消融包，本轮重新采集；它不是新版二进制中动态强制切换后的结果。

| 用例 | CANN原版 | 旧Streaming | 新Streaming | 本轮强制Cube | 新/旧Streaming加速比 |
|---|---:|---:|---:|---:|---:|
| ASC-024 | 6120.496 | 19202.455 | 1636.958 | 1762.016 | 11.73× |
| ASC-026 | 392.476 | 228.863 | 67.440 | 51.086 | 3.39× |
| ASC-024-FP32 | 6708.158 | 19194.613 | 1654.184 | 1764.250 | 11.60× |
| ASC-026-FP32 | 429.401 | 227.458 | 67.173 | 52.558 | 3.39× |
| ASC-024-NC | 6114.643 | 19191.039 | 1653.878 | 1762.890 | 11.60× |
| ASC-026-NC | 392.271 | 227.797 | 65.767 | 50.959 | 3.46× |
| ASC-024-FP32-NC | 6720.302 | 19193.851 | 1655.335 | 1766.200 | 11.60× |
| ASC-026-FP32-NC | 426.640 | 226.700 | 65.473 | 52.891 | 3.46× |

ASC-024新版耗时比内置版减少约73%～75%；相对本轮Cube耗时减少约6.2%～7.1%。ASC-026新版相对旧Streaming加速3.39～3.46×，但比Cube耗时仍高约24%～32%。ASC-024新版20次样本CV为0.73%～1.57%，ASC-026为2.89%～4.66%，不足以解释与旧Streaming的数量级差异。[逐例对比](rdv_perf/cross_version/selected_comparison.csv)、[逐次样本](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/selected/kernel_samples.csv)。

### 完整288例

- 性能执行 **288/288 PASSED**，共1728条CGDR记录（每例1次预热、5次正式测量），相对CANN原版 **288例均更快，0例退化**。
- 对CANN原版的几何平均加速比：**5.724×**（旧优化版5.437×）。
- 288例均值之和：旧优化版 **576.024ms → 新版 506.359ms**，降低 **12.09%**；相对CANN原版的该项加速比为 **7.084×**。这是测试集各case均值等权相加，不代表某个实际业务的端到端加速。
- 全量批跑有18例CV超过5%，因此保留逐次数据并对可能新增退化的控制组补测，不把全量均值当作无噪声结果。[全量对比](rdv_perf/cross_version/version_comparison.csv)、[统计JSON](rdv_perf/new_streaming/summary.json)。

首轮288例相对旧优化版有10例慢超过5%；同环境4次预热+20次测量后8例不再超过5%，仍有ARC-031-FP32/ARC-031-NC两例差异较大。这两例均为B=1、L=32768、Nv=4、Dk=Dv=128的稀疏head Cube场景。随后倒转顺序为新版→旧版，各20次预热+100次测量：

| 用例 | 旧版100次均值 µs | 新版100次均值 µs | 新版耗时变化 | 旧版CV | 新版CV |
|---|---:|---:|---:|---:|---:|
| ARC-031-FP32 | 1649.804 | 1724.752 | +4.54% | 6.97% | 3.91% |
| ARC-031-NC | 1554.811 | 1527.041 | -1.79% | 4.68% | 6.78% |

100次复核中两例新版/旧版均值变化为+4.54%和−1.79%，中位数仅分别高约0.12%和0.08%，未复现持续超过5%的差异。该组结果与首轮及20次复核一起保留，见 [控制组对比](rdv_perf/cross_version/control_comparison.csv)、[100次确认](rdv_perf/cross_version/confirmation.csv)。这些Cube场景存在跨轮波动，不能把单次批跑的5%差值全部归因Streaming实现变化，也不能据此宣称所有Cube形状逐例无退化。正式288例统计仍使用统一的1次预热+5次测量，没有用补测覆盖或挑选原样本。

## 改动为何有效

[批处理入口](../op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h#L942) 每次最多处理64个token。Q/K/V、beta/g通过带stride的多block请求批量搬入，一次VF内沿token顺序递推，输出积累后批量写回。ASC-024每head有157批，96个head合计15072批，源代码层面的输入输出DMA调用 **5,760,000→90,432**，VF调用 **960,000→15,072**，约减少63.7倍。每个请求仍含多个短block，不代表HBM transaction或有效张量字节数减少63.7倍。

[Dk≤8的PackedRows](../op_kernel/arch35/vf/chunk_gated_delta_rule_stage2_vf.h#L248) 把最多8个独立Dv行打包进一个FP32寄存器。ASC-024状态运算的有效lane由每行5/64变为40/64；每token的16次ReduceSum变为2次ReduceDataBlock，Q/K转换结果跨行复用。状态在64个token内保留在寄存器，批首Gather、批尾Scatter，显著减少UB状态读写。UB主状态本身仍是stride128，紧凑的是寄存器布局。

[Dk>8的BatchVF](../op_kernel/arch35/vf/chunk_gated_delta_rule_stage2_vf.h#L327) 将状态行放在外层、token放在内层，前后两个64特征片段在批内保留在寄存器，保持低半→高半的归约顺序。该分支不被288 RDV的8个Streaming用例覆盖，已通过补充边界测试验证。

分流公式、AIV的Dv切片和单缓冲依赖关系未改：ASC-024仍只有32/64个AIV承担计算，AIC不做矩阵乘；每批仍顺序执行MTE2→V→MTE3，尚未实现跨批预取。新版收益来自批量化、行打包和寄存器复用，而非改变分流或增加计算核数。

## 新版msopprof证据

对ASC-024采集 `Default,Occupancy,Roofline`，预热5次、匹配1个kernel并做指标重放。每张CSV有96条分核记录（32 AIC、64 AIV），下表对32个活跃vector0取均值，**不是96次独立测量**。

| 活跃AIV指标 | 旧Streaming | 新Streaming |
|---|---:|---:|
| Vector管线占比 | 28.591% | 62.710% |
| Scalar管线占比 | 7.142% | 2.386% |
| MTE2管线占比 | 23.575% | 25.914% |
| MTE3管线占比 | 45.248% | 10.648% |
| MTE2指令/核 | 300008 | 4718 |
| MTE3指令/核 | 30004 | 475 |
| GM→UB带宽利用率 | 0.512% | 6.089% |

新版每活跃核471批，MTE2约4718条、MTE3约475条，与批粒度变化吻合。AIC的Cube指令仍全为0，vector1的向量活动仍全为0。Memory.csv统计的读/写数据量与旧版接近，DMA指令却减少约64倍，支持收益主要来自减少固定搬运/同步开销和增加片上复用，而非减少GM逻辑张量大小。

本次msopprof记录时延为 **1845.500µs**，频率1650MHz；其采集/重放口径与正式msprof重复测量不同，不能混算加速比。Vector占比不是有效FLOPs峰值利用率，多个PIPE比例不能相加做墙钟时间分解。新版部分wait_ratio仍接近或超过100%，不能据此宣称几乎全部时间在等待；保留工具原字段，具体同步点耗时还需timeline或独立消融。

继续使用前轮的进程内ACL hook适配，正式采集在 `msopprof_clean/`，pytest和外层脚本退出码均为0。初次 `msopprof/` 的kernel采集成功，但主进程修改运行中的shell脚本造成收尾语法错误；该次保留为诊断记录，**本节仅使用重新完整采集的clean结果**。可导入MindStudio Insight的文件为 [visualize_data.bin](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/msopprof_clean/prof/OPPROF_20260908070335_IWOHOBZHSLVWWNJA/visualize_data.bin)。

## 精度覆盖与后续建议

- RDV中受影响的 **8/8例精度通过**，独立设置 `ENABLE_PROF=false`，使用仓库golden/benchmark及compare_cv检查attention和finalState；与性能模式PASSED分开记录。
- **24/24个补充边界用例通过**：L=1/63/64/65，Dk=1/5/8/9/64/65/128，Dv=1/8/9/64/65；包含PackedRows/通用BatchVF、Dk高半mask、Dv尾组及第二个AIV有效切片、BF16/FP32状态、NC、g开关、两个seed。脚本按分流公式断言每个用例会进入Streaming。它是有针对性的组合集合，并非上述维度的笛卡尔积。
- 本轮未执行288例全量精度回归，未覆盖全部正g/极值输入或任意混合变长分流；性能数据不能代替这些验证。[RDV精度日志](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/precision/run.log)、[边界精度结果](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/boundary_results.json)。

1. **保留ASC-024已证实有效的新版Streaming，撤回该形状禁用Streaming的旧建议。** ASC-026可评估走Cube或实现Dk=Dv=1专用VF，减少Gather、分段归约和padding转换的固定开销；按真实参数/测量交点分流，避免case名特判。
2. **先紧凑化小维度transport，再评估批间双缓冲。** 新版V/beta/out仍按每token128/16/64元素做转换，且out清零64元素；大量padding仍有成本。单槽Streaming UB末尾已到160KiB，不能把整套112KiB transport直接复制为双槽，否则48+112×2=272KiB超过248KiB限制。可缩小pitch、减少中间副本或只双缓冲关键输入输出。
3. **评估独立AIV分配head。** 当前ASC-024仍有一半AIV无计算；96个等长head改由64AIV分配时最长波数3→2，单调度改动理想上限约1.5×，需与现有mixed kernel协议协调，收益尚未实测。
4. **重新标定分流模型，并专门分析通用BatchVF。** 分流仍按旧算术权重，未体现PackedRows、批大小、has_g、有效lane和任务波数。通用路径现在每行每token计算Exp(g)，Dv较大时可能增加重复工作，可评估每批预计算alpha并复用。当前8个RDV Streaming结果不能代表该通用路径的性能。

## 本轮产物

[数据目录](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/) 包含构建/安装日志、完整profile、逐次kernel样本、运行环境与加载库、统计脚本、原报告备份和rebase核对。主要入口为 `run.sh`、`analyze_update.py`、`check_boundaries.py`；旧版和强制Cube复测脚本也随数据保留。静态机制详见 [源码复核](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/static_review.md)，分核原始字段详见 [msopprof审阅](../tests/pytest/output/rdv_perf_20260908/streaming_update_121832027/msopprof_review.md)。

---

# 历史记录：旧逐token Streaming（8432b2351）

**下文为更新前报告，耗时、源码行号和解决建议均对应旧版本；如与上文冲突，以新版实测结论为准。**


**结论：四个 ASC-024 变体的主要问题是错误选择了逐 token Streaming 分支。只改变分流，改走当前融合 Cube 后，设备 kernel 耗时由约 19.2 ms 降至 1.762 ms（10.9×），并比 CANN 内置原版快 3.47～3.81×。** 四个 ASC-026 对照用例也由约 227 µs 降至 53 µs。8 个消融用例均通过仓库现有 attention 和 final state 精度校验。

本次只在独立工作目录做分流消融；当前分支算子源码及默认安装的 custom 包保持原状。性能数据与消融 patch 位于 [regression_analysis](../tests/pytest/output/rdv_perf_20260908/regression_analysis/)。

## 1. 实测与可比性

原始 288 RDV 中只有 ASC-024、ASC-024-NC、ASC-024-FP32、ASC-024-FP32-NC 比 CANN 内置版慢；它们共有 B=3、每序列 L=10000、T=30000、Nk=16、Nv=32、Dk=5、Dv=8、has_g=True，输入 Q/K/V/beta 为 BF16。FP32 后缀仅指状态 dtype，NC 仅指初态不连续。

上轮复核四例约 18.60～18.62 ms、20 次样本 CV ≤0.132%，已排除偶发抖动。本轮重新用相同 pytest RDV 输入生成流程、seed=20260908，在 device 0 串行采集默认 Streaming 和强制 Cube。每例每个实现 4 次预热、20 次测量；两组各 192 条 CGDR kernel 记录，逐例校验输入输出 shape、顺序、seed 和 PASSED 状态。

表中单位均为 **µs**，统计的是 msprof `OP Type=ChunkGatedDeltaRule` 的 `Task Duration(us)`，不包含随机输入生成、外部 Cast、host 或其他算子。原版列沿用上轮 5 次均值，原版是 CANN 9.2.0 built-in，非 Git 父提交重新编译结果。

| 用例 | CANN 原版 | 本轮 Streaming | 强制融合 Cube | Cube 相对 Streaming | Cube 相对原版 |
|---|---:|---:|---:|---:|---:|
| ASC-024 | 6120.496 | 19202.455 | 1761.844 | 10.90× | 3.47× |
| ASC-026 | 392.476 | 228.863 | 53.309 | 4.29× | 7.36× |
| ASC-024-FP32 | 6708.158 | 19194.613 | 1762.928 | 10.89× | 3.81× |
| ASC-026-FP32 | 429.401 | 227.458 | 53.333 | 4.26× | 8.05× |
| ASC-024-NC | 6114.643 | 19191.039 | 1761.445 | 10.90× | 3.47× |
| ASC-026-NC | 392.271 | 227.797 | 53.094 | 4.29× | 7.39× |
| ASC-024-FP32-NC | 6720.302 | 19193.851 | 1762.479 | 10.89× | 3.81× |
| ASC-026-FP32-NC | 426.640 | 226.700 | 52.931 | 4.28× | 8.06× |

ASC-024 本轮 Streaming CV 为 0.065%～0.512%，Cube CV 为 0.027%～0.058%；CV 使用样本标准差（ddof=1）。本轮 Streaming 比上轮复核高约 3%，不影响数量级结论，未把两轮样本拼成同一个均值。ASC-026 的 Streaming 原先比内置版快 1.72～1.89×，但本次同输入消融表明融合 Cube 仍比 Streaming 快约 4.3×，因此不能仅凭其相对内置版有收益就保留现有分流判断。

消融只将非空序列的 `UseStreamingExact()` 返回值改为 false，空序列仍走状态复制逻辑。候选包使用独立 vendor 和安装目录，通过环境变量在测试进程中选择。加载库记录包含新 vendor，且消融 ASC-024 的 AIC MAC 管线时间指标由 0 变为约 562 µs，确认实际进入了 Cube 路径。默认 custom 二进制、头文件、op_api 库 SHA-256 及 vendors/config.ini 均与实验前相同。

精度单独设置 `ENABLE_PROF=false`，对同一批 8 例执行仓库 golden/benchmark/compare_cv，检查 attention 与 final state，**8/8 通过**；不是把性能模式 PASSED 当作精度通过。该验证覆盖本次 8 例、一个随机种子，尚未代表候选已通过完整 288 例精度回归或所有输入分布。

## 2. 原因链

### 分流模型低估固定成本

源码 [UseStreamingExact](../op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h#L328) 比较：

```text
streamScore = L × Dk × Dv × 7 × 256
cubeScore   = ceil(L / 64) × 2 × 5832704
```

ASC-024 得到 716800000 < 1831469056，因此选择 Streaming。L 较大时近乎相消，判断趋近 `Dk×Dv < 101.714`。此模型没有逐 token 的 DMA 描述符、VF 调用、同步、有效向量 lane 和活跃核数/任务波数等成本。低逻辑算术量被误当成低执行时间。对当前 288 例按代码重算，只有 ASC-024、ASC-026 及其变体（8 例）走 Streaming；实测这两组均更适合当前融合 Cube。

### 逐 token 微小搬运与串行依赖

[ProcessStreamingAivHead](../op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h#L931) 的单个 token 操作为：

```text
MTE2 Lock → Q/K/V/beta/g 五次短 DMA → Unlock
V Lock → Cast → Barrier → VF_CALL → Barrier → Cast → Barrier → Unlock
MTE3 Lock → 输出短 DMA → Unlock → 下一 token
```

ASC-024 每 token 有效输入只有 42 B（10+10+16+2+4），输出 16 B，却调用 6 次 DMA，使用同一 tokenMutex 和单组 token 缓冲区，无跨 token 预取/输出双缓冲。每个算子共 96×10000=960000 个 token-head，产生约 **576 万次 DMA 调用、288 万次 V barrier、96 万次 VF_CALL**。这些是源码级调用数量，不能等同最终硬件指令数量；55.68 MB 是逻辑输入输出量，不能等同实测 HBM 流量。

状态递推确实具有 token 间数据依赖，但未来 token 的输入搬运不依赖当前 state，已计算输出的批量写回也无需逐 token 完成。当前实现把可分块、可重叠的输入/输出成本放进每个 token 的串行关键路径。现有证据支持“细粒度搬运与同步延迟较高”，尚不足以把耗时归因于 HBM 带宽耗尽。

### 并行度及向量利用不足

[StateValidRows](../op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h#L318) 按两个 AIV 各处理 64 行状态切分。Dv=8 时第二个 AIV 有效行数为 0，直接返回；Streaming 下 [AIC 也退出计算分支](../op_kernel/arch35/chunk_gated_delta_rule_fused_exact_arch35.h#L801)。32 个 mixed group 只让 **32/64 个 AIV** 承担递推，96 个 head 分 3 波，每活跃核串行处理 30000 token。最终 SyncAll 使空闲核也可能显示长生命周期，不能把它当成有效计算。

[StreamingExactTokenVF](../op_kernel/arch35/vf/chunk_gated_delta_rule_stage2_vf.h#L252) 对 8 个 Dv 行逐行处理，每行只使用 64 个 FP32 lane 中的 5 个（7.8125%），两次归约，并重复加载/转换 Q/K。逻辑 Dk=5 并不意味着向量指令开销按 5/64 等比例减少。全例源码级 ReduceSum 调用约 1536 万次。

NC/BF16/FP32 主要改变初始及最终 state 搬运；Streaming 内部持续使用 FP32 状态，它们不改变 10000-token 主循环，所以四个变体耗时接近。NC 本身不构成这次约三倍退化的主因。

内置版按 chunk 执行三阶段，矩阵乘传实际逻辑 Dk/Dv，部分缓冲按 16 对齐；新融合 Cube 使用固定物理 128 tile。二者结构不同，本报告的止损建议依据上面的实际融合 Cube 消融，不是把内置版耗时当作新 Cube 耗时。

## 3. 管线证据与采集边界

本轮 ASC-024、20 次均值的 msprof 聚合计数如下。管线间可以重叠，跨核聚合方式也不同，**不能相加后把差值都归为同步等待**。

| 指标（µs） | Streaming | 强制 Cube |
|---|---:|---:|
| Kernel Task Duration | 19202.455 | 1761.844 |
| AIC MAC | 0.000 | 561.672 |
| AIV Vector | 3137.328 | 1704.669 |
| AIV Scalar | 784.384 | 188.787 |
| AIV MTE2 | 2251.233 | 526.534 |
| AIV MTE3 | 3784.510 | 212.015 |

数据与取消每 token 小输出、改为 chunk 流水的收益方向一致；但分流消融同时改变了计算/搬运/同步结构，不能据此给 DMA、VF 和同步各自分配独立加速百分比。

上轮内置版 1728 条记录有 1284 条各管线计数全部为 0，四个退化例亦如此，无法据此做可靠的旧版管线分解；默认 Streaming 还出现 cube_utilization≈99.99% 而 aic_mac_time=0 的字段组合，不能解释为 Cube 算力饱和。

已对 ASC-024 的默认 Streaming 与强制 Cube 分别完成 **msopprof `Default,Occupancy,Roofline`** 采集。每次匹配1个CGDR kernel、工具预热5次并进行指标重放，获得每核 `PipeUtilization.csv`、`Memory.csv`、`ArithmeticUtilization.csv`、`L2Cache.csv`、`ResourceConflictRatio.csv` 及可导入 MindStudio Insight 的 `visualize_data.bin`。每组有96条分核记录（32 AIC、64 AIV）；下表在同类32个子核内取均值。

| msopprof 分核指标 | Streaming vector0 | Streaming vector1 | Cube vector0 | Cube vector1 |
|---|---:|---:|---:|---:|
| Vector 管线时间占比 | 28.59% | 0.00% | 96.92% | 96.71% |
| MTE2 管线时间占比 | 23.58% | 0.009% | 33.50% | 32.05% |
| MTE3 管线时间占比 | 45.25% | 0.00% | 12.09% | 12.04% |
| MTE2 指令/核 | 300008 | 2 | 7526 | 7496 |
| MTE3 指令/核 | 30004 | 1 | 2830 | 2827 |
| GM→UB 带宽利用率 | 0.512% | 0.000027% | 3.816% | 3.806% |

Streaming 全部 AIC 的 Cube 指令总数为0，32个vector1的向量时间全部为0，与源码预测吻合。工具把 Streaming 判为 `latency bound:pipeline caused`；活跃vector0搬运管线占比不低但带宽利用率很低，支持大量短搬运/依赖链导致延迟，而非搬满内存带宽的解释。这里的 GM/L2 通路指标不能直接当作板卡 HBM 的独立带宽测量。切到 Cube 后AIC共135648条Cube指令，两组AIV均工作；Vector管线约97%提示后续可重点检查融合Cube中对小维度无效的固定128维Vector工作，**97%管线占比并不等于有效FLOPs达到峰值97%**。

硬件MTE指令数与前文源码DMA调用数不是同一计数口径。`ResourceConflictRatio` 部分等待字段接近99.8%，其分母/计数语义不能直接等同kernel墙钟时间，本报告没有把它写成“99.8%的时间都在等待”。未采逐指令或逐同步点timeline，因此尚未把具体锁/VF入口各自成本独立分解。

msopprof 的 Streaming 单次profile耗时约21.945ms，Cube约1.764ms，两组记录频率均1650MHz。本报告的正式加速比仍使用前面的msprof 4次预热+20次测量，不混合两种工具的样本：msopprof有重放/采集探针，其Streaming耗时明显高于msprof约19.2ms。

本机最初直接运行msopprof会得到空结果，原因定位为CANN 9.2的ACL函数表跳转绕过工具LD_PRELOAD Impl拦截。已使用公开 `aclrtApiInjectionGetFunc/SetFunc` 在被采集子进程内绑定工具的35个现有拦截函数，并逐项审计验证，随后成功采集；没有修改系统CANN库或全局配置。复现脚本为 `profile_hooked_stream.sh`、`profile_hooked_cube.sh` 和 `run_msopprof_hooked.py`，后者通过 `CGDR_MSOP_TARGET` 选择原pytest包装器或消融包装器。详见 [兼容性诊断](../tests/pytest/output/rdv_perf_20260908/regression_analysis/msopprof_diagnosis.md)、[分核汇总JSON](../tests/pytest/output/rdv_perf_20260908/regression_analysis/msopprof_core_summary.json)。工具指标与重放方式参考 [Ascend官方msopprof指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)。


## 4. 解决建议（按优先级）

1. **先修正分流，让本次已验证的两组参数走当前融合 Cube。** 本次 8 个 RDV 分流例全部获益，且精度通过。可把“非空序列暂统一走 Cube”作为 RDV 范围内的候选止损方案，保留空序列状态复制；也可保守限定已测参数范围。不能直接宣布所有极短序列都应禁用 Streaming。应以参数和实测代价决定，不按 case 名硬编码。提交正式修复前跑完整 288 例精度/性能，并扫描短序列交点。
2. **在融合 Cube 上探索小维度专用 tile/VF。** 新 msopprof 显示其 Vector 管线约97%繁忙，当前固定128维物理tile有大量padding。可评估16/32维特征tile、限制无效行的向量操作和归约工作，优先降低已实测最快路径的浪费。这需要独立模板/分支，协调UB/L1/L0布局、Cube维度、尾块mask及同步协议，不能只替换DIM常量。此方案尚未实现或测量收益。
3. **如果继续保留并优化 Streaming，先重做分块数据通路。** 一次搬入 16/32/64 个 token 的 Q/K/V/beta/g，在 UB 中串行递推，将输出积累后批量写回；用 ping-pong 两个槽重叠下一块 MTE2、当前块计算与上一块 MTE3。利用 TND 的 stride 做多块 DataCopy；只有 16/32/64-token 批量搬运成功才有机会降低描述符及锁的频率，不能只是把现有 token 循环外面再套一层循环。将多 token 递推合进一次 VF、复用已转换 Q/K，减少 VF 入口与状态装卸。
4. **给 Dk≤8 等小状态设计紧凑向量布局。** 可以按 8 个 Dv 行×8 个 padded Dk 打包 64 lane，做分段归约，避免逐行仅5 lane工作；进一步尝试状态跨 token 留寄存器。必须测量寄存器压力、归约成本和误差，不能假设 lane 填满就按比例提速。
5. **独立 AIV 分配 head，并实测校准代价模型。** 对本例，64 AIV 各处理完整 head 可把最长任务波数3→2，理想调度收益约1.5×，不是2×；仅把每 head 的8行拆为4+4行不会减少 token 固定开销。独立 AIV 调度应与当前 AIC/AIV 同步协议解耦。模型对等长序列的近似至少包含 `ceil(B×Nv/实际活跃AIV) × L × (每token固定成本 + 向量行/归约成本)`，变长序列应按各活跃核分配到的序列长度之和取最大值，Cube 模型包含 chunk 数、固定128物理tile与流水成本；has_g、状态dtype、尾块、空序列都需覆盖。

建议交点扫描：L=1/8/16/32/64/128/512/10000，Dk/Dv覆盖1/5/8/16/32/64/128及非对齐尺寸，B×Nv覆盖1/16/32/33/64/96，开/关g，BF16/FP32状态，连续/NC，变长与空序列。按照实测差值留出容忍抖动的分流余量；任何改变递推或归约顺序的候选均同时检查 attention 与 final state。

## 5. 复现与产物

- 源码提交：`8432b2351b719a9db294dfe1c6b91bff2bb5891b`；设备 Ascend950PR_9589，32 AIC / 64 AIV；CANN 9.2.0。
- 独立消融工作目录：`/home/s00940320/cgdr_cube_probe_20260908`；[force_cube.patch](rdv_perf/forced_cube/force_cube.patch)。
- 构建：`bash build.sh --pkg --soc=ascend950 --ops=chunk_gated_delta_rule --vendor_name=cgdr_cube_probe`。实验包安装在数据目录下的 `cube_opp/`，没有替换系统默认 vendor。
- [ablation_comparison.csv](rdv_perf/forced_cube/ablation_comparison.csv)、[ablation_summary.json](rdv_perf/forced_cube/ablation_summary.json)：8例统计。
- `msprof_cube/`、`msprof_stream_recheck/`：原始profile、逐次样本、逐例统计、pytest日志与加载库审计。脚本为 `profile_cube.sh`、`profile_stream_recheck.sh`、`analyze_ablation.py`。
- [精度日志](../tests/pytest/output/rdv_perf_20260908/regression_analysis/cube_precision/run.log)、[环境及二进制校验](../tests/pytest/output/rdv_perf_20260908/regression_analysis/environment_validation.json)。
- [msopprof指标独立审阅](../tests/pytest/output/rdv_perf_20260908/regression_analysis/msopprof_metrics_review.md)：逐角色均值/极值、等待字段语义边界与计数汇总。
- [源码详细分析](../tests/pytest/output/rdv_perf_20260908/regression_analysis/static_analysis.md)、[既有数据审计](../tests/pytest/output/rdv_perf_20260908/regression_analysis/existing_data_review.md)：独立子代理分析，形成于新消融结果之前。其中保留 ASC-026 的初步建议已被后续消融修正，以本报告结论为准。

注意：仓库 build.sh 会清理根目录 output/、build/，本次原始数据保存在算子 tests/pytest/output 下。原始profile、隔离包和实验脚本为本机数据产物，被git忽略；本Markdown文档保存主要结果与复现说明。
