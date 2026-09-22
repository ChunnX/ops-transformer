# AllGatherMatmulV2 性能采集框架（A8 备用，本期不设验收门槛）

> 依据：测试方案 2.1 §1.3 通道 F / §7 T7 —— 本环境无 NPU，性能采集框架
> 按可投入使用搭好备用；本期验收只含功能+精度，不产出性能数据、不设门槛。

## 组成

| 文件 | 用途 |
|------|------|
| `collect_perf.sh` | 多轮采集：对选中用例逐轮执行 example，采集 cannsim 仿真报告 / example 计时打印 / msprof（有卡）；msprof 模式每轮轮前执行 L2 cache flush |
| `perf_report.py` | 汇总解析：**瓶颈维度分解**（计算/搬入/搬出/标量占比，cannsim pipe 与 msprof op_summary 双路径）+ 多轮**中位数与离散度**（median/min/max/MAD/CV）+ msprof 主 kernel 逐 rank 聚合，输出 `perf_summary.json` + `perf_report.md` |
| `l2_flush.py` | L2 cache flush 助手（B9⑤-5）：每轮 msprof 前每卡 256MB bf16 ×2 遍 in-place add（> 950PR PCIE L2 112MB），FLUSH_OK 证据行由 collect_perf.sh 校验 |

## 用法

```bash
# 0) 先构建参数化 example（与精度通道共用，见 ../README.md；产物为
#    build/test_aclnn_all_gather_matmul_v2，example_name 区分在构建期完成）
cd <repo-root>
bash build.sh --run_example all_gather_matmul_v2 eager cust --vendor_name=custom \
     --soc=ascend950 --example_name=test_aclnn_all_gather_matmul_v2_fp16
# 无卡 cannsim 环境构建时追加 --simulator=camodel（与 run_sim_cases.sh --sim 同形态）

# 1) 仿真侧参考统计（本环境；多卡 HCCL 仿真支持能力属 A9 风险项）
bash tests/sim/perf/collect_perf.sh --case L0-01,L1-133 --rounds 5 --mode sim

# 2) 汇总（瓶颈维度分解 + 中位数/离散度）
python3 tests/sim/perf/perf_report.py --raw-root tests/sim/perf/perf_raw
```

有卡环境切 `--mode msprof`（输出 `prof/` 归档；pipe 级深度分析用
ops-profiling 的 `msprof_perf_summary.py` 解析 `PROF_GROUP_*`）。

## 双分支性能对比（4.1，USING_APACE_IMPL 开关）

```bash
# v1 分支（默认，显式 unset 开关 env）
bash tests/sim/perf/collect_perf.sh --case L0-01,L1-133 --rounds 10 --mode msprof
# APACE 分支（export ASCEND_MC2_AGMV2_APACE_IMPL=1，进程内 tiling 读取）
bash tests/sim/perf/collect_perf.sh --case L0-01,L1-133 --rounds 10 --mode msprof --apace on
# 汇总：同 case 双分支条目（L0-01[v1] / L0-01[apace]）同报告呈现，可直接对比
python3 tests/sim/perf/perf_report.py --raw-root tests/sim/perf/perf_raw
```

- round 目录按分支隔离：`perf_raw/<case>/branch_{v1|apace}/round_<i>/`；
- 框架侧 rank 键桥接：sim_cases 输出 `AGM_V2_RANK_SIZE`、种子 example 消费
  `AGM_V2_DEV_NUM`——collect_perf.sh 同值补发两键（否则 R=4/8 用例静默跑 R=2）。

## 就绪度核对结论（B9⑤-5 有卡采集更新，2026-09-20）

1. **example env 契约**：~~缺口~~ **已闭合**——参数化 example
   （v1.2 P2 处置后）消费全契约键（BIAS/TRANS_B/GATHER_OUT/
   VALUE_SPEC/VALUE_SCALE/REPEAT_LAUNCH/SEED/DUMP_DIR/INFORMATIONAL，
   B9⑤-1 34 例 + B9⑤-4 ST 56 运行实证）；含 bias/transB 因子的 perf 用例
   已可经 env 驱动（本 README 用例集口径仍为 L0-01/L1-133，不自行扩集）。
2. **计时行**：~~缺失~~ **已就绪**——example 每 launch（含同步）打印
   `AGM_V2_TIMING_US:` 行（B9⑤-1 33 例实测），perf_report.py 双源解析
   （example 墙钟 + msprof task 记录）。
3. **APACE 开关**：example 无需改动——开关 env 由框架导出，aclnn 进程内
   tiling 读取（单进程多线程形态；B9⑤-2 透明性 + §6.1 探针实证生效链）。
4. **msprof 解析**（B9⑤-5 新增）：perf_report.py 直接解析
   `prof/PROF_*/mindstudio_profiler_output/op_summary_*.csv`——主 kernel
   过滤 `all_gather_matmul`（arch35 单入口，v1/APACE 同名）、逐 rank
   Task Duration、PipeUtilization 维度列正则探测（命中列登记于输出
   `msprof.dimension_columns`）。

## L2 cache flush 口径（B9⑤-5 任务拍板，2026-09-20）

**每轮 msprof 前实际执行 L2 cache flush**（禁止只分配不 flush）：
`l2_flush.py` 每卡 256MB bf16 buffer ×2 遍 in-place add（> Ascend950PR
PCIE L2 112MB，npu-arch skill 唯一知识源），FLUSH_OK 证据行落
`round_*/flush.log` 并由 collect_perf.sh 校验（无证据行判该轮失败）。
flush 进程独立于 msprof 采集进程（flush kernel 不进 task 记录）。

方法学注记：flush 位于轮前（进程外），无法消除同轮 example 内 H2D 写入
的 L2 写分配效应——该效应 v1/APACE 两分支同构，主判据（双分支同条件
对比）公平；绝对带宽解读时注意此口径。本算子通信由 HCCL server
（CCU/AICPU 引擎）执行、无 AIV+URMA 数据路径，mc2 红线②（限
blaze-shmem/apace-UDMA 底座）不强制——本节 flush 为任务拍板的测量有效性
口径（B9⑤-5），非红线②合规要求。

## 瓶颈维度分解口径

cannsim（npusim）`report/summary.json` 的 `pipe_utilization` 映射：

| 维度 | pipe |
|------|------|
| compute（计算） | `AIC_CUBE` |
| copy_in（搬入） | `AIC_MTE2` + `AIC_MTE1` |
| copy_out（搬出） | `AIC_FIXPIPE` |
| scalar（标量） | `AIC_SCALAR` |

- **占比**：各维度 pipe busy 利用率归一化（逐轮归一后取中位数）；
- **绝对量**：`kernel_total_clocks` 与 example 计时（`AGM_V2_TIMING_US`
  打印行，参数化 example 契约）；
- AIV 侧 pipe（`AIVx_*`）如出现单独列出，不并入四维；
- 维度口径与 ops-simulator `performance-metrics-reference.md` 一致，
  阈值解读以其为唯一可信源。

## 多轮统计口径

对每个用例记录 `rounds` 轮，输出 **median / min / max / MAD（median
absolute deviation）/ CV（离散系数）**；波动用例（CV 高）以中位数为准，
并在报告中保留离散度供判断采集稳定性。

msprof 主 kernel 聚合（B9⑤-5，每轮独立进程单 launch）：
- **跳 warmup**：丢弃 round_1（首轮进程冷启动/频率爬升）；
- **去离群**：|v - median| > 3×MAD 剔除（MAD=0 全保留），剔除数登记；
- **per-rank median**（中位数为准）+ min/max/MAD/CV；
- **跨 rank 聚合（官方口径，木桶效应）**：主口径 = max(per-rank median)；
  参考 = mean(per-rank median)（卡均值）与 per-rank last-5 mean → max
  （msprof-guide「MC² 多 rank」字面口径——该口径假设进程内 10 次迭代，
  本框架为 10 轮独立进程，last-5 语义取末 5 轮）；rank 不可归属时回退
  per-round max 序列（每轮取最慢 rank）。

## MC2 特有注意事项（ascendc-mc2-best-practice / ops-profiling 口径）

1. **MC2 多 rank 算子必须用 `msprof`，禁止 `msprof op`**（`msprof op` 对
   fork 程序采集行为未定义）；
2. msprof 采集用 `--ai-core=on --aic-mode=task-based
   --aic-metrics=PipeUtilization --task-time=on --ascendcl=on`
   （B9⑤-5 起显式声明，对齐 msprof-guide「MC² 多 rank 算子采集」标准
   命令，防默认值跨版本漂移）；
3. **每轮轮前 L2 cache flush**（B9⑤-5 任务拍板，见上节——本算子为 HCCL
   高阶 in-kernel 通信（CCU/AICPU 引擎）无 AIV+URMA 直调路径，mc2 红线②
   不强制，flush 属测量有效性口径）；
4. 多 rank kernel 时间跨 rank 取 max（最慢者决定整体，官方口径），
   per-rank 统计不做跨 rank 平均为主口径（卡均值作参考列）——rank 间
   不均衡经 per-rank 表单独呈现。
