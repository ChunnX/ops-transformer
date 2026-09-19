# AllGatherMatmulV2 性能采集框架（A8 备用，本期不设验收门槛）

> 依据：测试方案 2.1 §1.3 通道 F / §7 T7 —— 本环境无 NPU，性能采集框架
> 按可投入使用搭好备用；本期验收只含功能+精度，不产出性能数据、不设门槛。

## 组成

| 文件 | 用途 |
|------|------|
| `collect_perf.sh` | 多轮采集：对选中用例逐轮执行 example，采集 cannsim 仿真报告 / example 计时打印 / msprof（有卡） |
| `perf_report.py` | 汇总解析：**瓶颈维度分解**（计算/搬入/搬出/标量占比）+ 多轮**中位数与离散度**（median/min/max/MAD/CV），输出 `perf_summary.json` + `perf_report.md` |

## 用法

```bash
# 0) 先构建参数化 example（与精度通道共用，见 ../README.md）
cd <repo-root>
bash build.sh --ops=all_gather_matmul_v2 --module=mc2 --soc=ascend950 --run_example --simulator

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

## 就绪度核对结论（4.1，有卡前置依赖——种子 example 侧待补）

1. **example env 契约缺口**：种子 example（A9 探针版）仅消费
   `AGM_V2_{DEV_NUM,M,K,N,COMM_MODE,DTYPE}` 6 键；`tests/sim/README.md` 契约中的
   `BIAS/TRANS_B/GATHER_OUT/VALUE_SPEC/REPEAT_LAUNCH/SEED/DUMP_DIR` 未消费（bias
   恒 nullptr、transB/gatherOut 恒默认）。性能用例选择须限于已消费因子
   （shape/dtype/engine/rank），含 bias/transB 因子的对比属 **example 契约补全的
   有卡前置依赖**（example 非 tests 目录，本框架侧不改）。
2. **计时行缺失**：example 无 `AGM_V2_TIMING_US` 打印——sim 模式的 example 墙钟
   统计不可用；msprof 模式不受影响（kernel 时长取 task 记录）。
3. **APACE 开关**：example 无需改动——开关 env 由框架导出，aclnn 进程内 tiling
   读取（单进程多线程形态）。

## L2 flush 口径（v2 APACE hcomm 变体）

本算子通信由 HCCL server（CCU/AICPU 引擎）执行、AIV 空转，**无 AIV+URMA 数据
路径**——mc2 红线②（L2 flush，限 blaze-shmem/apace-UDMA 底座）**不强制**。但
B（x2）L2 驻留对 MTE2 带宽与计算段绝对计时的影响属测量有效性事项：主判据
（v1 vs apace 同条件对比）两侧同等受影响、对比公平；涉及绝对带宽解读与
「通信 vs 计算占比」证伪判定时，建议有卡首采补一组 with-flush 对照轮
（heavy_add_kernel 扫 L2 容量级 buffer），口径由性能迭代阶段裁定。

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

## MC2 特有注意事项（ascendc-mc2-best-practice / ops-profiling 口径）

1. **MC2 多 rank 算子必须用 `msprof`，禁止 `msprof op`**（`msprof op` 对
   fork 程序采集行为未定义）；
2. msprof 采集用 `--aic-mode=task-based`；
3. 本算子为 HCCL 高阶 in-kernel 通信（950 CCU/AICPU 引擎，注册/aclnn
   形态），**无 AIV+URMA 直调路径，无需 L2 cache flush**（L2 flush 红线
   仅约束 blaze-shmem/apace 底座）；
4. 多 rank kernel 时间取多轮 last-N 的 max（跨 rank 最慢者），汇总脚本
   按 case 聚合、不做跨 rank 平均——rank 间不均衡经
   `top_level_diagnosis.imbalance_ratio` 单独呈现。
