---
name: ops-profiling
description: Use when collecting or analyzing ops-transformer NPU operator performance data, msprof or msprof op output, Ascend C hardware profiling, package/load evidence, performance regression evidence, or profiling matrix results.
---

# 上板性能采集与分析

本 skill 用于真实 NPU profiling、指标归档、case 映射、矩阵对比和回归判定。默认服务 `ops-transformer` 的 AscendC AI Core 算子。

## Repo Defaults

- SoC：`ascend910b`
- 架构目录：`arch22`
- custom vendor：`custom_transformer`
- 默认构建：`bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3`
- 默认 profiling target：`build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b` 生成的 eager example。

当前 `scripts/st_accuracy` 只优先覆盖 `lightning_indexer` 和 `recurrent_gated_delta_rule`。其他算子默认通过 example、UT、pytest 或后续自建 wrapper 采集，不要假设全仓库都有 ST native batch runner。

## Tool Selection

| 工具 | 何时使用 | 参考 |
| --- | --- | --- |
| `msprof op` | 默认上板算子 profiling，尤其是需要 `Task Duration(us)` 和 kernel 级指标时 | `references/msprof-op-guide.md` |
| `msprof` | 环境没有 `msprof op`/`msopprof`，或团队明确要求采集多组 `aic-metrics` 时 | `references/msprof-guide.md` |

如果用户显式指定工具，以用户指定为准。未指定时先探测环境；两者都可用时优先 `msprof op`，因为它更贴近当前 runbook。

## Preflight

采集前必须确认：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
sha256sum /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so 2>/dev/null || true
find /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer -path '*ascend910b*' -type f | rg '<op_name>|kernel|config'
```

如果 package/load evidence 缺失，先修安装或同步路径，不要继续解释 profiling 数据。

## Matrix And Sampling

- 固定 baseline/candidate 同一矩阵。
- 矩阵覆盖目标 shape、小 shape、边界 shape、常见中等 shape、相关 dtype/attrs。
- 每个 case 至少 3 次采样，默认用 median。
- 低于约 1% 的变化默认按噪声处理，除非同时间窗口 A/B 重复证明稳定收益。
- MC2 需要 rank、world size、HCCL 和拓扑；单卡结果只能写成 smoke 或局部指标。

## Launch Count And Mapping

使用 batch runner 时：

- `--launch-count` 必须覆盖真实 device task 数。
- `case_count` 只适用于一 case 一 task。
- ACLNN 组合链要比较 total device time，不能只看目标 kernel。

case 映射规则：

- 记录 case id、shape、dtype、attrs、kernel name、tiling key。
- 混合 kernel suffix、tiling key 或多个 kernel name 时，不按 CSV timestamp 直接排序。
- 优先用 `msprof op` 日志中的启动顺序和 launch index 重建映射。
- 映射不可信时，改成单 case 单输出目录。

## References And Scripts

| 文件 | 用途 |
| --- | --- |
| `references/msprof-op-guide.md` | `msprof op` 构建、采集、归档、判定、瓶颈和回归 |
| `references/msprof-guide.md` | 原生 `msprof` 多指标组采集和归档 |
| `references/csv_fields_reference.md` | CSV 字段定义和阈值 |
| `references/optimization_quickref.md` | 瓶颈类型和优化方向速查 |
| `scripts/perf_summary.py` | `msprof op` CSV 摘要和归档 |
| `scripts/msprof_profile_run.sh` | `msprof` 多指标组采集 wrapper |
| `scripts/msprof_perf_summary.py` | `msprof` 输出摘要 |

脚本来自可复用工具层。运行前先确认输入目录结构匹配当前 CANN/msprof 输出。

## Output

采集结束输出：

```text
operator:
tool:
application:
package evidence:
matrix:
samples:
case mapping:
kernel metrics:
total device time:
regressions:
decision:
retention action:
```

raw `msprof` dump、完整 CSV、`.o`、`.bin` 和长日志按 `agent/policies/evidence-retention.md` 处理。形成 reusable profiling 经验时更新本 skill；单算子局部事实写入 `agent_ops/<op>/` 或 `agent/lessons/`。
