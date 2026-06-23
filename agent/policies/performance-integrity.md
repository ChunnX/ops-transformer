# Performance Integrity Policy

性能结果必须可复现、可比较、语义有效。性能优化不能牺牲算子语义或验证可信度。

## Forbidden Changes

禁止的改动：

- 不删除必要计算、跳过必须写回的输出、改变 dtype/format 约束或改变算子语义。
- 不只测有利 shape、dtype、attrs 或 batch。
- 不遗漏小 shape、边界 shape、常见中等 shape 或目标大 shape 的代表性回归。
- 不把不同 package、不同输入、不同 build flags、不同 CANN/driver 环境或相隔很久的 baseline/candidate 直接比较。
- 不用单次最优样本代表结果。
- 不声称 `--opkernel` 性能结果有效，除非运行时 vendor 目录确实加载了新 binary/config。

## Required Evidence

一个可保留优化至少需要：

- 相关精度验证通过。
- selected case 全部真实执行；`unsupported`、跳过、runner gate 不算通过。
- baseline 和 candidate 在同一矩阵上测量。
- 每个 case 至少 3 次样本，默认用 median；如用 mean，要记录所有样本和离群处理。
- 目标 case 有有意义提升。
- 小 shape、边界 shape、常见中等 shape 和目标大 shape 没有稳定回退。
- profiling 日志或 package 检查证明运行加载的是目标 custom package、op_api、op_impl 和 kernel binary。

## Valid Optimization Definition

只有同时满足以下条件，才可以写成 `optimized_kept` 或“有效优化”：

1. 算子语义、必要计算、输出写回、dtype/format 支持范围没有被削减。
2. 精度验证在优化前后都可复核，且没有通过修改 golden、阈值、case JSON 或 runner 逻辑制造通过。
3. `custom_transformer` package/load 证据证明运行的是当前 candidate。
4. 目标 case 的 kernel time 或 total device time 有稳定收益。
5. 代表性矩阵没有稳定回退；目标 shape 变快但小/边界/中等 shape 稳定回退时不得保留。
6. case 到 profiling 记录的映射可信；混合 kernel suffix、tiling key 或多 task ACLNN 链时必须说明映射方法。

以下数据无效，必须重新采集或拒绝：

- baseline/candidate 来自不同 package、不同输入矩阵、不同 build flags 或明显不同环境。
- `msprof op --launch-count` 未覆盖真实 device task，导致只采集 batch 前缀。
- 使用旧 vendor 目录、旧 `libcust_opapi.so`、CANN 原始实现或未同步的 `--opkernel` binary。
- 只保留最优单次样本，或丢弃不利 case。
- 只比较目标 kernel，忽略 ACLNN 组合链中前后 runtime task 的 total device time 回退。

## Noise Rule

低于约 1% 的变化默认按噪声处理，除非同一时间窗口重复 A/B 采样能稳定证明收益，并且代表性矩阵没有稳定回退。

如果目标 case 提升接近噪声区间，但同时出现未触及 dtype/shape 大幅变化，先怀疑安装产物、机器状态、case 映射或 profiling 采集漂移，不要归因成代码优化。

## Transformer-Specific Requirements

- attention 类算子要记录 seq length、batch、head num、head dim、KV cache 状态、mask、dtype、layout、quant 参数。
- GMM/MoE 类算子要记录 expert 数、group 分布、token 分布、routing 参数和是否存在空 expert。
- MC2 类算子要记录 HCCL、rank、world size、通信拓扑、是否单机多卡，以及通信和计算 kernel 的分离指标。
- 如果一个 aclnn 调用触发多个 runtime task，要同时计算目标 kernel time 和 total device time。

## Result Record Template

```text
operator:
api:
soc:
arch:
CANN:
driver/firmware:
build command:
install/sync command:
run command:
profiling command:
shape/dtype/format:
attrs:
tiling key:
block dim:
kernel name:
baseline samples(us):
candidate samples(us):
summary metric:
gain:
correctness checks:
package load evidence:
decision:
```
