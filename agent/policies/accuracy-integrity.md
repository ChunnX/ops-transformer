# Accuracy Integrity Policy

精度验证优先于性能目标。后续优化 AscendC 算子时，任何性能改动都不能绕过精度语义。

## Allowed Changes

允许的改动：

- 修复真实算子 bug，例如 kernel 计算、边界处理、dtype/format 处理、workspace 或 tiling 逻辑错误。
- 修复真实 runner 或 example 的通用 bug，例如 dtype 映射、empty tensor、TensorList 生命周期、copy 长度、package 符号解析。
- 新增合法的 dtype、range、shape、package 或 SoC gate，但必须说明当前 runner、API contract、package 或 SoC 为什么不能执行该 case。
- 新增或修复真实 pytest/ST/example adapter，使其构造输入、调用 public API 并按算子语义比较输出。
- 修正本地新增的无效、非确定或超出官方文档/example/UT 可确定验证域的 ST case，但必须说明原因，并用测试或设备结果证明修正后仍覆盖明确场景。

## Forbidden Changes

禁止的改动：

- 不修改 ST golden、参考结果、case JSON、比较阈值来掩盖真实算子或 runner 失败。
- 不跳过失败 case 后把它统计为 passed。
- 不为了通过精度而改变输入 shape、dtype、attrs、range 或随机种子。
- 不把 `unsupported` 写成精度通过。`unsupported` 只表示当前 runner 或包声明不执行。
- 不用 example 正常退出替代 golden 对比，除非该 example 明确实现了语义比较。
- 不使用旧 custom package、旧 vendor 目录或无法证明来源的 op_api/kernel 产物写成当前修改后的设备侧精度通过。

## Required Wording

推荐表述：

- “当前 runner-supported scope passed”：只在 selected case 与预期支持范围一致，且 `failed=0,error=0` 时使用。
- “example smoke passed”：只表示样例调用链路成功，不能自动表示精度充分。
- “pytest/ST passed”：必须附带命令、case 范围和关键 summary。
- “scripts/st_accuracy device run passed”：必须确认真实 `run` 已触发 package/install，或明确记录等价的 package/load 证据；`coverage`、`taxonomy` 和 `--no-execute` 只能写成静态链路通过。

禁止表述：

- 未跑全量 case 时说“全部精度通过”。
- 只看到 `unsupported` 或 package gate 时说“精度通过”。
- 未确认 custom package 加载时说“当前修改后的算子通过”。

## Transformer-Specific Notes

- attention、MoE、GMM、MC2 算子常有多分支 tiling key。精度验证必须覆盖实际改动命中的分支。
- MC2 算子可能依赖 HCCL 和多卡拓扑。单卡失败或跳过不能直接写成算子精度失败，也不能写成通过。
- pytest 主要集中在部分 attention 算子；不要假设所有 transformer 算子都有 Python golden。
