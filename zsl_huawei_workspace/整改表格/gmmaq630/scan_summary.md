# GMMAQ 报错日志整改扫描汇总

- 基线分支：master
- 扫描/整改分支：gmmaq630
- 扫描范围：
  - `gmm/grouped_matmul_activation_quant/op_api`
  - `gmm/grouped_matmul_activation_quant/op_host`
- 整改数量：68
- 排除数量：31

## 整改原则

本次仅整改 GroupedMatmulActivationQuant 用户可感知的参数错误日志，将 dtype、format、shape、shape dim、tensorList 数量、attr value 等校验改为 `OP_LOGE_FOR_INVALID_*` 标准模板。校验条件、返回码、执行流程保持不变。

本轮按 `C:/workspace/CubeSkills_fork/skills/op-error-log-rectifier/SKILL.md` 的新版规则执行：dtype/value/shape/shape dim/format 类日志优先使用 `*_WITH_REASON` 模板；format 单参数 reason 使用 `OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON`，对应 `EZ0035`。量化模式条件 reason 采用类似 `when the quantization mode is MX, the dtype of x must be ...` 的句式。

复杂 incorrect/reason 字段使用 `gmm/common/op_host/log_format_util.h` 中的 `Ops::Transformer::Gmm::FormatString` 格式化，避免代码中出现长字符串拼接。

## 排除说明

以下类型保留原日志并记录到排除表：框架 context/attrs/desc/shape/compileInfo 空指针、InferShape/launcher/执行器失败、SetMKN/SetGroupNum/CheckCoreNum/CalL1Tiling 等内部流程失败。这些问题不是用户通过修正算子输入参数即可直接处理的外部参数错误。

## 验证命令和结果

- `rg -n "OP_LOGE_FOR_INVALID_FORMAT\\(" gmm\\grouped_matmul_activation_quant\\op_api gmm\\grouped_matmul_activation_quant\\op_host`
  - 结果：无剩余不带 reason 的单参数 format 错误模板。
- `rg -n "OP_LOGE_FOR_INVALID_(DTYPE|VALUE|SHAPE|SHAPEDIM)\\(" gmm\\grouped_matmul_activation_quant\\op_api gmm\\grouped_matmul_activation_quant\\op_host`
  - 结果：无剩余不带 reason 的 dtype/value/shape/shape dim 单参数错误模板。
- `git diff --check`
  - 结果：未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示。
- `bash build.sh --ophost --soc=ascend950`
  - 结果：当前 Windows PowerShell 环境未安装 bash，命令无法启动。

## 风险项

本地旧 opbase 头文件可能尚未包含 `OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON`，但新版整改 skill/canonical 已将该宏纳入 `EZ0035`；最终仍需在目标 CANN 编包环境执行完整 `bash build.sh --pkg --soc=ascend950` 验证。
