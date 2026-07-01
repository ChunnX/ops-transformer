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

## 排除说明

以下类型保留原日志并记录到排除表：框架 context/attrs/desc/shape/compileInfo 空指针、InferShape/launcher/执行器失败、SetMKN/SetGroupNum/CheckCoreNum/CalL1Tiling 等内部流程失败。这些问题不是用户通过修正算子输入参数即可直接处理的外部参数错误。

## 验证命令和结果

- `rg -n "OP_LOGE\\(|CHECK_COND|OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE|OP_CHECK_WRONG_DIMENSION" gmm\\grouped_matmul_activation_quant\\op_api gmm\\grouped_matmul_activation_quant\\op_host`
  - 结果：剩余普通日志均为内部错误或执行链路失败，已记录到排除表。
- `git diff --check`
  - 结果：未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示。
- `bash build.sh --ophost --soc=ascend950`
  - 结果：当前 Windows PowerShell 环境未安装 bash，命令无法启动。

## 风险项

未在本地执行完整 `bash build.sh --pkg --soc=ascend950` 编包验证；如 CI 对模板宏实参类型更严格，可能需要按编译日志微调个别 `std::string`/`const char *` 实参。
