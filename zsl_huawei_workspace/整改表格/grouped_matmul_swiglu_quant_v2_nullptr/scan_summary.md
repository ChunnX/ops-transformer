# GroupedMatmulSwigluQuantV2 空指针日志整改扫描总结

- 基线分支：`upstream/master`
- 扫描分支：`fix/gmmswigluquantv2-required-nullptr-check`
- 扫描范围：该分支相对基线新增的 aclnn 与 tiling 空指针/空列表日志
- 扫描源码：
  - `gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp`
  - `gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.cpp`
- 整改数量：6 个源码日志调用点
- 排除数量：0

## 整改结果

- 5 个空指针日志由 `OP_LOGE` 替换为 `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON`，对应 `EZ0037`。
- 1 个空 tensor list 日志由 `OP_LOGE` 替换为 `OP_LOGE_FOR_INVALID_TENSORNUM`，对应 `EZ0022`。
- aclnn 日志使用实际 workspace 接口名；空指针返回值保持 `ACLNN_ERR_PARAM_NULLPTR`。
- tiling 的判断条件、控制流和 `GRAPH_FAILED` 返回值保持不变。
- 空指针 reason 统一为 `does not support nullptr`，日志仍明确包含参数名。

## 风险与验证

- 未改变校验条件、返回值、变量计算或调用顺序。
- 需要执行 `git diff --check`、reason 句式扫描及现有单元测试/静态检查。
