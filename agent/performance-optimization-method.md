# Performance Optimization Method

本文定义 `ops-transformer` 中 AscendC AI Core 算子的通用性能优化方法。它吸收 `ops-math` 的闭环经验，但所有结论必须在本仓库的 `ascend910b -> arch22`、`custom_transformer` package、真实精度验证和真实 profiling 链路中重新验证。

## Scope

默认优化对象：

- `attention/`
- `ffn/`
- `gmm/`
- `mc2/`
- `moe/`
- `posembedding/`

只把最终运行在 AI Core 上的 AscendC kernel 计入主线优化。`op_kernel_aicpu/`、AICPU-only 调度、样例模板、测试代码和 third-party 目录不计入默认优化范围。

## Method

### 1. Candidate Gate

开始前确认：

- 目标算子在 [ascendc-operators.md](ascendc-operators.md) 主线列表或有新的等价证据。
- 优先选择 910B CI 覆盖、存在 eager example、存在 UT/pytest/ST 资产或用户提供真实业务输入的算子。
- 明确目标 API、shape、dtype、format、attrs、layout 和性能目标。
- 对 MC2 类算子，先确认 rank、world size、HCCL、拓扑和多卡环境；单卡 smoke 或 simulator 不能代表真实性能。

### 2. Package And Load Gate

性能数据必须来自当前 custom package 或明确同步过的新 binary/config。默认命令：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

至少记录：

- `.run` 包路径和生成时间。
- vendor 目录：`/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer`。
- `libcust_opapi.so` hash。
- 目标 kernel/config 是否存在。
- profiling 应用实际加载的 `LD_LIBRARY_PATH`。

`build.sh --opkernel` 只能证明 kernel binary 编译成功。除非显式同步 runtime vendor 目录中的 binary/config，并记录同步路径和 hash，否则不能把其结果写成有效运行时性能优化。

### 3. Accuracy Gate

性能优化前后都必须先过可用精度验证：

- `build.sh -u --ophost_test|--opapi_test|--opgraph_test|--opkernel_test --soc=ascend910b --ops=<op_name>`
- `build.sh --run_example <op_name> eager|graph cust --vendor_name=custom --soc=ascend910b`
- 算子已有 pytest 或 ST 资产时运行对应入口。
- 当前 `scripts/st_accuracy` 只优先覆盖 `lightning_indexer` 与 `recurrent_gated_delta_rule`，不能外推成全仓库 ST coverage。

`unsupported` 不是通过。只有 selected case 被真实执行且 `failed=0,error=0,unsupported=0`，才能进入性能判定。

### 4. Baseline Matrix

baseline 必须固定输入矩阵。最小矩阵覆盖：

- 目标 shape。
- 小 shape。
- 32/256/4096 等边界附近 shape，或目标算子的自然边界。
- 常见中等 shape。
- 相关 dtype、format、layout 和 attrs。

Transformer 相关维度：

- attention：batch、seq length、head num、head dim、KV cache、mask、layout、quant。
- GMM/MoE：expert 数、group/token 分布、routing/topk、空 expert。
- MC2：rank、world size、HCCL、拓扑、通信/计算 overlap。
- posembedding：rotary/rope 维度、sin/cos cache、KV cache 写回模式。

每个 case 至少采集 3 次，默认用 median。低于约 1% 的变化默认按噪声处理，除非同一时间窗口 A/B 重复采集稳定且矩阵无回退。

### 5. Profiling Mapping

默认使用 `msprof op` 对 eager example 或可用 runner wrapper 采集：

```bash
cd build
msprof op --application=./test_aclnn_<op_name> --output=../artifacts/<op_name>/msprof/<tag>
```

如果一个 aclnn 调用触发多个 runtime task，必须同时记录目标 kernel time 和 total device time。使用 batch runner 时，`--launch-count` 必须覆盖真实 device task 数；一 case 一 task 只是特例。

不能仅按 CSV 文件名、timestamp 或行顺序映射 case。混合多个 kernel name、suffix 或 tiling key 时，要用 profiling 日志中的启动顺序、kernel name、launch index 和输入矩阵重建映射；做不到时改成单 case 单输出目录。

### 6. Optimization Hypothesis

修改前先写清假设：

```text
operator:
target case:
suspected bound:
expected effect:
changed files:
risk:
rollback condition:
```

常见方向：

- tiling：blockDim、core split、UB tile、dtype/shape 分支、tail 处理。
- DataCopy：对齐、单次搬运量、stride copy、减少小 DMA 描述。
- memory：UB/L1/L0/BT/FP buffer 驻留复用、bank conflict 规避。
- pipeline：CopyIn/Compute/CopyOut overlap、double buffer、queue 同步。
- API usage：TQue/TBuf/TPipe 生命周期、counter mask、vector API 选择。

从 `ops-math` 或其他算子迁移来的经验只能作为假设。目标算子必须用本仓库精度和性能矩阵重新证明。

### 7. Decision

保留优化必须同时满足：

- 精度验证正常。
- package/load evidence 证明加载新产物。
- 目标 case 有稳定收益。
- 小/边界/中等/目标矩阵无稳定回退。
- case 映射可信，采样数量足够。
- 算子语义、输出、dtype/format 和必要计算没有改变。

以下情况必须拒绝或继续迭代：

- 只在单个有利 shape 变快，代表性矩阵稳定回退。
- 收益接近噪声且没有同窗口 A/B 证明。
- 只减少理论 GM 字节或指令数，但 `Task Duration(us)` 或 total device time 回退。
- 使用旧安装产物、错误 vendor、错误 case 映射或不完整 launch-count。
- 为通过精度修改 golden、阈值、case JSON 或 runner 语义。

### 8. Evidence Retention

`agent/` 只保留知识和摘要，不保留 raw dump。长期记录应包含：

- build/install/accuracy/profiling 命令。
- 环境和 package load evidence。
- baseline/candidate 矩阵摘要。
- 关键 samples、median、gain。
- 保留、拒绝或阻塞原因。
- raw artifact 路径和 hash，若决定保留。

当前已开始使用 `agent_ops/<op>/` 轻量账本。账本只放 `README.md`、`manifest.json`、`performance.md`、`workflow.md` 这类可读摘要；raw `msprof` dump、完整 CSV、`.o`、`.bin`、长日志应进入 artifact 边界。

### 9. Feedback

真实闭环结束后：

- 成功模式写入 [lessons/performance-optimization.md](lessons/performance-optimization.md) 或 `.agents/skills/ascendc-performance-best-practices/`。
- 失败反例写清拒绝原因，避免后续重复尝试。
- package、launch-count、case 映射、vendor 污染等流程类经验更新到 `.agents/skills/ops-profiling/` 或相关 runbook。

不要把长日志、单次流水或不可复核结论写进 `AGENTS.md`。
