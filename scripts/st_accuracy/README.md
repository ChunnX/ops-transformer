# ST Accuracy Runner

本目录是从 `ops-math` 的 `scripts/st_accuracy` 迁移并裁剪来的本地 ST 精度测试框架，当前优先服务 `ops-transformer` 的两个 attention 算子：

- `attention/lightning_indexer`
- `attention/recurrent_gated_delta_rule`

当前 runner 使用 Python 发现/筛选 ST JSON case，并在执行模式下编译一个 C++ native ACLNN batch runner。它不依赖 ATK/xrun。

## Commands

列出两个目标算子的 ST case：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py list \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --format json
```

生成 coverage 报告，不执行设备侧：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py coverage \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --out build/st_accuracy/static_coverage
```

只做筛选和报告，不调用 native runner：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --no-execute \
  --out build/st_accuracy/static_no_execute
```

构建/安装 custom package 后执行：

```bash
bash build.sh --pkg \
  --soc=ascend910b \
  --ops=lightning_indexer,recurrent_gated_delta_rule \
  --vendor_name=custom \
  -j16 -O3

bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --vendor custom \
  --soc ascend910b \
  --device 0 \
  --out build/st_accuracy/two_attention_ops
```

`--device` 是传给 `aclrtSetDevice()` 的 ACL runtime logical device id。

## Discovery Gate

`list`、`coverage`、`taxonomy` 和 `run` 的第一步都是 discovery gate：

1. 发现当前 scope 内的 AscendC attention 算子。
2. 装载 `<op>/tests/st/*/all_*.json` case。
3. 对显式 `--ops` 请求校验发现结果。

当传入 `--ops` 时，以下情况会返回非 0，并且 `run` 不会继续执行或写出成功形态报告：

- `missing_ops`：请求的算子没有被当前 AscendC attention 发现逻辑识别。
- `zero_case_ops`：请求的算子已发现，但没有任何有效 ST case。
- `skipped_assets`：请求算子的 ST 资产存在不可静默忽略的问题，例如缺少 `tests/st`、没有匹配 JSON、JSON 非法、payload 不是 list、case 不是 object 或 case id 非整数。

未传 `--ops` 时仍保持宽松扫描，用于观察当前默认 scope；不能把未请求算子的 0 case 当成执行失败。

## Outputs

`run` 写入：

- `summary.json`：`passed`、`failed`、`error`、`unsupported`、`skipped` 聚合计数。
- `summary.json.timings`：Python discovery/load/select/support、native bridge 和 report 写入的阶段耗时。
- `cases.jsonl`：每个 case 一行结构化结果。
- `cases.jsonl.extra.timings`：两个 attention API 的 C++ native 内部阶段耗时，包含 `parse_ms`、`input_generate_h2d_ms`、`golden_ms`、`workspace_ms`、`launch_sync_ms`、`d2h_ms`、`compare_ms`。
- `native/st_accuracy_aclnn_runner`：执行模式下缓存的 C++ native runner。
- `native/batch_cases.tsv`：传给 native runner 的 batch 输入。

`coverage` 写入：

- `coverage_summary.json`，包含 `timings` 阶段耗时。
- `coverage_by_op.jsonl`
- `coverage_by_api.jsonl`

`taxonomy` 写入：

- `taxonomy_summary.json`，包含 `timings` 阶段耗时。
- `taxonomy_by_api.jsonl`
- `taxonomy_by_op.jsonl`
- `taxonomy_priority.jsonl`

## Current Scope

当前新增 ST JSON：

- `attention/lightning_indexer/tests/st/aclnnLightningIndexer/all_aclnnLightningIndexer.json`：10 个 case。
- `attention/recurrent_gated_delta_rule/tests/st/aclnnRecurrentGatedDeltaRule/all_aclnnRecurrentGatedDeltaRule.json`：8 个 case。

当前 Python adapter 会把不满足文档约束的 shape/dtype/layout/optional input 归为 `unsupported`，不会把它们伪装成精度失败。

native runner 已接入两个 API 的 ACLNN 两段式调用和 host golden compare，参考各自 `examples/test_aclnn_*.cpp`：

- `aclnnLightningIndexerGetWorkspaceSize` / `aclnnLightningIndexer`
- `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` / `aclnnRecurrentGatedDeltaRule`

当前 golden 范围：

- `LightningIndexer`：按文档公式生成 host reference，覆盖 `BSND`、`TND`、`PA_BSND` 输入布局；校验 `sparseIndicesOut`，当 `returnValues=true` 时同时校验 `sparseValuesOut`。
- `RecurrentGatedDeltaRule`：按 kernel 的 recurrent 更新顺序生成 host reference；同时校验 `out` 和原地更新后的 `stateRef`。

当前已验证：

- `list --ops lightning_indexer,recurrent_gated_delta_rule`：2 个算子、18 个 case。
- `coverage --ops lightning_indexer,recurrent_gated_delta_rule`：18 executable、0 unsupported。
- `run --no-execute`：18 skipped、0 failed、0 error、0 unsupported。
- `bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer,recurrent_gated_delta_rule --vendor_name=custom -j16 -O3`：生成 `build_out/cann-ops-transformer-custom_linux-aarch64.run`。
- 安装后 `/usr/local/Ascend/cann-9.0.0/opp/vendors/config.ini` 包含 `custom_transformer`，`op_api/lib/libcust_opapi.so` 导出两个 API 的 `GetWorkspaceSize` 和 launch 符号。
- `run --ops lightning_indexer,recurrent_gated_delta_rule --out build/st_accuracy/device_full_after_case_fix`：18 passed、0 failed、0 error、0 unsupported。

历史根因：未安装 `custom_transformer` package 时，runner 会报告缺少 `${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so`。这不是算子源码缺失，而是 custom package 未构建/安装。只存在 CANN 内置 `libopapi_transformer.so` 或其他仓库的 `custom_math`，不能满足本仓库自定义算子的 ST 执行。

`RecurrentGatedDeltaRule` 的当前 ST case 保持 `ssmStateIndices` 全局唯一。重复 state index 会让 `stateRef` 原地更新出现读写顺序/覆盖依赖，不适合作为默认批量精度通过矩阵；如后续要研究该边界，应单独建 case 并先确认官方语义。

## Vendor Path

与 `ops-math` 的 `custom_math` 不同，本仓库 custom package vendor 路径是：

```text
${ASCEND_HOME_PATH}/opp/vendors/${vendor}_transformer/op_api/lib/libcust_opapi.so
```

runner 优先使用当前仓库 `build/libcust_opapi.so`；如果不存在，再 fallback 到上述安装路径。

注意：即使 runner 可以 `dlopen build/libcust_opapi.so`，设备执行仍需要 OPP vendor 目录中的 `op_proto`、`op_impl`、kernel config 和 kernel object。首次运行真实 ST 前应完整 `--pkg` 并安装。
