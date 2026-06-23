# Accuracy Runbook

本 runbook 面向后续验证单算子优化是否保持正确性。完整静态分析见 [../accuracy-test-flow.md](../accuracy-test-flow.md)。

## Default Order

建议按改动范围选择最小但有效的验证：

1. 修改 `op_kernel/`：跑 `opkernel` UT 和 eager example。
2. 修改 `op_host/*tiling*`：跑 `ophost` UT、eager example，必要时 graph example。
3. 修改 `op_host_aclnn*`：跑 `opapi` UT 和 eager example。
4. 修改 `op_graph/`：跑 `opgraph` UT 和 graph example。
5. 修改公共目录：识别受影响算子后扩大验证范围。

## C++ UT

```bash
bash build.sh -u --ophost_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opapi_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opgraph_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opkernel_test --soc=ascend910b --ops=<op_name> -j16
```

只编译不执行：

```bash
bash build.sh -u --noexec --soc=ascend910b --ops=<op_name> -j16
```

判断：

- GTest 输出 `[ PASSED ]` 才能写成 UT passed。
- 编译通过不能写成精度通过。
- `--noexec` 只能写成 UT build passed，不能写成运行通过。

## Example Smoke

eager：

```bash
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
```

graph：

```bash
bash build.sh --run_example <op_name> graph cust --vendor_name=custom --soc=ascend910b
```

判断：

- example 正常退出只代表调用链路成功。
- 只有 example 源码明确比较 golden，才能写成该 example 覆盖范围内的精度通过。

## Pytest

静态发现 pytest 主要在以下 attention 目录：

- `attention/flash_attention_score/tests/pytest`
- `attention/flash_attention_score_grad/tests/pytest`
- `attention/fused_infer_attention_score/tests/pytest`
- `attention/mla_prolog_v2/tests/pytest`
- `experimental/attention/fused_infer_attention_score/tests/pytest`

典型命令：

```bash
cd attention/fused_infer_attention_score/tests/pytest
python3 -m pytest -rA -s test.py
```

注意：

- pytest 可能依赖 `torch`、`torch_npu`、`torchair`，不一定由根目录 requirements 覆盖。
- pytest 结论必须写清 case 范围和失败/跳过数量。

## ST Assets

静态发现部分算子有 `tests/st/`，例如：

- `attention/attention_update`
- `attention/fused_infer_attention_score`
- `attention/incre_flash_attention`
- `attention/prompt_flash_attention`
- `mc2/all_gather_matmul`
- `mc2/allto_all_matmul`
- `mc2/matmul_all_reduce`
- `posembedding/apply_rotary_pos_emb`

当前仓库已有从 `ops-math` 迁移裁剪的局部 ST runner，优先覆盖 `lightning_indexer` 与 `recurrent_gated_delta_rule`：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py list \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --format json

python3 scripts/st_accuracy/run_st_accuracy.py coverage \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --out build/st_accuracy/static_coverage

python3 scripts/st_accuracy/run_st_accuracy.py run \
  --ops lightning_indexer,recurrent_gated_delta_rule \
  --no-execute \
  --out build/st_accuracy/static_no_execute
```

设备侧执行前需要先构建/安装 custom package：

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

当前 runner 只能写成两个重点 attention 算子的局部 ST 能力，不能外推为全仓库统一 ST coverage。

判断：

- `coverage` 只代表 JSON case 能被当前 adapter 接收，不能写成设备精度通过。
- `run --no-execute` 只代表筛选、序列化和报告路径可用。
- native C++ runner 会为 `LightningIndexer` 比较 `sparseIndicesOut`，并在 `returnValues=true` 时比较 `sparseValuesOut`。
- native C++ runner 会为 `RecurrentGatedDeltaRule` 比较 `out` 和原地更新后的 `stateRef`。
- 如果结果是缺少 `custom_transformer` op_api 库，只能写成运行环境未满足，不能写成算子失败或精度通过；根因通常是尚未构建/安装本仓库 custom package，而不是官方算子源码缺失。
- 当前已验证的两个重点算子结果：安装 `custom_transformer` 后，`run --ops lightning_indexer,recurrent_gated_delta_rule --out build/st_accuracy/device_full_after_case_fix` 为 18 passed、0 failed、0 error、0 unsupported。
- `RecurrentGatedDeltaRule` 默认矩阵不包含重复 `ssmStateIndices`，因为该场景会让 `stateRef` 原地更新存在覆盖/顺序依赖，应作为后续专项边界研究。

## CI Reference

批量 UT 脚本：

```bash
scripts/ci/ut_make.sh <code_path>
```

该脚本会参考 `tests/test_config.yaml` 并批量跑多 SoC UT。它是 CI 辅助入口，本地单算子优化默认先用 `build.sh -u --ops=<op_name> --soc=ascend910b`。
