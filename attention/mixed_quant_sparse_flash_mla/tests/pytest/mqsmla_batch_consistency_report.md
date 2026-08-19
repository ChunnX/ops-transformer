# MQSMLA Batch 一致性验证报告

## 1. 验证环境

- 自定义算子安装目录：`/home/p00835995/custom_wz`
- 用例目录：`/tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase`
- Batch 一致性等级：`torch_npu.npu.set_deterministic_level(3)`
- 验证模式：
  - Mode 1：Batch 重排一致性
  - Mode 2：Batch 拆分一致性
- Pytest 汇总：`4 failed, 17 passed, 20 skipped`
- 执行耗时：`10.42s`

说明：17 个 passed 中包含 12 个算子一致性测试和 5 个脚本辅助测试。

## 2. 通过的用例

以下 6 个 `.pt` 用例的 Batch 重排和 Batch 拆分均通过：

| 用例 | Batch 重排 | Batch 拆分 |
|---|---:|---:|
| `qsmla_case_multi_batch_full_cores` | PASS | PASS |
| `qsmla_case_residual` | PASS | PASS |
| `qsmla_case_tnd_padding_kv` | PASS | PASS |
| `qsmla_case_with_kvs_0` | PASS | PASS |
| `qsmla_case_with_qs_0` | PASS | PASS |
| `qsmla_case_with_tail` | PASS | PASS |

## 3. 失败的用例

以下 2 个 `.pt` 用例在两种模式下均失败，共产生 4 个失败项：

| 用例 | Batch 重排 | Batch 拆分 |
|---|---:|---:|
| `qsmla_case_1_core_has_2_reduce` | FAIL | FAIL |
| `qsmla_case_1_core_has_2_reduce_with_no_reduce_insight` | FAIL | FAIL |

### 3.1 `qsmla_case_1_core_has_2_reduce`

Batch 重排：

- 失败样本：`sample=27`
- 不一致元素：`3/32768`
- 首个不一致坐标：`(0, 4, 211)`
- 最大绝对误差：`0.000244140625`

Batch 拆分：

- 失败样本：`sample=2`
- 不一致元素：`2/32768`
- 首个不一致坐标：`(0, 16, 255)`
- 最大绝对误差：`0.03125`

### 3.2 `qsmla_case_1_core_has_2_reduce_with_no_reduce_insight`

Batch 重排：

- 失败样本：`sample=30`
- 不一致元素：`1/32768`
- 首个不一致坐标：`(0, 0, 41)`
- 最大绝对误差：`0.0001220703125`

Batch 拆分：

- 失败样本：`sample=2`
- 不一致元素：`2/32768`
- 首个不一致坐标：`(0, 16, 255)`
- 最大绝对误差：`0.03125`

初步现象：失败集中在“一个核包含两个 reduce block”的场景。目前只定位到最终输出的首个不一致坐标，尚未定位到 Kernel 内部数据链的首个分歧阶段。

## 4. 跳过的用例

以下 10 个用例因为 `B < 2`，不满足 Batch 重排或拆分测试条件，因此两种模式均跳过：

| 用例 | 跳过原因 |
|---|---|
| `qsmla_case_K_1024` | Batch size 小于 2 |
| `qsmla_case_N128` | Batch size 小于 2 |
| `qsmla_case_N2` | Batch size 小于 2 |
| `qsmla_case_N32_single_batch_3_cores` | Batch size 小于 2 |
| `qsmla_case_single_batch_2_cores` | Batch size 小于 2 |
| `qsmla_case_single_batch_3_cores` | Batch size 小于 2 |
| `qsmla_case_single_batch_full_cores` | Batch size 小于 2 |
| `qsmla_case_single_batch_full_cores_with_blocks` | Batch size 小于 2 |
| `qsmla_case_testest` | Batch size 小于 2 |
| `qsmla_case_tnd_padding_1` | Batch size 小于 2 |

每个用例分别跳过重排和拆分测试，因此合计 `20 skipped`。

## 5. 执行命令

### 5.1 安装自定义算子包

```bash
cd /home/panjie/codes/ops-transformer

build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --install-path=/home/p00835995/custom_wz
```

### 5.2 执行全部 Batch 一致性测试

```bash
cd /home/panjie/tmp/torch_ext/ops-transformer/attention/mixed_quant_sparse_flash_mla/tests/pytest

source /home/p00835995/custom_wz/vendors/custom_transformer/bin/set_env.bash

bash test_run.sh batch_consistency \
  --pt-dir /tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase \
  --consistency-mode all
```

Batch 一致性脚本内部已设置：

```python
torch_npu.npu.set_deterministic_level(3)
```

### 5.3 执行并保存日志

建议开启 `pipefail`，否则通过 `tee` 保存日志时，pytest 的失败退出码可能被掩盖：

```bash
cd /home/panjie/tmp/torch_ext/ops-transformer/attention/mixed_quant_sparse_flash_mla/tests/pytest

source /home/p00835995/custom_wz/vendors/custom_transformer/bin/set_env.bash
set -o pipefail

timeout --signal=TERM --kill-after=10s 3600s \
  bash test_run.sh batch_consistency \
    --pt-dir /tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase \
    --consistency-mode all \
  2>&1 | tee /tmp/mqsmla-pr8737-batch-consistency.log
```

## 6. 完整日志

完整验证日志：`/tmp/mqsmla-pr8737-batch-consistency.log`
