# MC2 算子测试选择工具

根据 Git PR 变更文件列表、白名单和算子依赖关系，自动确定需要执行的算子测试用例。

## 处理流程

```
1. 白名单过滤 → 过滤出 mc2 目录下的变更文件
2. 忽略规则过滤 → 忽略 .md 文件和 tests 目录
3. 非 mc2 目录检测 → 有 mc2 外文件则触发全量测试
4. common/3rd 目录检测 → 有变更则直接触发全量测试
5. 算子提取 → 从文件路径提取算子名称
6. 反向依赖搜索 → 找出依赖变更算子的算子
7. SOC 类型过滤 → 根据指定 SOC 类型过滤最终算子
8. 输出 JSON → 输出到 op_tests.json
```

## 输入文件

| 文件 | 路径 | 说明 |
|---|---|---|
| 变更文件列表 | `changed_files.txt` 或命令行指定 | PR 变更的文件路径列表，每行一个 |
| 白名单 | `op_filters/whitelist.txt` | 允许的路径模式列表 |

## 输出文件

| 文件 | 路径 | 说明 |
|---|---|---|
| 依赖关系 JSON | `op_filters/result.json` | 算子依赖关系，可通过 `mc2_analyzer.py` 或 `select_op_tests_main.py` 生成 |
| 测试算子 JSON | `op_tests.json` 或命令行指定 | 最终需要测试的算子列表 |
| 日志文件 | `output.log` 或命令行指定 | 处理过程日志 |

## 依赖关系生成

`result.json` 包含：
- 算子名称
- dependencies（已屏蔽 common/3rd/tools）
- supported_products（支持的产品类型）

### 方式一：单独生成（调试用）

```bash
# 使用默认参数，生成到 op_filters/result.json
python op_filters/mc2_analyzer.py

# 自定义参数
python op_filters/mc2_analyzer.py --mc2 /path/to/mc2 --output custom_result.json
```

### 方式二：自动生成（一键执行，推荐）

`select_op_tests_main.py` 每次执行都会自动调用 `mc2_analyzer.py` 生成 `result.json` 并覆盖已有文件。

**注意：** 逐步执行脚本（step1-7）不自动生成 result.json，需要预先通过上述方式生成。

## 使用方法

### 推荐方式（一键执行）

```bash
# 使用默认参数
python op_filters/select_op_tests_main.py --pr changed_files.txt

# 指定 SOC 类型过滤
python op_filters/select_op_tests_main.py --pr changed_files.txt --soc ascend910b ascend910_93

# 自定义所有参数
python op_filters/select_op_tests_main.py \
    --pr changed_files.txt \
    --whitelist custom_whitelist.txt \
    --mc2 /path/to/mc2 \
    --output-json result.json \
    --output-log output.log \
    --soc ascend910b ascend910_93
```

### 逐步执行（调试用）

每个步骤脚本都支持命令行参数：

```bash
# Step 1: 白名单过滤
python op_filters/op_whitelist_step1.py \
    --input changed_files.txt \
    --whitelist op_filters/whitelist.txt \
    --output filtered_files.txt \
    --log step1.log

# Step 2: 忽略规则过滤
python op_filters/op_ignored_step2.py \
    --input filtered_files.txt \
    --mc2 mc2 \
    --output remaining_files.txt \
    --log step2.log

# Step 3: 非 mc2 目录检测
python op_filters/op_non_mc2_step3.py \
    --input remaining_files.txt \
    --mc2 mc2 \
    --dependency-json op_filters/result.json \
    --output non_mc2_filtered.txt \
    --log step3.log

# Step 4: common/3rd 目录检测
python op_filters/op_common_3rd_step4.py \
    --input non_mc2_filtered.txt \
    --mc2 mc2 \
    --dependency-json op_filters/result.json \
    --output common_3rd_filtered.txt \
    --log step4.log

# Step 5: 算子提取
python op_filters/op_extractor_step5.py \
    --input common_3rd_filtered.txt \
    --output extracted_ops.txt \
    --soc ascend910b ascend910_93 \
    --log step5.log

# Step 6: 反向依赖搜索
python op_filters/op_dependency_step6.py \
    --input extracted_ops.txt \
    --dependency-json op_filters/result.json \
    --output dependent_ops.txt \
    --log step6.log

# Step 7: 输出 JSON
python op_filters/op_output_step7.py \
    --input dependent_ops.txt \
    --dependency-json op_filters/result.json \
    --output-json op_tests.json \
    --log step7.log
```

### 各脚本参数说明

| 脚本 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| **op_whitelist_step1.py** | `--input` | `op_filters/changed_files.txt` | 输入文件 |
| | `--whitelist` | `op_filters/whitelist.txt` | 白名单文件 |
| | `--output` | `filtered_files.txt` | 输出文件 |
| | `--log` | `step1.log` | 日志文件 |
| **op_ignored_step2.py** | `--input` | `filtered_files.txt` | 输入文件 |
| | `--mc2` | `mc2` | mc2 目录路径 |
| | `--output` | `remaining_files.txt` | 输出文件 |
| | `--log` | `step2.log` | 日志文件 |
| **op_non_mc2_step3.py** | `--input` | `remaining_files.txt` | 输入文件 |
| | `--mc2` | `mc2` | mc2 目录路径 |
| | `--dependency-json` | `op_filters/result.json` | 依赖关系 JSON |
| | `--output` | `non_mc2_filtered.txt` | 输出文件 |
| | `--log` | `step3.log` | 日志文件 |
| **op_common_3rd_step4.py** | `--input` | `non_mc2_filtered.txt` | 输入文件 |
| | `--mc2` | `mc2` | mc2 目录路径 |
| | `--dependency-json` | `op_filters/result.json` | 依赖关系 JSON |
| | `--output` | `common_3rd_filtered.txt` | 输出文件 |
| | `--log` | `step4.log` | 日志文件 |
| **op_extractor_step5.py** | `--input` | `common_3rd_filtered.txt` | 输入文件 |
| | `--output` | `extracted_ops.txt` | 输出文件 |
| | `--soc` | 无 | SOC 类型列表 |
| | `--log` | `step5.log` | 日志文件 |
| **op_dependency_step6.py** | `--input` | `extracted_ops.txt` | 输入文件 |
| | `--dependency-json` | `op_filters/result.json` | 依赖关系 JSON |
| | `--output` | `dependent_ops.txt` | 输出文件 |
| | `--log` | `step6.log` | 日志文件 |
| **op_output_step7.py** | `--input` | `dependent_ops.txt` | 输入文件 |
| | `--dependency-json` | `op_filters/result.json` | 依赖关系 JSON |
| | `--output-json` | `op_tests.json` | 输出 JSON |
| | `--full-test` | False | 全量测试模式 |
| | `--log` | `step7.log` | 日志文件 |

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pr` | 必填 | 变更文件列表路径 |
| `--whitelist` | `op_filters/whitelist.txt` | 白名单文件路径 |
| `--mc2` | `mc2/` | mc2 目录路径 |
| `--output-json` | `op_tests.json` | 输出 JSON 路径 |
| `--output-log` | `output.log` | 输出日志路径 |
| `--soc` | 无 | SOC 类型列表：`ascend910b`, `ascend910_93`, `ascend950` |

## 触发条件

| 触发条件 | 结果 |
|---|---|
| 白名单过滤后无剩余文件 | 无测试 |
| 忽略规则过滤后无剩余文件 | 无测试 |
| mc2 目录外有变更文件 | **全量测试** |
| common 目录有变更文件 | **全量测试** |
| 3rd 目录有变更文件 | **全量测试** |
| 仅算子目录变更 | 部分测试（变更算子 + 反向依赖算子） |

> **注意**: common/3rd 的特殊处理映射逻辑已禁用，如需恢复请参考 `op_step4_common_3rd.py` 中的注释代码。

## 特殊处理配置（已禁用）

> **以下特殊处理映射逻辑已禁用**，common/3rd 目录变更现在直接触发全量测试。
> 如需恢复，取消 `op_step4_common_3rd.py` 中 `COMMON_FILE_OP_MAPPING` 和 `THIRD_PARTY_OP_MAPPING` 的注释，
> 并恢复 `filter_common_3rd` 函数中的特殊处理分支逻辑。

### common 文件特殊处理映射（已禁用）

原位于 `op_step4_common_3rd.py` 的 `COMMON_FILE_OP_MAPPING`，变更这些文件只触发特定算子：

- `utils/mc2_exception_dump.h` → moe_distribute_combine_v2, moe_distribute_combine_v3, moe_distribute_dispatch_v2, moe_distribute_dispatch_v3
- `utils/mc2_moe_utils.h` → allto_all_all_gather_batch_mat_mul, batch_mat_mul_reduce_scatter_allto_all, grouped_mat_mul_allto_allv, quant_grouped_mat_mul_allto_allv
- `utils/context_util.h` → all_gather_matmul, allto_allv_grouped_mat_mul, allto_allv_quant_grouped_mat_mul, matmul_reduce_scatter, quant_grouped_mat_mul_allto_allv
- 更多配置见代码中的注释...

### 3rd 子目录特殊处理映射（已禁用）

原位于 `op_step4_common_3rd.py` 的 `THIRD_PARTY_OP_MAPPING`，变更这些子目录只触发特定算子：

- `quant_batch_matmul_v3` → all_gather_matmul_v2, allto_all_matmul, matmul_all_reduce...
- `batch_mat_mul_v3` → allto_all_all_gather_batch_mat_mul, batch_mat_mul_reduce_scatter_allto_all
- `rms_norm` → matmul_all_reduce_add_rms_norm, moe_distribute_combine_add_rms_norm
- 更多配置见代码中的注释...

## 输出格式

```json
{
  "ascend910b": [
    "all_gather_matmul",
    "matmul_all_reduce"
  ],
  "ascend910_93": [
    "allto_all_matmul"
  ]
}
```

若未指定 SOC 类型，输出格式：

```json
[
  {
    "op_name": "all_gather_matmul",
    "supported_products": ["ascend910b", "ascend910_93", "ascend950"]
  }
]
```