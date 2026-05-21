# MC2 算子测试选择工具

根据 Git PR 变更文件列表、白名单和算子依赖关系，自动确定需要执行的算子测试用例。

## 处理流程

```
1. 白名单过滤 → 过滤出 mc2 目录下的变更文件
2. 忽略规则过滤 → 忽略 .md 文件和 tests 目录
3. 非 mc2 目录检测 → 有 mc2 外文件则触发全量测试
4. common/3rd 目录检测 → 特殊处理或全量测试
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
| 依赖关系 JSON | `op_filters/result.json` | **动态生成**，每次运行覆盖 |
| 测试算子 JSON | `op_tests.json` 或命令行指定 | 最终需要测试的算子列表 |
| 日志文件 | `output.log` 或命令行指定 | 处理过程日志 |

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

```bash
python op_filters/op_whitelist_step1.py
python op_filters/op_ignored_step2.py
python op_filters/op_non_mc2_step3.py
python op_filters/op_common_3rd_step4.py
python op_filters/op_extractor_step5.py
python op_filters/op_dependency_step6.py
python op_filters/op_output_step7.py
```

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
| common 非特殊处理文件变更 | **全量测试** |
| 3rd 非特殊处理子目录变更 | **全量测试** |
| common/3rd 特殊处理 + 其他算子变更 | 部分测试 |

## 特殊处理配置

### common 文件特殊处理映射

位于 `op_common_3rd_step4.py` 的 `COMMON_FILE_OP_MAPPING`，变更这些文件只触发特定算子：

- `utils/mc2_exception_dump.h` → moe_distribute_combine_v2, moe_distribute_combine_v3, moe_distribute_dispatch_v2, moe_distribute_dispatch_v3
- `utils/mc2_moe_utils.h` → allto_all_all_gather_batch_mat_mul, batch_mat_mul_reduce_scatter_allto_all, grouped_mat_mul_allto_allv, quant_grouped_mat_mul_allto_allv
- `utils/context_util.h` → all_gather_matmul, allto_allv_grouped_mat_mul, allto_allv_quant_grouped_mat_mul, matmul_reduce_scatter, quant_grouped_mat_mul_allto_allv
- 更多配置见代码...

### 3rd 子目录特殊处理映射

位于 `op_common_3rd_step4.py` 的 `THIRD_PARTY_OP_MAPPING`，变更这些子目录只触发特定算子：

- `quant_batch_matmul_v3` → all_gather_matmul_v2, allto_all_matmul, matmul_all_reduce...
- `batch_mat_mul_v3` → allto_all_all_gather_batch_mat_mul, batch_mat_mul_reduce_scatter_allto_all
- `rms_norm` → matmul_all_reduce_add_rms_norm, moe_distribute_combine_add_rms_norm
- 更多配置见代码...

## 依赖关系生成

`result.json` 每次运行时动态生成，包含：
- 算子名称
- dependencies（已屏蔽 common/3rd/tools）
- supported_products（支持的产品类型）

依赖分析基于 `mc2_analyzer.py`，解析各算子目录下的 `op_host/CMakeLists.txt` 和 `_def.cpp` 文件。

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