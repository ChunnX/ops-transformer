# Evidence Retention Policy

本策略约束 AscendC 算子性能优化后的证据保留、归档和清理。目标是让后续 agent 能持续迭代，而不是把每轮 profiling 原始产物长期堆在上下文目录里。

## Core Rule

`agent/` 是知识库，不是原始数据仓库。

长期保留：

- 关键命令。
- 环境和 package 信息。
- 精度 summary。
- 性能矩阵摘要。
- baseline/candidate 关键数字。
- 最终结论、撤回原因或阻塞原因。
- 可复现入口和 artifact manifest。

不应长期放入 `agent/`：

- raw `msprof` 输出目录。
- `.bin`、`.o`、完整 CSV dump。
- 大体量 build/install/profiling log。
- 重复轮次的中间 profiling 产物。

原始证据需要保留时，建议放入 `artifacts/` 或后续单独的 `agent_ops/<op>/artifacts` 边界，并记录 source、target、sha256、生成命令和保留原因。

## Status-Based Retention

| 状态 | raw 证据处理 | 长期记录 |
| --- | --- | --- |
| `optimized_kept` | 归档最小可复核 raw 证据并记录 hash，删除重复/中间轮次 dump | 最终矩阵、baseline/candidate 摘要、关键 profiling 数据、安装产物 hash、精度 summary |
| `failed_candidate` | README/manifest 写清失败原因后删除 raw dump | 失败假设、验证命令、summary、性能矩阵结论、撤回说明 |
| `blocked` | 写清阻塞 gate 后删除 raw dump | 阻塞条件、复现命令、错误码或缺失符号/binary 证据 |
| `candidate` | 临时保留 raw dump | 当前假设、待验证矩阵、下一步 |
| `baseline` | 只保留可复用基线的最小证据 | 基线命令、关键指标、环境 |

## Operator Ledger

当前 `ops-transformer` 已开始使用 `agent_ops/<op>/` 账本。进入真实算子优化闭环后，建议每个算子建立：

```text
agent_ops/<op>/
  README.md
  manifest.json
  performance.md
  workflow.md
```

账本只记录摘要、命令、结论和 artifact hash；raw dump 放到 artifact 边界。

建议 `manifest.json` 字段：

```json
{
  "operator": "<op_name>",
  "domain": "attention|ffn|gmm|mc2|moe|posembedding",
  "status": "candidate|baseline|optimized_kept|failed_candidate|blocked",
  "soc": "ascend910b",
  "arch": "arch22",
  "vendor": "custom_transformer",
  "accuracy": {
    "command": "<command>",
    "summary": "failed=0,error=0,unsupported=0"
  },
  "package_load": {
    "build_command": "<command>",
    "install_command": "<command>",
    "opapi_sha256": "<hash>",
    "kernel_config_evidence": "<path or summary>"
  },
  "performance": {
    "matrix": "<path or summary>",
    "decision": "<kept/rejected/blocked reason>"
  },
  "artifacts": [
    {
      "path": "<artifact path>",
      "sha256": "<hash>",
      "retention_action": "archive_with_hash|delete_after_manifest_summary|keep_until_decision"
    }
  ]
}
```

如果没有真实 profiling 或精度闭环，不要创建“成功优化”账本；只记录候选、阻塞或 baseline 状态。

## Cleanup Rule

删除 raw 证据前必须确认：

- 已有足够 summary。
- 已记录复现命令。
- 已记录关键错误或关键性能数字。
- 成功优化的最小证据已归档。
- 失败或阻塞的原因足以避免后续重复踩坑。

不得删除尚未总结的唯一证据。
