# SparseFlashMla pytest_v2

`pytest_v2` 是原 pytest 的可检视副本：普通 single/save/load/load_graph 流程保持原样，batch
一致性直接接入 PT 加载、SMLA process、CPU golden 和原精度比较，而不是依赖 v1 的
`batch_consistency/` 包。

## 一致性数据流

```text
原 golden / PT save
  -> PT load
  -> batch/consistency/transform 同步处理 Q/KV/长度/索引
  -> DeterministicLevelGuard(level=3)
  -> sparse_flash_mla_metadata
  -> sparse_flash_mla（消费同一 metadata）
  -> origin 对齐 + MD5 + 原 check_result
  -> metadata 白盒覆盖 + baseline/derived 输入摘要
  -> 恢复旧 level
```

## 新增文件

| 文件 | 职责 |
| --- | --- |
| `test_sparse_flash_mla_batch_consistency.py` | 沿用原 batch PT 收集方式，提供模式 1-4 opt-in pytest 入口 |
| `test_batch_consistency_unit.py` | CPU 回归：四模式、双 case 格式、语义 oracle、MD5、FA/FD 展开 |
| `batch/sparse_flash_mla_consistency.py` | SMLA executor；调用原 process 并捕获实际 metadata |
| `batch/sparse_flash_mla_consistency_prepare.py` | Excel→原golden→baseline/derived PT的CPU-only验收入口 |
| `batch/consistency/flow.py` | level 3、baseline/derived、CPU 精度与 bitwise 双门禁编排 |
| `batch/consistency/transform.py` | BSND/TND/PA_BBND 模式变换与实际输入语义 oracle |
| `batch/consistency/compare.py` | `TokenOrigin` 对齐、有效 token 打包、MD5 和原精度复用 |
| `batch/consistency/metadata_layout.py` | 严格解析 1024-int32、36x9 FA、72x8 FD 和 reserved |
| `batch/consistency/metadata_coverage.py` | FA/FD token/core/workspace展开及floor/ceil公式审计 |
| `batch/consistency/reporting.py` | PT来源、shape/dtype/device/byte/digest/sample输入摘要 |
| `batch/consistency/offline.py` | origin-mapped expected output与derived PT round-trip |
| `batch/consistency/relation_manifest.py` | versioned baseline/derived TokenOrigin关系、内容指纹、预检与覆盖清单 |
| `batch/consistency/suite_config.py` | versioned per-PT suite配置、精确selector及shape-aware预检 |
| `batch/consistency/result_artifact.py` | 逐模式结果JSON、level3 PASS门禁、完整性检查与聚合 |
| `batch/consistency/metadata_stub.py` | CPU-only单核到多核候选规划及默认关闭hook约束 |
| 其余 `batch/consistency/*.py` | case格式、layout、schema、数据模型和命名配置 |

## 运行

```bash
source /mnt/workspace/gitCode/env.sh

# 需求示例：B=8 显式拆成 3+5。
bash test_run.sh consistency -P ./data \
  --modes reorder,split \
  --split-groups '0,1,2;3,4,5,6,7'

# 需求示例：单 batch 16K 拆成 4K+12K。
bash test_run.sh consistency -P ./data \
  --modes token-split \
  --token-split-sizes '4096,12288'

# 需求示例：300+100 与 300+200 比较共同前 300。
bash test_run.sh consistency -P ./data \
  --modes shape-change \
  --common-tokens 300 \
  --derived-extra 200

# 不用NPU，验证Excel、原golden、baseline/derived PT和四种模式。
bash test_run.sh consistency_prepare \
  --excel ../../../../requs/cases/SMLA_MQSMLA_batch一致性离线预检.xlsx \
  --sheet SMLA --pt-dir /tmp/smla_pt --report-dir /tmp/smla_report
```

数字 `-M 1,2` 仅为 v1 下游迁移兼容；正式配置使用 `--modes` 或 `--profile`。当前本机
torch_npu/CANN 在第一条 NPU 操作处拒绝 level 3，真实运行应报告 `UNSUPPORTED`，不能回退
level 0 伪造 PASS。

`consistency_prepare`只输出CPU-only预检状态，不代表NPU一致性通过。模式4没有origin的额外token
会在报告中记为`unmapped_tokens`，不会冒充完整CPU golden。每个baseline同时输出
`consistency_relation_manifest.json`，汇总目录输出`smla_consistency_relation_manifest.json`；
其中的PASS只表示TokenOrigin边界、scope、语义oracle、PT指纹和覆盖清单预检通过。

## 正式在线配置与结果

多个模式参数优先写入suite文件，不把一个`3+5`或`4K+12K`配置强套到所有PT：

```bash
bash test_run.sh consistency \
  --pt-dir /tmp/smla_pt/baseline \
  --config-file ../../../../requs/cases/SMLA_MQSMLA_batch一致性执行配置.json \
  --report-dir ./result/smla_consistency
```

suite只加载一次，但每个PT分别按operator/case name/PT basename/content SHA256精确匹配，并按
当前B和有效Q长度校验。`--config-file`不能和`--modes/--profile/--seed/groups/sizes`混用；
`-M 1,2`只保留在旧CLI转换层。

每次执行创建独立run目录，`results/*.json`对应一个`(operator,PT,mode)`，汇总为
`consistency_result_summary.json`；`pytest.log`仅用于诊断。只有执行报告通过且requested/observed
deterministic level均为3，逐模式状态才可写成`PASS`。聚合器还会检查预期artifact数和pytest
子进程异常，因此skip或退出码0不会被当成NPU通过。完整PASS返回0，真实FAIL返回1，
`UNSUPPORTED/PARTIAL/INCOMPLETE`等未完成状态返回2。汇总还校验唯一PT数和完整PT/mode矩阵；
损坏、重复或异算子旧artifact会写入汇总错误项，不能因文件总数刚好相等而通过。

当前模式1-4统一属于`same-source-cross-call`。独立编写PT之间的
`independent-cross-case`和同一次调用内位置互比`single-call-intra-case`是补充场景，artifact
会如实列为gap，不把模式1/3改名冒充。metadata stub planner目前只从自然metadata找候选并给出
拟拆分S2 block区间；FD契约、公式、安装包和level3关闭阻断前不会生成或执行synthetic tensor。
