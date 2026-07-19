# SparseFlashMla pytest_v2

`pytest_v2` 是原 pytest 的可检视副本：普通 single/save/load/load_graph 流程保持原样，batch
一致性直接接入 PT 加载、SMLA process、CPU golden 和原精度比较，而不是依赖 v1 的
`batch_consistency/` 包。

## 一致性数据流

```text
原 golden / PT save
  -> PT load
  -> consistency_transform 同步处理 Q/KV/长度/索引
  -> DeterministicLevelGuard(level=3)
  -> sparse_flash_mla_metadata
  -> sparse_flash_mla（消费同一 metadata）
  -> origin 对齐 + MD5 + 原 check_result
  -> metadata 白盒覆盖报告
  -> 恢复旧 level
```

## 新增文件

| 文件 | 职责 |
| --- | --- |
| `test_sparse_flash_mla_batch_consistency.py` | 沿用原 batch PT 收集方式，提供模式 1-4 opt-in pytest 入口 |
| `test_batch_consistency_unit.py` | CPU 回归：四模式、双 case 格式、语义 oracle、MD5、FA/FD 展开 |
| `batch/sparse_flash_mla_consistency.py` | SMLA executor；调用原 process 并捕获实际 metadata |
| `batch/consistency_flow.py` | level 3、baseline/derived、CPU 精度与 bitwise 双门禁编排 |
| `batch/consistency_transform.py` | BSND/TND/PA_BBND 模式变换与实际输入语义 oracle |
| `batch/consistency_compare.py` | `TokenOrigin` 对齐、有效 token 打包、MD5 和原精度复用 |
| `batch/metadata_layout.py` | 严格解析 1024-int32、36x9 FA、72x8 FD 和 reserved |
| `batch/metadata_coverage.py` | 将 FA/FD 展开到 batch/N2/query token/S2 block/workspace 并统计 Level-1/2 |
| 其余 `batch/consistency_*.py` | case 格式、layout、schema、数据模型和命名配置 |

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
```

数字 `-M 1,2` 仅为 v1 下游迁移兼容；正式配置使用 `--modes` 或 `--profile`。当前本机
torch_npu/CANN 在第一条 NPU 操作处拒绝 level 3，真实运行应报告 `UNSUPPORTED`，不能回退
level 0 伪造 PASS。
