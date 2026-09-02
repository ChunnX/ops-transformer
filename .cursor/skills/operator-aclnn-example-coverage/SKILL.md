---
name: operator-aclnn-example-coverage
description: 统计 ops-cv、ops-math、ops-nn、ops-transformer 四仓 aclnn API 的 example 覆盖率（以 docs/zh/op_api_list.md 接口列表为分母），产出总覆盖率、各仓覆盖率及未覆盖接口列表；可选按 soc 调用 build.sh 统计运行通过率。Use when the user asks for operator example coverage, aclnn example statistics, test_aclnn example coverage, op_api_list coverage, or 算子 example 覆盖率统计 across the four ops repos.
---

# 算子 ACLNN Example 覆盖率统计

## 统计口径

### 静态模式（默认）

| 项 | 规则 |
| --- | --- |
| 四仓 | `ops-cv`、`ops-math`、`ops-nn`、`ops-transformer`（可用 `--repos` 筛选，默认全跑） |
| 接口来源 | 各仓 `docs/zh/op_api_list.md`「接口列表」表格 |
| 分母 | 表格「接口列表」每一行（`aclnnFoo&aclnnInplaceFoo` 计 1 条）；排除 `## 废弃接口` |
| 不计入分母 | `## 废弃接口` 专节；主列表中与废弃专节同名的条目 |
| 分子 | 对应算子目录 `examples/**/test_aclnn_*.cpp` 中能检索到完整接口标识符 |
| 算子目录 | 由接口文档链接解析：`.../<op_dir>/docs/aclnnXxx.md` → `<op_dir>` |
| 匹配规则 | 匹配完整标识符 `aclnnFoo` 或 `aclnnFooGetWorkspaceSize`（不把 `aclnnFooV2` 等更长名算作覆盖）；组合名按 `&` 拆开，任一命中即覆盖 |

### 运行模式（`--exec <soc>`）

| 项 | 规则 |
| --- | --- |
| 平台支持 | 解析接口文档「产品支持情况」（`<!-- npu="910b\|A3\|950" -->` / 产品文案） |
| 分母 | 文档标明**支持**该 soc 的接口 |
| 跳过 | 文档标明**不支持**该 soc → 不跑、不计分母 |
| 分子 | 分母内 `bash build.sh --run_example <op> eager --soc=<soc>` **执行成功** |
| 无 example | 支持该 soc 但无匹配 example → 计入分母，状态 `no_example`（未通过） |
| 同算子 | 同一 `op_dir` 只跑一次 build.sh，结果复用 |

soc 映射：`ascend910b`(A2) / `ascend910_93`(A3) / `ascend950`(A5)。

## 工作流

```
Task Progress:
- [ ] Step 1: 确认四仓根目录与 op_api_list.md
- [ ] Step 2: 运行统计脚本
- [ ] Step 3: 检查报告并回复摘要
```

**Step 1: 确认根目录**

默认根目录为包含四仓的父目录。当前工作区示例：

```
example_Statistics/
├── ops-cv/docs/zh/op_api_list.md
├── ops-math/docs/zh/op_api_list.md
├── ops-nn/docs/zh/op_api_list.md
├── ops-transformer/docs/zh/op_api_list.md
```

**Step 2: 运行统计脚本**

Markdown 全量报告（默认四仓）：

```bash
python3 .cursor/skills/operator-aclnn-example-coverage/scripts/collect_aclnn_example_coverage.py \
  --root /path/to/example_Statistics
```

只统计部分仓：

```bash
python3 .cursor/skills/operator-aclnn-example-coverage/scripts/collect_aclnn_example_coverage.py \
  --root /path/to/example_Statistics --repos ops-math,ops-nn
```

按 soc 执行 example（需本机 CANN / 设备或仿真就绪）：

```bash
python3 .cursor/skills/operator-aclnn-example-coverage/scripts/collect_aclnn_example_coverage.py \
  --root /path/to/example_Statistics --exec ascend950 --repos ops-cv --format both
```

自动化场景仅要摘要 JSON（打印到 stdout）：

```bash
python3 .cursor/skills/operator-aclnn-example-coverage/scripts/collect_aclnn_example_coverage.py \
  --root /path/to/example_Statistics --format json --pull
```

可选参数：

- `--repos LIST`：参与统计的仓，逗号分隔；默认四仓全跑
- `--exec SOC`：运行模式；`ascend910b` / `ascend910_93` / `ascend950`
- `--exec-timeout SEC`：单个算子 `build.sh` 超时（默认 300）
- `--pull`：统计前对**选定仓**分别执行 `git pull --ff-only`；失败则非 0 退出
- `--output`：Markdown 路径（静态默认 `<root>/aclnn_example_coverage_report.md`；`--exec` 时为 `<root>/aclnn_example_exec_<soc>_report.md`）
- `--json-output`：JSON 路径；`--format json` 且未指定时写 stdout
- `--format both`：同时产出 Markdown 与 JSON

**Step 3: 向用户汇报**

读取生成的报告，回复：

1. 总覆盖率 / 运行通过率（分子/分母 + 百分比）
2. 各仓覆盖率表格
3. 未覆盖 / 未通过接口（如有；可摘要数量并列出或指向报告）
4. 报告文件路径

## 报告结构

### Markdown（`--format markdown`）

静态模式：

1. **总览** — 合计覆盖率
2. **各仓覆盖率** — 分仓分子/分母/百分比
3. **接口 example 覆盖情况** — 全量接口及是否有 example
4. **未覆盖接口列表** — 无匹配 example 的接口
5. **不计入分母的废弃接口**

运行模式（`--exec`）：

1. **总览** — 该 soc 支持数（分母）与执行成功数（分子）
2. **各仓运行通过率**
3. **接口执行明细** — 支持 / 有 example / 状态 / 详情
4. **支持但未通过列表**

### JSON（`--format json`，仅摘要）

静态：

```json
{
  "total": {
    "coverage_rate": 91.23,
    "in_scope_count": 993,
    "with_example_count": 906
  },
  "repos": {
    "ops-cv": {
      "coverage_rate": 100.0,
      "in_scope_count": 46,
      "with_example_count": 46
    }
  }
}
```

运行模式额外含 `soc`，`in_scope_count`/`with_example_count` 分别表示支持数与跑通数。

## 注意事项

- 实际文件名为 `docs/zh/op_api_list.md`（不是 `ap_api_list.md`）
- 仅解析表格「接口名」列的链接行，忽略说明列中的嵌套链接
- 同一算子目录下可有多个接口；每个接口独立判定是否有 example
- 跳过 `## 废弃接口` 专节；主列表若与废弃专节同名（如部分 MlaProlog）也不计入分母
- 仅当 example 源码中出现完整接口标识符（`aclnnFoo` 或 `aclnnFooGetWorkspaceSize`）才算覆盖；裸子串命中更长名（如 `aclnnFooV2`）不算
- V2/V3 等版本接口需自身名字被匹配到（不会因同目录其它版本 example 自动算覆盖）
- `--exec` 依赖仓内 `build.sh` 与本地 CANN/设备环境；平台支持以 aclnn 文档为准（不用 `AddConfig`）
- `--exec` 运行中会在 **stderr** 实时打印 `[exec] OK/FAIL`、`[result] OK/FAIL`；最终报告仍整次结束才落盘。建议：`python3 ... --exec ... 2>&1 | tee run.log`
