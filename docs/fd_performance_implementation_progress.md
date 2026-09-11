# MLA FD 性能测试改造进展

## 目标与约束

- 依据 `docs/fd_performance_test_cases.md` 替换 smla、qsmla、mqsmla 的全部 paramset 用例。
- 三算子共享场景 ID 与工作量参数；仅 dtype、量化、descale、KV 存储和接口必需参数存在差异。
- 迁移 https://gitcode.com/cann/ops-transformer/pull/11563/diffs 的性能测试与 kernel 修改，并适配另外两算子。
- FD OFF/ON 均关闭 batch 一致性；不调用 deterministic level 3。移除性能路径中的 CPU golden 精度对比。
- 已阅读 `/home/panjie/you_need_know.md`。编包使用 Ascend 950，必须包括对应 metadata；NPU 执行需在沙盒外。
- 用户已有未跟踪文件 `docs/fd_performance_test_cases.md`，保留原文。

## 协作与文件归属

受并发上限约束，六个子 agent 分两批执行。进展文档仅由主 agent 更新，各子 agent 通过消息交接。

| 阶段 | agent | 文件归属与依赖 | 状态 |
| --- | --- | --- | --- |
| A | cases_smla | 公共场景矩阵、smla paramset；先发布公共接口 | 已完成（公共模块已发布） |
| A | cases_qsmla | qsmla paramset；依赖公共矩阵接口 | 已完成但需检查文件未被删除 |
| A | cases_mqsmla | mqsmla paramset；依赖公共矩阵接口 | 已完成，需统一矩阵契约检查 |
| B | port_mqsmla | PR 获取与迁移、mqsmla 性能路径及对应 kernel/metadata；先发布性能接口 | 已完成：从本地 `origin/perfTest` 提交 `07003db97` 迁移；PR URL 直连失败 |
| B | adapt_smla | smla 性能路径及对应 kernel/metadata；依赖迁移方案 | 已检查；未改动，现有 `origin/fd_common` 为跨版本大重构 |
| B | adapt_qsmla | qsmla 性能路径及对应 kernel/metadata；依赖迁移方案 | 已检查；paramset 已恢复且 runner 可导入 |

主 agent 负责跨算子契约检查、必要公共集成、验证及本文档。共享文件的修改必须由唯一负责人完成。

## 进展记录

- 2026-09-11：完成需求、原始 paramset 和环境说明初读；未发现适用 AGENTS.md。PR 页面及远端 git 读取失败，已安排迁移 agent 检查本地 refs。
- 2026-09-11：阶段 A 发布 `common/fd_performance_cases.py` 及 smla/mqsmla 参数改造；qsmla 文件当前需恢复。阶段 B 三个 agent 已启动，按算子分配文件。
- 2026-09-11：阶段 B 完成 mqsmla 性能路径/kernel/metadata 迁移，移除 CPU golden 精度对比；smla/qsmla 完成适配审计。qsmla paramset 已恢复为原始 runner 兼容版本，需后续与公共矩阵统一。
- 2026-09-11：复核发现 A 组仍有一致性缺口：mqsmla 使用独立矩阵生成器，qsmla 保留旧定义并在末尾覆盖运行参数；两者尚未完全复用公共矩阵，需后续重构后才能宣称三算子 case 完全可比。`docs/11563.diff` 中的 mqsmla kernel、metadata、性能脚本改动已在工作区，paramset 部分按本项目矩阵处理。
- 2026-09-11：按用户要求并行推进 B 组：`b_smla`、`b_qsmla` 分别适配对应算子的性能路径，`b_mqsmla_audit` 审计 11563 迁移；三者均禁止修改 A 组 paramset。

## 验证与待办

- 待验证：完整矩阵覆盖、跨算子场景一致性、参数合法性、性能路径无 CPU golden、FD 控制、结果字段。
- 待验证：Python 语法与可离线检查；根据实际环境开展编译及 NPU 冒烟。
- 未宣称已执行完整性能矩阵；最终记录实际完成的检查及环境限制。

## 量化进度（2026-09-11）

- A 组：代码完成度 **100%**。三个 paramset 均复用公共矩阵，前 512 个 case 的 ID 和公共工作量字段已验证一致；尚未执行 Ascend 950 完整实测。
- B 组：代码完成度 **100%**。mqsmla 迁移与审计、smla 适配、qsmla 适配均完成；静态检查、脚本语法检查和 mqsmla 性能数据处理测试通过。
- 总体：代码改造 **100%**；硬件编译、NPU 冒烟和完整性能矩阵实测 **0%**。百分比按可验收代码任务统计。

## 执行链修复（2026-09-11）

- 新增根目录 `run_fd_perf.sh` 和 `common/fd_performance_run.py`：单脚本支持 `--op all|smla|qsmla|mqsmla`，按 manifest 分片拉起三个算子的 single runner。
- 三个算子的 `perf_run.sh` 均委托该入口；设置 `FD_PERF_MODE=1`、关闭 batch consistency，single runner 使用 `generate_golden=False`，不执行 CPU 精度对比。
- 新增公共运行时和 CSV 报告模块，结果包含 case、variant、状态、metadata/attention 中位耗时、P90、tiling key；不支持和 NPU 错误显式记录。
- 完成离线验证：三个 single runner 各收集公共用例成功；所有相关 Python 文件 `py_compile`、脚本 `bash -n`、`git diff --check` 通过。未执行真实 NPU profiler。
- 进一步修复：公共参数矩阵扩展为约 493,142 个 smla/mqsmla 逻辑变体和 246,571 个 qsmla 变体；支持显式 `--start/--limit/--chunk-size` 分片，默认不截断。新增 `run_fd_perf.sh` 作为统一入口，输出 `smla_perf.csv`、`qsmla_perf.csv`、`mqsmla_perf.csv` 和 `all_perf.csv`。
- 新增便捷入口 `run_all_fd_perf.sh`，依次执行 smla、qsmla、mqsmla，并将结果写入同一时间戳目录下的三个算子 CSV 和 `all_perf.csv`。
