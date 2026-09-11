# Pdist 开发过程文档（PyPTO-Pro 流程产物迁移）

本目录是 pdist 算子 PyPTO-Pro 五阶段开发流程（Stage 1–5）的过程产物存档，自
`pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/custom/pdist`
迁移而来（开发时间 2026-09-01 ~ 09-02），作为本仓实现的历史依据与维护参考。

## 与本仓实现的对应关系

| 流程产物 | 本仓落点 |
|---|---|
| SPEC.md（Stage 1 需求规格，冻结合同） | `../../README.md`、`op_host/pdist_tiling.cpp` 的校验合同 |
| DESIGN.md / DESIGN_BINDINGS.json / module_interfaces.yaml（Stage 3 设计） | `op_kernel/pdist.py`（tile 布局 / PBranch / 切分方案的设计来源） |
| pdist_golden*.py（Stage 2 golden，未随迁） | `tests/assets/golden.py`（同源精简版） |
| test_pdist.py（Stage 4 交付件，未随迁） | `op_kernel/pdist.py`（移植源） |
| PERFORMANCE_CASES.json / PERFORMANCE_REPORT.md（Stage 5 性能证据） | 无运行时依赖，仅作性能基线参考 |

## 阅读注意

- 文档中的 `.devkit` 缓存路径、`custom/pdist` 同级文件引用（test_pdist.py、
  pdist_golden.py、prof/ 证据目录等）指向原开发环境，在本仓中不存在
- KB_SELECTION.json / KB_USAGE.json / PRO_MATERIAL_INDEX.md / EXPLORE_REPORT.md
  为流程内部知识管理产物，路径引用同样仅具历史意义
- 唯一已声明的实现偏差（NaN 常量以 ln(0) 惯用法替代字面量）见
  `op_kernel/pdist.py` 模块 docstring
