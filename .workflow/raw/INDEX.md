# raw/ — 原始文档索引

<!-- 更新目录内容时更新索引文档 -->

> 本目录保存所有原始文档，**不可修改**。其他文件是衍生物，这里是 truth source。
> 本索引帮助 agent 按需检索，实现渐进式披露。

## 目录结构

```
raw/
├── INDEX.md        ← 本文件（检索入口）
├── spec/           ← 需求规格说明书（实现无关的原理说明，视项目需求而定）
├── docs/           ← 参考文档（工作中产生的走读、分析、设计文档等）
└── design/         ← 完成本项目任务的设计方案（实现方案、软件架构、编码方案、伪代码等）
```

## spec/ — 需求规格

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [spec/gmmfr_op.md](spec/gmmfr_op.md) | 参考算子原型设计说明书样例（GMMFR MxA8W4 weightNz） | 需要理解参考算子原理、算子规格文档格式时 |
| [spec/gmm_a16w4_950_requirement.md](spec/gmm_a16w4_950_requirement.md) | GMM A16W4(INT4) weight ND 数据流 Ascend950 场景需求 | 需要理解本项目目标、约束和实现方案要点时 |
| [spec/gmm_a16w4nd_op.md](spec/gmm_a16w4nd_op.md) | GMM A16W4 weight ND pergroup 算子原型描述文档 | 需要理解 A16W4 ND 数据流的输入输出规格时 |

## docs/ — 参考文档

<!-- 工作中产生的文档在此登记。包括代码走读、差异分析、设计方案、学习文档等 -->

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [docs/gmm_a16w8_tiling_walkthrough.md](docs/gmm_a16w8_tiling_walkthrough.md) | GMM A16W8 Tiling 走读文档：GMMWeightQuantParam 字段、tiling 计算流程、groupSize 设置、A16W4 适配点 | 需要了解 A16W8 tiling 实现或 A16W4 tiling 适配方案时 |
| [docs/gmm_a16w8_kernel_walkthrough.md](docs/gmm_a16w8_kernel_walkthrough.md) | GMM A16W8 Kernel 走读文档：Controller 流程、SetGmAddr 地址偏移、L1/UB 空间划分、Cube/Vec 调度、A16W4 适配点 | 需要了解 A16W8 kernel 实现或 A16W4 kernel 适配方案时 |
| [docs/wqbmmv2_nd_antiquant_walkthrough.md](docs/wqbmmv2_nd_antiquant_walkthrough.md) | WQBMMv2 ND INT4 PerGroup 反量化走读文档：NK/KN 排布实现、VF 函数、迁移适配方案 | 需要了解 ND int4 pergroup 反量化实现或 GMM 迁移方案时 |

## development/ — 开发辅助文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [development/gmm_a16w4_pergroup_nd_mindmap.md](development/gmm_a16w4_pergroup_nd_mindmap.md) | GMM A16W4 PerGroup ND 脑图（输入/输出/UT/ST 总览，162 个用例分类和条数） | 需要快速了解算子全貌、测试用例分类时 |
| [development/gmm_a16w4_pergroup_nd_stc.md](development/gmm_a16w4_pergroup_nd_stc.md) | STC 用例设计方案文档：测试策略、用例设计方法论、数量统计（162 个） | 需要了解用例设计方案时 |
| [development/gmm_a16w4_pergroup_nd_srs.md](development/gmm_a16w4_pergroup_nd_srs.md) | SRS 算子实现方案文档：tiling 切分 SVG、buffer 分配 SVG、软件算法到硬件映射 SVG、计算流程伪代码、关键修改点 | 需要向他人介绍算子实现方案时 |
| [development/wqbmmv2_pergroup_dequant_research.md](development/wqbmmv2_pergroup_dequant_research.md) | wqbmmv2 PerGroup反量化实现方案分析、GMM A16W4 PerGroup ND反量化迁移方案 | 需要了解反量化实现方案时 |
| [development/code_review_report.md](development/code_review_report.md) | GMM A16W4 PerGroup ND 代码检视报告：26 个问题（2 高/21 中/3 低），含安全/性能/API/风格/重复代码检查 | 需要查看代码检视结果、修复优先级时 |

## design/ — 设计文档

<!-- 项目任务实现方案的设计文档，包括实现方案、架构设计、编码方案、伪代码等 -->

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [design/01_deliverables.md](design/01_deliverables.md) | 交付件清单：11 个需修改文件、8 个 tiling key 组合、4 个 WqmmConfig 实例、2 个 VF 函数 | 需要了解完整交付件范围时 |
| [design/02_platform_independent_design.md](design/02_platform_independent_design.md) | 与硬件无关的实现方案：地址偏移公式推导（含 bias 偏移分离）、兼容性策略、测试方案（11+8+6 UT 用例） | 需要了解总体设计和跨平台参考时 |
| [design/03_tiling_design.md](design/03_tiling_design.md) | Tiling 拦截校验与切分方案：GMMWeightQuantParam 结构、IsA16W4NDPergroup 判断、8 个 tiling key 组合注册 | 需要了解 tiling 实现方案时 |
| [design/04_kernel_design.md](design/04_kernel_design.md) | Kernel 实现方案：SetGmAddr pergroup 偏移、WqmmConfig 新增、GetNdPerGroupBufferInfo、VF 函数签名和伪代码 | 需要了解 kernel 实现方案时 |
| [design/05_test_case_design.md](design/05_test_case_design.md) | 测试用例设计文档：162 个用例（8 种类型），含 tiling 异常拦截/精度/功能/性能/泛化/打桩/tiling白盒/kernel白盒 | 需要了解具体测试用例设计时 |
| [design/test_cases_gmm_a16w4_pergroup_nd.xlsx](design/test_cases_gmm_a16w4_pergroup_nd.xlsx) | 测试用例 Excel 表格（8 个 sheet，162 个用例） | 需要查看/导入测试用例参数时 |
| [design/generate_test_cases.py](design/generate_test_cases.py) | 测试用例生成脚本（pandas + openpyxl） | 需要重新生成或修改 Excel 时 |
