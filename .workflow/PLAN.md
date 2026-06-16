# PLAN — 实现计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段计划。
> 仅关注 tiling + kernel，不涉及 Aclnn 通路（API checker、infershape）。
> **初期方案计划不进行编码，主要用于确认编码方案，保证方案正确有效。**

## 阶段一：规划与设计（当前阶段）

> 本阶段产出完整的算子规格文档、代码走读文档和 SRS 实现方案文档，为后续编码提供充分依据。

### 任务 1：编写算子原型描述文档
**目标**: 完成 `spec/gmm_a16w4nd_op.md`，参照 `gmmfr_op.md` 格式编写 GMM A16W4 weight ND 的算子规格文档
**产出**: `.workflow/raw/spec/gmm_a16w4nd_op.md`
**详细任务说明**: [reference/next_steps/1_op_spec.md](reference/next_steps/1_op_spec.md)

### 任务 2：走读 GMM A16W8 代码实现
**目标**: 走读 kernel 实现（`GMMWeightQuantBasicController`）和 tiling 实现（`grouped_weight_quant_batch_matmul_tiling`），产出完整走读文档
**产出**: `.workflow/raw/docs/` 下的走读文档
**详细任务说明**: [reference/next_steps/2_walkthrough_gmm_a16w8.md](reference/next_steps/2_walkthrough_gmm_a16w8.md)

### 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现
**目标**: 走读 ops-nn 仓库中 weight ND int4 的反量化方法，了解如何迁移到 GMM
**产出**: `.workflow/raw/docs/` 下的走读文档
**详细任务说明**: [reference/next_steps/3_walkthrough_wqbmmv2.md](reference/next_steps/3_walkthrough_wqbmmv2.md)

### 任务 4：编写 SRS 算子开发实现方案
**目标**: 基于走读结果，编写详细的 GMM A16W4 实现方案，按模块分文档产出
**产出**: `.workflow/raw/design/` 下的 4 个设计文档（按编号顺序阅读）：
  - `01_deliverables.md` — 交付件清单（新增 tiling 模板、注册分发、def.cpp 适配、_apt.cpp 适配、kernel 实现）
  - `02_platform_independent_design.md` — 与硬件无关的实现方案（地址偏移推导、兼容性、测试方案）
  - `03_tiling_design.md` — Tiling 拦截校验与切分方案
  - `04_kernel_design.md` — Kernel 实现方案（Controller/BasicBlock/VF/L1/UB）
**详细任务说明**: [reference/next_steps/4_srs_design.md](reference/next_steps/4_srs_design.md)

### 任务 5：设计 GMM A16W4 PerGroup ND 测试用例
**目标**: 基于算子约束说明和 01/02/03/04 设计文档，设计具体的测试用例（tiling 异常拦截、精度、功能、性能、泛化、kernel 打桩验证、tiling 白盒、kernel 白盒），给出具体参数设置，产出用例设计文档和输入 Excel 表格
**产出**: `.workflow/raw/design/` 下的用例设计文档和 Excel 表格：
  - `05_test_case_design.md` — 用例设计文档
  - `test_cases_*.xlsx` — 用例输入 Excel 表格（8 个 sheet）
  - `generate_test_cases.py` — 用例生成脚本
**详细任务说明**: [reference/next_steps/5_test_case_design.md](reference/next_steps/5_test_case_design.md)

### 任务 6：总结归纳设计文档
**目标**: 对方案设计文档、用例文档和脑图进行总结归纳，形成留档查看的设计文档。包括更新脑图、创建 STC 用例设计方案文档、创建 SRS 算子实现方案文档（含 SVG 图、软件算法映射到硬件的计算流程、关键修改点）
**产出**: `.workflow/raw/development/` 下的 3 个文档：
  - `gmm_a16w4_pergroup_nd_mindmap.md` — 更新后的脑图（对齐用例设计）
  - `gmm_a16w4_pergroup_nd_stc.md` — STC 用例设计方案文档
  - `gmm_a16w4_pergroup_nd_srs.md` — SRS 算子实现方案文档
**详细任务说明**: [reference/next_steps/6_summary_docs.md](reference/next_steps/6_summary_docs.md)

## 阶段二：编码实现（当前阶段）

> 在阶段一方案确认后启动编码。采用 TDD 方式：先写测试，再实现代码，最后验证。

### 任务 7：Tiling 实现 + UT 验证
**目标**: 在 arch35 tiling 中为 A16W4 pergroup ND 场景生成正确的 tiling 参数，并通过 UT 验证
**详细任务说明**: [reference/next_steps/7_tiling_impl.md](reference/next_steps/7_tiling_impl.md)

#### 7.1 UT 测试框架搭建 + 用例编写（先写测试）
**目标**: 建立完整的测试基线，确保后续 tiling 实现可立即验证
**产出**:
- `gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.csv` — CSV 用例数据
- `gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.cpp` — CSV 驱动测试代码
**用例覆盖**: ~60 个用例（正常通过 ~30 + 异常拦截 ~20 + tiling 白盒 ~10）
**验证**: 编译通过，新增用例预期 FAIL（tiling 代码未修改）

#### 7.2 Tiling 代码实现
**目标**: 在现有 tiling 模板中新增 A16W4 PerGroup ND 分支
**修改文件**:
- `grouped_weight_quant_batch_matmul_tiling.h` — 新增 `IsA16W4NDPergroup()` 声明
- `grouped_weight_quant_batch_matmul_tiling.cpp` — pergroup 校验/计算分支（6 个修改点）
- `weight_quant_tiling_key.h` — 新增 8 个 `ASCENDC_TPL_ARGS_SEL` 组合
**验证**: 编译通过

#### 7.3 UT 验证 + 迭代
**目标**: 运行 UT，验证所有用例通过，回归验证原有用例不受影响
**验证命令**:
- 编译: `bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul -j8 --noexec`
- UT: `ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_GLOBAL_LOG_LEVEL=0 BUILD_PATH=$(pwd)/build ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut`

### 任务 8：Kernel 适配
**目标**: 在 `weight_quant_basic_block` 体系中适配 A16W4 pergroup ND 计算逻辑
**目标文件**:
- `grouped_matmul_weight_quant_basic_controller.h` — Controller 适配
- `basic_block_vf_nd.h` — ND 反量化 VF 函数实现
- `weight_quant_vec_compute.h` — 向量计算适配
- `basic_block_config.h` — WqmmConfig 新增

## 阶段三：验证（待启动）

### 任务 9：单元测试
**目标**: 覆盖 tiling 和 kernel 的正确性

### 任务 10：兼容性回归
**目标**: 确保已有 950 测试不受影响

## 当前进度

- [x] 阶段一：规划与设计 ✅
  - [x] 任务 1：算子原型描述文档 ✅
  - [x] 任务 2：走读 GMM A16W8 代码实现 ✅
  - [x] 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现 ✅
  - [x] 任务 4：SRS 算子开发实现方案 ✅
  - [x] 任务 5：设计测试用例（162 个用例，8 种类型） ✅
  - [x] 任务 6：总结归纳设计文档（脑图更新 + STC 文档 + SRS 实现方案） ✅
- [ ] 阶段二：编码实现（当前阶段）
  - [x] 任务 7：Tiling 实现 + UT 验证 ✅
    - [x] 7.1 UT 测试框架搭建 + 用例编写（37 个 CSV 用例） ✅
    - [x] 7.2 Tiling 代码实现（9 个修改点 + 8 个 tiling key） ✅
    - [x] 7.3 UT 验证 + 迭代（37/37 pergroup 通过，13/13 回归通过） ✅
  - [ ] 任务 8：Kernel 适配
- [ ] 阶段三：验证

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| `SetGmAddr` 的 pergroup 地址偏移与 perchannel 差异大 | Controller 需要较大修改 | 走读阶段仔细分析偏移公式，设计阶段明确 pergroup 偏移方案 |
| `basic_block_vf_nd.h` 中 pergroup 反量化逻辑缺失 | 需要新增 pergroup ND 向量计算 | 走读 wqbmmv2 的 `AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`，设计迁移方案 |
| WqmmConfig 缺少 A16W4 pergroup ND 配置 | 需要新增配置项 | 参照 `S8S4_NZKN_G` 新增 ND 版本 |
| int4 字节偏移计算容易出错 | 地址偏移需要减半处理 | 设计阶段明确 int4 偏移计算公式，编码阶段重点验证 |
