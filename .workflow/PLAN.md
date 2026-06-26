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
**详细任务说明**: [reference/next_steps/8_kernel_impl.md](reference/next_steps/8_kernel_impl.md)

#### 8.1 基础配置层（WqmmConfig + Buffer + Kernel 入口 + config950）
**目标**: 新增 4 个 WqmmConfig 实例 + Buffer 配置 + 8 个 Kernel 入口分支 + config950 新增 4 个 dtype 组合
**修改文件**:
- `basic_block_config.h` — 新增 A16W4_NDKN_PG/A16W4_NDKN_PG_OFF/A16W4_NDNK_PG/A16W4_NDNK_PG_OFF
- `grouped_matmul_apt.cpp` — 新增 8 个 if constexpr 分支（2 transB × 2 offset × 2 bias）
- `grouped_matmul_def.cpp` — config950 新增 4 个 antiquant_offset=null 的 dtype 组合
**验证**: 编译通过

#### 8.2 Controller 层适配
**目标**: SetGmAddr pergroup scale/offset 偏移 + bias 偏移独立
**修改文件**:
- `grouped_matmul_weight_quant_basic_controller.h` — pergroup 偏移公式 + biasBaseOffset
**验证**: 编译通过，perchannel 路径不受影响

#### 8.3 VecCompute 层适配
**目标**: pergroup scale/offset UB 地址计算 + offset 搬运 + pergroup 分发
**修改文件**:
- `weight_quant_vec_compute.h` — CalLocalAddrForVf + CopyAntiQuantParamsGmToUb + AntiQuantProcessNd
**2026-06-26 修复**: KN pergroup scale/offset 搬运和 VF 寻址统一使用当前 N tile 对齐行步长，修复 gs32 泛化精度
**验证**: 编译通过

#### 8.4 VF 层 — KN 路径反量化（bTrans=false）
**目标**: 实现 pergroup KN 统一 VF 函数（所有 groupSize 共用）
**修改文件**:
- `basic_block_vf_nd.h` — NdKnLoadScaleOffset PER_GROUP 分支 + AntiQuantInt4PerGroupNdKn
**参考**: ops-nn `AntiquantPerGroupKN`（vf.h L347-443）
**验证**: 编译通过

#### 8.5 VF 层 — NK 路径反量化（bTrans=true）
**目标**: 实现 pergroup NK 按 groupSize 分派 4 个专用 VF 函数
**修改文件**:
- `basic_block_vf_nd.h` — NdNkLoadScaleOffset PER_GROUP 分支 + AntiQuantInt4PerGroupNdNk（含 gs=32/64/128/256）
**参考**: ops-nn `AntiquantComputeW4NKPerGroup`（reg_base_common.h L1185-1409）
**核心差异**: GMM scale/offset 固定 `[G,N]` 布局，地址公式为 `gLocal * nTileAlign + nLocal`
**2026-06-24 修复**: NK weight N 行步长固定为 `vecConfig.ubMte2InnerSize >> 1`，输出 stride 固定 `WEIGHT_F16_UB_NZ_STRIDE`；gs=32 scale/offset 使用 Select 拼接 32 元素分组
**2026-06-24 gs128 补充修复**: `AntiQuantInt4PerGroupNdNkGs128` 每个 group 仅处理一个 128K 向量，移除误按 gs256 处理的第二段 128K 加载/写回；`K=128` 时 group 循环只执行一次
**2026-06-24 gs32 重实现**: `AntiQuantInt4PerGroupNdNkGs32` 按 256K inner 分块处理 8 个 32K group；scale/offset 使用逐 group `DIST_BRC_B16` + `Select` 拼接，并对 `K<256` 尾块避免读取不存在的第二个 128K 向量
**2026-06-24 gs128 调整**: `AntiQuantInt4PerGroupNdNkGs128` 因 `__VEC_SCOPE__` 内不使用运行期 `if`，改为 `groupNum -> N` 双层 `for`；weight 按转置 `[N,K]` UB 行步长读取，scale/offset 按非转置 `[K//GS,N]` UB 布局寻址，输出 stride 保持 NK NZ 布局
**2026-06-24 gs64 修复**: `AntiQuantInt4PerGroupNdNkGs64` 保持 wqbmmv2 的 `inner(K/128) -> outer(N)` 结构，每个 inner 只处理一个 128-lane weight 寄存器；scale/offset 按 GMM `[G,N]` 读取相邻两个 group 后用 `Select(maskH)` 拼接
**2026-06-24 gs32 修复**: `AntiQuantInt4PerGroupNdNkGs32` 改为每个 inner 处理一个 128-lane weight 寄存器，按 GMM `[G,N]` 读取 4 个 group 后用 `Select(mask32/mask96/maskH)` 构造 `[A|B|C|D]` 四段 32-lane scale/offset
**验证**: 本轮不编译，仅执行 `git diff --check` 静态检查

#### 8.6 静态/编译验证
**目标**: 完整验证 kernel 修改
**编译命令**: `bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`
**验证**:
- 历史 kernel 编译曾通过（exit code 0）
- 本轮 gs128 `groupNum -> N`、gs64 与 gs32 修改未编译（用户环境不运行）
- 2026-06-26 A16W4 ND pergroup 显式切到 `{bufferNum=2, ubMte2InnerSize=256}` tiling key，未编译
- 2026-06-26 `AntiQuantInt4PerGroupNdKn` 参考 wqbmmv2 `AntiquantComputeKNPerGroup` 重写为无运行时 `if` 的 `AddrReg` 数据流，未编译
- 静态检查：`git diff --check` 通过

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
  - [x] 任务 8：Kernel 适配 ✅
    - [x] 8.1 基础配置层（4 WqmmConfig + 8 apt 分支 + config950 4 条目） ✅
    - [x] 8.2 Controller 层适配（pergroup 偏移 + biasBaseOffset_） ✅
    - [x] 8.3 VecCompute 层适配（CalLocalAddrForVf + CopyAntiQuantParamsGmToUb + AntiQuantProcessNd） ✅
    - [x] 8.4 VF — KN 路径反量化（AntiQuantInt4PerGroupNdKn 参考 wqbmmv2 重写，无运行时 if） ✅
    - [x] 8.5 VF — NK 路径反量化（AntiQuantInt4PerGroupNdNk + Gs32/Gs64/Gs128/Gs256） ✅
    - [x] 8.6 静态/编译验证（本轮 gs128/gs64/gs32 修改未编译，仅静态检查） ✅
- [ ] 阶段三：验证

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| `SetGmAddr` 的 pergroup 地址偏移与 perchannel 差异大 | Controller 需要较大修改 | 走读阶段仔细分析偏移公式，设计阶段明确 pergroup 偏移方案 |
| `basic_block_vf_nd.h` 中 pergroup 反量化逻辑缺失 | 需要新增 pergroup ND 向量计算 | 走读 wqbmmv2 的 `AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`，设计迁移方案 |
| WqmmConfig 缺少 A16W4 pergroup ND 配置 | 需要新增配置项 | 参照 `S8S4_NZKN_G` 新增 ND 版本 |
| int4 字节偏移计算容易出错 | 地址偏移需要减半处理 | 设计阶段明确 int4 偏移计算公式，编码阶段重点验证 |
