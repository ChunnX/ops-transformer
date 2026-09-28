# aclnnGridSampler2D 测试设计总结

## 设计过程

1. **参数功能说明内容清洗**（01_parameter_description.md）：从 `docs/aclnnGridSampler2D.md` 提取函数原型（GetWorkspaceSize 两段式）、参数说明表与错误码表，清洗为 6 参数 × 11 列标准表格；提取约束关系 R1~R9（dtype 等值 ×2、shape 公式 ×2、枚举值域、bicubic 条件 dtype、H×W 上界、空 tensor 边界、grid 归一化坐标功能场景），并完成约束提取反向校验（9 项 EXACT_MATCH/INFERENCE，无 MISMATCH）。
2. **测试因子生成**（02_test_factors.yaml）：operator_name=aclnnGridSampler2D；3 个 Tensor（input/grid/out）+ 3 个属性参数（interpolationMode/paddingMode/alignCorners，.value 枚举）；grid 定义归一化坐标值域分段（场景A [-1,1] / 场景B 越界 ±3 / 边界 ±1 退化段）；中间因子 _n/_c/_h_in/_w_in/_out_h/_out_w（分段值域，空 tensor 维含 0 段）驱动 shape 公式。
3. **约束模块编写**（03 + 04）：场景枚举（C3 shape 公式 4 子场景、C5 bicubic 条件 2 子场景 + ET×4 / BD×5 / EX×4 增强场景）；追溯表登记 R1~R9 全部策略；@solves 实现 dtype 等值（模式1）、shape 公式（模式3）、bicubic 条件过滤（模式8B，含 tag 标记）；validate_constraints 独立转录 9 条校验。
4. **约束校验与用例生成**：阶段 A --validate 通过（约束满足率 100%）；阶段 B 引擎拓扑求解生成 L0/L1/L2（三层自检通过，含 L0 值域覆盖自动补全）；阶段 C qa_verify_oracle 退出码 0（21 PASS / 0 FAIL）；阶段 D 评审报告归档。

## 结果

| 级别 | 文件 | 用例数 |
|------|------|--------|
| L0 | testcases/aclnnGridSampler2D_l0_functional.csv | 69 |
| L1 全量 | testcases/aclnnGridSampler2D_l1_all_functional.csv | 619（含非连续 12） |
| L1 精简 | testcases/aclnnGridSampler2D_l1_functional.csv | 603 |
| L2 | testcases/aclnnGridSampler2D_l2_exception.csv | 21 |

关键覆盖：dtype 4 种（bicubic 收窄 f32/fp16）、三属性全枚举（3×3×2）、空 tensor 4 场景、边界 5 场景（含 H×W 上界 46340×46340）、grid 值域场景 A/B、非连续 4 类、异常 4 显式 + 自动域外/维度异常。

详见 `design/06_review_report.md` 与 `design/qa_auto_report.md`。
