# aclnnGridSampler3D 测试设计总结

## 设计过程

1. **参数功能说明内容清洗**（01_parameter_description.md）：从 `docs/aclnnGridSampler3D.md` 提取函数原型、参数说明表与错误码表，清洗为 6 参数 × 11 列标准表格；提取约束关系 R1~R10（dtype/format 等值 ×3、5 维 shape 公式 ×2、枚举值域（3D 无 bicubic）、D×H×W 上界、空 tensor 边界（空间三维 ≥1 为推断，标注待确认）、NDHWC 条件格式（f32+mode0+C%4==0）、grid 归一化坐标功能场景），并完成约束提取反向校验（10 项，无 MISMATCH）。
2. **测试因子生成**（02_test_factors.yaml）：operator_name=aclnnGridSampler3D；input.format 三格式 [NCDHW, NDHWC, ND] 完整保留（格式交叉覆盖）；3 个属性参数 .value 枚举；grid 归一化坐标值域分段；中间因子 _n/_c/_d_in/_h_in/_w_in/_out_d/_out_h/_out_w 驱动 5 维 shape 公式。
3. **约束模块编写**（03 + 04）：场景枚举（C4 shape 公式 4 子场景、C8 NDHWC 条件 2 子场景 + ET×5 / BD×6 / EX×4 增强场景）；追溯表登记 R1~R10 全部策略；@solves 实现 format→dimensions 查找表（模式13）、dtype/format 等值（模式1）、NDHWC 三重条件过滤（模式8B：dtype 固定 f32、mode 固定 0、C 轴向上对齐 4 的倍数，含 tag 标记）；validate_constraints 独立转录 10 条校验。
4. **约束校验与用例生成**：阶段 A --validate 通过（约束满足率 100%）；阶段 B 一次通过（三层自检 + L0 值域覆盖自动补全）；阶段 C qa_verify_oracle 一次通过（21 PASS / 0 FAIL）；阶段 D 评审报告归档。

## 结果

| 级别 | 文件 | 用例数 |
|------|------|--------|
| L0 | testcases/aclnnGridSampler3D_l0_functional.csv | 72 |
| L1 全量 | testcases/aclnnGridSampler3D_l1_all_functional.csv | 628（含非连续 12） |
| L1 精简 | testcases/aclnnGridSampler3D_l1_functional.csv | 609 |
| L2 | testcases/aclnnGridSampler3D_l2_exception.csv | 21 |

关键覆盖：dtype 4 种（NDHWC 收窄 f32）、format 三格式交叉（NDHWC 条件组合正向构造）、interpolationMode {0,1}（无 bicubic）、paddingMode {0,1,2}、alignCorners {true,false}、空 tensor 5 场景、边界 6 场景（含 D×H×W 上界 1290³、NDHWC C=4 对齐下界）、grid 值域场景 A/B、非连续 4 类、异常 4 显式 + 自动域外/维度异常。

详见 `design/06_review_report.md` 与 `design/qa_auto_report.md`。
