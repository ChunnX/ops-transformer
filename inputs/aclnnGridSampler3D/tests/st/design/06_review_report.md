# aclnnGridSampler3D 测试设计评审报告

## 1. 评审概述

- 算子名称：aclnnGridSampler3D（GridSample 算子 3D aclnn 接口）
- 目标产品：Ascend 950PR/Ascend 950DT
- 接口类型：aclnn（两段式接口 aclnnGridSampler3DGetWorkspaceSize + aclnnGridSampler3D）
- L0 用例数：72（门槛用例，含 ET 空tensor 5 场景、BD 边界 6 场景）
- L1 用例数：628（全量，含 12 条非连续 tensor 用例：transpose/broadcast/slice/asstrided 各 3 条）；精简版 609 条
- L2 用例数：21（异常用例，含 EX-S1~S4 显式异常 + 引擎自动域外/维度异常）
- 评审轮次：1

## 2. 自动校验结果

> 引用 qa_auto_report.md（阶段 C `qa_verify_oracle.py` 退出码 = 0，全部通过）

| 状态 | 数量 |
|------|------|
| PASS | 21 |
| FAIL | 0 |
| SKIP | 10 |

关键通过项：
- Q2/Q3（L0+L1）：标量参数值域覆盖 100%（interpolationMode 2 值、paddingMode 3 值、alignCorners 2 值）；tensor 组数一致性 0/72、0/628 不一致
- Q4_yaml_domain：YAML 域值与 01 文档一致（含 input.format 三格式 NCDHW/NDHWC/ND 完整保留）
- Q7/Q7b/Q8：ET 场景完备性与变量存在性
- Q9b/Q10/Q11/Q12：BD 边界与 01 一致、EX 异常有效性与预期错误码、场景 ID 格式
- Q13/Q13b/Q14/Q14b/Q15b：ET/BD/EX 场景用例端到端覆盖（含 BD-S2 D×H×W 上边界 1290×1290×1290 用例、BD-S6 NDHWC C 对齐边界用例）
- Q17/Q18：locked 一致性、ET 场景约束合法性
- 阶段 A：约束追溯 R1~R10 全部登记、约束满足率 100%、无 MISMATCH、[CONSTRAINT-CONSERVATION] 通过

## 3. 修复历史

| 轮次 | 修复问题 | 回退步骤 | 修复动作 |
|------|---------|---------|---------|
| 1 | 无（复用 2D 接口本轮已沉淀的工程规范：中间因子命名规避 `out.value` 子串、intermediate 显式声明 `_{name}.value` 占位键、ET/BD/EX 表格无空格单元格、生成命令显式传入 `--scenarios`） | — | 一次通过 |

## 4. 总结

- 最终状态：全部通过（阶段 A 退出码 0、阶段 B 三层自检通过、阶段 C 退出码 0）
- 用例规模：L0=72 ≤ 200、L1 全量=628 ≥ 500、L2=21 ≤ 200，符合用例级别标准
- 覆盖要点：dtype 4 种（NDHWC 时收窄 float32 由 C8-S2 覆盖）、format 三种交叉（NCDHW/NDHWC/ND，NDHWC 条件组合 f32+mode0+C%4==0 正向构造）、interpolationMode {0,1}（3D 无 bicubic）、paddingMode {0,1,2}、alignCorners {true,false} 全枚举；空 tensor 5 场景（N/C/D_out/H_out/W_out 轴）；边界 6 场景（空间维下界 1、D×H×W 上界 1290³、输出下界、坐标 ±1、NDHWC C=4 对齐下界）；grid 值域场景 A/B 经 YAML 分段覆盖；非连续 4 类×3 条
- 已知噪声：L0 日志 "[WARN] L0 未覆盖场景: C4-S*/C8-S*" 为校验器仅支持 ET/BD 锁定值匹配的结构性噪声，C 子场景经 @solves tag() 标记实际全覆盖（见 03 场景覆盖验证表说明）
- 待确认项（继承自 01）：R8 input 空间三维 ≥1 为推断约束（aclnn 文档"支持空Tensor"与 README"所有维度都必须大于0"存在文档间差异），本设计按保守取值编码
