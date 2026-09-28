# aclnnGridSampler2D 测试设计评审报告

## 1. 评审概述

- 算子名称：aclnnGridSampler2D（GridSample 算子 2D aclnn 接口）
- 目标产品：Ascend 950PR/Ascend 950DT
- 接口类型：aclnn（两段式接口 aclnnGridSampler2DGetWorkspaceSize + aclnnGridSampler2D）
- L0 用例数：69（门槛用例，含 ET 空tensor 4 场景、BD 边界 5 场景）
- L1 用例数：619（全量，含 12 条非连续 tensor 用例：transpose/broadcast/slice/asstrided 各 3 条）；精简版 603 条
- L2 用例数：21（异常用例，含 EX-S1~S4 显式异常 + 引擎自动域外/维度异常）
- 评审轮次：2

## 2. 自动校验结果

> 引用 qa_auto_report.md（阶段 C `qa_verify_oracle.py` 退出码 = 0，全部通过）

| 状态 | 数量 |
|------|------|
| PASS | 21 |
| FAIL | 0 |
| SKIP | 10 |

关键通过项：
- Q2/Q3（L0+L1）：标量参数值域覆盖 100%（interpolationMode 3 值、paddingMode 3 值、alignCorners 2 值）；tensor 组数一致性 0/69、0/619 不一致
- Q4_yaml_domain：YAML 域值与 01 文档一致
- Q7/Q7b/Q8：ET 场景完备性与变量存在性
- Q9b/Q10/Q11/Q12：BD 边界与 01 一致、EX 异常有效性与预期错误码、场景 ID 格式
- Q13/Q13b/Q14/Q14b/Q15b：ET/BD/EX 场景用例端到端覆盖（含 BD-S2 H×W 上边界 [1,1,46340,46340] 用例）
- Q17/Q18：locked 一致性、ET 场景约束合法性
- 阶段 A：约束追溯 R1~R9 全部登记、约束满足率 100%、无 MISMATCH、[CONSTRAINT-CONSERVATION] 通过

## 3. 修复历史

| 轮次 | 修复问题 | 回退步骤 | 修复动作 |
|------|---------|---------|---------|
| 1 | L0 约束校验违规：引擎默认边界目标锁定 grid.shape=[1,1,1,1]（自决定 shape 绕过校验），违反 R3 末维=2 | 步骤 3 | solve_grid_shape 的 sources 由 `_n.value` 改为 `input.shape`（N 取自 input.shape[0]，消除自决定 shape 判定，非法锁定被校验拒绝） |
| 2 | 阶段 C 5 项 FAIL：Q7（ET 表格首列含空格不匹配）、Q8（中间因子 `_X.value` 键名未在 YAML 声明）、Q13b（`_h_out/_w_out` 含 `out.value` 子串触发预期输出幻影解析）、Q14b/Q15b（`--scenarios` 未传入导致 ET/BD/EX 目标未加载） | 步骤 2/3 | ① 中间因子改名 _h_out→_out_h、_w_out→_out_w；② 每个 intermediate 显式声明 `_{name}.value: []` 占位键（空域回退 value_range 采样，运行时行为不变）；③ ET/BD/EX 表格改为无空格单元格且描述列去除 `=` 模式；④ 生成命令显式传入 `--scenarios` |

## 4. 总结

- 最终状态：全部通过（阶段 A 退出码 0、阶段 B 三层自检通过、阶段 C 退出码 0）
- 用例规模：L0=69 ≤ 200、L1 全量=619 ≥ 500、L2=21 ≤ 200，符合用例级别标准
- 覆盖要点：dtype 4 种（float16/float32/float64/bfloat16，bicubic 收窄 float32/float16 由 C5-S2 覆盖）、interpolationMode {0,1,2}、paddingMode {0,1,2}、alignCorners {true,false} 全枚举；空 tensor 4 场景（N/C/H_out/W_out 轴）；边界 5 场景（空间维下界 1、H×W 上界 46340×46340、坐标 ±1）；grid 值域场景 A（[-1,1] 正常采样）/B（越界按 paddingMode 处理）经 YAML 分段覆盖；非连续 4 类×3 条
- 已知噪声：L0 日志 "[WARN] L0 未覆盖场景: C3-S*/C5-S*" 为校验器仅支持 ET/BD 锁定值匹配的结构性噪声，C 子场景经 @solves tag() 标记实际全覆盖（见 03 场景覆盖验证表说明）
- 待确认项（继承自 01）：文档内部差异——bicubic 时 BFLOAT16 支持性（错误码表 vs A2/A3 产品段），950 上以第一段接口实际校验为准
