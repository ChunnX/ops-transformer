# aclnnFusedFloydAttentionGrad 测试设计评审报告

## 1. 评审概述

- 算子名称：aclnnFusedFloydAttentionGrad
- 接口类型：aclnn（csv_mode=aclnn，13 列 ACLNN 格式）
- 目标产品：Ascend 950PR/Ascend 950DT（用户指定 A5 新增支持需求；接口文档当前标注不支持，属新增支持验证场景）
- L0 用例数：62（≤200 ✓，BD-S1~S14 边界场景全覆盖）
- L1 用例数：651（≥500 ✓；641 条功能用例 + 10 条非连续用例，10 个输入 tensor 各 1 条 broadcast 场景；含 410 基础 + 211 正态分布 + 20 均匀分布）
- L2 用例数：107（≤200 ✓，EX-S1~S16 算子专属异常全部在 CSV shape 可见 + 引擎自动异常：dtype/format 域外、维度边界、空 tensor、dtype 不匹配、输出 shape 不匹配等）
- 评审轮次：4A 修复 2 轮通过；4B 修复 5 轮通过；4C 首轮通过（最终配置：PYTHONHASHSEED=0 + seed=5）

## 2. 自动校验结果

> 引用 qa_auto_report.md，禁止改写。

- qa_verify_oracle.py 退出码：**0（全部通过）**
- 结果统计：PASS 16 / FAIL 0 / SKIP 15
- 通过项：Q2_trace_coverage（L0/L1）、Q3_output_indexes（L0/L1）、Q4_yaml_domain、Q7_param_coverage（L0/L1，全部必选因子已覆盖）、Q9b_bd_boundary_yaml、Q10_ex_validity、Q11_ex_expected_error、Q12_scenario_id_format、Q14_bd_case_coverage（14 条）、Q17_locked_consistency、Q13b_et_e2e、Q14b_bd_e2e、Q15b_ex_e2e
- SKIP 项说明：Q1_product_support（信息性检查）、Q5/Q6（无适用检查项）、Q7/Q7b/Q8/Q13/Q18（ET 场景——本算子所有输入 tensor 各轴下界为正，空 tensor 不支持，support_empty_tensor: false，无 ET 场景为设计预期）、Q15/Q16（EX 覆盖由 CSV 逐条核验 16/16 出现且违反值可见）、Q20（输出 shape 由 @solved 推导，公式忠实性由 4A 保障）

约束校验（阶段 A，generate_factor_values.py --validate，exit=0）：
- 约束满足率 100%（16 个 shape 因子 + 11 个 dtype 因子 + exist/value_range 因子全部 1000/1000）
- 可达率 100%（solved 因子域值全部可达：dtype 2/2、mask value_range 3/3、scaleValue value_range 27/27）
- 追溯表完整性：R1~R20 全部登记；[CONSTRAINT-CONSERVATION] 通过；反向校验表无 MISMATCH
- 无 SOLVE-SKIP 告警（预算链收窄式采样后 0 丢弃）；_m.value/_k.value 满足度 68.5%/79.8% 的 [WARN] 为预算收窄的预期表现（YAML 声明文档完整值域 + @solves 按 2G 预算收窄，符合值域保真原则与场景收窄模式）

## 3. 修复历史

| 轮次 | 修复问题 | 回退步骤 | 修复动作 |
|------|---------|---------|---------|
| 4A-1 | YAML 科学计数法 `3.4028235e38` 被 PyYAML 解析为字符串，引擎 `_expand_wide_range` int() 崩溃 | 步骤 2 | 指数补正号（e38→e+38，e308→e+308） |
| 4A-2 | scaleValue.value_range_float64 含 ['inf','inf']/['nan','nan'] 段触发引擎 ValueError（inf/inf=nan 绕过比例检查） | 步骤 2 | 移除 inf/nan 字符串段（scaleValue 为缩放系数属性，01/spec 值域定义即不含 inf/nan） |
| 4B-1 | 预算链 raise 式采样导致 SOLVE-SKIP 复合跳过（_b 17%→_k 57%），L2 单次回退求解成功率仅 ~14%，93/107 条 L2 用例 shape 为 nan | 步骤 3 | 预算链改收窄式采样（段超预算回落文档下界）+ _m 补 key2 的 K 下界预算预留（数学保证链上每级 cap ≥ 文档下界，单次求解成功率 ~100%） |
| 4B-2 | EX-S1~S11 原用中间因子锁（`_*_value_range`），L2 管道过滤 `_` 前缀锁导致违反值不进入 tensor shape（CSV 不可见） | 步骤 3 | EX 场景改为 tensor 级 shape 锁定（16 个 tensor shape 一致锁定，违反值直接体现在 tensor_view_shapes 列） |
| 4B-3 | L2 回退填充时 attenMaskOptional.exist 重新采样与 case 已记录值不一致（exist=True ∧ shape=None）→ 转换层 has_required_none_shape 丢行 22 条 → 行数错位 → meta 按位置回填产生 NaN 空行 + remark/名字错位污染 | 步骤 3 | solve_attenMask_shape 始终返回模板 shape（不按 exist 返回 NOT_APPLICABLE；CSV 渲染由 exist 列决定 absent 输出 None，语义不变），消除丢行与错位 |
| 4B-4 | NC 生成器对输出 tensor 注入非连续用例，与约束说明"该接口不支持非连续输出"冲突 | 步骤 2 | 5 个输出 tensor 设置 support_non_contiguous: false（来源：01 参数表"非连续"列"不支持"） |
| 4B-5 | non_contiguous.py 使用 hash(operator_name)（Python 哈希随机化）导致 NC 用例生成跨进程不确定 | 执行环境 | 固定 PYTHONHASHSEED=0 + seed=5，确定性获得 10 条输入 tensor 非连续用例（broadcast 场景；transpose/slice/asstrided 因引擎 NC 数据量预算（约 137M 元素/tensor）跳过，属引擎数据保护行为） |

## 4. 总结

- 最终状态：**全部通过**（4A exit=0 无阻断告警 / 4B L0+L1+L2 生成成功且三层自检通过 / 4C exit=0）
- 产物清单（tests/st/testcases/）：
  - `aclnnFusedFloydAttentionGrad_l0_functional.csv`（62 条）
  - `aclnnFusedFloydAttentionGrad_l1_all_functional.csv`（651 条）
  - `aclnnFusedFloydAttentionGrad_l1_functional.csv`（651 条精简集）
  - `aclnnFusedFloydAttentionGrad_l2_exception.csv`（107 条）
  - 对应 `*_factors.json` sidecar 文件
- 设计要点：
  1. 预算分配链（_h/_d 锚点 → _b → _n → _m → _k 级联 2G 收窄 + key2 K 下界预算预留），YAML 声明文档完整值域（值域保真），@solves 按预算收窄（场景收窄模式）
  2. N/M/K 对齐约束（%16/%128）通过 @solves 约束推导实现；对齐/越界异常通过"区间无对齐值回退原值"与 tensor 级 shape 锁定构造，违反值在 CSV 可见
  3. BD 场景锁定 `_*_value_range` 锚点段 + 相关中间变量同步锁定（上边界规范）；EX 场景锁定完整一致的 tensor shape 组
  4. 复现命令需固定环境：`PYTHONHASHSEED=0`、`--seed 5`（否则 NC 用例数因 hash 随机化不确定）
- 残留问题：
  1. [待确认] dy.dtype 与 query 一致（R17）为语义推断（文档未显式声明），已在 01 标注
  2. [待确认] 参数表"非连续Tensor"列输出行标 √ 与约束说明"不支持非连续输出"矛盾，已按约束说明处理（输出排除 NC 注入）
  3. N/M/K 文档上界（1M）的极端组合超出单 tensor 2G 元素限制，不可构造可执行用例（见 03 说明），上界方向由大值段随机采样覆盖
  4. NC 用例仅覆盖 broadcast 场景（10 条/每输入 tensor 1 条）：transpose/slice/asstrided 场景受引擎 NC 数据量预算限制（L1 模板行 shape 普遍大于 ~137M 元素预算）被跳过；如需全场景 NC 覆盖，可在执行阶段通过 non-contiguous-tensor 技能补充
  5. 接口文档"产品支持情况"当前标注 950PR/950DT 不支持，本设计按用户确认的新增支持需求执行，执行阶段需在 950PR/950DT 环境部署验证
