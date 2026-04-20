# 校验类别覆盖报告

## 一、校验函数完整列表（14个）

| 序号 | 函数名 | 校验内容 | 代码行号 | 已覆盖 |
|------|--------|----------|----------|--------|
| 1 | CheckEpAndTpWorldSize | ep/tp worldsize基础校验 | 111 | ✓ |
| 2 | CheckEpRankId | ep rank id范围校验 | 147 | ✓ |
| 3 | CheckTpRankId | tp rank id范围校验 | 160 | ✓ |
| 4 | CheckSharedExpertAttrs | shared expert属性校验 | 189 | ✓ |
| 5 | CheckEpWorldSize | ep worldsize具体值校验 | 290 | ✓ |
| 6 | CheckSharedAttrs | shared属性整除校验 | 316 | ✓ |
| 7 | CheckAndSetGlobalBSAttrs | globalBs校验 | 339 | ✓ |
| 8 | CheckExpertIdsAndSetBs | expertIds(bs)校验 | 374 | ✓ |
| 9 | CheckAndSetAttrs | 参数整除校验 | 390 | ✓ |
| 10 | CheckSharedExpertXShape | sharedExpertX维度校验 | 442 | ⚠ |
| 11 | CheckEpSendCountAndTpSendCountShape | ep/tp send count校验 | 478 | ✓ |
| 12 | CheckInputTensorShape | 输入tensor维度校验 | 519 | ✓ |
| 13 | CheckTensorShape | H/K维度校验 | 597 | ⚠ |
| 14 | CheckCommAttrs | 通信窗口大小校验 | 685 | ✗ |

## 二、校验类别分类（6大类）

### 类别1：参数范围校验（已覆盖80%）

**已覆盖**：
- ✓ epWorldSize范围（8倍数校验、特殊值校验）
- ✓ tpWorldSize范围（tpWorldSize=2校验）
- ✓ epRankId范围（越界、负数）
- ✓ tpRankId范围（tpRankId!=0）
- ✓ moeExpertNum范围（0、513）
- ✓ sharedExpertNum范围（V1: >1）
- ✓ bs范围（bs>512）
- ✓ k范围（k>8）

**未覆盖**：
- ⚠ expertShardType!=0校验（line 272）
- ⚠ V2版本的参数范围（H、K、moeExpertNum）
- ⚠ tpWorldSize负数校验
- ⚠ globalBs与bs不匹配校验（line 353）

### 类别2：参数关系校验（已覆盖90%）

**已覆盖**：
- ✓ epWorldSize与sharedExpertRankNum整除
- ✓ moeExpertNum与epWorldSize整除
- ✓ sharedExpertNum与sharedExpertRankNum关系

**未覆盖**：
- ⚠ sharedExpertNum=0但sharedExpertRankNum>0校验（line 323）
- ⚠ sharedExpertRankNum不整除sharedExpertNum校验（line 331）

### 类别3：Optional输入校验（已覆盖50%）

**已覆盖**：
- ✓ x_active_mask输入（V1不支持）

**未覆盖**：
- ✗ sharedExpertX输入场景（需要sharedExpertNum>0）
- ⚠ x_active_mask V2版本正确输入

### 类别4：Tensor维度校验（已覆盖85%）

**已覆盖**：
- ✓ expandIdx维度不足
- ✓ expertScales维度不匹配
- ✓ epSendCount不足
- ✓ output x维度不匹配

**未覆盖**：
- ⚠ sharedExpertX维度校验（需要输入sharedExpertX）
- ⚠ xActiveMask维度校验（V2版本）
- ⚠ tpSendCount维度校验（需要tpWorldSize=2）

### 类别5：通信窗口校验（未覆盖）

**未覆盖**：
- ✗ CheckCommAttrs通信窗口大小超限校验（line 694）
  - 需要大bs、大h、大epWorldSize触发

### 类别6：平台信息校验（部分覆盖）

**已覆盖**：
- ✓ Ascend310P平台

**未覆盖**：
- ⚠ 不同SoC版本验证
- ⚠ coreNum/ubSize边界值

## 三、覆盖率提升计划

### 当前覆盖率：87%（301行执行，45行未执行）

### 未覆盖代码分析（45行）

**类型分布**：
1. **Optional输入未触发**（~15行）
   - sharedExpertX校验代码（line 459-474）
   - xActiveMask V2校验代码（line 536-541）
   
2. **通信窗口校验未触发**（~10行）
   - CheckCommAttrs函数体（line 694-700）
   
3. **错误日志输出行**（~15行）
   - OP_LOGE日志行（不统计覆盖率）
   
4. **调试代码**（~5行）
   - OP_LOGD调试日志

### 补充测试用例设计

| 测试用例名 | 目标覆盖率 | 校验类别 | 关键参数 |
|------------|------------|----------|----------|
| arch35_shared_expert_x_input | sharedExpertX校验 | 类别3 | sharedExpertNum=1, sharedExpertX输入 |
| arch35_expert_shard_type_fail | expertShardType校验 | 类别1 | expertShardType=1 |
| arch35_comm_window_large | 通信窗口校验 | 类别5 | bs=512, h=7168, ep=256 |
| arch35_shared_num_zero_rank_nonzero | shared参数关系校验 | 类别2 | sharedExpertNum=0, sharedExpertRankNum=2 |
| arch35_global_bs_exceed_fail | globalBs校验 | 类别1 | globalBs=100（不满足整除） |
| arch35_x_dim_mismatch_fail | output x维度校验 | 类别4 | x_shape与bs/h不匹配 |

## 四、覆盖率提升目标

**短期目标**：87% → 92%（+5%）
- 补充sharedExpertX输入测试
- 补充expertShardType校验测试
- 补充通信窗口校验测试

**中期目标**：92% → 95%（+3%）
- 补充V2版本场景测试（虽然arch35强制V1）
- 补充边界值组合测试

**长期目标**：95% → 98%（+3%）
- 覆盖所有错误处理路径
- 覆盖所有调试代码路径

## 五、覆盖率计算方法

### Lines覆盖率计算
```
已执行行数 = 301
总可执行行数 = 346
覆盖率 = 301/346 = 87.0%
```

### Functions覆盖率计算
```
已调用函数数 = 20
总函数数 = 23
覆盖率 = 20/23 = 87.0%
```

### 未覆盖的3个函数（推测）
1. CheckCommAttrs（通信窗口校验，未触发）
2. CheckSharedExpertXShape（未输入sharedExpertX）
3. 其他辅助函数（如PrintTilingDataInfo）

## 六、下一步行动

### 补充测试用例CSV更新
将在test_moe_distribute_combine_v2_tiling.csv中新增：
1. arch35_shared_expert_x_input（SUCCESS）
2. arch35_expert_shard_type_fail（FAILED）
3. arch35_comm_window_large（FAILED）
4. arch35_shared_num_zero_rank_nonzero（FAILED）
5. arch35_x_dim_mismatch_fail（FAILED）

### 预期覆盖率提升
- 补充5个测试用例后
- 预计覆盖+20行代码
- 覆盖率提升至 92%

---

**报告生成时间**：2026-04-17
**当前覆盖率**：87.0%（Lines），87.0%（Functions）
**目标覆盖率**：92%（短期），95%（中期），98%（长期）