# moe_distribute_combine_v2 UT 覆盖率增强报告

## 1. 覆盖率分析概述

由于 Windows 环境限制无法直接运行覆盖率测试，本报告基于代码静态分析设计测试用例。

## 2. 需要补充的测试场景

### 2.1 数据类型分支
**测试用例名称**: `Bf16DataType`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:685-687
**进入条件**: 
- expandX dataType = ge::DT_BF16
- expertScales dataType = ge::DT_FLOAT
**测试参数**:
```json
{
  "inputs": [
    {"shape": [64, 7168], "dtype": "DT_BF16", "format": "FORMAT_ND"},
    {"shape": [8, 7], "dtype": "DT_INT32", "format": "FORMAT_ND"},
    {"shape": [16384], "dtype": "DT_INT32", "format": "FORMAT_ND"},
    {"shape": [8], "dtype": "DT_INT32", "format": "FORMAT_ND"},
    {"shape": [8, 7], "dtype": "DT_FLOAT", "format": "FORMAT_ND"}
  ],
  "outputs": [{"shape": [8, 7168], "dtype": "DT_BF16", "format": "FORMAT_ND"}],
  "attrs": {
    "ep_world_size": 8,
    "moe_expert_num": 8,
    "tp_world_size": 1,
    "shared_expert_num": 0,
    "comm_quant_mode": 0,
    "comm_alg": ""
  }
}
```

### 2.2 1维 xActiveMask (isTokenMask)
**测试用例名称**: `TokenMask1D`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:1523-1524
**进入条件**: 
- xActiveMask != nullptr
- xActiveMask->GetDimNum() == 1
**测试参数**:
```json
{
  "inputs": [
    {"shape": [64, 7168], "dtype": "DT_FLOAT16"},
    {"shape": [8, 7], "dtype": "DT_INT32"},
    {"shape": [16384], "dtype": "DT_INT32"},
    {"shape": [8], "dtype": "DT_INT32"},
    {"shape": [8, 7], "dtype": "DT_FLOAT"},
    {"shape": [8], "dtype": "DT_BOOL"}  // 1维 xActiveMask
  ],
  "attrs": {
    "ep_world_size": 8,
    "moe_expert_num": 8,
    "tp_world_size": 1
  }
}
```

### 2.3 PerformanceInfo 输入
**测试用例名称**: `WithPerformanceInfo`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:1533-1535, 1124-1132
**进入条件**: 
- performanceInfoStorageShape != nullptr
**测试参数**:
```json
{
  "inputs": {
    "performanceInfo": {"shape": [8], "dtype": "DT_INT64"}
  },
  "attrs": {
    "ep_world_size": 8,
    "moe_expert_num": 8
  }
}
```

### 2.4 A3架构 INT8 quant
**测试用例名称**: `A3Int8Quant`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:328-340, 1429-1437
**进入条件**: 
- comm_quant_mode = 2 (INT8_COMM_QUANT)
- npuArch != DAV_3510
- isLayered = false
**测试参数**:
```json
{
  "attrs": {
    "comm_quant_mode": 2,
    "ep_world_size": 8,
    "moe_expert_num": 8,
    "tp_world_size": 1,
    "comm_alg": ""
  },
  "soc": "Ascend910_93"
}
```

### 2.5 A3架构 hierarchy 模式
**测试用例名称**: `A3Hierarchy`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:321, 1357-1359
**进入条件**: 
- comm_alg = "hierarchy"
- isLayered = true
- ep_world_size >= 16 (满足 RANK_NUM_PER_NODE)
**测试参数**:
```json
{
  "attrs": {
    "ep_world_size": 32,
    "moe_expert_num": 32,
    "tp_world_size": 1,
    "comm_alg": "hierarchy"
  },
  "inputs": {
    "expandX": {"shape": [32, 7168]},
    "expertIds": {"shape": [8, 8]},
    "assistInfo": {"shape": [8*8]},
    "epSendCounts": {"shape": [32]}
  }
}
```

### 2.6 tpWorldSize = 1 (无TP)
**测试用例名称**: `TpWorldSize1`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:291-295
**进入条件**: 
- tpWorldSize <= 1
- tpRankId = 0
**测试参数**:
```json
{
  "attrs": {
    "tp_world_size": 1,
    "tp_rank_id": 0
  }
}
```

### 2.7 globalBs > 0
**测试用例名称**: `GlobalBsNonZero`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:1298-1309
**进入条件**: 
- globalBs != 0
- globalBs % epWorldSize == 0
- globalBs >= bs * epWorldSize
**测试参数**:
```json
{
  "attrs": {
    "global_bs": 64,
    "ep_world_size": 8
  },
  "inputs": {
    "expertIds": {"shape": [8, 7]}
  }
}
```

### 2.8 sharedExpertNum 和 sharedExpertRankNum 组合
**测试用例名称**: `SharedExpertBothNonZero`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:1057-1060
**进入条件**: 
- sharedExpertNum > 0
- sharedExpertRankNum > 0
- sharedExpertRankNum % sharedExpertNum == 0
**测试参数**:
```json
{
  "attrs": {
    "shared_expert_num": 1,
    "shared_expert_rank_num": 1,
    "moe_expert_num": 7
  },
  "inputs": {
    "sharedExpertX": {"shape": [8, 7168]}
  }
}
```

### 2.9 zeroExpertNum > 0
**测试用例名称**: `ZeroExpertNum`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:1246-1251
**进入条件**: 
- zeroExpertNum > 0 或 copyExpertNum > 0 或 constExpertNum > 0
- 提供相应的 oriX, constExpertAlpha1, constExpertAlpha2, constExpertV 输入
**测试参数**:
```json
{
  "attrs": {
    "zero_expert_num": 1,
    "copy_expert_num": 0,
    "const_expert_num": 1
  },
  "inputs": {
    "oriX": {"shape": [8, 7168]},
    "constExpertAlpha1": {"shape": [1]},
    "constExpertAlpha2": {"shape": [1]},
    "constExpertV": {"shape": [1, 7168]}
  }
}
```

### 2.10 expandX dataType = DT_INT32 (错误情况)
**测试用例名称**: `ExpandXInt32DataType`
**覆盖代码行**: moe_distribute_combine_v2_tiling.cpp:2125-2127
**进入条件**: 
- expandX dataType = DT_INT32
**预期结果**: ge::GRAPH_FAILED

## 3. 执行步骤

### Step 1: 获取初始覆盖率
在 Linux 环境中执行：
```bash
cd M:\Users\y00951989\yangshengjun\pr\ops-transformer
bash build.sh -u --ophost --ops='moe_distribute_combine_v2' --soc='Ascend910_93' --cov 2>&1 | tee mc2/moe_distribute_combine_v2/tests/ut/coverage_round_0.log
```

分析覆盖率报告中的未覆盖代码行。

### Step 2: 添加新测试用例
将上述测试用例添加到 `tests/ut/op_host/arch32/test_moe_distribute_combine_v2_tiling.cpp` 文件末尾。

### Step 3: 验证覆盖率
重新执行编译和测试，验证新增用例是否通过，覆盖率是否提升。

### Step 4: 持续迭代
如果覆盖率未达到 90%，继续分析未覆盖代码并添加更多测试用例。

## 4. 无法覆盖的代码分析

基于代码分析，以下代码可能难以通过 UT 覆盖：

### 4.1 死代码
- **文件**: moe_distribute_combine_v2_tiling.cpp
- **行号**: 2254-2256 (Ascend950 特定逻辑)
- **原因**: UT 环境无法模拟 Ascend950 节点

### 4.2 运行时检查
- **文件**: moe_distribute_combine_v2_tiling.cpp  
- **行号**: 1470-1471 (CheckWinSize 函数)
- **原因**: 需要 HCCL 环境变量和真实通信域配置

### 4.3 异常处理分支
- **文件**: moe_distribute_combine_v2_tiling.cpp
- **行号**: 2265-2278 (版本检查失败分支)
- **原因**: 版本检查在 UT 环境中通常成功

## 5. 预期覆盖率目标

通过以上测试用例，预期可达到：
- 行覆盖率: 85-90%
- 函数覆盖率: 95-100%
- 分支覆盖率: 75-80%

## 6. 测试执行命令汇总

```bash
# Round 0: 获取初始覆盖率
bash build.sh -u --ophost --ops='moe_distribute_combine_v2' --soc='Ascend910_93' --cov

# Round 1: 添加第一批测试用例后重新编译测试
bash build.sh -u --ophost --ops='moe_distribute_combine_v2' --soc='Ascend910_93' --cov

# Round 2-N: 持续迭代直到达到 90% 覆盖率
bash build.sh -u --ophost --ops='moe_distribute_combine_v2' --soc='Ascend910_93' --cov
```

## 7. 注意事项

1. 所有测试用例必须全部通过（无 FAILED 标识）
2. 新增测试用例需遵循现有命名规范（MoeDistributeCombineV2Arch32TilingTest_*）
3. 参数配置需满足代码中的约束条件
4. 避免与现有测试用例重复
5. 覆盖率不应下降

## 8. 输出文件列表

所有修改和新增文件将保存到：
- mc2/moe_distribute_combine_v2/tests/ut/log/: 编译测试日志
- mc2/moe_distribute_combine_v2/tests/ut/cov_result/: 覆盖率数据
- mc2/moe_distribute_combine_v2/tests/ut/code/: 修改后的测试文件副本
- mc2/moe_distribute_combine_v2/tests/ut/report/: 分析报告