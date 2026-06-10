# Engram算子测试用例设计总结（更新版）

## 一、设计变更说明

### 1.1 五项核心调整

| 调整项 | 原设计 | 新设计 | 说明 |
|--------|--------|--------|------|
| **移除num_qps** | 包含QP数测试 | 完全移除 | 不再测试QP数量参数 |
| **移除HYBRID_MODE** | 包含hybrid模式测试 | 完全移除 | 不再测试hybrid模式切换 |
| **简化world_size** | 拆分成NUM_PEERS×RANKS_PER_PEER | 直接使用2,4,8,16 | 不再拆分拓扑参数 |
| **增加output_dtype** | 仅支持bf16 | 支持bf16/fp16/fp32 | fetch输出支持三种dtype |
| **新增sheet** | 仅level0/level1 | 增加performance/exception | 性能和失败用例 |

### 1.2 参考依据

- **DeepEP test_engram.py**：真实使用场景
- **xlsx-format-spec.md**：MC2测试框架规范
- **用户需求**：简化参数、增加dtype、新增sheet

## 二、参数定义（简化版）

### 2.1 核心参数

| 参数名 | 取值范围 | 说明 | 来源 |
|--------|----------|------|------|
| **hidden** | [64, 128, 256, 512, 1024, 2048, 4096] | 隐藏层维度 | DeepEP典型值 |
| **num_entries** | [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288] | 每rank表行数 | DeepEP典型值（2^12~2^19） |
| **num_tokens** | [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] | Fetch token数 | DeepEP典型值 |
| **world_size** | [2, 4, 8, 16] | 分布式进程数 | 常见配置（不拆分） |
| **output_dtype** | ['bf16', 'fp16', 'fp32'] | Fetch输出类型 | 新增支持fp16/fp32 |
| **phase** | ['write', 'fetch', 'write_fetch'] | 测试阶段 | write/fetch分离测试 |

### 2.2 移除参数

| 参数名 | 原设计 | 移除原因 |
|--------|--------|---------|
| **num_qps** | [1, 4, 8, 16, 32] | 不影响Python API层面 |
| **HYBRID_MODE** | [0, 1] | 简化world_size，不再拆分 |
| **NUM_PEERS_TIER** | [1, 2, 4, 8, 16] | 直接用world_size替代 |
| **RANKS_PER_PEER_TIER** | [1, 2, 4, 8] | 直接用world_size替代 |

## 三、Sheet结构与用例统计

### 3.1 Sheet结构

| Sheet名 | 用例数 | 用途 | 说明 |
|---------|--------|------|------|
| **level0** | 27条 | 基础功能覆盖 | 单因子独立覆盖（≤200 ✓） |
| **level1** | 500条 | 组合覆盖 | 两两因子组合（500-700 ✓） |
| **performance** | 30条 | 性能测试 | 仅关注fetch接口性能 |
| **exception** | 10条 | 失败用例 | 异常场景和边界失败 |

### 3.2 Level0覆盖详情

**单因子独立覆盖**：

1. HIDDEN_TIER覆盖（7档） - 7条
2. ENTRIES_TIER覆盖（5档） - 5条
3. NUM_TOKENS_TIER覆盖（5档） - 5条
4. WORLD_SIZE_TIER覆盖（4档） - 4条
5. OUTPUT_DTYPE_TIER覆盖（3档） - 3条
6. PHASE_TIER覆盖（3档） - 3条

**总计**：27条基础覆盖

### 3.3 Level1覆盖详情

**两两组合覆盖**：

1. HIDDEN × NUM_TOKENS（5×5 = 25条）
2. HIDDEN × WORLD_SIZE（4×4 = 16条）
3. OUTPUT_DTYPE × HIDDEN（3×5 = 15条）
4. PHASE × HIDDEN（3×4 = 12条）
5. WORLD_SIZE × OUTPUT_DTYPE（4×3 = 12条）
6. 随机组合补充（至500条）

**总计**：500条组合覆盖

### 3.4 Performance覆盖详情

**仅关注fetch接口性能**：

1. 不同hidden维度性能测试
   - hidden=128/256/1024 × output_dtype=bf16/fp16/fp32
   - 9条基础性能用例

2. 不同world_size性能测试
   - world_size=2/4/8/16 × hidden=256
   - 3条规模性能用例

3. 不同num_tokens性能测试
   - num_tokens=128/1024/8192 × hidden=256
   - 3条负载性能用例

4. 边界组合性能测试
   - hidden=64×32tokens×ws2
   - hidden=4096×8192tokens×ws16
   - 2条极限性能用例

5. 梯度性能测试
   - hidden梯度：128~4096
   - num_tokens梯度：32~8192
   - 13条梯度性能用例

**总计**：30条性能用例

### 3.5 Exception覆盖详情

**失败场景分类**：

1. **indices越界失败**（2条）
   - indices负数越界
   - indices超大越界

2. **tensor参数错误**（2条）
   - storage dtype错误
   - storage shape错误

3. **状态错误**（2条）
   - buffer未构造
   - buffer已销毁

4. **时序错误**（1条）
   - fetch先于write

5. **并发错误**（1条）
   - 并发fetch同一buffer

6. **device错误**（1条）
   - tensor device不匹配

7. **分布式错误**（1条）
   - ProcessGroup未初始化

**总计**：10条失败用例

## 四、xlsx列字段定义

### 4.1 通用列（所有sheet）

| 字段名 | 类型 | 示例 | 说明 |
|--------|------|------|------|
| **case_name** | string | `h256_tok1024_e65536_ws8_bf16_write_fetch_ID000` | 用例命名 |
| **input_tensor_shape** | list | `[[65536, 256], [1024]]` | 输入tensor shape |
| **input_tensor_dtype** | list | `['bf16', 'int32']` | 输入tensor dtype |
| **input_tensor_range** | list | `[(-5, 5), (0, 524287)]` | 输入tensor取值范围 |
| **output_tensor_dtype** | list | `['bf16']` | 输出tensor dtype |
| **hidden** | int | 256 | 隐藏层维度 |
| **num_entries** | int | 65536 | 每rank表行数 |
| **num_tokens** | int | 1024 | Fetch token数 |
| **world_size** | int | 8 | 分布式进程数 |
| **output_dtype** | string | 'bf16' | Fetch输出类型 |
| **phase** | string | 'write_fetch' | 测试阶段 |
| **seed** | int | 0 | 随机种子 |
| **graph_type** | int | 0 | 图模式类型 |

### 4.2 Exception专用列

| 字段名 | 类型 | 示例 | 说明 |
|--------|------|------|------|
| **exception_type** | string | 'indices_negative' | 异常类型分类 |
| **exception_detail** | string | 'indices包含负数索引，预期失败' | 异常详细描述 |

## 五、用例示例

### 5.1 Level0典型用例

```yaml
case_name: h256_tok1024_e65536_ws8_bf16_write_fetch_ID000
input_tensor_shape: [[65536, 256], [1024]]
input_tensor_dtype: ['bf16', 'int32']
input_tensor_range: [(-5, 5), (0, 524287)]
output_tensor_dtype: ['bf16']
hidden: 256
num_entries: 65536
num_tokens: 1024
world_size: 8
output_dtype: bf16
phase: write_fetch
seed: 0
graph_type: 0
```

**解释**：
- 完整write+fetch流程
- 256维度hidden，65536 entries，1024 tokens
- 8卡分布式，bf16输出
- 输入包含storage和indices两个tensor

### 5.2 Level1组合用例

```yaml
case_name: h512_tok2048_e65536_ws8_fp16_fetch_ID200
input_tensor_shape: [[2048]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, 524287)]
output_tensor_dtype: ['fp16']
hidden: 512
num_entries: 65536
num_tokens: 2048
world_size: 8
output_dtype: fp16
phase: fetch
seed: 200
graph_type: 0
```

**解释**：
- 仅fetch阶段测试
- HIDDEN×NUM_TOKENS组合（512×2048）
- fp16输出类型测试

### 5.3 Performance性能用例

```yaml
case_name: perf_h128_tok4096_ws8_bf16_ID000
input_tensor_shape: [[4096]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, 524287)]
output_tensor_dtype: ['bf16']
hidden: 128
num_entries: 65536
num_tokens: 4096
world_size: 8
output_dtype: bf16
phase: fetch
seed: 0
graph_type: 0
```

**解释**：
- 性能测试仅关注fetch
- 128维度hidden，4096 tokens
- 测试不同hidden维度的fetch性能

### 5.4 Exception失败用例

```yaml
case_name: fail_fetch_ID000
input_tensor_shape: [[1024]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, 524287)]
output_tensor_dtype: ['bf16']
hidden: 256
num_entries: 65536
num_tokens: 1024
world_size: 8
output_dtype: bf16
phase: fetch
seed: 0
graph_type: 0
exception_type: indices_negative
exception_detail: indices包含负数索引，预期失败
```

**解释**：
- indices负数越界失败场景
- exception_type标记异常类型
- exception_detail描述失败原因

## 六、Phase阶段说明

### 6.1 Phase定义

| Phase | 输入tensor | 输出tensor | 测试场景 |
|-------|-----------|-----------|---------|
| **write** | storage: (num_entries, hidden) bf16 | - | 仅测试Write阶段 |
| **fetch** | indices: (num_tokens,) int32 | fetched: (num_tokens, hidden) output_dtype | 仅测试Fetch阶段 |
| **write_fetch** | storage + indices | fetched | 完整流程测试 |

### 6.2 Phase×output_dtype组合

| Phase | output_dtype | 说明 |
|-------|--------------|------|
| write | ''（空） | Write阶段无输出，output_dtype为空 |
| fetch | 'bf16'/'fp16'/'fp32' | Fetch阶段支持三种输出dtype |
| write_fetch | 'bf16'/'fp16'/'fp32' | 完整流程，fetch输出支持三种dtype |

## 七、与原设计对比

### 7.1 参数简化对比

| 对比项 | 原设计参数 | 新设计参数 | 简化效果 |
|--------|-----------|-----------|---------|
| **参数数量** | 9个参数 | 6个参数 | 减少33% |
| **world_size表示** | num_peers×ranks_per_peer | 直接world_size | 简化拓扑表示 |
| **dtype支持** | 仅bf16 | bf16/fp16/fp32 | 增加2种dtype |
| **sheet数量** | 2个 | 4个 | 增加性能和失败sheet |

### 7.2 用例数量对比

| Sheet | 原设计 | 新设计 | 变化 |
|-------|--------|--------|------|
| level0 | 33条 | 27条 | 减少6条（移除num_qps等） |
| level1 | 500条 | 500条 | 保持不变 |
| performance | 无 | 30条 | 新增性能sheet |
| exception | 无 | 10条 | 新增失败sheet |

## 八、测试覆盖验证

### 8.1 参数覆盖验证

- [x] HIDDEN_TIER全覆盖（7档）
- [x] ENTRIES_TIER覆盖（8档选5）
- [x] NUM_TOKENS_TIER覆盖（9档选5）
- [x] WORLD_SIZE_TIER全覆盖（4档）
- [x] OUTPUT_DTYPE_TIER全覆盖（3档）
- [x] PHASE_TIER全覆盖（3档）

### 8.2 组合覆盖验证

- [x] HIDDEN × NUM_TOKENS组合（25条）
- [x] HIDDEN × WORLD_SIZE组合（16条）
- [x] OUTPUT_DTYPE × HIDDEN组合（15条）
- [x] PHASE × HIDDEN组合（12条）
- [x] WORLD_SIZE × OUTPUT_DTYPE组合（12条）

### 8.3 Sheet统计验证

- [x] level0用例数27（≤200 ✓）
- [x] level1用例数500（500-700 ✓）
- [x] performance用例数30（新增sheet ✓）
- [x] exception用例数10（新增sheet ✓）

## 九、后续工作建议

### 9.1 Executor代码更新

需要更新executor_engram.py以适配新参数：
- 移除num_qps、hybrid_mode参数处理
- 简化world_size处理（直接使用）
- 增加output_dtype参数支持
- 增加phase处理逻辑
- 增加performance和exception测试逻辑

### 9.2 测试框架集成

- ATK配置更新：atk_engram.json
- 支持performance sheet执行
- 支持exception sheet验证

### 9.3 性能测试补充

- 增加性能指标测量（带宽、延迟）
- 对标DeepEP性能基线
- 不同dtype性能对比

---
**生成时间**：2026-06-09
**设计依据**：DeepEP test_engram.py + 用户需求
**关键改进**：移除QP/hybrid参数、简化world_size、增加dtype支持、新增sheet
**文件路径**：mc2/result/excel/engram.xlsx