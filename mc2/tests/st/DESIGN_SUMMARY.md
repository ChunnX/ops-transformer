# Engram算子测试设计总结

## 一、算子定义

### 1.1 核心概念

**Engram算子包含两个阶段**：
- **Write阶段**：将本rank的表分片写入host pinned共享段
- **Fetch阶段**：通过RDMA从远程rank抓取数据到本rank

**设计理念**：将Fetch和Write视为同一算子的不同阶段，而非独立算子。

### 1.2 参考实现

https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py

## 二、测试参数定义

### 2.1 六维TilingKey

| 参数 | 取值范围 | 说明 |
|------|----------|------|
| **NUM_QPS_TIER** | {1, 4, 8, 16, 32} | QP数量（每QP占用1 SM） |
| **HIDDEN_TIER** | {64, 128, 256, 512, 1024, 2048, 4096} | 隐藏层维度 |
| **NUM_PEERS_TIER** | {1, 2, 4, 8, 16} | RDMA peer数 |
| **RANKS_PER_PEER_TIER** | {1, 2, 4, 8} | 每peer内rank数 |
| **ENTRIES_TIER** | {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288} | 每 rank表行数 |
| **HYBRID_MODE** | {0, 1} | Hybrid模式标志（flat/hybrid） |
| **NUM_TOKENS** | {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192} | Fetch token数 |

### 2.2 测试阶段（Phase）

| Phase | 输入Tensor | 输出Tensor | 测试场景 |
|-------|-----------|-----------|---------|
| **write** | storage: (num_entries, hidden) bf16 | - | 仅测试Write阶段 |
| **fetch** | indices: (num_tokens,) int32 | fetched: (num_tokens, hidden) bf16 | 仅测试Fetch阶段 |
| **write_fetch** | storage + indices | fetched | 完整流程测试 |

## 三、测试用例生成

### 3.1 Level0基础覆盖（33条）

**设计原则**：单因子独立覆盖

**覆盖内容**：
1. NUM_QPS_TIER全覆盖（5档） - 5条
2. HIDDEN_TIER全覆盖（7档） - 7条
3. ENTRIES_TIER代表性覆盖（5档） - 5条
4. NUM_TOKENS_TIER代表性覆盖（5档） - 5条
5. HYBRID_MODE全覆盖（2档） - 2条
6. 测试阶段覆盖（3个phase） - 3条
7. world_size组合覆盖（6种配置） - 6条

**示例用例**：
```
case_name: qps1_h256_tok1024_e65536_write_fetch_ID000
input_tensor_shape: [[65536, 256], [1024]]
input_tensor_dtype: ['bf16', 'int32']
input_tensor_range: [(-5, 5), (0, 524287)]
num_qps: 1
hidden: 256
num_entries: 65536
num_tokens: 1024
world_size: 8
phase: write_fetch
```

### 3.2 Level1两两组合覆盖（500条）

**设计原则**：两两因子组合覆盖

**重点组合**：
1. **NUM_QPS × HIDDEN**（5×7 = 35条）
2. **HIDDEN × ENTRIES**（7×5 = 35条）
3. **NUM_TOKENS × HIDDEN**（9×5 = 45条）
4. **HYBRID_MODE × NUM_PEERS**（2×5 = 10条）
5. **phase × HIDDEN**（3×5 = 15条）
6. **随机组合补充**（至500条）

## 四、xlsx文件格式

### 4.1 Sheet结构

| Sheet名 | 用例数 | 用途 |
|---------|--------|------|
| level0 | 33 | 基础功能覆盖（≤200 ✓） |
| level1 | 500 | 两两组合覆盖（500-700 ✓） |

### 4.2 列字段

**通用列**：
- case_name
- input_tensor_shape
- input_tensor_dtype
- input_tensor_range
- output_tensor_dtype
- seed
- graph_type

**算子特定列**：
- num_qps
- hidden
- num_entries
- num_tokens
- num_peers
- ranks_per_peer
- world_size
- hybrid_mode
- phase

### 4.3 数据类型编码

**Tensor dtype**：
- storage/fetched: 'bf16'
- indices: 'int32'

**Shape编码**：
- write: [[num_entries, hidden]]
- fetch: [[num_tokens]]
- write_fetch: [[num_entries, hidden], [num_tokens]]

**Range编码**：
- storage: (-5, 5)
- indices: (0, num_entries × world_size - 1)

## 五、Executor实现

### 5.1 核心类

```python
@register("function_engram")
class Engram(BaseApi):
    """
    Engram算子 - 包含Write和Fetch两个阶段
    """
    
    def __call__(self, input_data):
        phase = input_data.kwargs['phase']
        
        if phase == 'write':
            # 仅Write阶段
            buffer.engram_write(local_storage)
            
        elif phase == 'fetch':
            # 仅Fetch阶段
            hook = buffer.engram_fetch(indices)
            fetched = hook()
            
        else:  # write_fetch
            # 完整流程
            buffer.engram_write(local_storage)
            hook = buffer.engram_fetch(indices)
            fetched = hook()
```

### 5.2 ElasticBuffer API

```python
from torch_npu.contrib import ElasticBuffer

# 构造buffer
buffer = ElasticBuffer(
    group=group,
    num_bytes=num_gpu_bytes + num_cpu_bytes,
    num_cpu_bytes=num_cpu_bytes,
    num_allocated_qps=num_qps,
    allow_hybrid_mode=hybrid_mode
)

# Write阶段
buffer.engram_write(local_storage)

# Fetch阶段（issue + wait）
hook = buffer.engram_fetch(indices)
fetched = hook()
```

## 六、文件清单

### 6.1 生成的文件

| 文件路径 | 说明 |
|---------|------|
| mc2/result/excel/engram.xlsx | 测试用例xlsx文件 |
| mc2/tests/st/executor_engram.py | Executor实现 |
| mc2/tests/st/atk_engram.json | ATK测试配置 |
| mc2/tests/st/README.md | 测试说明文档 |
| mc2/generate_engram_cases.py | 测试用例生成脚本 |

### 6.2 文件统计

| 类型 | 文件数 | 总行数 |
|------|--------|--------|
| executor | 1 | ~200 |
| config | 1 | ~40 |
| doc | 2 | ~200 |
| script | 1 | ~360 |

## 七、验证检查清单

### 7.1 xlsx格式验证

- [x] 包含level0和level1 sheets
- [x] level0用例数33（≤200 ✓）
- [x] level1用例数500（500-700 ✓）
- [x] 所有shape/dtype/range可eval()解析
- [x] case_name遵循命名规范
- [x] 布尔值使用0/1编码（hybrid_mode）

### 7.2 测试覆盖验证

- [x] NUM_QPS_TIER全覆盖（5档）
- [x] HIDDEN_TIER全覆盖（7档）
- [x] ENTRIES_TIER覆盖（8档选5）
- [x] HYBRID_MODE全覆盖（2档）
- [x] 测试阶段全覆盖（3个phase）
- [x] world_size组合覆盖（6种配置）

## 八、运行说明

### 8.1 ATK框架运行

```bash
# Level0测试
python mc2_test.py --xlsx mc2/result/excel/engram.xlsx --sheet level0

# Level1测试
python mc2_test.py --xlsx mc2/result/excel/engram.xlsx --sheet level1
```

### 8.2 简化测试脚本

```bash
# 分布式测试（参考DeepEP）
python mc2/tests/st/test_engram.py --num-processes 8 --phase write_fetch

# 单进程测试
python mc2/tests/st/test_engram.py --num-processes 1 --phase write
```

## 九、关键设计决策

### 9.1 Fetch和Write统一设计

**决策**：将Fetch和Write视为同一算子的不同阶段

**理由**：
1. Write和Fetch共用同一个ElasticBuffer
2. Write必须先于Fetch执行
3. 实际使用场景中两者总是成对出现
4. 简化测试设计，避免重复配置

### 9.2 无小算子级联标杆

**决策**：不设计npu_bm小算子级联标杆

**理由**：
1. 参考DeepEP直接使用真实API
2. EngramWrite是host runtime操作，无独立ACLNN接口
3. 简化测试逻辑，贴近实际应用

### 9.3 Phase参数设计

**决策**：引入phase参数区分测试阶段

**设计**：
- phase='write'：仅测试Write阶段
- phase='fetch'：仅测试Fetch阶段
- phase='write_fetch'：完整流程测试

**好处**：
- 支持阶段独立测试
- 支持完整流程验证
- 符合实际使用场景

## 十、后续工作

### 10.1 待torch_npu实现

- ElasticBuffer类
- engram_write方法
- engram_fetch方法
- RDMA one-sided Get支持

### 10.2 性能测试补充

- issue/wait时间统计
- 带宽利用率测试
- kineto profiling

### 10.3 异常场景测试

- indices越界测试
- 并发冲突测试
- Write-Fetch时序测试

---
**生成时间**：2026-06-09
**参考框架**：MC2 Test Design + DeepEP test_engram.py
**测试框架**：ATK + torch_npu
**核心理念**：Fetch和Write作为同一算子的不同阶段