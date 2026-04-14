# aclnnAdvanceStep

## 功能描述

### 算子功能
AdvanceStep 是推测解码（Speculative Decoding）中的步骤推进算子。它根据采样得到的 token ID 更新输入 token、位置编码、序列长度和槽位映射等 KV Cache 管理相关数据结构，支持无推测和带推测两种模式。

### 计算公式

**无推测模式（Legacy）：** 对每个 query 序列：
- `input_tokens[index] = sampled_token_ids[index]`
- `seq_lens[index] = seq_lens[index] + 1`
- `input_positions[index] = seq_lens[index]`
- `slot_mapping[index] = (block_tables[block_index] + blockTablesStride * index) * blockSize + block_offset`

其中 `block_index = input_positions[index] / blockSize`，`block_offset = input_positions[index] % blockSize`。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.advance_step()` 函数形式提供：

```python
def advance_step(input_tokens: Tensor, sampled_token_ids: Tensor,
                 input_positions: Tensor, seq_lens: Tensor,
                 slot_mapping: Tensor, block_tables: Tensor,
                 spec_token: Tensor, accepted_num: Tensor,
                 num_seqs: int, num_queries: int, block_size: int) -> List[Tensor]
```

## 使用案例
```python
import torch
import kernel_gen_ops

num_seqs, num_queries, block_size = 8, 4, 16
input_tokens = torch.randint(0, 1000, (num_seqs,), dtype=torch.int64).npu()
sampled_token_ids = torch.randint(0, 1000, (num_queries, 1), dtype=torch.int64).npu()
input_positions = torch.randint(0, 128, (num_seqs,), dtype=torch.int64).npu()
seq_lens = torch.randint(1, 128, (num_seqs,), dtype=torch.int64).npu()
slot_mapping = torch.full((num_seqs,), -1, dtype=torch.int64).npu()
block_tables = torch.randint(0, 1024, (num_seqs, 8), dtype=torch.int64).npu()
spec_token = torch.zeros(0, dtype=torch.int64).npu()
accepted_num = torch.zeros(0, dtype=torch.int64).npu()

result = kernel_gen_ops.advance_step(input_tokens, sampled_token_ids, input_positions,
                                      seq_lens, slot_mapping, block_tables,
                                      spec_token, accepted_num,
                                      num_seqs, num_queries, block_size)
```

## 约束与限制
- 支持的数据类型：仅 `int64`
- 支持的数据格式：`ND`
- `num_seqs` 必须 >= `num_queries`（无推测模式要求 `num_seqs` > `num_queries`）
- `block_size` 必须 >= 1
- 所有输入张量必须在同一设备上
