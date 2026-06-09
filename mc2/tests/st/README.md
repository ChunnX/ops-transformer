# Engram算子ST测试

## 参考

https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py

## 文件说明

| 文件 | 说明 |
|------|------|
| executor_engram.py | Executor实现（3个类合并） |
| atk_engram.json | ATK测试配置 |
| test_engram.py | 简化测试脚本 |
| README.md | 本文档 |

## 测试流程

参考 DeepEP test_engram.py：

```python
# 1. 准备数据
local_storage = torch.randn((num_entries, hidden), dtype=torch.bfloat16, device=device)
global_storage = torch.empty((world_size * num_entries, hidden), dtype=torch.bfloat16, device=device)
dist.all_gather_into_tensor(global_storage, local_storage)

# 2. 构造buffer
buffer = ElasticBuffer(group, num_bytes, num_cpu_bytes, num_allocated_qps, allow_hybrid_mode)

# 3. engram_write
buffer.engram_write(local_storage)

# 4. engram_fetch
indices = torch.randint(0, world_size * num_entries, (num_tokens,), device=device)
hook = buffer.engram_fetch(indices)  # issue
fetched = hook()                      # wait

# 5. 验证
ref_fetched = global_storage[indices]
assert torch.equal(ref_fetched, fetched)
```

## 三个测试类

| 类名 | 注册名 | 功能 |
|------|--------|------|
| Engram | function_engram | write + fetch完整流程 |
| EngramWrite | function_engram_write | 仅write操作 |
| EngramFetch | function_engram_fetch | 仅fetch操作 |

## 关键特点

- **无小算子级联标杆**
- **直接使用ElasticBuffer API**
- **合并到一个文件**
- **简化清晰**

## 运行方式

```bash
# 分布式测试
torchrun --nproc_per_node=8 test_engram.py

# 单进程
python test_engram.py --num-processes 1 --num-entries 65536 --hidden 256
```

## 参数说明

- `--num-processes`: 进程数
- `--num-qps`: QP数量
- `--num-entries`: 每rank表行数
- `--hidden`: 隐藏层维度
- `--num-tokens`: fetch token数
- `--allow-hybrid-mode`: hybrid模式

## 验证点

- 数据一致性：`torch.equal`（二进制一致）
- 分布式拓扑：world_size和hybrid_mode
- 边界条件：indices范围验证

## 依赖

需要torch_npu提供ElasticBuffer实现。

---
生成时间：2026-06-09
参考：DeepEP test_engram.py