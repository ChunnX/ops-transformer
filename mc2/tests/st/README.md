# Engram算子ST测试

## 算子定义

Engram算子包含两个阶段：
- **Write阶段**：将本rank的表分片写入host pinned共享段
- **Fetch阶段**：通过RDMA从远程rank抓取数据到本rank

参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py

## 测试设计

### 六维TilingKey参数

| 参数 | 取值范围 | 说明 |
|------|----------|------|
| NUM_QPS_TIER | {1, 4, 8, 16, 32} | QP数量 |
| HIDDEN_TIER | {64, 128, 256, 512, 1024, 2048, 4096} | 隐藏层维度 |
| NUM_PEERS_TIER | {1, 2, 4, 8, 16} | RDMA peer数 |
| RANKS_PER_PEER_TIER | {1, 2, 4, 8} | 每peer内rank数 |
| ENTRIES_TIER | {4096..524288} | 每 rank表行数 |
| HYBRID_MODE | {0, 1} | Hybrid模式标志 |

### 测试阶段

| Phase | 输入 | 输出 | 说明 |
|-------|------|------|------|
| write | storage (num_entries, hidden) bf16 | - | Write阶段独立测试 |
| fetch | indices (num_tokens,) int32 | fetched (num_tokens, hidden) bf16 | Fetch阶段独立测试 |
| write_fetch | storage + indices | fetched | 完整流程测试 |

### 测试用例统计

- **Level0**: 33条基础覆盖（单因子）
- **Level1**: 500条两两组合覆盖

## 文件结构

```
mc2/tests/st/
├── executor_engram.py     # Executor实现
├── atk_engram.json        # ATK配置
└── README.md              # 本文档

mc2/result/excel/
└── engram.xlsx            # 测试用例文件
```

## 运行方式

```bash
# ATK框架运行
python mc2_test.py --xlsx mc2/result/excel/engram.xlsx --sheet level0

# 简化测试
python mc2/tests/st/test_engram.py --num-processes 8 --phase write_fetch
```

## 验证点

- 数据一致性：torch.equal（二进制一致）
- 六维TilingKey全覆盖
- Write/Fetch阶段独立测试
- 完整流程正确性验证

---
生成时间：2026-06-09
参考：DeepEP test_engram.py
测试框架：MC2 + ATK