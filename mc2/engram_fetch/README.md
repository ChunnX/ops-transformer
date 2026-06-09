# EngramFetch 算子

## 概述

EngramFetch 是一个基于 HCCL one-sided RMA 的数据抓取算子，用于从远程 rank 的 host pinned storage 获取 Engram 数据到本地 NPU。

## 当前版本（Skeleton）

这是 **极简骨架版本**，仅验证 ACLNN 调用路径：

- ✅ **Kernel 仅打印输出**，不做实际数据处理
- ✅ **单一 TilingKey**，无模板系统
- ❌ **不包含 HCCL RDMA 实现**
- ❌ **不包含实际数据搬运逻辑**

## 两段式设计

符合设计文档要求，分为 issue + wait 两个阶段：

1. **EngramFetch (issue)**: 发起 RDMA Get 请求
2. **EngramFetchWait (wait)**: 等待请求完成

## Python调用方式

```python
from npu_ops_transformer.ops.elastic_buffer import ElasticBuffer

# 创建 ElasticBuffer
eb = ElasticBuffer(...)

# 调用 engram_fetch (issue)
indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="npu")
wait_func = eb.engram_fetch(indices, hidden_size=128)

# 调用 wait 阶段
fetched = wait_func()
```

## 文件结构

```
engram_fetch/
├── op_api/          # ACLNN 接口层
│   ├── aclnn_engram_fetch.h/cpp       # issue 接口
│   └── aclnn_engram_fetch_wait.h/cpp  # wait 接口
├── op_host/         # 算子定义与 Tiling
│   ├── engram_fetch_def.cpp           # issue 定义
│   └── engram_fetch_wait_def.cpp      # wait 定义
├── op_kernel/       # Kernel 实现（当前仅打印）
│   ├── engram_fetch.cpp               # issue kernel
│   └── engram_fetch_wait.cpp          # wait kernel
└── examples/        # 测试用例
    └── test_aclnn_engram_fetch.cpp
```

## 后续迭代

1. **Data Movement**: 添加实际数据搬运逻辑
2. **HCCL RDMA**: 接入 HCCL one-sided RMA API
3. **TilingKey**: 添加模板系统优化性能
4. **Barrett**: 除法优化

## 参考

详细设计文档：https://gitcode.com/Allan_Yu/design_work/blob/master/engram_design/operators/engram_fetch/docs/DESIGN.md