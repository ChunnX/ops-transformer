# MC2 Context 创建流程

## Context 结构

```cpp
struct Mc2ContextStru {
    uint64_t epRankId;           // 当前 rank ID
    uint64_t kfcContextAddr;     // KFC 通信上下文地址
    uint64_t epHcclBuffer[1024]; // 各 rank 的 HCCL Buffer 地址数组
};
```

## 创建流程

```
update_context(group_ep, ep_world_size)
    │
    ▼
GetMc2Context(mc2ContextHost, epWorldSize, bufferSize, groupEpStr)
    │
    ├─► HcclKfcAllocOpArgs(&opArgs)           // 1. 分配通信配置对象
    │
    ├─► HcclKfcOpArgsSetCommEngine(opArgs, AIV) // 2. 设置通信引擎为 AIV
    │
    ├─► CreateHcclContext(comm, opArgs, worldSize, groupName)
    │       │
    │       ├─► HcclKfcOpArgsSetAlgConfig(opArgs, algConfig)  // 设置算法配置
    │       ├─► HcclCommGetHandleWithName(groupName, &comm)   // 获取通信句柄
    │       ├─► HcclCreateOpResCtx(comm, opType, opArgs)      // 创建操作资源上下文
    │       ├─► HcclGetRankSize(comm, &worldSize)             // 获取 world size
    │       └─► HcclGetRankId(comm, &rankId)                  // 获取 rank ID
    │
    ├─► CreatMc2Context(comm, worldSize, bufferSize, mc2Context)
    │       │
    │       ├─► HcclGetRankId(comm, &rankId)   // 获取当前 rank
    │       │
    │       └─► for (remoteRankId = 0; remoteRankId < worldSize; remoteRankId++)
    │               │
    │               ├─► rankId == remoteRankId:
    │               │   HcclGetHcclBuffer(comm, &addr, &size)     // 本地 buffer
    │               │
    │               └─► rankId != remoteRankId:
    │                   HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &addr, &size)  // 远程 IPC buffer
    │               │
    │               └─► mc2Context.epHcclBuffer[remoteRankId] = addr  // 存入结构体
    │
    ├─► HcclKfcFreeOpArgs(opArgs)             // 3. 释放通信配置对象
    │
    ▼
at::Tensor output = copy mc2ContextHost to NPU
return (output, bufferSize)
```

### 两阶段函数说明

创建流程分为两个核心函数：

#### CreateHcclContext - 建立 HCCL 通信框架

**职责**: 创建 HCCL 操作资源上下文，建立通信框架

| API | 作用 |
|-----|------|
| HcclKfcOpArgsSetAlgConfig | 设置 All-to-All 算法配置 |
| HcclCommGetHandleWithName | 根据 group 名称获取通信句柄 |
| HcclCreateOpResCtx | 创建操作资源上下文 |
| HcclGetRankSize / HcclGetRankId | 获取 world size 和 rank ID，验证参数 |

**产出**: `HcclComm commHandle` - 有效的通信句柄

#### CreatMc2Context - 填充具体通信地址

**职责**: 基于已建立的通信框架，填充 Mc2ContextStru 结构体

| API | 作用 |
|-----|------|
| HcclGetRankId | 获取当前 rank ID → `mc2Context.epRankId` |
| HcclGetHcclBuffer | 获取本地 buffer 地址和大小 |
| HcclGetRemoteIpcHcclBuf | 获取远程 rank 的 IPC buffer 地址 |

**产出**: 完整的 `Mc2ContextStru`，包含所有 rank 的 buffer 地址

#### 逻辑关系

- **顺序依赖**: `CreateHcclContext` 先执行建立通信框架，产出 `commHandle`；`CreatMc2Context` 后执行，依赖 `commHandle` 查询 buffer 地址

## 缓存机制

```cpp
static bool isInit = false;
static std::string last_group_ep_str = "";

// group 未变化时直接返回缓存结果，以适应重复调用
if (isInit && new_group_ep_str == last_group_ep_str) {
    return std::make_tuple(output, bufferSize);
}
```

## Context 的作用

1. **预获取通信地址**: 初始化时一次性获取所有 rank 的 HCCL buffer 地址，避免每次 kernel 调用时重复 HCCL API 查询
2. **支持零拷贝**: Kernel 通过 `epHcclBuffer[rank_id]` 直接访问目标 rank 的 buffer 地址
3. **通信资源缓存**: 同一 group 重复调用时直接返回已创建的 context
