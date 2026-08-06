# Dispatch / Combine V2 与 V3 差异：Context 是如何切换的

## 1. 先给结论

V2 和 V3 的核心计算流程没有完全重写。它们的主要差异是**通信上下文从哪里来、以什么格式传给 kernel**：

| 项目 | V2 | V3 |
|---|---|---|
| 对外通信参数 | `group_ep`、`group_tp` 字符串 | `context` Tensor |
| CCL buffer size | 不作为显式算子属性传入 | 必选属性 `ccl_buffer_size` |
| kernel 收到的 context | 运行框架提供的 `HcclOpParam` | 上层提前创建的 `Mc2MoeContext` |
| kernel context 适配器 | `HcclContextHolder` | `Mc2MoeContextHolder` |
| 核心 dispatch/combine 模板 | V2 模板 | 仍复用 V2 模板，只替换 ContextHolder |
| A2 | 支持 | 当前 V3 OpDef 不注册 A2 |
| A3/A5 | 支持 | 支持 |

最简化地看：

```cpp
// V2
MoeDistributeDispatchV2<HcclContextHolder, ...>
MoeDistributeCombineV2<HcclContextHolder, ...>

// V3
MoeDistributeDispatchV2<Mc2MoeContextHolder, ...>
MoeDistributeCombineV2<Mc2MoeContextHolder, ...>
```

也就是说，V3 主要通过模板参数把“如何解释 context、如何取得 rank/window 地址”的策略切掉，而不是复制一套核心通信算法。

## 2. 算子接口层的差异

### 2.1 Dispatch

DispatchV2 没有 context Tensor 输入，而是接收通信域名称：

- [`group_ep` 是必选属性](mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp#L98)
- [`group_tp` 是可选属性](mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp#L102)

DispatchV3 删除这两个通信域名称，增加显式 context Tensor 和 buffer size：

- [`context` Tensor 输入](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L23)
- [`ccl_buffer_size` 必选属性](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L106)

其余主要业务输入输出，例如 `x`、`expert_ids`、`expand_x`、`assist_info_for_combine`、`ep_recv_count` 等基本保持一致。

### 2.2 Combine

CombineV2 同样通过字符串指定通信域：

- [`group_ep`](mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp#L138)
- [`group_tp`](mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp#L142)

CombineV3 改为：

- [`context` Tensor 输入](mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L23)
- [`ccl_buffer_size` 必选属性](mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L146)

### 2.3 平台注册差异

V2 的 OpDef 包含 A2、A3、A5 配置：

- [DispatchV2 平台配置起点](mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp#L118)
- [CombineV2 平台配置起点](mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp#L158)

V3 当前只包含 A3、A5配置：

- [DispatchV3 平台配置起点](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L122)
- [CombineV3 平台配置起点](mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L162)

## 3. 两种 Context 分别是什么

### 3.1 V2：HcclContextHolder

V2 的 A3 kernel 实例化核心模板时选择 `HcclContextHolder`：

- [DispatchV2 使用 `HcclContextHolder`](mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a3.cpp#L43)
- [CombineV2 使用 `HcclContextHolder`](mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a3.cpp#L44)

它把 kernel 收到的 context 地址解释为 `HcclOpParam`：

- [`HcclContextHolder`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L36)
- [`ctxPtr` 转成 `HcclOpParam*`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L41)

后续通过 HCCL helper 读取：

- 当前 rank：[`GetEpRankId()`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L59)
- 通信域大小：[`GetEpWorldSize()`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L64)
- 指定 rank 的数据窗口：[`GetWindAddrByRankId()`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L69)
- 指定 rank 的状态窗口：[`GetWindStateAddrByRankId()`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L74)

V2 holder 还会使用 HCCL context 中的 window size 做越界检查，见 [`Check()`](mc2/moe_distribute_dispatch_v2/op_kernel/hccl_context_holder.h#L49)。

### 3.2 V3：Mc2MoeContextHolder

V3 的 kernel 入口改用 `Mc2MoeContextHolder`：

- [DispatchV3 使用 `Mc2MoeContextHolder`](mc2/moe_distribute_dispatch_v3/op_kernel/arch22/moe_distribute_dispatch_v3_a3.cpp#L48)
- [CombineV3 使用 `Mc2MoeContextHolder`](mc2/moe_distribute_combine_v3/op_kernel/arch22/moe_distribute_combine_v3_a3.cpp#L49)

它把显式传入的 context Tensor 地址解释为 `Mc2MoeContext`：

- [`Mc2MoeContextHolder`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L35)
- [`ctxPtr` 转成 `Mc2MoeContext*`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L41)

它直接从结构体读取：

- 当前 rank：[`mc2Context_->epRankId`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L57)
- 本 rank 状态区：[`GetStatusDataSpaceGm()`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L52)
- 指定 rank 数据窗口：[`epHcclBuffer_[rankId] + STATE_SIZE`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L68)
- 指定 rank 状态窗口：[`epHcclBuffer_[rankId]`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L74)

`Mc2MoeContext` 本身不保存 `epWorldSize`，所以 holder 从 tiling data 接收并暂存它，见 [`epWorldSize_`](mc2/moe_distribute_dispatch_v2/op_kernel/mc2_moe_context_holder.h#L41)。

## 4. Kernel 层是怎么“切 Context”的

核心 dispatch/combine 类将 ContextHolder 作为模板参数，因此内部算法只依赖统一接口：

```text
InitAndCheck()
GetEpRankId()
GetEpWorldSize()
GetStatusDataSpaceGm()
GetWindAddrByRankId()
GetWindStateAddrByRankId()
```

V2 和 V3 分别提供这套接口的不同实现。

### Dispatch 切换

```cpp
// V2 kernel 入口
MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, ...> op;
op.Init(contextGM0, ...);

// V3 kernel 入口
MoeDistributeDispatchV2<Mc2Kernel::Mc2MoeContextHolder, ...> op;
op.Init(mc2Context, ...);
```

对应源码：

- [DispatchV2 入口](mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a3.cpp#L43)
- [DispatchV3 入口](mc2/moe_distribute_dispatch_v3/op_kernel/arch22/moe_distribute_dispatch_v3_a3.cpp#L48)

### Combine 切换

```cpp
// V2 kernel 入口
MoeDistributeCombineV2<Mc2Kernel::HcclContextHolder, ...> op;
op.Init(contextGM0, ...);

// V3 kernel 入口
MoeDistributeCombineV2<Mc2Kernel::Mc2MoeContextHolder, ...> op;
op.Init(mc2Context, ...);
```

对应源码：

- [CombineV2 入口](mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a3.cpp#L44)
- [CombineV3 入口](mc2/moe_distribute_combine_v3/op_kernel/arch22/moe_distribute_combine_v3_a3.cpp#L49)

因此，context 的切换发生在**kernel 入口实例化模板时**，不是在核心 token dispatch/combine 循环中反复判断版本。

## 5. Mc2MoeContext 是如何创建的

`GetMc2ContextTensor()` 的流程如下：

```text
groupEp
  -> 获取 HCCL communicator handle
  -> 用 groupEp + opName 组成 context tag
  -> 查询 HCCL Engine Context 缓存
  -> 命中：把缓存地址包装成 aclTensor
  -> 未命中：创建 Engine Context、取得各 rank buffer 地址，再包装成 aclTensor
```

关键位置：

- [`GetMc2ContextTensor()`](mc2/common/op_api/mc2_context.cpp#L499)
- [context tag = `groupEp + opName`](mc2/common/op_api/mc2_context.cpp#L510)
- [取得 HCCL communicator](mc2/common/op_api/mc2_context.cpp#L517)
- [查询 context 缓存](mc2/common/op_api/mc2_context.cpp#L521)
- [创建新的 context](mc2/common/op_api/mc2_context.cpp#L533)
- [包装成 context Tensor](mc2/common/op_api/mc2_context.cpp#L538)

创建出的 Tensor 数据类型是 `INT32`，其 device 地址实际指向 `Mc2MoeContext` 内存，见 [`CreatMc2ContextTensor()`](mc2/common/op_api/mc2_context.cpp#L348)。所以这个 Tensor 不是模型数据，而是为了让通信资源能够作为算子输入进入 eager/GE 图的一种载体。

## 6. 重要细节：当前 V2 API 自己也可能切到 V3

当前代码为了兼容旧 V2 API，在 ACLNN host 层加入了内部路由。

### Dispatch 路由

```text
非 Ascend 950，或者 commAlg == "ccu"
  -> aclnnInnerMoeDistributeDispatchV2

Ascend 950 且 commAlg != "ccu"
  -> 根据 groupEp 创建/取得 Mc2Context Tensor
  -> aclnnInnerMoeDistributeDispatchV3
```

源码：

- [Dispatch V2/V3 分支条件](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L190)
- [走 inner DispatchV2](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L191)
- [由 `groupEp` 取得 context Tensor](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L201)
- [改走 inner DispatchV3](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L203)

### Combine 路由

Combine 使用同样的规则：

- [Combine V2/V3 分支条件](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L187)
- [走 inner CombineV2](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L188)
- [由 `groupEp` 取得 context Tensor](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L200)
- [改走 inner CombineV3](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L202)

这意味着：**用户调用旧 V2 ACLNN API，不保证底层一定执行 V2 GE 算子。Ascend 950 的非 CCU 路径会在 host 侧把 group 名称转换成 context Tensor，并内部转发给 V3。**

## 7. DeepEP 显式 V3 路径与兼容 V2 路径的区别

```text
DeepEP / 显式 V3：
上层长期持有 context Tensor
  -> 每次 dispatch/combine 直接传 context
  -> V3 kernel + Mc2MoeContextHolder

兼容 V2 API 在 Ascend 950 非 CCU 上：
每次从 groupEp 进入 V2 ACLNN wrapper
  -> wrapper 查询或创建缓存 context
  -> 转发 inner V3
  -> V3 kernel + Mc2MoeContextHolder

传统 V2 路径：
groupEp/groupTp
  -> inner V2
  -> V2 kernel 收到 contextGM0
  -> HcclContextHolder
```

显式 V3 的重点不是通信算法完全不同，而是 context 的生命周期和所有权上移：调用方可预先创建、保存，并让 dispatch 与 combine 复用同一个 context。

## 8. 一句话总结

V2 到 V3 的 context 切换分为两层：**host 层把 `groupEp` 转成 `Mc2MoeContext` Tensor并选择 inner V3；kernel 层把模板参数从 `HcclContextHolder` 换成 `Mc2MoeContextHolder`。核心 dispatch/combine 模板继续复用 V2 实现。**
