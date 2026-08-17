# Ascend C 与 MC2 算子小结

## 1. 算子是什么

算子是一个独立的计算或数据搬运单元。Ascend C 算子编译后运行在昇腾 NPU 的 AI Core 上，通常分成以下几部分：

| 模块 | 作用 |
|---|---|
| `op_api` | 对外提供 aclnn 两段式接口：先获取 Workspace 和执行器，再提交执行 |
| `op_host` | 定义输入、输出、属性；完成 Shape 推导和 Tiling |
| `op_graph` | 生成运行时可调度的通信、Kernel 任务 |
| `op_kernel` | 在 NPU 上执行真正的计算和数据搬运 |

真实执行顺序大致为：

```text
aclnn API → 算子定义 → Host Tiling → Graph Task → Kernel
```

### Host Tiling

Host 端负责运行前规划，不处理 token：校验 Shape、dtype、format 和属性，计算 Window/Workspace 大小，配置 HCCL，选择 TilingKey，设置核数，最后把 Kernel 需要的参数写入 TilingData。

### Kernel

Kernel 端按 TilingData 执行。常见结构是：

```text
Init：绑定 GM、读取 Tiling、申请 UB/队列
Process：分核 → CopyIn → Compute/寻址 → CopyOut
```

- Global Memory 对应 `GlobalTensor`，容量大但访问较慢。
- Local Memory（主要是 UB）对应 `LocalTensor`，计算前通常先从 GM 搬入。
- `TPipe` 管理片上内存和同步资源，`TQue` 用 `AllocTensor/EnQue/DeQue/FreeTensor` 串联流水任务。
- Double Buffer 让搬入、计算、搬出尽量重叠。
- Barrier/Sync 用于保证有依赖的异步指令按正确顺序完成。

### MC2

MC2 把计算与通信放进同一执行流程，利用 MTE 直接读写各 rank 的 HCCL Window。Window 一般分为数据区和状态区：发送方先写数据，再写 ready 标志；接收方通过状态判断数据是否可读。

## 2. AttentionToFFN

AttentionToFFN和FFNToAttention应该都是做token的分发（搬运？），并不涉及token的计算。

### 作用

Attention 根据 `expertIds` 和 `expertRankTable`，把 token 分发到承载对应专家的 FFN rank。支持共享专家、动态量化、active mask 以及同步/异步模式。

### 流程

```text
读取 token 和专家 ID
→ 查询 expertRankTable，得到目标 FFN rank 和本地专家号
→ 等待目标 Window 槽可写
→ 可选：按专家 scale 将 FP16/BF16 动态量化为 INT8
→ 将 token 写入目标 FFN Window
→ 写 token ready 标志
→ 汇总发送状态，通知相关 FFN rank 本批次/层数据已就绪
```


## 3. FFNToAttention

### 作用

FFN 计算完成后，根据 token 携带的回传信息，把结果送回对应 Attention Worker 的原位置。它不产生普通输出 Tensor，而是直接写 Attention rank 的 HCCL Window。

### 流程

```text
按 AIV 数量分摊 actualTokenNum
→ 读取 sessionId、microBatchId、tokenId、expertOffset
→ 可选：通过 attnRankTable 将逻辑 Worker 映射为物理 rank
→ 计算目标 Window 中的 Info/Data 地址
→ 将 x[token, 0:H] 写入 Data 区
→ Barrier
→ 将对应 Info 槽写为 1
```

地址核心公式：

```text
slot = (microBatchId × BS + tokenId) × expertNumPerToken + expertOffset
dataIndex = slot × HS
```



## 4. 两个算子的共性与区别


共性：两者都没有普通输出 Tensor；Host 负责校验、Window 规划、通信配置和 Kernel 选择；Kernel 按元数据寻址远端 Window，先写数据，再发布状态。

Kernel入口都主要做两件事:op.init(),op.process()

大体流程都是：API/算子注册 → Host Tiling 准备参数 → Graph Task 生成可调度任务 → 运行时启动 Kernel

调用都是两段式接口，先做准备工作如申请workspace，然后再提交任务。

## 5. MC2 路由搬运算子的通用模板

```text
Host：
1. 定义输入和属性
2. 校验 Shape/dtype/format
3. 提取通信布局和路由模式
4. 检查 Window 容量，设置 Workspace
5. 生成 HCCL 配置
6. 选择 TilingKey，设置 BlockDim，写入 TilingData

Graph：
7. 准备 KFC/MC2 通信资源
8. 生成运行时任务

Kernel：
9. Init：绑定 GM、HCCL Context 和 UB Buffer
10. Process：按 AIV 分核
11. 读取 token 及路由元数据
12. 计算目标 rank 和 Window 槽位
13. 必要时等待旧状态或执行量化/计算
14. 写 Data → Barrier → 写 Info/完成标志
```
