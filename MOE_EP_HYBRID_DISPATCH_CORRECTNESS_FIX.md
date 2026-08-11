# Moe EP Hybrid Dispatch 精度与卡死问题修复记录

## 1. 问题现象

- cache 和 non-cache 的 `recv_x`、`recv_topk_weights` 随机出现连续整槽为 0，错误起点通常是完整输出行边界。
- non-cache 的 `recv_src_metadata` 同时随机错误。
- 相同输入重复运行时错误 rank 和错误区间不固定。
- `is_continue=1` 可能卡死。

这些现象说明问题发生在 slot 分配、payload 发布或状态复用阶段，不是单个 hidden 分片的搬运长度错误。

## 2. 已确认根因与修复

### 2.1 Counter Reduce 的输入输出非法重叠

原 `ReduceCounterRange` 使用：

```text
ReduceSum(counterTmpTensor, counterTmpTensor, ...)
```

高阶 `ReduceSum` 不允许 source 和 destination 重叠。该结果同时用于：

- 每个 AIV 的 scaleup slot prefix；
- 每个 AIV 的 scaleout slot prefix；
- 每个目标 server 的 scaleout 总 slot 数。

错误 prefix 会导致不同 AIV 写同一个 slot、产生未写空洞，或把错误数量的 slot 发给 proxy。这是连续整槽为 0、cache/non-cache 都随机错误的直接根因之一。

修复后复用已结束预统计生命周期的 `countRankHitTensor_` 作为独立 reduce 输出：

```text
counterTmpTensor  --ReduceSum-->  counterSumTensor
dstTensor         --Add-------->  dstTensor
```

### 2.2 Slot 预统计缺少流水依赖

补齐以下依赖：

- `Duplicate` 初始化 rank/server 去重表后，Scalar `GetValue` 前增加 `V_S`；
- 上一 token 的 Scalar 读取结束后，下一 token 的 MTE2 覆盖同一 top-k UB 前增加 `S_MTE2`；
- scaleup prefix 和 scaleout prefix 复用同一 counter 临时 UB，第二次 MTE2 前增加 `V_MTE2`；
- count 的 Vector 结果写 workspace 前增加 `V_MTE3`；
- count workspace 在跨核读取前排空 MTE3。

### 2.3 Payload metadata 缺少 Scalar 到 MTE3 同步

`src_rank_id` 和 `src_token_idx` 由 Scalar `SetValue` 写入 token UB，随后 payload 由 MTE3 写入本地 window 或 staging workspace。原代码缺少 `S_MTE3`，可能把上一 token 的 metadata 搬出。

修复后在 metadata `SetValue` 与任何 payload MTE3 消费之间增加 `S_MTE3`。

### 2.4 Fanout 与其他 AIV 并发使用 HCOMM handle

原流程没有在 Stage1 发送和 AIV0 fanout 之间做全核汇合：

```text
WriteToRemoteWindow
SendScaleoutPayloadsToProxy
FanoutReceivedScaleoutPayloads
```

AIV0 可能已经开始向本 server rank fanout，其他 AIV 仍在向同一个 rank 做普通 scaleup 发送，从而并发操作同一个 HCOMM handle。

修复后流程为：

```text
WriteToRemoteWindow
SendScaleoutPayloadsToProxy
SyncAll<true>
FanoutReceivedScaleoutPayloads
```

`SyncAll` 的所有 AIV 都能到达；提前结束本核发送范围的 AIV 也只从子函数返回，不会退出 `ProcessHybridDispatch`。

### 2.5 Dispatch 提前消费 epilogue 状态

原 hybrid `Process` 末尾调用 `WaitDispatch`，等待并清除 `slotWinState`。但该状态本来是 dispatch 发布给下一算子 `dispatch_epilogue` 的 ready 标记；dispatch 自己先清零后，epilogue 会一直等待已经被消费的状态。

修复后删除 hybrid dispatch 内的 `WaitDispatch`，状态只由 epilogue 等待和清理。算子仍按同一 stream 的 dispatch、epilogue、combine 顺序执行，dispatch 不再越过下游抢占状态所有权。

### 2.6 Scaleout 汇总状态跨轮覆盖

原 scaleout 协议使用可复用的 `{ready=1, slotCount}`。Proxy 消费后清 0，但发送端不知道清 0 是否完成。快 rank 进入下一轮后可能发生：

```text
sender:   写下一轮 payload 和 ready=1
proxy:    清上一轮 ready=0
result:   下一轮通知丢失，或下一轮 payload 被提前覆盖
```

修复采用显式消费确认，不依赖 host 侧 rank 同步：

```text
source                         proxy
  | payload + ready ------------> |
  |                              | fanout payload
  |                              | clear scaleout ready
  | <----------------------- ack |
  | clear local ack              |
  | 下一轮才可复用接收区          |
```

所有 source 都先完成 notify 发送，之后才进入 fanout/ack 阶段，因此不会形成“等待 ack 导致无人接收 notify”的环。

新增 hybrid-only window 区：

```text
修改前
scaleoutRecvData
scaleupFinalRecvData
scaleoutRecvStatus      S * Nmax * 512
scaleupRecvStatus       E * Nmax * 512
combineData

修改后
scaleoutRecvData
scaleupFinalRecvData
scaleoutRecvStatus      S * Nmax * 512
scaleupRecvStatus       E * Nmax * 512
scaleoutAckStatus       S * 512
combineData
```

`dispatch`、`dispatch_epilogue`、`combine` 三处 tiling 以及 Python CCL buffer size helper 已使用同一公式。Direct 模式下新增区域大小为 0，原 direct window 格式不变。

### 2.7 `route_count` 多核 DCache 伪共享

原 cache/non-cache 路径均由多个 AIV 对连续 `route_count[token]` 执行 GM `SetValue`。相邻 AIV 的 token 边界可能落在同一 64B cache line，核私有 DCache 写回会相互覆盖。

修复后使用固定 32B UB 搬出块保存当前 token 的 route count，再通过 4B `DataCopyPad` 精确写入对应 GM 元素。不同 AIV 的 MTE3 写地址互不重叠，不再经过核私有 DCache。

新增 UB 为：

```text
32B
```

该空间位于 SendPhase 的 `numRecvPerRankBuf_` 尾部，不随 token 数增长，也不增加 TQue 数量。相邻 token 复用同一 UB 前增加 `MTE3 -> Scalar` 依赖，避免下一 token 覆盖尚未搬出的 route count。

### 2.8 小规模配置的分核除 0

原分核先做整数除法，`moeExpertNum + 2 * epWorldSize < aivNum` 时 `numPerCore` 为 0，后续 `Ceil(..., numPerCore)` 除 0。

修复后仅将每核工作量改为向上取整，保证除数至少为 1；rank/expert 的原分核比例和 expert 侧预留核策略保持不变。

### 2.9 Torch shape 配置一致性

增加以下校验，防止 dispatch 和 epilogue 使用不同 slot stride 或分配不足的输出：

- `x.shape[1] == configured hidden`；
- `topk_idx.shape[1] == configured num_topk`。

### 2.10 CCL buffer size 的未知拓扑低估

`get_moe_ep_ccl_buffer_size` 在创建 context 前无法取得真实 `rank_num_per_server`。原逻辑默认按单 server 计算，真实多 server 时可能低估 scaleout data/status/ack 区域。

修复后：

- 设置调试环境变量时，按指定拓扑精确计算；
- 未设置时，枚举 `world_size` 的全部合法 server 划分，返回其中最大的 hybrid window 需求。

这样不会要求业务额外传入拓扑，也不会因为 context 尚未创建而分配不足。

## 3. Direct 兼容性

- 未修改 `moe_ep_dispatch.h` direct kernel。
- Ack 状态区仅在 hybrid tiling 下分配，direct 下为 0。
- Epilogue 和 combine kernel 未改；只同步调整其 host 侧 hybrid `combineDataOffset` 计算。
- 原 direct 的 workspace、window data/status 地址和大小保持不变。

## 4. 验证建议

使用 `moe_train_test` 的 `explosiveDawn` 分支和同一组 shape，依次执行：

1. non-cache，`is_performance=0`，只跑 dispatch；
2. cache，`is_performance=0`，只跑 dispatch；
3. non-cache，`is_performance=1`，至少 100 轮；
4. cache，`is_performance=1`，至少 100 轮；
5. 上述四种模式分别打开 `is_continue=1`；
6. direct 模式重复 1-5，确认旧模板无回归。

重点检查：

- 每次运行的 `recv_x`、`recv_topk_weights`、`recv_src_metadata`；
- 错误是否仍从完整 row 边界开始；
- continue 是否仍停在 dispatch/epilogue/combine 任一状态轮询；
- cache 与首次 non-cache handle 的 route 表逐项一致。

## 5. 静态检查状态

- `git diff --check` 通过。
- `python3 -m py_compile torch_extension/cann_ops_transformer/ops/elastic_buffer.py` 通过。
- 本机无法执行 Ascend 编译：`/usr/local/Ascend/cann` 指向不存在的 `cann-9.0.0`，没有 `setenv.bash` 和 `bisheng`。
- 本机没有多 NPU 环境，未做设备侧精度和死锁验证。

## 6. 独立审查发现但未纳入本次修复

以下问题与当前连续零/continue 卡死不是同一直接根因，需要单独确认接口约束后处理：

- `debug_out` 多 AIV 计数使用非原子 GM 读改写，调试计数可能丢失且有性能开销；
- ACL 入口没有明确限制超大 `num_experts`，极端配置下 `SetRecvNumPerExpert` 的 int64 临时区可能超过固定 `dstExpBuf_`；
- 同 group 创建第二个 `ElasticBuffer` 时，已有 HCCL engine context 的复用分支没有回填本地 `MoeCommContext`；
- combine 对负 expert id 的 ready 数量协议需要单独定义；
- 当前预留的 per-slot `scaleoutRecvStatus` / `scaleupRecvStatus` 尚未按设计做细粒度流式消费，当前实现仍是 server/source 聚合 ready。
