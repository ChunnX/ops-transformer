# URMA P2P Benchmark 双模式设计

## 1. 设计目标

1. 建立两个节点间的基础P2P通信连接
2. 精确测量通信头开销时间
3. 计算并记录链路往返延迟(RTT)
4. **支持两种通信模式切换**：SQE级 vs Hcomm(WriteNbi+Drain)，通过Python `comm_alg`参数选择

## 2. 双模式架构

### SQE级模式 (comm_mode=0, comm_alg="sqe")

- 使用P2pUrmaSqeComm，基于HcclCombinOpParam上下文
- 通信流程：InitRankComm → P2pSend → TriggerDoorBell → DrainWaitCompletion
- 手动组装SQE、敲DoorBell、轮询CQ、管理PICI
- 9个timing slot

### Hcomm模式 (comm_mode=1, comm_alg="hcomm")

- 使用P2pHcommComm，基于Mc2MoeContext上下文
- 通信流程：DataCopy写本地窗口 → WriteNbi推送远端 → Drain等待完成 → P2pSendNotify推送状态
- Hcomm内部封装SQE/DoorBell/CQ/PICI，无需手动管理
- 5个timing slot

### 关键区别

| | SQE级 (P2pUrmaSqeComm) | Hcomm (P2pHcommComm) |
|---|---|---|
| 上下文 | HcclCombinOpParam (GetHcclContext) | Mc2MoeContext (Host端Mc2Context创建) |
| 发送API | PutCommNotifySQE (SQE写入GM环形缓冲区) | WriteNbi (非阻塞写入远端) |
| 等待完成 | PollCommCQUpdateSQCI (轮询CQ) | Drain (等待WriteNbi完成) |
| 状态通知 | WriteWithNotify SQE中的notify_addr/data | P2pSendNotify (先Drain再WriteNbi状态) |
| DoorBell | 手动SendJFSDoorBell | Hcomm内部处理 |
| PICI管理 | 手动GetPICI/UpdatePICI | Hcomm内部处理 |
| 地址模型 | windowsIn[rankId] (远端窗口) | epHcclBuffer[rankId] (远端URMA Buffer) |
| 首次通信 | 需InvalidateCqeStatus + GetIsFirstInComm | 无需(通道由Host端建立) |

## 3. SQE模式 Timing Slot布局

| Slot索引 | 名称 | 含义 |
|----------|------|------|
| 0 | PROCESS_START | Process()入口 |
| 1 | DUPLICATE_END | Duplicate(dstRankIds)完成 |
| 2 | INIT_SQ_CQ_LOAD_END | InitRankComm步骤1-6完成 |
| 3 | SQE_TEMPLATE_END | InitRankComm步骤7-8完成 |
| 4 | PROCESS_TOKEN_END | ProcessToken完成 |
| 5 | P2P_SEND_END | P2pSend完成 |
| 6 | DOORBELL_END | TriggerDoorBell完成 |
| 7 | POLL_CQ_END | PollCQ完成 |
| 8 | UPDATE_PICI_END | UpdatePICI完成 |

输出12个int64（9 slot + 3 padding）

## 4. Hcomm模式 Timing Slot布局

| Slot索引 | 名称 | 含义 |
|----------|------|------|
| 0 | PROCESS_START | Process()入口 |
| 1 | DATA_COPY_END | 数据从GM写入本地窗口完成 |
| 2 | WRITE_NBI_END | WriteNbi发送完成 |
| 3 | DRAIN_END | Drain等待完成 |
| 4 | NOTIFY_END | 状态通知WriteNbi完成 |

输出8个int64（5 slot + 3 padding）

## 5. Hcomm模式流程

### OverheadLatencyTest (mode=0)

```
DataCopyPad: GM输入 → UB → 本地窗口   [SLOT[1]=DATA_COPY_END]
WriteNbi:    本地窗口 → 远端URMA Buffer [SLOT[2]=WRITE_NBI_END]
Drain:       等待WriteNbi完成           [SLOT[3]=DRAIN_END]
Scalar写:    本地状态区(flag=1.0f, count=1)
SyncFunc<S_MTE3>
P2pSendNotify: Drain+WriteNbi推送状态到远端 [SLOT[4]=NOTIFY_END]
```

### BandwidthTest (mode=1)

```
所有token:   DataCopyPad写入本地窗口   [SLOT[1]=DATA_COPY_END]
逐rank:      WriteNbi批量推送数据      [SLOT[2]=WRITE_NBI_END]
逐rank:      Drain + P2pSendNotify     [SLOT[3]=DRAIN_END, SLOT[4]=NOTIFY_END]
```

## 6. Python端选择逻辑

```python
# 新增comm_alg参数
def urma_p2p_benchmark_op(x, group, world_size, max_bs, mode,
                          ccl_buffer_size, dst_rank_id_attr, aiv_num,
                          comm_alg="sqe"):  # "sqe" or "hcomm"
    comm_mode = 0 if comm_alg == "sqe" else 1
    y, receive_cnt, performance_info = npu_ops_transformer.ops.npu_urma_p2p_benchmark(
        x, group, world_size, max_bs, mode, ccl_buffer_size,
        dst_rank_id_attr, aiv_num, comm_mode
    )
    return y, receive_cnt, performance_info
```

### Hcomm模式性能数据解析

```python
def compute_hcomm_phase_breakdown(performance_info_list):
    CYCLES_PER_US = 50
    phases = {
        'data_copy_us':  (1, 0),   # GM→本地窗口
        'write_nbi_us':  (2, 1),   # WriteNbi发送
        'drain_us':      (3, 2),   # Drain等待完成
        'notify_us':     (4, 3),   # 状态通知
        'total_us':      (4, 0),   # 总耗时
    }
    # 通信头开销 = WriteNbi
    # 单向延迟 = data_copy + write_nbi + drain
```

## 7. 需要的Host端修改

### Tiling层新增comm_mode

- `UrmaP2pBenchmarkTilingData` 新增 `int32_t comm_mode` 字段
- Tiling函数根据Python传入的`comm_alg`设置`comm_mode`

### Hcomm模式需要Mc2Context

Hcomm模式需要Host端通过`Mc2Context::GetMc2ContextTensor`创建`Mc2MoeContext`，
而非SQE模式使用的`GetHcclContext<HCCL_GROUP_ID_0>()`。
需要修改op_host/op_api层，在`comm_alg=="hcomm"`时使用`COMM_PROTOCOL_UBC_CTP`协议创建上下文。

## 8. 文件变更清单

| 文件 | 变更 |
|------|------|
| p2p_hcomm_comm.h | **新增** Hcomm通信类 |
| urma_p2p_benchmark.h | 修改：加入Hcomm分支、timing slot、成员变量 |
| urma_p2p_benchmark_tiling.h | 修改：新增comm_mode字段 |
| urma_p2p_benchmark_tiling_key.h | 可能需要新增Hcomm的TilingKey |
| op_host tiling.cpp | 需修改：设置comm_mode |
| op_api aclnn | 需修改：Hcomm模式创建Mc2Context |
| urma_p2p_benchmark.py | 需修改：新增comm_alg参数、Hcomm模式解析 |
