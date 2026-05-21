# MatmulReduceScatterV2 AICPU All2All 通路批跑精度问题定位记录

## 现象

| 通路 | 场景 | 精度 |
|------|------|------|
| CCU | 批跑/单跑 | 均 PASS |
| AICPU ReduceScatter | 批跑/单跑 | 均 PASS |
| AICPU All2All | 单跑 | PASS |
| AICPU All2All | 同 shape 批跑 | PASS |
| AICPU All2All | 不同 shape 批跑 | FAIL (第一个 PASS, 后续 FAIL) |
| AICPU All2All + `ASCEND_GLOBAL_LOG_LEVEL=0` | 批跑 | PASS (日志延迟掩盖竞态) |

## 排除的根因

| 怀疑方向 | 验证方式 | 结论 |
|----------|----------|------|
| recvBuf 跨 tile 流水线竞态 | AIV 侧完全串行化 | 排除 |
| AlltoAll<true> vs AlltoAll<false>+Commit | 改为两步 Prepare+Commit 模式 | 排除 |
| InitV2/Finalize 不对称 | 对照参考算子 matmul_allto_all / allto_all_matmul | 模式一致, 排除 |
| ReduceSum 多核分派 | 改为单核 (block 0 only) | 排除 |

## 根因定位

**AICPU 模式下，Cube 核写入 GM 的 MatMul 结果对 KFC Server DMA 引擎不可见。**

在 `VecWaitCube()` (对 flag 8 的 CrossCoreWaitFlag) 之后、AlltoAll 启动之前，增加 `SyncAll<true>()` 内存栅栏后，批跑精度恢复 PASS。

### 根因分析

```
AIC(Cube): MatMul → 写 sendBuf_(GM) → SetFlag(8, MODE2)
AIV:       WaitFlag(8, MODE2)  → HCCL AlltoAll DMA 读 sendBuf_
                                   ↑ KFC Server 独立硬件, 看不到 Cube 写
```

- **CCU 模式**：HCCL 引擎与 Cube 核在同一缓存一致性域，`CrossCoreWaitFlag` 的同步语义隐式包含了 GM 数据可见性保证。
- **AICPU 模式**：KFC Server 是独立硬件单元，与 Cube 核不在同一缓存一致性域。`CrossCoreWaitFlag` 只提供逻辑同步（通知 AIC 计算完成），不保证 AIC 的 GM 写入对 KFC DMA 引擎可见。

### 为什么不同 shape 切换才暴露

不同 shape 的 tile 大小不同，MatMul 写入 sendBuf_ 的数据量不同。同 shape 重复时，sendBuf_ 的缓存状态在连续调用间残留（cache hit），恰好避免了部分不可见的问题；shape 切换后变 cache miss，暴露了 KFC 读不到新数据的问题。

### 为什么 `ASCEND_GLOBAL_LOG_LEVEL=0` 后 PASS

日志 I/O 引入了隐式 pipeline stall / 内存屏障，恰好将 Cube 侧缓存冲刷到 GM。

## 建议修复路径

**通信组件侧**：AICPU KFC Server 在收到 AIC notify 信号、启动 AlltoAll DMA 之前，对 sendBuf 对应的 GM 地址范围执行 cache flush/invalidate，补齐数据可见性。

**算子侧短期规避**：`matmul_reduce_scatter_tiling_base.cpp` 中 AICPU 模式下强制 `isA2APath_=false`，走 ReduceScatter 通路。

## 涉及文件

- `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h` — All2All 通路 kernel
- `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h` — ReduceScatter 通路 kernel
- `mc2/matmul_reduce_scatter_v2/op_host/op_tiling/matmul_reduce_scatter_tiling_base.cpp` — 基类 tiling

## 验证分支

- `debug/a2a-syncall-barrier` — 在 VecWaitCube 后加 SyncAll<true>(), 批跑 PASS
- `debug/a2a-serialize-pipeline` — AIV 流水线串行化, 排除 recvBuf 竞态
- `fix/force-rs-path-aicpu-workaround` — tiling 侧规避, AICPU 强制走 RS 通路
