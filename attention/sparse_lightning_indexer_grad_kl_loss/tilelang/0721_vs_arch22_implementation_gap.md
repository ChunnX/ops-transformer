# 0721 TileLang 与 arch22 AscendC 实现对照

## 范围

本文对比当前精度基线 `0721.py` 与 A3 上官方 `arch22` AscendC
实现。官方源码主要位于：

- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector2.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h`

目标不是逐字替换 API，而是在 TileLang-Ascend 实际可表达的范围内保持
数学、数据所有权和关键流水一致。生成的 `0721_kernel_source.cpp` 是最终
执行语义的依据。

## 当前 0721 的实现

1. 调度按 `(B, S1)` task 分配到 `launch_core_num`，核心数由运行时参数
   提供，不写死为 24。
2. V0 在两个 AIV 上分片 gather Key、KeyRope、KeyIndex；C1 完成 BMM1/BMM2。
3. V1 保留官方的职责切分：AIV0 计算 PReduced，AIV1 计算 ReduceSumY 和
   SoftmaxI。二者写入 GM workspace，随后 V1 backward 的两个 AIV 各处理
   一半 `Nidx1` 行的 dWeight/reluGrad/Loss。
4. C2 以显式 `T.mma` 完成 dKey partial 与 dQuery accumulation。L1 到 L0
   的转置在 copy 时完成。
5. V2 已按 16 行批量读取 `dk_matmul_partial`，两个 UB slot 交替服务，
   但因 sparse index 目标地址一般不连续，仍逐行 atomic add 到
   `dKeyIndexAccum`。dKey 最终转换也为 16 行批量。Loss 在 kernel 内归约。
6. 关闭 `TL_ASCEND_AUTO_SYNC`，用显式 RAW/WAR 和 cross-core flag 管理。

## 与官方的主要差异

| 区域 | 官方 arch22 | 当前 0721 | 原因与影响 |
| --- | --- | --- | --- |
| V1 backward ping-pong | 使用 `mm1Tbuf/mm2Tbuf` 原生 `LocalTensor` 队列，FP32 输入和 FP16 reluGrad 在确定的 ping/pong 物理地址上复用 | 使用单 `32x64` BMM2 UB 路径 | TileLang 的静态 UB planner 可重新分配 `alloc_ub` 的物理地址，且没有暴露官方队列的 no-alias/固定地址语义。此前双 slot 版本即使串行化输入仍破坏精度，不能保留。 |
| ReLU gradient mask | Compare/Select | `Muls(-10000) -> Exp -> 1-Exp` 近似 | 当前 TileLang-Ascend 版本的 Compare/Select 路径曾出现精度或性能回退；保留已验证的近似链。 |
| V0 gather | 官方 `DataCopyPad`、队列和 C++ 标量访问 | TileLang 标量 sparse index 加显式 GM/UB copy | TileLang 没有等价、已验证的 queue/DataCopyPad 组合。当前实现已使用双 slot，但指令形态仍不同。 |
| Cube service | 官方封装的 TPipe/queue 服务 | 显式 L1/L0A/L0B/L0C、`T.mma`、flag | 这是有意选择。`gemm_v0` 会引入内部同步，无法与全手动 event 协议安全混用。 |
| V2 scatter | 官方可使用本地 scatter buffer 与实现私有的搬运接口 | 16 行 GM->UB ping-pong 后逐行 `T.tile.atomic_add` | TileLang V1 atomic add 只支持 local tensor 到 GM，且 region rank 必须精确；稀疏目标不连续时不能安全合成一个多行 atomic region。 |
| 跨 task 流水 | 官方通过 run-info、task ratio 和 queue 实现三级重叠 | 0721 使用 task slot、cross flag 和固定深度调度 | 小 shape 的 `B*S1=1` 没有相邻 task 可重叠，结构收益只会在多 task shape 显现。 |
| 最终输出 | 官方服务内完成 dWeight/Loss/dKey 输出 | 0721 已在 kernel 内完成 | 这一项已基本对齐。 |

## 为什么不能直接一比一翻译

### 1. UB 地址不是 TileLang 源码可指定的资源

官方 AscendC 通过 `TPipe::InitBuffer` 和 `GetWithOffset` 控制本地队列及其
物理地址。TileLang 的 `T.alloc_ub` 只声明逻辑 buffer，实际地址由编译器
根据其推断的生命周期分配。异步 MTE3 读取尚未结束时，planner 对 FP32 和
FP16 buffer 的 coalesce 可能与源码直觉不一致。

因此，只有在生成 CPP 中确认物理 offset、容量和 event 生命周期后，buffer
复用才可接受。V1 backward 的双 slot 试验曾出现 NaN 和大误差，串行化
MTE2 预取后仍错误，已证明不能用补 barrier 的方式模拟官方队列语义。

### 2. V1 UB 容量决定 staging 方式

精度基线生成 CPP 的 V1 UB 峰值为 `196352 B`，A3 的 192 KiB UB 上限为
`196608 B`，余量仅 `256 B`。当前关键区域为：

```text
v1_bw_bmm2      FP32 32x64  = 8192 B, offset 20992
v1_bw_diff_2d   FP32 32x64  = 8192 B, offset 29184
v1_bw_mask      FP32 32x64  = 8192 B, offset 37376
v1_bw_rg        FP32 32x64  = 8192 B, offset 45568
v1_bw_rg_half   FP16 32x64  = 4096 B, offset 20992 (与 bmm2 生命周期复用)
```

基线中直接把 BMM2 扩成 `32x128` 会额外需要 8192 B，并覆盖 `diff_2d` 的
物理区间。当前实验版本因此同时移除了独立的 `diff_2d`：`rg` 先保存
column-wise broadcast 的 `diff`，dWeight 规约完成后再原地乘 `weight_2d` 与
ReLU mask。该变换释放 8192 B，同时消除一次 `weight_2d -> rg` UB copy，
使 BMM2 可以安全地按 `32x128` 一次搬入、按两个 `32x64` 子块计算。

这个方案仍须以新生成 CPP 为准：`v1_bw_bmm2` 必须为 4096 个 FP32 元素，
`v1_bw_rg_half` 只能覆盖已消费的前半输入，且下一次整块 BMM2 覆盖前必须
等待对应 MTE3 写回完成。

### 3. TileLang intrinsic 约束不同

- 归约 fast path 对输入/输出 shape 有额外限制。
- 部分 vector 参数以 `uint8_t` 编码；例如 repeat=256 会截断为 0。
- Vector 不进行 FP16 算术；需要 FP32 vector 运算并用 `CAST_RINT` 输出。
- `atomic_add` 不接受任意 rank 或任意 local/global 方向。
- 某些高层 intrinsic 会插入内部 event，不能与外部手动同步混用。

## 已验证的取舍

保留：显式 MMA、copy 时转置、手动 flag、P/SY AIV 分工、V2 16 行批量
scatter、kernel 内 dWeight/Loss/dKey 完成。

放弃：V1 backward 双 slot ping-pong、直接 Compare/Select mask、单独把
`block_n` 改为 128、dQuery finalizer 的粗粒度 16 行批处理。后者虽然精度
正确，但破坏原单行 MTE2/V/MTE3 的细粒度滚动流水，端到端性能从约 0.58x
退化到约 0.53x，已回退。

## 后续优化前提

1. 先用 profile 确认 V1 AIV1/SY、V1 backward、C2 或 V2 atomic 哪一个是
   当前关键路径。
2. `32x128` BMM2 staging 已与 `diff_2d` 消除绑定实现；必须检查生成 CPP
   的所有 `GetWithOffset`、两个 64 列 slice 的 copy 与 MTE3/MTE2 生命周期。
3. 任何重新引入 ping-pong 的尝试必须先有可证明的物理隔离策略，不能仅靠
   增加 `set_flag`/`wait_flag`。
4. 每次只改一个结构变量，执行多次精度和中位数性能测试，再保留或回退。
