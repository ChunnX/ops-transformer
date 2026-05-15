# 冗余搬运与物化

当优化涉及 GM/L1/L0/L0C 搬运、resident inverse/result buffer、Fixpipe copy-out、identity/zero tile、layout-copy MMAD 时读取本文件。

## 消费者优先检查

保留或新增 copy 前先回答：

- 下一步谁消费这份数据？
- 消费者从哪个内存层级读取？
- 消费者需要什么 layout、block 顺序和 stride？
- 这次 copy 是搬数据、提供可见性副作用，还是两者都有？
- 消费者能否直接读取已有 resident buffer？

如果唯一理由是“后面旧代码读这个 GM/L1 对象”，先把后面消费者改到目标 resident 源，再把旧物化作为单独 phase 删除。

## 常见冗余形态

- 用 `X @ I` 或 `I @ X` 只为 reshape/copy。
- 同一个 kernel 内先写 GM 再读回，而 L1/L0C resident 可行。
- 写 BF16/FP32 外部 workspace 只是为了沿用旧 readiness/cache 边界。
- 当前计算只消费 diagonal，却搬完整 off-diagonal。
- 同形同 stride block 通过多个局部 helper 重复搬运。
- identity/zero tile 在每个 helper 里重建，而不是在已验证 epoch 内复用。

## Packed 目标规则

Packed execution 本身不是冗余。若目标是 packed block，应保留 packed 目标，删除它周围的重复 scaffolding：

- per-`16x16` load/store 循环；
- layout-copy MMAD replay；
- diagonal-only copy-out 后再 repack；
- 只为跨 helper 桥接而写 GM 的临时物化。

不要因为 shape 相同就合并不同方向的 load。L0A no-transpose 和 L0B transpose 可能需要不同 source layout 与 stride 处理。

优化 matmul operand 搬运时先分层：

- GM->L1 `DataCopy`/`Nd2Nz`：关注 GM readiness、L1 预算、是否可提前 prefetch 以及预取 buffer 生命周期。
- L1->L0 `LoadData`：关注 L0A/L0B 各自所需 layout、transpose、repeatTimes、srcStride/dstGap，以及是否存在完整 B tile 可单条 `LoadData` 覆盖的条件。

不要把 GM->L1 已经一条 `DataCopy` 搬完整 tile，误判成 L1->L0 也没有优化空间；也不要把 L1->L0 多条 `LoadData`，误判成 GM->L1 没有 prefetch 空间。

## Identity/Zero 生命周期

identity/zero tile 应按证明过的生命周期准备：

- kernel object/init epoch：buffer 不会被 reset 或复用；
- raw-MMAD local-buffer epoch：`pipe_->Reset()` 或 alias 可能让内容失效；
- batch/para epoch：存储会被覆盖或更长生命周期未证明。

不要在未检查 buffer 分配域、alias、reset 边界前加一个“already initialized” flag。

## 通用教训

- 更大的 preload 可降低 MTE2 replay，但如果带来 scalar cache clean、setup、sync 开销，端到端仍可能变慢。
- 只搬下一消费者需要的 lane，往往比搬完整 tile 更好；full-tile 搬运必须有明确消费者。
- GM 写回即使不是数学输入，也可能承载可见性、顺序或生命周期副作用；删除前要先拆清数据依赖和副作用。
