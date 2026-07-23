# Compressor grouped decode x_cache 实现说明

本文记录 Compressor 算子侧 grouped decode x_cache 的接口、host tiling 和 kernel 实现边界。模型侧职责与 metadata 构造见 `cann-recipes-infer/docs/design/grouped_compressor_decode_xcache.md`，本文只描述 `ops-transformer/experimental/attention/compressor/` 内的实现。

## 当前职责边界

grouped decode 的目标是在 decode 阶段将连续 `cmp_ratio` 个 token 的原始 `x` 拼成完整压缩组，再复用原 Compressor 投影、state cache、softmax、reduce、RMSNorm 和 Rope 逻辑。

模型层负责：

- 分配并初始化每个 Attention 层自己的 `x_cache`。
- 每轮 decode 传入当前 token `x`、`x_cache`、`x_cache_slot_mapping`、`start_pos`、`seqused` 和 `x_cache_update`。
- HCA Compressor 层使用各自独立的 `x_cache`。同一 KV/LI 层内，KV
  Compressor 与 LI Compressor 使用相同 `x` 和同一份 `x_cache`：HCA/KV
  调用是各自 cache 的写入方，LI 调用只读同层 KV 已更新的 cache。

算子侧负责：

- 根据 `x_cache_slot_mapping` 将当前 decode token 写入 ring cache。
- 根据每个 batch 的 `start_pos` 和 `seqused` 判断是否跨过压缩边界。
- 非压缩轮次只更新 `x_cache`，不执行 cube 投影。
- 压缩轮次从 `x_cache` 读取完整窗口，继续执行 Compressor 原有压缩流程。

## 接口变更

`op_host/compressor_def.cpp` 新增两个可选输入和一个可选属性：

```text
x_cache: optional Tensor, shape [B, cache_len, H], dtype float16/bfloat16
x_cache_slot_mapping: optional Tensor, shape [B, 1] 或 [B], dtype int32
x_cache_update: optional bool, default true
```

约束：

- `x_cache` 和 `x_cache_slot_mapping` 必须成对传入；只传其中一个会在 host tiling 校验阶段报错。
- `x_cache_update=true` 表示本次调用先写入当前 token；用于 HCA/KV Compressor。
- `x_cache_update=false` 表示本次调用只读取同层已更新的 cache window；用于 LI Compressor，避免同一 token 被重复写入。
- prefill 或旧 decode 路径可以不传 `x_cache`，此时 kernel 保持 legacy direct-x 行为。
- `state_cache` 保持 `IgnoreContiguous()`，算子通过 `state_cache_stride_dim0` 支持第 0 维 stride，避免对状态缓存做隐式临时连续化。

`op_kernel/compressor.cpp` 中保留 arch 分流：

- `__CCE_AICORE__ == 220` 的 arch22 路径仍调用旧 kernel Init，不接收 `xCache/xCacheSlotMapping`，不启用本优化。
- arch35 路径将 `xCache/xCacheSlotMapping` 传入 `CompressorKernel` 和 `CompressorKernelFullLoad`。

## Host tiling

arch35 tiling 在 `op_host/arch35/compressor_tiling.cpp` 中处理 grouped decode：

1. 读取可选输入：
   - `xCache`
   - `xCacheSlotMapping`
   - `xCacheUpdate`

2. BSH decode 且存在 `x_cache` 时，将逻辑计算长度设为 `cmp_ratio`：

   ```text
   xCacheReadSize = has_x_cache ? cmp_ratio : 0
   computeSeqSize = has_x_cache ? xCacheReadSize : seqSize
   ```

   这样 decode 的物理 `x` 仍是 `[B, 1, H]`，但压缩流程看到的是一个完整的 `[B, cmp_ratio, H]` 窗口。

3. `xCacheUpdateBatchSize` 控制是否写当前 token：

   ```text
   xCacheUpdateBatchSize = has_x_cache && x_cache_update ? batchSize : 0
   ```

   KV/HCA 调用写 cache；LI 调用设置为 0，只读 cache。

4. shape 校验只固定 `x_cache` 的 batch 和 hidden：

   ```text
   x_cache.shape[0] == batchSize
   x_cache.shape[2] == hiddenSize
   ```

   中间维 `cache_len` 由模型选择，可以是 `cmp_ratio` 或更深的 ring cache。

5. arch35 grouped decode 可以选择 Full-Load 模板。除原有 BSH、物理
   `seq_size <= 4` 和逻辑 `token_size <= 256` 条件外，grouped Full-Load
   还要求 batch 为不超过 AIV 核数的 2 的幂。该约束保证 Full-Load vector
   的静态分组仍是一组对应一个 batch，使不同 batch 的压缩 phase 可以独立跳过。
   不满足条件时继续使用 Normal 模板。

   常用场景中：

   - C4、BS=16 的逻辑 `token_size=16*4=64`，进入 Full-Load。
   - C128、BS=16 的逻辑 `token_size=16*128=2048`，回退 Normal。
   - C128、BS=1/2 的逻辑 token 数不超过 256，可以进入 Full-Load。

## Kernel 流程

arch35 的 Normal `CompressorKernel` 和 `CompressorKernelFullLoad` 都以
`xCache != nullptr` 表示启用 grouped decode x_cache 路径。不存在 dummy
x_cache 兼容逻辑。

### 更新 x_cache

Normal 路径的 `compressor_block_vec.h` 与 Full-Load 路径的
`compressor_block_vec_full_load.h` 都在 AIV 侧执行当前 token 写入：

```text
slot = x_cache_slot_mapping[b]
dst = slot * hidden + h_start
x_cache[dst : dst + h_deal] = x[b, 0, h_start : h_start + h_deal]
```

任务按 `(batch, hidden block)` 切分。非压缩轮次也会执行该步骤，保证后续压缩边界到达时 cache 中已有完整窗口。

### 判断压缩边界

`op_kernel/arch35/compressor_tools.h` 中按 batch 判断是否激活压缩：

```text
active = (start_pos + seq_used) / cmp_ratio > start_pos / cmp_ratio
```

每个 batch 可以有不同 decode position；只要该 batch 跨过压缩边界，就使用完整 `x_cache` 窗口参与投影。未跨边界时 `GetSeqUsed()` 返回 0，该 batch 不执行投影。
这里的 `seq_used` 是模型传入的原始 query 长度，非 MTP decode 通常为 1，MTP decode
通常为 `next_n + 1`；它不再是模型侧构造的 `0/ratio` active flag。kernel 内部
`GetSeqUsed()` 会把原始长度转换成逻辑压缩长度：active batch 返回 `xCacheReadSize`，
inactive batch 返回 0。`compressor_kernel.h` 中按尾部 `GetSeqUsed()==0` 裁剪
`constInfo.projectionBatchSize`，它只表示后续投影/压缩工作的 batch 上界，
不改变 host tiling 记录的输入 shape batch。当前 token 写入由
`xCacheUpdateBatchSize` 控制。

### 读取完整窗口

压缩轮次读取窗口起点：

```text
complete_group_end = floor((start_pos + seq_used) / cmp_ratio) * cmp_ratio
window_start = complete_group_end - xCacheReadSize
```

kernel 从 ring cache 中按 `window_start % xCacheSize` 读取，窗口跨 ring 尾部时拆成两段拷贝。随后复用原有 Matmul、state cache 更新、softmax/reduce、RMSNorm/Rope 流程。

### Full-Load 的静态 M 处理

Full-Load 的 host tiling 和 workspace 按输入的完整 `B * ratio` 静态展开 M
轴，因此 kernel 不裁剪 `projectionBatchSize`。kernel 另行扫描各 batch 的
`GetSeqUsed()`，判断本轮是否存在投影工作：

- 所有 batch 都未跨边界时，AIV 只执行 `UpdateXCache()`，AIC 不进入投影。
- 存在 active batch 时，Cube 从每个 batch 的 ring cache 与当前 `x` 组合出
  完整逻辑窗口。
- Vector 根据每个 batch 的逻辑 `GetSeqUsed()` 跳过 inactive batch 的
  state cache 与输出处理。

这样既保留 Full-Load 既有的静态 M/workspace 布局，也支持一个 batch 内各
序列 decode phase 不一致的场景。

## state cache 语义

`state_cache` 仍是 Compressor 的状态输入，kernel 会原地读写。算子通过 `state_cache_stride_dim0` 计算第 0 维块偏移：

```text
state_offset = block_id * stateCacheStrideDim0 + inner_offset
```

因此 OpDef 不应对 `state_cache` 使用 `AutoContiguous()`。如果框架为状态缓存生成临时连续副本，可能破坏写回原 cache 的语义。调用侧需要保证第 1/2 维连续，并正确传入第 0 维 stride。

## 代码路径

- `op_host/compressor_def.cpp`
  - 注册 `x_cache`、`x_cache_slot_mapping`、`x_cache_update`。
  - 保持 `state_cache.IgnoreContiguous()`。

- `op_host/compressor_proto.cpp`
  - 维护新增 optional input 和 attr 的索引。
  - 输出 shape 仍只推导 `cmp_kv`。

- `op_host/arch35/compressor_tiling.cpp`
  - 读取 optional x_cache 参数。
  - 校验 x_cache 与 slot mapping 成对出现。
  - 生成 `xCacheReadSize`、`xCacheUpdateBatchSize`、`xCacheSize`。

- `op_kernel/compressor.cpp`
  - arch22 走旧 Init。
  - arch35 传入 xCache/xCacheSlotMapping。

- `op_kernel/arch35/compressor_kernel.h`
  - 将 tiling 中的 x_cache 参数写入 `ConstInfo`。
  - update-only 轮次只运行 `UpdateXCache()`。

- `op_kernel/arch35/compressor_block_vec.h`
  - 实现 x_cache 当前 token 写入。
  - state cache 读写继续使用 `stateCacheStrideDim0`。

- `op_kernel/arch35/compressor_block_cube.h`
  - cube 侧从 x_cache 读取完整压缩窗口。

- `op_kernel/arch35/compressor_kernel_full_load.h`
  - 在 Full-Load 中区分 x_cache 更新工作与投影工作。
  - update-only 轮次只初始化并运行 AIV 更新路径。

- `op_kernel/arch35/compressor_block_vec_full_load.h`
  - 实现 Full-Load 的 x_cache 当前 token 写入。
  - 按 batch 跳过未跨压缩边界的 state cache 与输出处理。

- `op_kernel/arch35/compressor_block_cube_full_load.h`
  - 从 ring cache 与当前物理 `x` 组合 Full-Load 的静态 M 输入。

- `op_kernel/arch35/compressor_tools.h`
  - 按 batch 判断 grouped decode 是否跨压缩边界。
  - 为 active batch 返回对齐后的窗口起点和 `xCacheReadSize`。

## 验证入口

静态检查：

```bash
cd /home/jinxh/deepseek-v4
git -C ops-transformer diff --check
```

单算子 pytest：

```bash
cd /home/jinxh/deepseek-v4/ops-transformer/experimental/attention/compressor/tests/pytest
bash run_grouped_decode_update_only.sh
bash run_grouped_decode_compress.sh
```

现有 grouped decode 用例的 shape 同时覆盖 Full-Load 新分支：

- update-only 的 C4/C128、BS=2 用例进入 Full-Load。
- active read 的 C4/C128、BS=1 用例进入 Full-Load，并与 legacy full-x
  输出及 state cache 对比。
- 模型常用 C128、BS=16 仍由 Normal 路径覆盖。

workspace-only ACLNN 示例：

```bash
cd /home/jinxh/deepseek-v4/ops-transformer
./build/binary/ascend950/tests/experimental/attention/compressor/test_aclnn_compressor \
  prefill-model-null-decode-update-workspace
```

完整编译和板上验证需按仓库 CANN 环境执行。未实际运行硬件测试时，不应把静态检查或代码阅读结论当作性能/精度结论。
