# Compressor grouped decode x_cache 实现说明

本文记录 Compressor 算子侧 grouped decode x_cache 的接口、host tiling 和 kernel 实现边界。模型侧职责与 metadata 构造见 `cann-recipes-infer/docs/design/grouped_compressor_decode_xcache.md`，本文只描述 `ops-transformer/experimental/attention/compressor/` 内的实现。

## 当前职责边界

grouped decode 的目标是在 decode 阶段将连续 `cmp_ratio` 个 token 的原始 `x` 拼成完整压缩组，再复用原 Compressor 投影、state cache、softmax、reduce、RMSNorm 和 Rope 逻辑。

模型层负责：

- 分配并初始化每个 Attention 层自己的 `x_cache`。
- 每轮 decode 传入本轮 BSH `x`、`x_cache`、`x_cache_slot_mapping`、`start_pos`、`seqused` 和 `x_cache_update`。
- 同一 C4 层内 KV Compressor 与 LI Compressor 使用相同 `x` 和同一份 `x_cache`；KV/HCA 调用负责更新，LI 调用只读。

算子侧负责：

- 根据 `x_cache_slot_mapping` 将本轮有效 decode token 写入 ring cache。
- 根据每个 batch 的 `start_pos` 和 `seqused` 判断是否跨过压缩边界。
- 非压缩轮次只更新 `x_cache`，不执行 cube 投影。
- 压缩轮次从 `x_cache` 读取完整窗口，继续执行 Compressor 原有压缩流程。

## 接口变更

`op_host/compressor_def.cpp` 新增两个可选输入和一个可选属性：

```text
x_cache: optional Tensor, shape [B, cache_len, H], dtype float16/bfloat16
x_cache_slot_mapping: optional Tensor, shape [B] 或 [B, S], dtype int32
x_cache_update: optional bool, default true
```

约束：

- `x_cache` 和 `x_cache_slot_mapping` 必须成对传入；只传其中一个会在 host tiling 校验阶段报错。
- `x_cache` 只用于 BSH decode，即 `x.shape == [B, S, H]`；`S` 可以是普通 decode 的 1，
  也可以是 MTP verification 的 `next_n + 1`。
- `x_cache.dtype` 必须与 `x.dtype` 相同，且
  `cache_len >= cmp_ratio + S - 1`，避免本轮边界后的 token 覆盖刚完成的压缩组。
- `x_cache.shape[0]`、`x_cache_slot_mapping.shape[0]` 必须等于 `B`，`x_cache.shape[2]` 必须等于 `H`。
- `x_cache_update=true` 表示本次调用先写入本轮有效 token；用于 HCA/KV Compressor。
- `x_cache_update=false` 表示本次调用只读取同层已更新的 cache window；用于 LI Compressor，避免同一轮 token 被重复写入。
- `x_cache_update=false` 时必须传入 `x_cache`，不能把只读调用退化为 legacy direct-x 路径。
- prefill 或旧 decode 路径可以不传 `x_cache`，此时 kernel 保持 legacy direct-x 行为。
- `state_cache` 保持 `IgnoreContiguous()`，算子通过 `state_cache_stride_dim0` 支持第 0 维 stride，避免对状态缓存做隐式临时连续化。

`op_kernel/compressor.cpp` 中保留 arch 分流：

- `__CCE_AICORE__ == 220` 的 arch22 路径仍调用旧 kernel Init，不接收 `xCache/xCacheSlotMapping`，不启用本优化。
- arch35 的公共 kernel ABI 包含 `xCache/xCacheSlotMapping`。host 在存在 `x_cache` 时只选择 normal
  模板；Full-Load 仅保留公共 ABI 形参，不实现 grouped decode cache 读写。

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

3. `xCacheUpdateBatchSize` 控制本次调用是否允许写 cache：

   ```text
   xCacheUpdateBatchSize = has_x_cache && x_cache_update ? batchSize : 0
   ```

   KV/HCA 调用写 cache；LI 调用设置为 0，只读 cache。逐 batch 的有效 token 数仍由
   原始 `seqused[b]` 决定。

4. host 对 grouped decode 输入做以下约束：

   ```text
   x.shape == [B, S, H]
   x_cache.shape[0] == batchSize
   x_cache.shape[1] >= cmpRatio + S - 1
   x_cache.shape[2] == hiddenSize
   x_cache.dtype == x.dtype
   x_cache_slot_mapping.shape == [B, S]
   # S == 1 时也接受展平的 [B]
   ```

   `cache_len` 由模型选择，可以等于 `cmp_ratio`，也可以是包含 MTP lookahead 的更深 ring cache。

5. 只读 LI 调用在所有 batch 都未跨压缩边界时，kernel 入口会在创建 `TPipe` 和模板对象前
   直接返回；混合 phase 下只要任一 batch 到达边界，仍进入 normal 模板并按 batch 跳过无效投影。

## Kernel 流程

arch35 `CompressorKernel` 中，`xCache != nullptr` 表示启用 grouped decode x_cache 路径。不存在 dummy x_cache 兼容逻辑。

### 更新 x_cache

`op_kernel/arch35/compressor_block_vec.h` 的 `UpdateXCache()` 在 AIV 侧写入本轮有效 token：

```text
for s in range(seq_used[b]):
    slot = x_cache_slot_mapping[b, s]
dst = slot * hidden + h_start
x_cache[dst : dst + h_deal] = x[b, s, h_start : h_start + h_deal]
```

任务按 `(batch, token, hidden block)` 切分，并用原始 `seq_used[b]` 跳过 BSH padding token。
非压缩轮次也会执行该步骤，保证后续压缩边界到达时 cache 中已有完整窗口。

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

AIV 完成本轮 token 更新后先执行全核同步；cube 随后从 ring cache 中按
`window_start % xCacheSize` 读取整个完整组。窗口跨 ring 尾部时拆成两段拷贝。
投影不再假设本轮 `x` 位于完整组末尾，因此 MTP 在本轮中途跨边界时也沿用同一读取逻辑。
随后复用原有 Matmul、state cache 更新、softmax/reduce、RMSNorm/Rope 流程。

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
  - 对全 inactive 的 LI read-only normal 轮次做入口短路。

- `op_kernel/arch35/compressor_kernel.h`
  - 将 tiling 中的 x_cache 参数写入 `ConstInfo`。
  - update-only 轮次只运行 `UpdateXCache()`。

- `op_kernel/arch35/compressor_block_vec.h`
  - 实现 x_cache 当前 token 写入。
  - state cache 读写继续使用 `stateCacheStrideDim0`。

- `op_kernel/arch35/compressor_block_cube.h`
  - cube 侧从 x_cache 读取完整压缩窗口。

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

workspace-only ACLNN 示例：

```bash
cd /home/jinxh/deepseek-v4/ops-transformer
./build/binary/ascend950/tests/experimental/attention/compressor/test_aclnn_compressor \
  prefill-model-null-decode-update-workspace
```

完整编译和板上验证需按仓库 CANN 环境执行。未实际运行硬件测试时，不应把静态检查或代码阅读结论当作性能/精度结论。
