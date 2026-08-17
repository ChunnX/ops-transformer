# 核间同步设计

本文定义 `QkvRmsNormRopeCacheWithKScale` 当前 arch35 kernel 的 AIC/AIV CrossCore 同步规则。代码路径为：

```text
ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_kernel/arch35/
```

执行模型固定为：

```text
KERNEL_TYPE_MIX_AIC_1_2
1 AIC + 2 AIV per group
```

## 1. 核组与身份

核组关系：

```text
AIC cubeIndex = GetBlockIdx()
AIV cubeIndex = GetBlockIdx() / 2
AIV local id  = GetSubBlockIdx()
```

约定：

```text
GetSubBlockIdx() == 0 -> AIV0
GetSubBlockIdx() == 1 -> AIV1
```

两个 AIV 对同一个 token tile 做连续 token 切分。tail tile 可以不均衡，但 padding token 不进入后处理有效写回。

## 2. Flag 语义

核间同步使用 `CrossCoreSetFlag/CrossCoreWaitFlag`。

当前规则：

1. flag id 直接硬编码。
2. flag id 不按 `bufferId`、tile id 或 generation 派生。
3. 同一个 id 可以跨 tile 反复 `Set/Wait`，依赖固定循环顺序和计数配平保证正确性。
4. AIV0 使用基础 id；AIV1 使用 `基础 id + QKV_K_SCALE_AIV1_FLAG_OFFSET`，当前 offset 为 16。
5. AIC 等待 AIV->AIC 方向时，对同一基础 id wait 两次，分别消费 AIV1/AIV0 的事件。
6. AIC->AIV 方向由 AIC set 两次，每个 AIV 侧 wait 自己对应的事件。

## 3. Flag ID

当前代码中的 flag id：

```cpp
constexpr uint64_t SYNC_A_READY = 0;
constexpr uint64_t SYNC_FIX_OUTPUT_READY = 1;
constexpr uint64_t SYNC_FIX_OUTPUT_CONSUMED = 2;
constexpr uint64_t SYNC_A_CONSUMED = 3;
constexpr uint64_t SYNC_MROPE_K_CONSUMED = 4;
```

五个 id 均在当前 arch35 CrossCore flag 基础 id 范围内。当前实现直接使用语义常量，不再保留针对这些常量的额外 `static_assert`。

这些常量定义在 `common.h`，CrossCore set/wait 语义函数集中在 `basic_block.h`。`common.h` 不保留额外的
`SetAicToAiv`/`WaitAivToAic` 公共封装层，也不保留针对 flag 范围的 common 级 static_assert。

## 4. 同步边

### 4.1 A_CONSUMED

```text
data      = A_rot AL1 logical buffer reusable credit
location  = L1
producer  = AIC MTE1 side after consuming A_rot
consumer  = AIV MTE3 before writing next A_rot
direction = AIC -> AIV
```

同步 pipe：

```cpp
AIC: SetAicMte1ToAivMte3AConsumed();
AIV: WaitAicMte1ToAivMte3AConsumed();
```

当前流程：

1. AIC 在 `PrepareBeforeLoop()` 中循环 owner-local `A_ROT_L1_BUFFER_NUM` 次调用 `SetAicMte1ToAivMte3AConsumed()`，发出两个初始 credit。该槽数由 AL1 pool/logical slot 容量局部推导。
2. AIV 每次前处理前先 wait 一个 `A_CONSUMED`，确认当前 AL1 logical buffer 可写。
3. AIC 每完成一个 tile 的 AL1 消费后 set 一个新的 `A_CONSUMED`。
4. AIV 在 `EndAiv()` 中循环 `A_ROT_L1_BUFFER_NUM` 次调用 `WaitAicMte1ToAivMte3AConsumed()`，配平初始/剩余 credit。

`A_CONSUMED` 是当前实现避免 AL1 double buffer 复用错位的关键同步边。

### 4.2 A_READY

```text
data      = BasicBlock 传入的 aRotL1Nz
location  = L1
producer  = AIV0/AIV1 MTE3
consumer  = AIC MTE1/Cube
direction = AIV -> AIC
```

同步 pipe：

```cpp
AIV: SetAivMte3ToAicMte1AReady();
AIC: WaitAivMte3ToAicMte1AReady();
```

AIV 必须在 MTE3 写 AL1 完成后再 set `A_READY`。AIC wait 两个 AIV 的 `A_READY` 后再启动当前 tile 的 Q/K matmul。AL1 buffer 由 `BasicBlock::aRotL1BufferUseId_` 与 owner-local `A_ROT_L1_BUFFER_NUM` 选择，不来自 token tile id 或 `TileParam` ring id。

### 4.3 FIX_OUTPUT_READY

```text
data      = current scene's Fixpipe output
location  = AIV UB
producer  = AIC FIX
consumer  = AIV PIPE_V
direction = AIC -> AIV
```

同步 pipe：

```cpp
AIC: SetAicFixToAivOutputReady();
AIV: WaitAicFixToAivOutputReady();
```

RoPE 中，AIC Fixpipe 使用 `dualDstCtl = 1`、`subBlockId = false`，按 M 轴把 Q/K segment 分发到两个
AIV 的本地 output DB slot。M-RoPE 中，同一正向边发布独立 K rotation slot，Q 由后续 Fixpipe 直接写 GM。
两个场景复用同一个固定语义 flag；旧的独立 `MROPE_K_READY` flag 和封装函数已经删除。删除重复 flag
不等于删除 FIX->V 数据可见性同步。

### 4.4 FIX_OUTPUT_CONSUMED（RoPE）

```text
data      = output DB pool reusable credit
location  = AIV UB
producer  = AIV MTE3 after Q/K postprocess writes are complete
consumer  = AIC FIX before reusing output path
direction = AIV -> AIC
```

同步 pipe：

```cpp
AIV: SetAivMte3ToAicFixOutputConsumed();
AIC: WaitAivMte3ToAicFixOutputConsumed();
```

RoPE output DB 由 BasicBlock 根据 pool/slot 容量局部推导出 2 个物理槽。AIV `PrepareBeforeLoop` 先 set 两个
`FIX_OUTPUT_CONSUMED` 初始 credit；AIC 每个 tile 覆盖当前 output slot 前 wait 一个 credit；AIV 在 tile 内
或 End 中完成对应 Q/K 后处理后返还一个 credit；AIC End 再 drain 两个 credit。当前实现不再有一份
output credit，也没有 `isLastTile` 的提前返还特判。

### 4.5 MROPE_K_CONSUMED（M-RoPE）

M-RoPE 的 direct-Q 实例不把 Q 输出放入 shared output DB，而是由 Fixpipe 直接写 `q_out` GM。K 的
正向发布复用第 4.3 节的 `FIX_OUTPUT_READY`，反向复用使用独立 `MROPE_K_CONSUMED`：

```text
FIX_OUTPUT_READY:
  AIC FIX set flag 1 -> AIV V wait flag 1
  语义：当前 K rotation FP32 slot 已可读

MROPE_K_CONSUMED:
  AIV V set flag 4 -> AIC FIX wait flag 4，2 个初始 credit
  语义：K rotation 已被 Stage 完整读取，可以复用对应 rotation slot
```

compact 的两个 CrossCore credit 对应 K rotation 双缓冲。V 完成 K rotation 读取和 K quant staging 后即可
返还 rotation credit；随后在同一控制迭代立即发起 K scatter。K quant 自身仍为双缓冲，其 slot 复用由
逐 slot 的 `MTE3 -> V` reusable event 保护，必须等 K cache/K scale 两路 MTE3 写回完成后才能返还。
flag 1/4 固定不随 tile、buffer 或 generation 派生。

## 5. Tile 级时序

下列 AIC/AIV 时序以 RoPE 为示例；M-RoPE 复用相同 `FIX_OUTPUT_READY` 正向边，并将反向
`FIX_OUTPUT_CONSUMED` 替换为第 4.5 节的 `MROPE_K_CONSUMED`，其余 A 路握手不变。

AIC：

```text
PrepareBeforeLoopAic:
  1. cube.PrepareRotationBeforeLoop()
  2. set two A_CONSUMED credits

for tile i:
  1. select aRotL1Nz by aRotL1BufferUseId_
  2. select output slot by outputBufferUseId_
  3. wait one FIX_OUTPUT_CONSUMED credit
  4. wait A_READY from AIV0/AIV1
  5. cube.ComputeTile(tile i, aRotL1Nz, qkAfterCubeUb)
  6. set A_CONSUMED
  7. set FIX_OUTPUT_READY
  8. advance the A_rot and output resource counters independently

EndAic:
  drain two FIX_OUTPUT_CONSUMED credits
```

AIV：

```text
PrepareBeforeLoopAiv:
  1. vec.PrepareBeforeLoop()
  2. set two initial FIX_OUTPUT_CONSUMED credits

for tile i:
  1. select aRotL1Nz by aRotL1BufferUseId_
  2. select outputUb by outputBufferUseId_
  3. wait A_CONSUMED
  4. vec.ComputeTile(tile i, aRotL1Nz, outputUb, outputBufferId)
     - wait input UB reusable
     - quant V and signal V-cache source ready
     - build Q/K RoPE BF16/NZ in UB
     - MTE3 write V cache
     - MTE3 write Q/K BF16/NZ to AL1
     - release input UB reusable
  6. set A_READY
  7. if hasPreviousTile_:
       select previous output slot from the output resource counter
       wait FIX_OUTPUT_READY of tile i-1
       vec.PostprocessQk(tile i-1, qkAfterCubeUb, outputBufferId)
       set FIX_OUTPUT_CONSUMED
  8. advance the A_rot and output resource counters independently
  9. hasPreviousTile_ = true

EndAiv:
  if hasPreviousTile_:
    wait FIX_OUTPUT_READY of last tile
    vec.PostprocessQk(last tile, qkAfterCubeUb, outputBufferId)
    set FIX_OUTPUT_CONSUMED
  drain two A_CONSUMED credits
```

M-RoPE 保留相同的一拍 AIC/AIV rolling：控制迭代 `i` 先完成当前 tile 的 preprocess/A_READY，再处理
`lastTile=i-1`。上一拍存在时，其 K 顺序固定为：

```text
wait FIX_OUTPUT_READY(lastTile)
kQuantBufferId = StageMropeK(lastTile)
set MROPE_K_CONSUMED                 # 返还 rotation credit
ScatterMropeK(lastTile, kQuantBufferId)
```

`Stage` 与 `Scatter` 位于同一控制迭代，不额外保存跨迭代的 K tile 状态。End 对最后一个 `lastTile`
执行同一序列。`ScatterMropeK` 通过 K quant slot 自身的 V->MTE3 ready / MTE3->V reusable event 保证两路
GM 写回完成后才允许该 slot 再次量化。

### 5.1 Set/Wait 计数配平

设一个核组处理 `N` 个 tile。下表按单个 AIV 的事件流统计；AIC 对 AIV0/AIV1 的实际 CrossCore 指令数是对应单 AIV 计数的两倍：

| Flag | 单个 AIV Set | 单个 AIV Wait | AIC 对应操作 | 配平依据 |
|------|-------------:|--------------:|--------------|----------|
| `A_READY` | `N` | `0` | 每 tile 对两个 AIV 各 Wait 一次 | 每个 tile 固定一次 ready。 |
| `FIX_OUTPUT_READY` | `0` | `N` | 每 tile 对两个 AIV 各 Set 一次 | RoPE Q/K output 与 M-RoPE K rotation 共享该固定正向边。 |
| `A_CONSUMED` | `0` | `N + 2` | 每个 AIV 在 loop 前获得 2 个初始 credit，之后每 tile 返还 1 个 | `N` 次循环 Wait 加 End 的 2 次 drain。 |
| `FIX_OUTPUT_CONSUMED` | RoPE: `N + 2` | `0` | AIC 对应 `N` 次循环 Wait 加 End 的 2 次 drain | 2 个初始 credit 加每 tile 最终返还 1 个 credit。 |
| `MROPE_K_CONSUMED` | M-RoPE: `N + 2` | `0` | AIC 对应 `N` 次循环 Wait 加 End 的 2 次 drain | compact 的 2 个初始 credit，配平 K rotation 生命周期；K quant 复用由核内 event 管理。 |

`N = 0` 时，`A_CONSUMED` 的两个初始 credit，以及当前场景的两个 output/K-rotation credit 仍分别在 End 阶段
被消费，不会遗留事件。tail 或 AIV1 empty slice 只跳过数值计算，不改变上述计数。

## 6. Buffer 复用口径

当前不再使用统一的：

```text
bufferId = tileId % 2
```

来选择所有数据 buffer。buffer 选择由 owner 自己维护：

| 资源 | 选择方式 |
|------|----------|
| TileParam ring | `Controller::tileParamBufferUseId % TILE_PARAM_BUFFER_NUM`；Controller 局部固定两槽 |
| A_rot AL1 logical buffer | `BasicBlock::aRotL1BufferUseId_ % A_ROT_L1_BUFFER_NUM`；由 pool/slot 容量局部推导 |
| RoPE/M-RoPE input UB | `Vec::inputBufferUseId_ % INPUT_BUFFER_NUM`；按场景由 pool/slot 容量局部推导为 2/4 |
| RoPE metadata / M-RoPE raw cos/sin | `Vec::metadataBufferUseId_ % METADATA_BUFFER_NUM`；按场景局部推导为 2/4 |
| RoPE output DB | `BasicBlock::outputBufferUseId_ % ROPE_OUTPUT_BUFFER_NUM`；局部推导 |
| M-RoPE output DB | `BasicBlock::outputBufferUseId_ % MROPE_OUTPUT_BUFFER_NUM`；局部推导 |
| RoPE/M-RoPE V output UB | `Vec::vOutBufferUseId_ % V_OUT_BUFFER_NUM`；按场景局部推导，当前均为 2 |
| M-RoPE K rotation DB | `BasicBlock::kRotationBufferUseId_ % MROPE_K_ROTATION_BUFFER_NUM`；局部推导 |
| M-RoPE K quant DB | `Vec::kQuantBufferUseId_ % K_QUANT_BUFFER_NUM`；局部推导 |
| L0C | `Cube::l0cBufferUseId_ % AIC_L0C_BUFFER_NUM` |

`QKV_K_SCALE_DOUBLE_BUFFER_NUM` 以及按每类资源扩散的 common 全局 `*_BUFFER_NUM` 已删除。common 保留
pool/slot bytes、offset 和 elements，owner 类根据容量局部得到槽数。每个 physical buffer id 都只能从该
资源自己的 use counter 通过 `% BUFFER_NUM` 生成，不依赖槽数为 2 的幂，不能依赖 token tile id，也不能
借用其他 pool 的 counter。空 M-RoPE slice 不占 K quant slot，因此不推进
`kQuantBufferUseId_`；它仍参与 `FIX_OUTPUT_READY/MROPE_K_CONSUMED` 的固定 CrossCore 协议。

Controller 的 `GetCurrentTileParamBufferId/GetPreviousTileParamBufferId` 只使用
`tileParamBufferUseId` 管理两槽 `TileParam` ring，不代表任何数据 buffer id。BasicBlock 的
`hasPreviousTile_` 只表达 pipeline 是否有上一拍，也不参与 slot 选择。

## 7. 正确性检查

review 时重点检查：

1. `A_CONSUMED` 初始 credit、tile 内 set/wait、End drain 是否配平。
2. AIV set `A_READY` 前，AL1 MTE3 写入是否已由核内 event 保证完成。
3. AIC set `FIX_OUTPUT_READY` 前，RoPE output DB 或 M-RoPE K rotation slot 是否完成 Fixpipe 发布。
4. RoPE 下 AIV set `FIX_OUTPUT_CONSUMED` 前，Q/K postprocess 的 GM 写回是否完成；M-RoPE compact 下确认
   `MROPE_K_CONSUMED` 的返还时机。
5. `SYNC_*` 基础 id 与 `QKV_K_SCALE_AIV1_FLAG_OFFSET` 偏移后的 AIV1 id 不冲突。
6. tail 中 AIV1 token 数可能为 0，但仍必须参与固定 CrossCore Set/Wait。

## 8. `MROPE_BF16_Q_INT8_K` 同步约束

M-RoPE/BF16-Q/INT8-K 复用 flag 1 的 `FIX_OUTPUT_READY`，并使用 flag 4 的
`MROPE_K_CONSUMED`。当前唯一的 compact direct-Q 路径使用两个 consumed credit：

1. M-RoPE 调用中 `query_start_loc/seq_lens` GM anchor 为空且由编译期模板消除读取；mrope_position、T/H/W raw
   cos_sin 和 Gather 都是 AIV 本地前处理，不引入额外跨核数据边。
2. `A_READY` 仍只在 Q/K BF16 NZ 已写入 AL1 后发送；`FIX_OUTPUT_READY` 在 AIC Fixpipe 发布 K rotation 后发送；
   `mrope_section[0]` 不影响同步路径。
3. `q_quant_mode=NoQuant` 的 BF16 模式不计算或写出 q_scale，Q 由 Fixpipe F32->BF16 直写 GM，旧 Vec
   QCast 路径已删除。compact 的 `MROPE_K_CONSUMED` 只表达 K rotation 已被 V 读取且 K quant staging
   已完成；BasicBlock 在返还该 rotation credit 后、同一控制迭代立即调用 `ScatterMropeK`。K quant slot
   由核内 reusable event 保护，直到 K cache/K scale 两路写回都完成。
4. compact 的 `MROPE_K_CONSUMED` 在 V 侧返还两个初始 credit 对应的 slot。表中的 “QK scale staging” 按
   `Q_QUANT_MODE/K_QUANT_MODE` 模板解释：
   `(PER_TOKEN_PER_HEAD,FP8)` 包含 Q/K scale，`(NO_QUANT,INT8)` 只包含 K scale；buffer owner 和
   credit 规则不变。
5. tail/empty slice 的 `A_READY/A_CONSUMED/FIX_OUTPUT_READY/MROPE_K_CONSUMED` 次数按 compact 固定协议配平；
   不因有效 token 数为 0 而跳过必要握手。
