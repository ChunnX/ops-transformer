# 核内同步设计

## 1. 目标

本文定义 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 kernel 的核内同步规则。核内同步用于同一个 AIC 或 AIV 内不同 pipe 之间的数据依赖和 buffer 复用保护，不替代 `CrossCoreSetFlag/CrossCoreWaitFlag` 的核间同步。

相关 pipe：

```text
AIC: MTE2, MTE1, MMAD, FIXP
AIV: MTE2, V, MTE3
```

本文中的 `PIPE_V` 或 `V pipe` 指 AIV 的 vector pipe，不是 Q/K/V 输入中的 V 分支。V 分支的语义是
`V * v_scale -> v_cache`：RoPE 使用 `v_scale[n]`，M-RoPE 使用 `v_scale[n,d]`；该分支不参与
`a_rot_l1_nz` 生成，`a_rot_l1_nz` 来自 Q/K 分支的 RMSNorm/RoPE 结果。

核内同步主要分为两类：

1. 正向同步：保护 producer pipe 产出的数据能被 consumer pipe 正确读取。
2. 反向同步：保护一个 buffer 在旧 consumer pipe 使用完成后，才能被下一轮 producer pipe 复用。

## 2. 基本规则

### 2.1 正向同步

正向同步讨论两块 buffer 或同一数据流的 producer/consumer 关系。

例子：

```text
buffer A 的数据由 pipeA 产生
buffer A 的数据随后被 PIPE_V 消费并转移或计算到 buffer B
```

需要：

```cpp
Set<pipeA, PIPE_V>(eventId);
Wait<pipeA, PIPE_V>(eventId);
```

放置语义：

```text
producer pipe 写完 buffer A 后：Set<producer, consumer>(eventId)
consumer pipe 读 buffer A 前： Wait<producer, consumer>(eventId)
```

典型例子：

```cpp
// MTE2 copies GM -> UB, PIPE_V consumes UB.
Set<PIPE_MTE2, PIPE_V>(eventId);
Wait<PIPE_MTE2, PIPE_V>(eventId);
```

### 2.2 反向同步

反向同步讨论一个 buffer 什么时候可以被复用。

例子：

```text
buffer A 由 MTE2 写入
buffer A 被 PIPE_V 消费
下一轮 MTE2 要复用 buffer A
```

需要：

```cpp
// Init: buffer is initially reusable.
Set<PIPE_V, PIPE_MTE2>(eventId);

// Before MTE2 writes buffer A in each round.
Wait<PIPE_V, PIPE_MTE2>(eventId);

// After V finishes consuming buffer A in each round.
Set<PIPE_V, PIPE_MTE2>(eventId);

// End: consume the last reusable event to keep Set/Wait paired.
Wait<PIPE_V, PIPE_MTE2>(eventId);
```

放置语义：

```text
new producer 复用 buffer 前：Wait<old consumer, new producer>(eventId)
old consumer 用完 buffer 后：Set<old consumer, new producer>(eventId)
```

## 3. Event ID 规则

核内同步不允许连续 `Set` 相同 event id。因此每个可并行或 pingpong 的 buffer 必须有独立 event id。

规则：

1. 每个具体 buffer 使用独立 event id。
2. pingpong buffer 使用 `BASE + bufId`。
3. 不使用数组分配 event id，优先使用常量 `BASE + bufId`。
4. 同一个 `<src_pipe, dst_pipe>` 组合内，event id 不能冲突。
5. 不同 `<src_pipe, dst_pipe>` 组合可以复用相同数值的 event id，因为 pipe pair 本身区分同步通道。
6. 正向和反向同步即使数值相同，也必须使用不同的常量名表达不同语义。

event id 的取值范围为：

```text
0..5
```

命名格式：

```cpp
EVT_<SRC>_TO_<DST>_<BUFFER_NAME>_BASE
```

例子：

```cpp
constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0;  // + bufId
constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0;  // + bufId
```

这两个常量数值可以相同，因为它们属于不同 pipe pair：

```text
<PIPE_MTE2, PIPE_V>
<PIPE_V, PIPE_MTE2>
```

但命名必须分开，避免把正向同步和反向同步混淆。

本算子 AIV 侧当前用到的 event id 计划如下：

| Pipe pair | Buffer | Event id |
|-----------|--------|----------|
| `S -> PIPE_V` | resident Q/K scatter and M-RoPE gather indices ready | `0` |
| `PIPE_V -> MTE3` | V cache source ready | `0` |
| `PIPE_V -> MTE3` | Q/K AL1 or output source ready | `1` |
| `MTE3 -> PIPE_V` | `v_out_ub[2]` reuse | `0/1` |
| `MTE3 -> PIPE_V` | `qk_after_cube_ub[2]` reuse | `2/3` |

`a_rot_l1_nz` / AL1 的 4 个物理 buffer 不占用 AIV 核内 event id；AL1 到 AIC 读取的跨核可见性由 `A_READY` 硬同步保证。

当前实现中 `PIPE_V -> MTE3` 使用两个固定语义 ready event：

```cpp
constexpr uint32_t EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0;
constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_READY = 1;
```

两者分别保护 V cache 写出和 Q/K AL1/输出写出，便于 V-first 流水在不同语义点排队 MTE3 工作。它们不是双缓冲 reusable id，也不按 `bufferId` 派生。MTE3 写出完成后的可复用状态由对应的 `<MTE3, PIPE_V>` 反向事件表达：V output 使用 `0/1`，QK output 使用 `2/3`。

`S -> PIPE_V` 的 event 只在 `PrepareBeforeLoop` 中使用一次，用于发布标量流水构造的常驻索引表。索引表在
tile loop 中只读且不会被覆盖，因此不需要反向 reusable event，也不计入双缓冲的 Init/End credit。

## 4. 初始化与结束

所有需要反向同步保护的 buffer，在 kernel 初始化时可认为处于可复用状态，需要先为每个 buffer set 一次 reusable event。

示例：

```cpp
for (uint32_t bufId = 0; bufId < BUFFER_NUM; ++bufId) {
    Set<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);
}
```

每轮复用前：

```cpp
Wait<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);
```

consumer 用完后：

```cpp
Set<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);
```

kernel 结束时，为保证 set/wait 配平，需要消费最后一次 reusable event：

```cpp
for (uint32_t bufId = 0; bufId < BUFFER_NUM; ++bufId) {
    Wait<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);
}
```

## 5. AIV 核内同步点

### 5.1 Resident index: S -> PIPE_V

`PrepareBeforeLoop` 通过标量 `SetValue` 构造常驻 UB 中的 Q/K BF16 NZ scatter index；M-RoPE 场景还会
构造 T/H/W gather index。后续 VF 从 `PIPE_V` 读取这些索引，因此所有标量写入完成后必须建立一次
`S -> PIPE_V` 可见性边界：

```cpp
constexpr uint32_t EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY = 0;

SetFlag<HardEvent::S_V>(EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY);
WaitFlag<HardEvent::S_V>(EVT_S_TO_PIPE_V_PERSISTENT_INDEX_READY);
```

这组索引在整个 tile loop 中只读，RoPE 和 M-RoPE 共用同一条发布边；无需按场景、tile 或 buffer id
重复 Set/Wait，也不需要 `PIPE_V -> S` 反向事件。

### 5.2 Input UB: MTE2 -> PIPE_V

输入 UB 承载 Q/K/V、`cos_sin`、`slot_mapping` 等由 MTE2 从 GM 搬入、由 `PIPE_V` 消费的数据。

正向同步：

```cpp
constexpr uint32_t EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE = 0;  // + bufId

Set<PIPE_MTE2, PIPE_V>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufId);
Wait<PIPE_MTE2, PIPE_V>(EVT_MTE2_TO_PIPE_V_INPUT_UB_BASE + bufId);
```

反向同步：

```cpp
constexpr uint32_t EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE = 0;  // + bufId

// Init
Set<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);

// Before MTE2 reuses input UB
Wait<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);

// After PIPE_V consumes input UB
Set<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);

// End
Wait<PIPE_V, PIPE_MTE2>(EVT_PIPE_V_TO_MTE2_INPUT_UB_BASE + bufId);
```

### 5.3 Q/K A L1 write: CrossCore-protected Cube AL1

AIV 前处理在 `PIPE_V` 中完成 Q/K 分支的 RMSNorm/RoPE，并生成需要写到 Cube AL1 的 BF16/NZ A 数据；MTE3 将该数据写入 `a_rot_l1_nz[bufferId]`。这里的 `PIPE_V` 是 vector pipe，不是 V 分支；V 分支只按场景执行 head scale 或 head/channel scale 后输出。

`a_rot_l1_nz` / AL1 属于 Cube 侧资源，不纳入 AIV 本核 event id 分配。AIV 写 AL1 后通过 `SetAivMte3ToAicMte1AReady()` 通知 AIC；AIC 通过 `WaitAivMte3ToAicMte1AReady()` 等两个 AIV 的 AL1 写入完成，再启动 MTE1/Cube 读取。因此 AL1 写入到 AIC 读取之间的可见性由 `A_READY` 这条 CrossCore 硬同步保证。

实现中 `PIPE_V` 到 MTE3 写 AL1 之间的 pipe handoff 属于 AIV 核内同步，当前用以下 event 表示：

```cpp
constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_READY = 1;
```

这组 event 保护的是 AIV 内部 `PIPE_V` 到 MTE3 的写出启动边界；不得为 4 个 AL1 物理 buffer 分配 AIV 核内 event id。`vec.ComputeTile` 会在写 AL1 前等待当前 output DB slot 的 `MTE3 -> V` reusable event，写 AL1 后再 set 该 slot 的 reusable event。AIV 随后发送 `A_READY`，AIC 仍通过 `WaitAivMte3ToAicMte1AReady()` 获得跨核可见性。

### 5.4 V branch output UB: PIPE_V -> MTE3

V 分支量化结果写入 `v_out_ub[bufferId]`，随后由 MTE3 写入 `v_cache`。

正向同步：

```cpp
constexpr uint32_t EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0;

Set<PIPE_V, PIPE_MTE3>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY);
Wait<PIPE_V, PIPE_MTE3>(EVT_PIPE_V_TO_MTE3_V_CACHE_READY);
```

反向同步：

```cpp
constexpr uint32_t EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE = 0;  // + bufId

// Init
Set<PIPE_MTE3, PIPE_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufId);

// Before V reuses v_out_ub
Wait<PIPE_MTE3, PIPE_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufId);

// After MTE3 writes v_out_ub to GM/cache
Set<PIPE_MTE3, PIPE_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufId);

// End
Wait<PIPE_MTE3, PIPE_V>(EVT_MTE3_TO_PIPE_V_V_OUT_UB_BASE + bufId);
```

### 5.5 Q/K output UB: FIXP -> PIPE_V and PIPE_V/MTE3 reuse

`qk_after_cube_ub[bufferId]` 是 AIC Fixpipe 分发到 AIV UB 的 C 输出，AIV `PIPE_V` 读取它做 dynamic quant。

从 AIV 本核视角，Fixpipe 写 UB 后由核间 `FIX_OUTPUT_READY` 通知 AIV。AIV
`WaitAicFixToAivOutputReady()` 通过后，`PIPE_V` 才能消费 `qk_after_cube_ub[bufferId]`。

首期固定让 `qk_after_cube_ub` 原地覆盖复用为 Q/K FP8 staging，不额外开独立 FP8 staging 区。因此需要定义 `PIPE_V` 处理和 MTE3 写出之间的同步：

```cpp
constexpr uint32_t EVT_PIPE_V_TO_MTE3_QK_READY = 1;
constexpr uint32_t EVT_MTE3_TO_PIPE_V_QK_OUT_UB_BASE = 2;  // + bufId
```

同步语义：

```text
PIPE_V quant/staging 完成 -> MTE3 写 q_out/k_cache/k_scale_cache 使用 EVT_PIPE_V_TO_MTE3_QK_READY
MTE3 写出完成 -> PIPE_V 可复用 qk_after_cube_ub
```

`SetAivMte3ToAicFixCConsumed()` 应放在 Q/K 后处理相关 MTE3 写出完成之后，保证 AIC FIXP 可以安全复用 C 输出路径或 buffer。

## 6. AIC 核内同步点

### 6.1 L1 inputs: MTE2 -> MTE1 -> MMAD

AIC 使用 MTE2 将 `rotation_l1_nz` 从 GM 搬入 L1 并常驻；`a_rot_l1_nz` 由 AIV MTE3 写入 L1。AIC 内部随后使用 MTE1 将 L1 数据搬入 L0，MMAD 消费 L0 数据。具体实现可参考 `grouped_matmul_finalize_routing` 的 arch35 `gmm_fr_weight_quant_cube_compute.h`。

下面的 `MTE2 -> MTE1` L1 event 只保护 AIC 自己通过 MTE2 写入的 L1 数据，例如 `rotation_l1_nz` 常驻区或后续若增加的 AIC-owned L1 搬运区。`a_rot_l1_nz` / AL1 属于 Cube 侧资源，它的跨核可见性由 `A_READY` 硬同步保证；AIC 等到 `A_READY` 后再用 MTE1 读取它。这里的 `l1BufId` 不对应 4 个 AL1 物理 buffer。

MTE2 写 L1、MTE1 读 L1 的正向同步：

```cpp
constexpr uint32_t EVT_MTE2_TO_MTE1_L1_BASE = 3;  // + l1BufId

Set<PIPE_MTE2, PIPE_MTE1>(EVT_MTE2_TO_MTE1_L1_BASE + l1BufId);
Wait<PIPE_MTE2, PIPE_MTE1>(EVT_MTE2_TO_MTE1_L1_BASE + l1BufId);
```

MTE1 不再读取 L1、MTE2 可复用 L1 的反向同步：

```cpp
constexpr uint32_t EVT_MTE1_TO_MTE2_L1_BASE = 3;  // + l1BufId

Set<PIPE_MTE1, PIPE_MTE2>(EVT_MTE1_TO_MTE2_L1_BASE + l1BufId);   // Init
Wait<PIPE_MTE1, PIPE_MTE2>(EVT_MTE1_TO_MTE2_L1_BASE + l1BufId);  // Before MTE2 reuses L1
Set<PIPE_MTE1, PIPE_MTE2>(EVT_MTE1_TO_MTE2_L1_BASE + l1BufId);   // After MTE1 consumes L1
Wait<PIPE_MTE1, PIPE_MTE2>(EVT_MTE1_TO_MTE2_L1_BASE + l1BufId);  // End
```

`rotation_l1_nz` 首期固定常驻 L1 时，初始化阶段搬入一次即可；若不复用该 L1 区域，则不需要为 rotation resident 区设计反向复用同步。

当前 T05 只在 AIC 侧初始化 L0 pingpong (`MMAD -> MTE1`, ids `3/4`) 的反向复用事件。`rotation_l1_nz` 固定常驻，暂不初始化 `MTE1 -> MTE2` L1 反向复用事件；L0C 当前也未接入 `FIXP -> MMAD` 反向事件。后续若增加 AIC-owned 可复用 L1 搬运区或加深 AIC 流水，需要分别补齐对应同步。

MTE1 写 L0、MMAD 消费 L0 的正向同步：

```cpp
constexpr uint32_t EVT_MTE1_TO_MMAD_L0_BASE = 3;  // + l0BufId

Set<PIPE_MTE1, PIPE_MMAD>(EVT_MTE1_TO_MMAD_L0_BASE + l0BufId);
Wait<PIPE_MTE1, PIPE_MMAD>(EVT_MTE1_TO_MMAD_L0_BASE + l0BufId);
```

MMAD 完成 L0A/L0B 使用、MTE1 可复用 L0 buffer 的反向同步：

```cpp
constexpr uint32_t EVT_MMAD_TO_MTE1_L0_BASE = 3;  // + l0BufId

Set<PIPE_MMAD, PIPE_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + l0BufId);   // Init
Wait<PIPE_MMAD, PIPE_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + l0BufId);  // Before MTE1 reuses L0
Set<PIPE_MMAD, PIPE_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + l0BufId);   // After MMAD consumes L0
Wait<PIPE_MMAD, PIPE_MTE1>(EVT_MMAD_TO_MTE1_L0_BASE + l0BufId);  // End
```

### 6.2 L0C/C path: MMAD -> FIXP

MMAD 产生 L0C 结果，FIXP 搬出到 AIV UB。

正向同步：

```cpp
constexpr uint32_t EVT_MMAD_TO_FIXP_L0C_BASE = 0;  // + cBufId

Set<PIPE_MMAD, PIPE_FIX>(EVT_MMAD_TO_FIXP_L0C_BASE + cBufId);
Wait<PIPE_MMAD, PIPE_FIX>(EVT_MMAD_TO_FIXP_L0C_BASE + cBufId);
```

反向同步：

```cpp
constexpr uint32_t EVT_FIXP_TO_MMAD_L0C_BASE = 0;  // + cBufId

Set<PIPE_FIX, PIPE_MMAD>(EVT_FIXP_TO_MMAD_L0C_BASE + cBufId);   // Init
Wait<PIPE_FIX, PIPE_MMAD>(EVT_FIXP_TO_MMAD_L0C_BASE + cBufId);  // Before MMAD reuses L0C/C buffer
Set<PIPE_FIX, PIPE_MMAD>(EVT_FIXP_TO_MMAD_L0C_BASE + cBufId);   // After FIXP completes
Wait<PIPE_FIX, PIPE_MMAD>(EVT_FIXP_TO_MMAD_L0C_BASE + cBufId);  // End
```

`SetAicFixToAivOutputReady()` 使用 `PIPE_FIX` 发出，已足够表达 AIC Fixpipe 输出完成后再通知 AIV。AIC
内部的 MMAD/FIXP event 仍属于 AIC 核内流水管理，不额外改变 CrossCore `FIX_OUTPUT_READY` 协议。

## 7. 与核间同步的衔接

核内同步与核间同步的推荐顺序：

```text
producer pipe writes data
local intra-core Set/Wait if the data crosses local buffers or local pipe ownership
CrossCoreSetFlag notifies remote core
CrossCoreWaitFlag passes on remote core
remote consumer starts
```

关键边：

1. `A_READY` 前：AIV 需要完成 MTE3 对 `a_rot_l1_nz[bufferId]` / AL1 的写入；AL1 属于 Cube 侧资源，AIC 通过 `WaitAivMte3ToAicMte1AReady()` 获得跨核可见性保证。
2. `FIX_OUTPUT_READY` 前：AIC 需要保证 FIXP 已将 RoPE C 输出或 M-RoPE K rotation 分发到 AIV UB。
3. `FIX_OUTPUT_CONSUMED` 前：RoPE AIV 需要保证 Q/K 后处理相关 MTE3 写出完成，output buffer 可以被
   AIC 复用；M-RoPE 改由 `MROPE_K_CONSUMED` 在 V 完成 K quant staging 后保护 K rotation 复用。

## 8. 当前确认项

1. AIV 侧不为 `PIPE_V -> MTE3` 分配双缓冲 id；`EVT_PIPE_V_TO_MTE3_V_CACHE_READY = 0` 和 `EVT_PIPE_V_TO_MTE3_QK_READY = 1` 是两个固定语义正向 ready event。
2. AIC rotation 的 `MTE2 -> MTE1`、`MTE1 -> MMAD` ready event 使用 id `5`；L0 pingpong 的 `MMAD -> MTE1`/`MTE1 -> MMAD` 使用 `3/4`。
3. 当前代码没有启用 `FIXP -> MMAD` L0C 反向事件；L0C buffer 由 `l0cBufferUseId_` 轮转和 tile 顺序保护。后续若引入更深 AIC 内流水，需要补充相应 event 设计和实现。

## 9. `MROPE_BF16_Q_INT8_K` 核内约束

1. M-RoPE 模板不绑定、不读取空的 `query_start_loc/seq_lens` anchor；`mrope_position` 和 T/H/W raw cos_sin DB
   复用现有 input/cos-sin 的 `MTE2 -> PIPE_V` 与
   `PIPE_V -> MTE2` 事件对。为保留 V-first overlap 可以拆开已有 Set/Wait 的位置，但不分配新 event id。
2. RegBase Gather 在 `PIPE_V` 内消费 raw cos/sin；令 $D_{\mathrm{half}}=D/2$，cos/sin 分别应用同一份
   $D_{\mathrm{half}}$ 项 id，cos 从 `rawBase` 读取，sin 从 `rawBase` 偏移 $D_{\mathrm{half}}$ 后读取。
3. M-RoPE gather index 与 Q/K NZ scatter index 都由标量流水在 `PrepareBeforeLoop` 构造，并在第一次 VF
   使用前通过同一条一次性 `S -> PIPE_V` event 发布；tile loop 内只读，不引入额外 reusable event。
4. Q/K output staging 由 `Q_QUANT_MODE/K_QUANT_MODE` 编译期模板选择，kernel 不做 runtime 判断。RoPE
   使用 output DB 的 `FIX_OUTPUT_READY` 和 `FIX_OUTPUT_CONSUMED`；M-RoPE Q 由 Fixpipe 直接写 GM，
   不经过 Vec QCast。K rotation 复用 `FIX_OUTPUT_READY`，并由 `MROPE_K_CONSUMED` 管理反向复用。
5. `q_quant_mode=NoQuant` 的 BF16 模式没有 q_scale staging 和 q_scale MTE3 写出。M-RoPE compact 的两个
   credit 在 V 完成 K rotation 读取和独立 K quant staging 后由 V 返还。BasicBlock 在
   `StageMropeK(lastTile)` 后、`ScatterMropeK(lastTile)` 前返还 credit，并在同一控制迭代立即发起
   scatter；K quant 双缓冲 slot 的 `MTE3 -> V` reusable event 只在 K cache/K scale 两路写回完成后返还。
6. event id、pipe pair、Init/End 配平规则不变，不因新增 mode 模板增加范围 `static_assert`。
7. compact M-RoPE 的 input 与 raw cos/sin 分别使用独立 use counter；Vec 根据各自物理 pool/slot 容量
   局部推导二者均为 4 槽。独立 K rotation/K quant buffer 和 1024-token metadata window 只改变 buffer
   owner/lifetime，不新增核内 event id；metadata window 每次 refill 仍沿用现有
   `MTE2 -> PIPE_V` 边界，K rotation 的跨核可见性与复用分别由 `FIX_OUTPUT_READY`（flag 1）和
   `MROPE_K_CONSUMED`（flag 4）负责。
