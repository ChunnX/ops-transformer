# 核间同步设计

## 1. 目标

本文单独定义 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 kernel 的 AIC/AIV 核间同步规则。同步规则服务于 `KERNEL_TYPE_MIX_AIC_1_2` 执行模型：

```text
1 AIC + 2 AIV per group
```

AIC 负责 `rope_out @ rotation` 的 Cube matmul 和 Fixpipe 输出；两个 AIV 共同负责同一个 token tile 的 Q/K/V 前后处理，并按连续 token 子区间写入同一个 `A_rot[M_tile,D]` 的 disjoint row 区间。

## 2. 核组与身份

核组关系：

```text
AIC: GetBlockIdx() gives cube core index
AIV: GetBlockIdx() / 2 gives corresponding cube core index
AIV local id: GetSubBlockIdx()
```

约定：

```text
GetSubBlockIdx() == 0 -> AIV0
GetSubBlockIdx() == 1 -> AIV1
```

两个 AIV 对同一个 token tile 做连续切分。tail tile 可以不均衡，但不能在 A 矩阵 M 方向留下空洞：

```text
actualAiv0 = ceil(actualTokenTile / 2)
actualAiv1 = actualTokenTile - actualAiv0

AIV0 token range: [tileStart, tileStart + actualAiv0)
AIV1 token range: [tileStart + actualAiv0, tileStart + actualTokenTile)

AIV0 A rows: [0, actualAiv0 * Hqk)
AIV1 A rows: [actualAiv0 * Hqk, actualTokenTile * Hqk)

M_tile = actualTokenTile * Hqk
```

若 tail 中某个 AIV 的实际 token 数为 0，该 AIV 也必须参与固定的核间同步流程，避免 AIC 等待缺失。

## 3. Flag 语义

核间同步使用 `CrossCoreSetFlag/CrossCoreWaitFlag`。

已确认语义：

1. flag id 可以复用。
2. `Wait(id)` 会一直阻塞，直到对应 `Set(id)` 发生。
3. 多次 `Set` 同一个 id 是合法的，只要有相同次数的 `Wait` 去消费。
4. 首期不通过不同 flag id 区分 pingpong buffer。
5. pingpong buffer 复用依赖同一个 flag id 上的事件计数和 AIC/AIV 固定循环顺序。
6. `a_rot_l1_nz` / AL1 属于 Cube 侧资源。AIV MTE3 写 AL1 后通过 `A_READY` 通知 AIC，AIC `WaitAivToAic<PIPE_MTE1>(SYNC_A_READY)` 通过后再读取 AL1；该跨核可见性由 `A_READY` 硬同步保证。AIV 内部若存在临时 UB staging，其复用仍由核内同步管理。

## 4. Flag ID

flag id 直接硬编码，不使用数组，不按 `bufferId` 或 `generation` 派生：

```cpp
constexpr uint64_t SYNC_A_READY = 0;
constexpr uint64_t SYNC_C_READY = 1;
constexpr uint64_t SYNC_C_CONSUMED = 2;
```

AIV0 使用基础 id。AIV1 使用：

```text
syncFlag + FLAG_ID_MAX
```

当前 `FLAG_ID_MAX` 对应 `+16` 规则。框架根据 `+ FLAG_ID_MAX` 推导 AIV1，不需要额外指定 target core。

## 5. 建议封装

封装参考 `grouped_matmul_finalize_routing` 的 arch35 `weight_quant_basic_block`：

```cpp
template <const pipe_t pipe>
__aicore__ inline void SetAivToAic(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
}

template <const pipe_t pipe>
__aicore__ inline void WaitAivToAic(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
}

template <const pipe_t pipe>
__aicore__ inline void SetAicToAiv(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
    CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
}

template <const pipe_t pipe>
__aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
}
```

含义：

1. AIV -> AIC 方向：每个 AIV 只 set 基础 id；AIC 对同一基础 id wait 两次，先消费 AIV1，再消费 AIV0。
2. AIC -> AIV 方向：AIC 对同一基础 id set 两次，分别通知 AIV1 与 AIV0；每个 AIV 只 wait 基础 id。
3. AIV0/AIV1 的差异由框架和 `FLAG_ID_MAX` 规则处理，AIV 侧代码不需要手动加 `FLAG_ID_MAX`。

## 6. 同步边

### 6.1 A_READY

```text
data      = a_rot_l1_nz[bufferId]
location  = L1
producer  = AIV0/AIV1 MTE3
consumer  = AIC MTE1/Cube
direction = AIV -> AIC
```

两个 AIV 各自写 `a_rot_l1_nz[bufferId]` 的连续 disjoint M rows。AIC 必须等两个 AIV 的 A 数据都 ready 后，才能启动完整 `M_tile` matmul。

同步 pipe：

```cpp
AIV: SetAivToAic<PIPE_MTE3>(SYNC_A_READY);
AIC: WaitAivToAic<PIPE_MTE1>(SYNC_A_READY);
```

注意：`a_rot_l1_nz` / AL1 属于 Cube 侧资源。AIV 必须在 MTE3 写 AL1 完成后再 `SetAivToAic<PIPE_MTE3>(SYNC_A_READY)`；AIC 通过 `WaitAivToAic<PIPE_MTE1>(SYNC_A_READY)` 等两个 AIV 的 AL1 写入完成后再读取，AL1 到 AIC 的可见性由该 CrossCore 硬同步保证。若 AIV 内部存在额外 UB staging，其复用由 AIV 核内同步单独管理。

### 6.2 C_READY

```text
data      = qk_after_cube_ub[bufferId]
location  = AIV UB
producer  = AIC FIX
consumer  = AIV PIPE_V
direction = AIC -> AIV
```

AIC Fixpipe 在 M 方向将 C 输出分发给 AIV0/AIV1 对应 UB 分片。两个 AIV 各自等待 `C_READY` 后，只消费自己的 row 范围。

同步 pipe：

```cpp
AIC: SetAicToAiv<PIPE_FIX>(SYNC_C_READY);
AIV: WaitAicToAiv<PIPE_V>(SYNC_C_READY);
```

### 6.3 C_CONSUMED

```text
data      = qk_after_cube_ub[bufferId] reusable state
location  = AIV UB
producer  = AIV MTE3 after Q/K postprocess side effects are complete
consumer  = AIC FIX before reusing C output path/buffer
direction = AIV -> AIC
```

AIV 完成 Q/K 后处理后通知 AIC。AIC 必须等待两个 AIV 的 `C_CONSUMED` 后，才能复用对应 C 输出路径或 buffer。

同步 pipe：

```cpp
AIV: SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED);
AIC: WaitAivToAic<PIPE_FIX>(SYNC_C_CONSUMED);
```

## 7. Tile 级时序

AIV rolling 分支：

```text
for tile i in range:
  1. preprocess tile i, write a_rot_l1_nz[bufferId]
  2. complete MTE3 write to Cube AL1
  3. SetAivToAic<PIPE_MTE3>(SYNC_A_READY)
  4. PostprocessV(tile i)
  5. if tile i - 1 exists:
       WaitAicToAiv<PIPE_V>(SYNC_C_READY)
       PostprocessQk(tile i - 1)
       SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED)

drain:
  6. WaitAicToAiv<PIPE_V>(SYNC_C_READY) for last tile
  7. PostprocessQk(last tile)
  8. SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED)
```

AIC rolling 分支：

```text
for tile i in range:
  1. WaitAivToAic<PIPE_MTE1>(SYNC_A_READY)
  2. Cube Matmul(a_rot_l1_nz[bufferId], rotation_l1_nz)
  3. Fixpipe C to AIV UB by M range
  4. SetAicToAiv<PIPE_FIX>(SYNC_C_READY)
  5. WaitAivToAic<PIPE_FIX>(SYNC_C_CONSUMED)
```

## 8. Pingpong Buffer 复用

`bufferId = tileId % 2` 只用于选择数据 buffer：

```text
a_rot_l1_nz[bufferId]
qk_after_cube_ub[bufferId]
v_ub[bufferId]
v_out_ub[bufferId]
slot_mapping_ub[bufferId]
cos_sin_ub[bufferId]
```

核间 flag id 不随 `bufferId` 改变。示例：

```text
Tile 0 uses buffer 0 and SYNC_A_READY/SYNC_C_READY/SYNC_C_CONSUMED
Tile 1 uses buffer 1 and the same sync ids
Tile 2 uses buffer 0 and the same sync ids again
```

正确性来自：

1. AIC/AIV 双方对 tile 的遍历顺序一致。
2. 每个 tile 固定产生一次 AIV0 ready、一次 AIV1 ready。
3. 每个 tile 固定产生一次 AIC to AIV0 C ready、一次 AIC to AIV1 C ready。
4. 每个 tile 固定产生一次 AIV0 consumed、一次 AIV1 consumed。
5. 同一 flag id 上的 `Set/Wait` 次数严格配平。

## 9. 待补充项

1. AIC Fixpipe 分发 C 到 AIV UB 时的具体 M 方向 offset 和 UB 地址计算。

`rotation_l1_nz` 只由 AIC 内部使用，AIV 不等待 rotation 常驻完成；当前 rotation 常驻不需要核间 flag。若后续确有 AIV 侧依赖，再新增独立硬编码 id，并重新证明 Set/Wait 计数。
