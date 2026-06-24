# mmL0CBuffers 同步方式修改方案：eventId → bufferId

## 目标文件

```
experimental/attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_cube.h
```

## 背景

当前 `mmL0CBuffers`（L0C buffer，double-buffered）使用 eventId 方式（`SetFlag`/`WaitFlag`）进行 M 与 FIX 之间的同步。需要将其改为 bufferId 方式（`get_buf`/`rls_buf`），与 `l1QBuffers` 和 `l1RightBuffers` 保持一致。

## 同步映射规则

参考 `l1QBuffers`（MTE1↔MTE2）和 `l1RightBuffers`（MTE1↔MTE2）的已有写法：

| eventId 方式 | bufferId 方式 | 说明 |
|---|---|---|
| `WaitFlag<HardEvent::X_Y>(id)` | `get_buf(PIPE_Y, id, false);` `rls_buf(PIPE_Y, id, false);` | 消费者阻塞等待生产者信号 |
| `SetFlag<HardEvent::X_Y>(id)` | `get_buf(PIPE_X, id, true);` `rls_buf(PIPE_X, id, true);` | 生产者等前序完成后发射信号 |

对于 mmL0C，涉及的 HardEvent 和 Pipe 映射：

| 方向 | HardEvent | 生产者 Pipe | 消费者 Pipe |
|---|---|---|---|
| Fixpipe → Matmul | `FIX_M` | `PIPE_FIX` | `PIPE_M` |
| Matmul → Fixpipe | `M_FIX` | `PIPE_M` | `PIPE_FIX` |

## 涉及变量

```cpp
uint32_t l0CBufId = 0;              // double-buffer 索引，0/1 交替
uint32_t l0CFixToMBufId = 0;        // FIX→M 方向的 bufId（原 l0CFixToMFlagId）
uint32_t l0CMToFixBufId = 0;        // M→FIX 方向的 bufId（原 l0CMToFixFlagId）
```

> 注：变量名已从 `FlagId` 改为 `BufId`（见 line 101-102），无需再改。

---

## 修改点一览

共 **7 处**修改，分布在 3 个函数中：

| # | 函数 | 行号 | 原代码 | 修改后 |
|---|---|---|---|---|
| 1 | `InitLocalBuffer` | 153-154 | `SetFlag<HardEvent::FIX_M>` ×2 | 移除或替换（见下方详述） |
| 2 | `IterateBmm1SCFA` | 255 | `WaitFlag<HardEvent::FIX_M>` | `get_buf`/`rls_buf` PIPE_M false |
| 3 | `IterateBmm1SCFA` | 311-312 | `SetFlag`+`WaitFlag` `M_FIX` | `get_buf`/`rls_buf` PIPE_M true |
| 4 | `IterateBmm1SCFA` | 327 | `SetFlag<HardEvent::FIX_M>` | `get_buf`/`rls_buf` PIPE_FIX true |
| 5 | `IterateBmm2SCFA` | 341 | `WaitFlag<HardEvent::FIX_M>` | `get_buf`/`rls_buf` PIPE_M false |
| 6 | `IterateBmm2SCFA` | 353-354 | `SetFlag`+`WaitFlag` `M_FIX` | `get_buf`/`rls_buf` PIPE_M true |
| 7 | `IterateBmm2SCFA` | 371 | `SetFlag<HardEvent::FIX_M>` | `get_buf`/`rls_buf` PIPE_FIX true |

---

## 修改详情

### 修改点 1：InitLocalBuffer（line 153-154）

**原代码：**
```cpp
SetFlag<HardEvent::FIX_M>(l0CFixToMBufId);     // {0, 1}, 用于L0C
SetFlag<HardEvent::FIX_M>(l0CFixToMBufId + 1);
```

**修改后：**
```cpp
// 移除以上两行
```

**说明：** 原 eventId 方式需要在初始化时预置信号，使第一轮 Matmul 的 `WaitFlag<FIX_M>` 不被阻塞。bufferId 方式下，buffer 初始状态为"已释放"（released），第一轮 `get_buf(PIPE_M, ..., false)` 不会被阻塞，因此无需预置信号。

> **风险提示：** 如果硬件 buffer 初始状态不是 released，需要在 InitLocalBuffer 中为 bufId 0 和 1 各执行一次 `rls_buf(PIPE_FIX, l0CFixToMBufId + i, true)`（i=0,1）来初始化。请在实际测试中验证首轮 Matmul 是否阻塞。

---

### 修改点 2：IterateBmm1SCFA — Matmul 等待 Fixpipe（line 255）

**原代码：**
```cpp
WaitFlag<HardEvent::FIX_M>(l0CFixToMBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_M, l0CFixToMBufId + l0CBufId, false);
rls_buf(PIPE_M, l0CFixToMBufId + l0CBufId, false);
```

**说明：** Matmul（PIPE_M）阻塞等待 Fixpipe（PIPE_FIX）完成上一轮 Fixpipe 并发出 rls_buf 信号。

---

### 修改点 3：IterateBmm1SCFA — Matmul 通知 Fixpipe（line 311-312）

**原代码：**
```cpp
SetFlag<HardEvent::M_FIX>(l0CMToFixBufId + l0CBufId);
WaitFlag<HardEvent::M_FIX>(l0CMToFixBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_M, l0CMToFixBufId + l0CBufId, true);
rls_buf(PIPE_M, l0CMToFixBufId + l0CBufId, true);
```

**说明：** Matmul（PIPE_M）等待本流水前序 Matmul 完成及相同 bufId 的前序 rls_buf 发射完成后，发射本轮 rls_buf 信号通知 Fixpipe。mode=true 保证了 Matmul 指令已发射后才发出信号。

---

### 修改点 4：IterateBmm1SCFA — Fixpipe 完成后释放（line 327）

**原代码：**
```cpp
SetFlag<HardEvent::FIX_M>(l0CFixToMBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_FIX, l0CFixToMBufId + l0CBufId, true);
rls_buf(PIPE_FIX, l0CFixToMBufId + l0CBufId, true);
```

**说明：** Fixpipe（PIPE_FIX）等待前序 Fixpipe 完成后发射 rls_buf 信号，通知下一轮 Matmul 可以开始。

---

### 修改点 5：IterateBmm2SCFA — Matmul 等待 Fixpipe（line 341）

**原代码：**
```cpp
WaitFlag<HardEvent::FIX_M>(l0CFixToMBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_M, l0CFixToMBufId + l0CBufId, false);
rls_buf(PIPE_M, l0CFixToMBufId + l0CBufId, false);
```

**说明：** 与修改点 2 相同，Bmm2 中 Matmul 等待 Fixpipe 释放 L0C buffer。

---

### 修改点 6：IterateBmm2SCFA — Matmul 通知 Fixpipe（line 353-354）

**原代码：**
```cpp
SetFlag<HardEvent::M_FIX>(l0CMToFixBufId + l0CBufId);
WaitFlag<HardEvent::M_FIX>(l0CMToFixBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_M, l0CMToFixBufId + l0CBufId, true);
rls_buf(PIPE_M, l0CMToFixBufId + l0CBufId, true);
```

**说明：** 与修改点 3 相同。

---

### 修改点 7：IterateBmm2SCFA — Fixpipe 完成后释放（line 371）

**原代码：**
```cpp
SetFlag<HardEvent::FIX_M>(l0CFixToMBufId + l0CBufId);
```

**修改后：**
```cpp
get_buf(PIPE_FIX, l0CFixToMBufId + l0CBufId, true);
rls_buf(PIPE_FIX, l0CFixToMBufId + l0CBufId, true);
```

**说明：** 与修改点 4 相同。

---

## 同步流程图

### 修改前（eventId）

```
InitLocalBuffer:
  SetFlag<FIX_M>(0), SetFlag<FIX_M>(1)     ← 预置信号

IterateBmm1SCFA / IterateBmm2SCFA (per iteration):
  [M]   WaitFlag<FIX_M>(bufId)             ← 等 Fixpipe 完成
  [M]   Matmul → L0C
  [M]   SetFlag<M_FIX>(bufId)              ← 通知 Fixpipe
  [M]   WaitFlag<M_FIX>(bufId)             ← 等信号发射
  [M]   Fixpipe ← L0C
  [M]   SetFlag<FIX_M>(bufId)              ← 释放给下一轮
  [M]   l0CBufId ^= 1
```

### 修改后（bufferId）

```
InitLocalBuffer:
  （无预置信号，buffer 初始为 released 状态）

IterateBmm1SCFA / IterateBmm2SCFA (per iteration):
  [M]   get_buf(PIPE_M, FixToM+bufId, false)   ← 阻塞等 Fixpipe rls_buf
  [M]   rls_buf(PIPE_M, FixToM+bufId, false)
  [M]   Matmul → L0C
  [M]   get_buf(PIPE_M, MToFix+bufId, true)    ← 等前序完成再发信号
  [M]   rls_buf(PIPE_M, MToFix+bufId, true)
  [M]   Fixpipe ← L0C
  [FIX] get_buf(PIPE_FIX, FixToM+bufId, true)  ← 等前序 Fixpipe 完成
  [FIX] rls_buf(PIPE_FIX, FixToM+bufId, true)  ← 通知下一轮 Matmul
  [M]   l0CBufId ^= 1
```

---

## 不变部分

以下同步逻辑 **不在本次修改范围**：

| 代码位置 | 同步方式 | 说明 |
|---|---|---|
| `InitLocalBuffer` line 151-152 | `SetFlag<HardEvent::M_MTE1>` | L0AB 同步，保持 eventId |
| `InitLocalBuffer` line 155-157 | `SetFlag<HardEvent::MTE1_MTE2>` | l1Q 初始信号，保持 eventId |
| `CopyQGmToL1` 全部 | `get_buf`/`rls_buf` | 已是 bufferId，无需修改 |
| `IterateLoadQK` 全部 | `get_buf`/`rls_buf` | 已是 bufferId，无需修改 |
| `IterateBmm1SCFA` l1Q/l1K 部分 | `get_buf`/`rls_buf` | 已是 bufferId，无需修改 |
| `IterateBmm2SCFA` l1K 部分 | `get_buf`/`rls_buf` | 已是 bufferId，无需修改 |

---

## 验证要点

1. **首轮 Matmul 不阻塞**：确认 buffer 初始状态是否为 released，否则需在 InitLocalBuffer 添加初始 rls_buf
2. **double-buffer 交替正确**：l0CBufId 在 0/1 之间交替，确认 get_buf/rls_buf 的 bufId 匹配正确
3. **无死锁**：M→FIX→M 的环形依赖不会导致双方同时阻塞
4. **性能对比**：bufferId 方式与 eventId 方式的性能差异
