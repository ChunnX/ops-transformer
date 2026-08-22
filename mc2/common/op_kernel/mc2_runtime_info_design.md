# Mc2RuntimeInfo 重设计方案（方案 B：环形 Buffer + Policy 模板引擎）

## 1. 背景与问题

### 1.1 现状

`mc2/common/op_kernel/mc2_runtime_info.h` 中的 `Mc2RuntimeInfo` 类在每个 core 的 workspace
`WS_SEG_RUNTIME_INFO` 段（64KB，per-core 512B）中维护一份 `RuntimeInfoPerCore` 结构体，
记录当前执行状态。异常 dump 时 host 侧（`parse_allto_all_matmul_tiling.py`）读取该段
定位卡死位置。

### 1.2 核心问题

| 问题 | 说明 |
|---|---|
| **单槽位覆盖，无历史轨迹** | 每次 `UpdatePhase` 覆盖前值，流水线 comm→trans→comm 交替，dump 只能看到最后一个写入点，无法还原"谁先停、谁在等谁"的因果链 |
| **commPhase 语义不纯** | 当前 `RuntimePhase` 枚举把 HCCL 通信生命周期（Init/Prepare/Commit/Wait/Finalize）和算子计算子阶段（TRANSPOSE/REDUCE/COMPUTE_*）混在一起，缺 Commit 独立态 |
| **execPosition 1B 截断** | 裸 tile 索引，无法表达"哪一段的哪个 task"；1B 最多 255，tileCnt 可能超限 |
| **AIC cube 核卡点不可见** | AIC 不参与通信，pipeline 里 AIC 分支在 `CrossCoreWaitFlag` 等转置完成，若无插桩 dump 只看到顶层 PREPARE |
| **无时序信息** | 看不到各核停了多久、谁先停 |

### 1.3 参考：moe ExceptionDump 机制

`mc2/common/op_kernel/mc2_exception_dump.h` 的 `ExceptionDump<Policy>` 模板引擎：

- **BlockStage（每核一份）**：`{ currentStage, stageCycle[END] }`，`UpdateStage` 写当前阶段 + 进入 cycle。
- **StageEnum**：算子级大粒度枚举（11 阶段），Policy 注入。
- **DumpParams**：记录关键 GM 区域搬运参数，host 侧 `ParseStructuredDump` 三阶段搬回。
- **Host 解析**：`[Header][Tiling][BlockStage[]][DumpParams[]][dump data]` 固定布局。

moe 的 `stageCycle[END]` 数组适合"每阶段只进一次"的粗粒度场景。alltoallmm 循环内
Prepare→Commit→Wait 反复进入，cycle 数组会覆盖丢历史，不适合。因此采用**环形 buffer**。

---

## 2. 设计目标

1. **commPhase 记录 HCCL 五态**：Init / Prepare / Commit / Wait / Finalize。
2. **execPosition 记录执行位置**（position tag）：编码 `[part][taskIdx]`。
3. **execTurn 记录执行轮次**。
4. **字段数不增加**：复用现有 `RuntimeInfoPerCore` 的 `reserved[503]` 做环形 buffer，不新增字段。
5. **per-core 空间不变**：保持 512B / `alignas(512)`。
6. **Policy 模板引擎**：独立 `Mc2RuntimeInfo<Policy>`，各算子提供 `StageEnumT`，通用引擎复用。
7. **历史轨迹**：环形 buffer 追加 `{turn, pos, phase, cycle}` 条目，保留最近 N 次状态变更。

---

## 3. 数据结构设计

### 3.1 通信阶段枚举（HCCL 五态）

**文件**：`mc2/common/op_kernel/mc2_tiling_struct.h`

```c
// 通信阶段枚举（参考 HCCL: Init / Prepare / Commit / Wait / Finalize）
enum RuntimePhase : uint8_t {
    RT_PHASE_INIT         = 0,   // 通信初始化 (hccl.InitV2)
    RT_PHASE_COMM_PREPARE = 1,   // 通信任务准备 (primitive.Prepare 返回 handle)
    RT_PHASE_COMM_COMMIT  = 2,   // 通信任务提交 (hccl.Commit)
    RT_PHASE_COMM_WAIT    = 3,   // 等待通信完成 (hccl.Wait)
    RT_PHASE_FINALIZE     = 4,   // 通信结束 (hccl.Finalize)
    RT_PHASE_RESERVED     = 255,
};
```

HCCL API 对应关系（`mc2_templates/communication/communicator/hccl_communicator.h`）：

| HCCL 阶段 | 代码 | 说明 |
|---|---|---|
| Init | `hccl_.InitV2(GetHcclContext<0>(), &mc2InitTiling)` + `SetCcTilingV2` | 通信器初始化 |
| Prepare | `primitive_.Prepare(&hccl_, &context_, i)` → 返回 `HcclHandle` | 构造任务描述，不启动 |
| Commit | `hccl_.Commit(handle)` | 任务入队异步启动 |
| Wait | `hccl_.Wait(handle)` | 阻塞等待完成 |
| Finalize | `hccl_.Finalize()` | 释放收尾 |

### 3.2 Position Tag 编码

**文件**：`mc2/common/op_kernel/mc2_tiling_struct.h`

`execPosition`（1 字节）编码为 position tag：

```c
// execPosition 位置 tag 编码: [高 3bit part][低 5bit taskIdx]
//   part:   执行段 (scale / tile / tail)
//   taskIdx: 段内任务索引 (0~31)
constexpr uint8_t RT_POS_PART_SCALE = 0;
constexpr uint8_t RT_POS_PART_TILE  = 1;
constexpr uint8_t RT_POS_PART_TAIL  = 2;

constexpr uint8_t RT_POS_IDX_MASK = 0x1F;   // 低 5bit
constexpr uint8_t RT_POS_PART_SHIFT = 5;    // 高 3bit

constexpr uint8_t MakePosTag(uint8_t part, uint8_t taskIdx)
{
    return static_cast<uint8_t>((part << RT_POS_PART_SHIFT) | (taskIdx & RT_POS_IDX_MASK));
}
constexpr uint8_t GetPosPart(uint8_t tag) { return tag >> RT_POS_PART_SHIFT; }
constexpr uint8_t GetPosTaskIdx(uint8_t tag) { return tag & RT_POS_IDX_MASK; }
```

- part 用高 3bit（最多 8 种段），taskIdx 用低 5bit（0~31）。
- tileCnt 超过 32 时 taskIdx 回绕，但 part + turn + phase 组合仍能粗定位；
  死锁通常卡在早期 task，32 条粒度足够。
- `constexpr` 函数 host/kernel 双用。

### 3.3 RuntimeInfoPerCore 结构体（字段不变，复用 reserved）

**文件**：`mc2/common/op_kernel/mc2_tiling_struct.h`

```c
// 环形 buffer 单条记录 (8 字节)
struct RtTraceEntry {
    uint8_t  turn;       // 执行轮次
    uint8_t  pos;        // 位置 tag ([part][taskIdx])
    uint8_t  phase;      // 通信阶段 (RuntimePhase)
    uint8_t  reserved;   // 对齐
    uint32_t cycle;      // GetSystemCycle() 低 32bit
};

// 环形 buffer 容量计算:
//   per-core 512B - 固定头 16B = 496B
//   496B / 8B = 62 条
constexpr uint32_t RT_TRACE_CAPACITY = 62;

struct alignas(512) RuntimeInfoPerCore {
    // ---- 固定头 (16B，最新状态快照) ----
    uint32_t magicNum;        // 4B  魔数 0x5A5A5A5A
    uint16_t coreId;          // 2B  core ID
    uint8_t  execTurn;        // 1B  执行轮次 (最新)
    uint8_t  execPosition;    // 1B  执行位置 tag (最新)
    uint8_t  commPhase;       // 1B  通信阶段 (最新, RuntimePhase)
    uint8_t  reserved1;       // 1B  对齐
    uint16_t traceHead;       // 2B  环形 buffer 写头索引
    uint16_t traceCount;      // 2B  已写入条目数 (饱和到 RT_TRACE_CAPACITY)
    uint8_t  reserved2[2];    // 2B  对齐到 16B 边界

    // ---- 环形 buffer (62 × 8B = 496B) ----
    RtTraceEntry trace[RT_TRACE_CAPACITY];
}; // 16 + 496 = 512B
```

布局验证：
- 固定头 16B，全部落在第一个 cacheline（64B）内。
- `trace[]` 从 offset 16 开始，62 × 8 = 496B，总计 512B。
- DCCI 刷写：写固定头时刷第一个 cacheline；写 trace 条目时刷对应条目所在 cacheline。

### 3.4 Policy 模板

**文件**：`mc2/common/op_kernel/mc2_runtime_info.h`

```c
namespace Mc2Kernel {

// 默认 Policy: 通用通信算子
struct DefaultRtPolicy {
    using StageEnumT = RuntimePhase;  // 默认用 HCCL 五态
};

template <typename Policy = DefaultRtPolicy>
class Mc2RuntimeInfo {
public:
    __aicore__ inline Mc2RuntimeInfo() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM,
                                Mc2Tiling::WorkspaceLayoutInfo *wsLayout,
                                uint32_t aicCoreNum);

    // 全量更新: turn + position + phase，同时追加 trace 条目
    __aicore__ inline void Record(uint8_t turn, uint8_t position, uint8_t phase);

    // 只更新 phase (保持 turn/position)，追加 trace
    __aicore__ inline void RecordPhase(uint8_t phase);

    // 只更新 position (保持 turn/phase)，追加 trace
    __aicore__ inline void RecordPosition(uint8_t turn, uint8_t position);

private:
    __aicore__ inline void AppendTrace(uint8_t turn, uint8_t position, uint8_t phase);
    __aicore__ inline void FlushCacheLine(GM_ADDR addr, size_t size);

    __gm__ Mc2Tiling::RuntimeInfoPerCore *runtimeInfoBase_ = nullptr;
    uint32_t slotIdx_ = 0;
};

} // namespace Mc2Kernel
```

### 3.5 核心方法实现逻辑

#### Init

```
1. 从 wsLayout 找到 WS_SEG_RUNTIME_INFO 段，设置 runtimeInfoBase_
2. 计算 slotIdx_ (AIC: blockIdx, AIV: aicCoreNum + blockIdx)
3. 写固定头: magicNum, coreId, execTurn=0, execPosition=0, commPhase=INIT
4. 清空 trace: traceHead=0, traceCount=0
5. DCCI 刷前两个 cacheline (覆盖固定头 + trace 起始)
```

#### Record(turn, position, phase)

```
1. 更新固定头: execTurn=turn, execPosition=position, commPhase=phase
2. AppendTrace(turn, position, phase)
```

#### AppendTrace(turn, position, phase)

```
1. idx = traceHead % RT_TRACE_CAPACITY
2. trace[idx] = {turn, position, phase, 0, GetSystemCycle() & 0xFFFFFFFF}
3. traceHead++
4. if (traceCount < RT_TRACE_CAPACITY) traceCount++
5. DCCI 刷 trace[idx] 所在 cacheline
```

#### FlushCacheLine(addr, size)

```
DataCacheCleanAndInvalid<uint8_t, SINGLE_CACHE_LINE, CACHELINE_OUT>(addr, size)
```

---

## 4. alltoallmm mxfp8 kernel 插桩方案

### 4.1 涉及文件

| 文件 | 角色 |
|---|---|
| `mc2/common/op_kernel/mc2_tiling_struct.h` | 枚举 + 结构体 + position tag 编码 |
| `mc2/common/op_kernel/mc2_runtime_info.h` | Policy 模板引擎 |
| `mc2/allto_all_matmul/op_kernel/arch35/allto_all_mx_quant_matmul_arch35.h` | 算子主流程插桩 |
| `mc2/allto_all_matmul/op_kernel/arch35/allto_all_mx_quant_matmul_pipeline.h` | pipeline 循环内插桩 |

### 4.2 AlltoAllMxQuant Policy 定义

**文件**：`allto_all_mx_quant_matmul_arch35.h`（或单独 policy 头文件）

```c
namespace Mc2Kernel {
struct AlltoAllMxQuantRtPolicy {
    using StageEnumT = RuntimePhase;  // HCCL 五态
};
using AlltoAllMxQuantRuntimeInfo = Mc2RuntimeInfo<AlltoAllMxQuantRtPolicy>;
}
```

### 4.3 插桩点清单

#### allto_all_mx_quant_matmul_arch35.h

| 位置 | 行号(当前) | 调用 | 说明 |
|---|---|---|---|
| `Init()` | L108 | `runtimeInfo_.Init(...)` | 写 INIT + 清空 trace |
| `Process()` scale 启动前 | L130 | `Record(0, MakePosTag(SCALE,0), PREPARE)` | scale 段通信准备 |
| `Process()` tile 启动前 | L135 | `Record(0, MakePosTag(TILE,0), PREPARE)` | tile 段通信准备 |
| `Process()` tail 启动前 | L141 | `Record(1, MakePosTag(TAIL,0), PREPARE)` | tail 段通信准备 |
| `Process()` End 前 | L146 | `RecordPhase(FINALIZE)` | 通信结束 |

#### allto_all_mx_quant_matmul_pipeline.h

pipeline 的 `Process(taskCnt)` / `ProcessScale()` 内部循环按 task 推进，
需要在循环内每个 stage 调用前插桩。由于 `runtimeInfo_` 在 arch35.h 中，
pipeline 需要访问它。方案: **给 pipeline 传入 runtimeInfo 指针**。

pipeline 修改：

```c
class AlltoAllMXQuantMatmulPipeLine {
public:
    __aicore__ inline void SetRuntimeInfo(Mc2Kernel::AlltoAllMxQuantRuntimeInfo *rt) { rtInfo_ = rt; }
    // ...
private:
    Mc2Kernel::AlltoAllMxQuantRuntimeInfo *rtInfo_ = nullptr;
};
```

插桩点（`Process(taskCnt)` 循环内）:

| 位置 | 行号(当前) | 调用 | 说明 |
|---|---|---|---|
| `commStage_->PrepareAll(taskCnt)` 后 | L87 后 | `rtInfo_->RecordPhase(COMMIT)` | 通信任务已 commit |
| AIV: `commStage_->Process(index)` 前 | L90 前 | `rtInfo_->Record(turn, MakePosTag(part,index), WAIT)` | 等待第 index 个 task |
| AIV: `transStage_->Process(index)` 前 | L92 前 | `rtInfo_->RecordPosition(turn, MakePosTag(part,index))` | 转置阶段 |
| AIC: `computeStage_->Process(index)` 前 | L101 前 | `rtInfo_->RecordPosition(turn, MakePosTag(part,index))` | matmul 阶段 |

插桩点（`ProcessScale()` 内）:

| 位置 | 行号(当前) | 调用 | 说明 |
|---|---|---|---|
| `commStage_->PrepareAll(1)` 后 | L112 后 | `rtInfo_->RecordPhase(COMMIT)` | scale 通信已 commit |
| `commStage_->Process(0)` 前 | L113 前 | `rtInfo_->Record(0, MakePosTag(SCALE,0), WAIT)` | scale 通信等待 |
| `scaleTransStage_->Process(0)` 前 | L115 前 | `rtInfo_->RecordPosition(0, MakePosTag(SCALE,0))` | scale 转置 |

> turn 传递: tile 段 turn=0, tail 段 turn=1。pipeline 需从 arch35.h 获取当前段信息。
> 方案: pipeline `Process()` 增加 `uint8_t turn` 参数，或通过 context 传递。

### 4.4 死锁定位示例

异常 dump 后 host 侧解析 trace，典型死锁场景输出:

```
=== core 3 (AIV, slotIdx=3) ===
  head=8, count=8
  [0] turn=0 pos=TILE[0] phase=PREPARE  cycle=0x00001000
  [1] turn=0 pos=TILE[0] phase=COMMIT   cycle=0x00001010
  [2] turn=0 pos=TILE[0] phase=WAIT     cycle=0x00001020
  [3] turn=0 pos=TILE[0] phase=(pos)    cycle=0x00001030  ← transpose done
  [4] turn=0 pos=TILE[1] phase=PREPARE  cycle=0x00001100
  [5] turn=0 pos=TILE[1] phase=COMMIT   cycle=0x00001110
  [6] turn=0 pos=TILE[1] phase=WAIT     cycle=0x00001120
  [7] turn=0 pos=TILE[1] phase=(pos)    cycle=0x00001130  ← [STOP] transpose done, waiting flag
  latest: commPhase=WAIT, pos=TILE[1], turn=0

=== core 7 (AIC, slotIdx=7) ===
  head=5, count=5
  [0] turn=0 pos=TILE[0] phase=(pos)    cycle=0x00001200  ← matmul[0]
  [1] turn=0 pos=TILE[1] phase=(pos)    cycle=0x00001300  ← matmul[1]
  [2] turn=0 pos=TILE[2] phase=(pos)    cycle=0x00001400  ← [STOP] waiting CrossCoreWaitFlag(9)
  latest: commPhase=PREPARE, pos=TILE[2], turn=0
```

判读:
- AIV core 3 跑到 tile[1] 的 WAIT 后完成 transpose，停在等 AIC 的 flag。
- AIC core 7 跑到 tile[2] 的 matmul，停在 `CrossCoreWaitFlag(9)` 等转置。
- 结论: AIC 跑太快超过 AIV，AIV tile[1] transpose 完成但 AIC 已在 tile[2] 等 → flag 时序错配。

---

## 5. Host 侧解析工具改动

### 5.1 文件

`mc2/tools/dump_analysis/parse_allto_all_matmul_tiling.py`

### 5.2 改动点

1. **RUNTIME_PHASE_MAP 更新为 HCCL 五态**:

```python
RUNTIME_PHASE_MAP = {
    0: "INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "FINALIZE",
    255: "RESERVED",
}
```

2. **position tag 解码**:

```python
RT_POS_PART_SHIFT = 5
RT_POS_IDX_MASK = 0x1F
RT_POS_PART_MAP = {0: "SCALE", 1: "TILE", 2: "TAIL"}

def decode_pos_tag(tag):
    part = tag >> RT_POS_PART_SHIFT
    idx = tag & RT_POS_IDX_MASK
    return RT_POS_PART_MAP.get(part, f"PART({part})"), idx
```

3. **RuntimeInfoPerCore 解析** (结构体布局变更):

```python
RT_TRACE_ENTRY_SIZE = 8    # turn(1)+pos(1)+phase(1)+pad(1)+cycle(4)
RT_TRACE_CAPACITY = 62

def parse_runtime_info_per_core(data, offset):
    # 固定头 16B
    magic, core_id, exec_turn, exec_pos, comm_phase, _, trace_head, trace_count = \
        struct.unpack_from("<I H B B B B H H", data, offset)
    # trace 数组 62×8B
    entries = []
    for i in range(min(trace_count, RT_TRACE_CAPACITY)):
        entry_off = offset + 16 + i * RT_TRACE_ENTRY_SIZE
        turn, pos, phase, _, cycle = struct.unpack_from("<B B B B I", data, entry_off)
        part_name, task_idx = decode_pos_tag(pos)
        entries.append({
            "turn": turn, "part": part_name, "taskIdx": task_idx,
            "phase": RUNTIME_PHASE_MAP.get(phase, f"UNKNOWN({phase})"),
            "cycle": cycle,
        })
    return {
        "magic": magic, "coreId": core_id,
        "execTurn": exec_turn, "execPosition": exec_pos,
        "commPhase": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "traceHead": trace_head, "traceCount": trace_count,
        "trace": entries,
    }
```

4. **输出格式**: 按 core 打印 trace 时间线表格。

---

## 6. 迁移范围

### 6.1 需修改的文件

| 文件 | 改动内容 |
|---|---|
| `mc2/common/op_kernel/mc2_tiling_struct.h` | RuntimePhase 枚举改 HCCL 五态；新增 position tag 编码；RuntimeInfoPerCore 结构体改 reserved 为 trace |
| `mc2/common/op_kernel/mc2_runtime_info.h` | 改为 Policy 模板；新增 Record/RecordPhase/RecordPosition/AppendTrace；Init 清空 trace |
| `mc2/allto_all_matmul/op_kernel/arch35/allto_all_mx_quant_matmul_arch35.h` | Policy 定义；插桩点改用 Record API |
| `mc2/allto_all_matmul/op_kernel/arch35/allto_all_mx_quant_matmul_pipeline.h` | 增加 rtInfo 指针；循环内插桩 |
| `mc2/allto_all_matmul/op_kernel/allto_all_matmul_kernel_base.h` | 调用点迁移到新 API |
| `mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_matmul_fp16_bf16.h` | 调用点迁移 |
| `mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_matmul_base_v2.h` | 调用点迁移 |
| `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h` | 调用点迁移 |
| `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h` | 调用点迁移 |
| `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h` | 调用点迁移 |
| `mc2/matmul_reduce_scatter_v2/op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h` | 调用点迁移 |
| `mc2/tools/dump_analysis/parse_allto_all_matmul_tiling.py` | RUNTIME_PHASE_MAP 改五态；trace 解析；position tag 解码 |

### 6.2 枚举值变化（向后不兼容）

| 旧值 | 旧名 | 新值 | 新名 |
|---|---|---|---|
| 0 | RT_PHASE_INIT | 0 | RT_PHASE_INIT |
| 1 | RT_PHASE_COMM_PREPARE | 1 | RT_PHASE_COMM_PREPARE |
| 2 | RT_PHASE_COMM_WAIT | 2 | RT_PHASE_COMM_COMMIT |
| 3 | RT_PHASE_COMM_PROCESS | 3 | RT_PHASE_COMM_WAIT |
| 4 | RT_PHASE_COMPUTE_LOCAL | — | 删除 |
| 5 | RT_PHASE_COMPUTE_GATHER | — | 删除 |
| 6 | RT_PHASE_TRANSPOSE | — | 删除 |
| 7 | RT_PHASE_REDUCE | — | 删除 |
| 8 | RT_PHASE_FINALIZE | 4 | RT_PHASE_FINALIZE |

> 计算子阶段不再用 commPhase 记录，改用 `RecordPosition` 只更新位置（phase 保持上一个通信状态）。

### 6.3 调用点迁移规则

| 旧调用 | 新调用 |
|---|---|
| `UpdatePhase(0, 0, RT_PHASE_COMM_PREPARE)` | `Record(0, MakePosTag(TILE,0), RT_PHASE_COMM_PREPARE)` |
| `UpdatePhase(isTail?1:0, i, RT_PHASE_COMM_WAIT)` | `Record(isTail?1:0, MakePosTag(part,i), RT_PHASE_COMM_WAIT)` |
| `UpdatePhase(0, 0, RT_PHASE_COMPUTE_LOCAL)` | `RecordPosition(0, MakePosTag(TILE,0))` |
| `UpdatePhase(0, 0, RT_PHASE_FINALIZE)` | `RecordPhase(RT_PHASE_FINALIZE)` |
| `WriteGmByPassDCache(turn, pos, phase)` | `Record(turn, pos, phase)` |

---

## 7. API 兼容性

### 7.1 旧 API 保留过渡

为减少迁移风险，保留旧 API 作为薄封装（标记 `__deprecated`），内部转调新 API:

```c
// 旧 API (过渡期保留)
[[deprecated("use Record instead")]]
__aicore__ inline void WriteGmByPassDCache(uint8_t turn, uint8_t position, uint8_t phase)
{
    Record(turn, position, phase);
}

[[deprecated("use RecordPhase instead")]]
__aicore__ inline void UpdatePhase(uint8_t phase)
{
    RecordPhase(phase);
}
```

### 7.2 Host 侧 workspace 分配

host 侧 5 处通过 `Mc2Tiling::RUNTIME_INFO_TOTAL_SIZE` 常量引用分配，结构体大小不变（512B），
无需改动 host 侧分配逻辑。

---

## 8. 性能开销评估

### 8.1 写开销

| 操作 | DCCI 刷写 | 频率 |
|---|---|---|
| Record (全量) | 固定头 cacheline + trace cacheline = 2 次 DCCI | 每 stage 入口 |
| RecordPhase | 同上 2 次 | 通信状态变更 |
| RecordPosition | 同上 2 次 | 计算阶段入口 |

每次 DCCI 为单 cacheline 刷写，开销 ~数十 cycle。alltoallmm 典型 tileCnt × 4 点插桩，
总开销在可接受范围（< 总执行时间 1%）。

### 8.2 workspace 开销

per-core 512B × 128 core = 64KB，与现有完全一致，无增量。

---

## 9. 未决问题

1. **pipeline 传递 turn/part 的方式**: 是给 `pipeline::Process()` 增加 `uint8_t turn` 参数，
   还是通过 context 结构体传递？倾向后者（不改 Process 签名）。
2. **trace 条目数 62 是否够**: tileCnt × 4 点（PREPARE+COMMIT+WAIT+TRANSPOSE）若 tileCnt=16
   则 64 条略超。可接受（最旧被覆盖，死锁卡早期 task）。若需更多可考虑压缩 entry 到 4B
   （去掉 cycle，但损失时序）。
3. **AIC 侧插桩**: AIC 不参与通信，`commPhase` 在 AIC 上意义不大。AIC 只用 `RecordPosition`
   记录 matmul 位置，commPhase 保持 INIT 或上一次值。是否需要为 AIC 单独定义 phase？
   倾向不单独定义，AIC 的 commPhase 字段语义为"当前通信视角的状态"。
