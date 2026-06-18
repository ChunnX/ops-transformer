# Flash Attention 公共流程架构方案设计文档

> 目标：将 cube/vec 两侧的 while 主循环抽取为公共流程，使用控制流+函数指针（不使用继承）实现差异注入。

## 1. 当前架构分析

### 1.1 现有流程

```
Cube侧: regbase → Init(cube) → Process(cube) → FlashAttention → while循环(任务分发+执行)
Vec侧:  regbase → Init(vec)  → Process(vec)  → FlashAttention → while循环(任务分发+执行) → FlashDecode
```

### 1.2 循环内公共逻辑（cube/vec 一致）

| 函数 | 说明 | 一致性 |
|------|------|--------|
| `ShouldDispatchTask` | 判断是否还有任务需要分发 | **完全一致** |
| `GetTaskDealMode` | 获取当前任务处理模式 | **主体逻辑一致**，vec 侧有 `mloop++` hook |
| `GetFASectionInfo` | 从 metadata 读取起点终点 | **主体一致**，vec 多读 `coreFirstTmpOutWsPos_` |
| while 循环骨架 | `while(shouldDispatch || validTaskCount)` | **结构一致** |

### 1.3 循环内差异逻辑（cube/vec 不同）

| 函数/流程 | Cube侧 | Vec侧 | 差异点 |
|------|------|------|------|
| `CreateTask` | 填充 `RunInfoXCube` | 填充 `RunInfoXVec`（多 `mloop, gIdx, preTokens` 等字段） | RunInfo 类型不同、字段数不同 |
| `ExecuteTask` | `ComputeMm1` + `ComputeMm2` | `ComputeVec1` + `ComputeVec2` | 完全不同的计算逻辑 |
| `UpdateAxisInfo` | 3 参数 `(bN2, gS1, s2)` | 4 参数 `(taskDealMode, bN2, gS1, s2)` + 追踪 `accGS1Loops` | 签名不同、vec 有额外计数 |
| `DealZeroActSeqLen` | **不存在** | 清零输出行 | vec 独有 |
| `DEAL_ZERO` 分支 | 统一走 `UpdateAxisInfo + continue` | 单独分支 `DealZeroActSeqLen` + `UpdateAxisInfo` + `continue` | 分支结构不同 |
| 循环初始化 | 无额外状态 | `mloop=0, accGS1Loops=0, headS2Split=false, tailS2Split=false` | vec 有额外状态初始化 |
| `GetTaskDealMode` 内 | 无 `mloop++` | `s2Cur == curS2Start` 时 `mloop++` | vec 有 hook |
| FlashDecode | **不存在** | section 循环内调用 | vec 独有（在公共循环外） |

### 1.4 目标架构

```
Vec侧:  regbase → Init(vec) → Process(vec) → 公共流程(common) → 任务分发(vec) → 任务执行(vec)
Cube侧: regbase → Init(cube) → Process(cube) → 公共流程(common) → 任务分发(cube) → 任务执行(cube)
```

公共流程包含 while 主循环：

```
获取metadata起点终点 → 开始循环 → 获取任务类型 → 任务分发 → 跳到下一个循环 → 结束循环
                                       ↓ (不需要分发)
                                    执行任务 → 结束任务
```

---

## 2. 方案A: 模板策略 + Kernel引用注入

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    flash_attn_task_loop_policy.h                     │
│                       (公共头文件，两侧共享)                           │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐   │
│  │   TaskLoopState   │    │        CommonTaskLoop<Kernel,Policy> │   │
│  │   (循环状态结构体)  │    │  (公共 while 循环模板函数)            │   │
│  │                    │    │                                      │   │
│  │  bN2Start_        │    │  ShouldDispatchTask  ← 公共          │   │
│  │  gS1OStart_       │    │  GetTaskDealMode     ← 公共+Policy   │   │
│  │  ...              │    │  Policy::CreateTask  ← 差异注入      │   │
│  │  mloop (vec only) │    │  Policy::ExecuteTask ← 差异注入      │   │
│  │  accGS1Loops      │    │  Policy::UpdateAxis  ← 差异注入      │   │
│  │  headS2Split      │    │  Policy::DealZero    ← vec only     │   │
│  │  tailS2Split      │    │  Policy::OnS2Start   ← vec only     │   │
│  └──────────────────┘    │  Policy::InitState   ← 差异注入      │   │
│                           │  Policy::GetSection  ← 差异注入      │   │
│                           └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

        ↕ Policy::xxx(self, state, ...)  ↕ ShouldDispatchTask/GetTaskDealMode(self, state)

┌──────────────────────────┐          ┌──────────────────────────────┐
│  flash_attn_gqa_cube.h   │          │  flash_attn_gqa_vec.h        │
│                          │          │                              │
│  struct CubePolicy {     │          │  struct VecPolicy {          │
│    HAS_DEAL_ZERO = false │          │    HAS_DEAL_ZERO = true      │
│    CreateTask  → Cube    │          │    CreateTask  → Vec         │
│    ExecuteTask → Cube    │          │    ExecuteTask → Vec         │
│    UpdateAxis  → Cube    │          │    UpdateAxis  → Vec(+mode)  │
│    OnS2Start   → no-op   │          │    OnS2Start   → mloop++    │
│    DealZero    → N/A     │          │    DealZero    → DealZero..  │
│    InitState   → empty   │          │    InitState   → mloop=0..   │
│    GetSection  → Cube    │          │    GetSection  → Vec(+wsPos) │
│  }                        │          │  }                            │
│                          │          │                              │
│  Process() {             │          │  Process() {                 │
│    InitBuffers();        │          │    CrossCoreBufferInit();    │
│    AllocEventID();       │          │    InitBuffers();            │
│    for (s) {             │          │    AllocEventID();           │
│      CommonTaskLoop      │          │    for (s) {                 │
│        <Cube, CubePolicy>│          │      CommonTaskLoop          │
│        (*this, s);       │          │        <Vec, VecPolicy>      │
│    }                     │          │        (*this, s);           │
│    FreeEventID();        │          │      FlashDecode(s);         │
│  }                       │          │    }                         │
│                          │          │    FreeEventID();            │
└──────────────────────────┘          └──────────────────────────────┘
```

### 2.2 核心数据结构

```cpp
// TaskLoopState: 从两侧 Kernel 成员变量中提取的循环状态
struct TaskLoopState {
    // ──── 公共字段 ────
    uint32_t bN2Start_, gS1OStart_, s2OStart_;
    uint32_t bN2End_, gS1OEnd_, s2OEnd_;
    uint32_t bN2Cur, gS1Cur, s2Cur;
    uint32_t prevBN2Idx, prevGS1Idx, prevBIdx;
    int64_t actSeqLensKv = 0;
    int64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t createdTaskCount = 0;
    uint32_t executedTaskCount = 0;
    uint32_t validTaskCount = 0;
    bool shouldDispatchTask = true;
    // ──── Vec扩展字段(cube侧不使用但存在) ────
    uint32_t mloop = 0;
    uint32_t accGS1Loops = 0;
    bool headS2Split = false;
    bool tailS2Split = false;
    uint32_t coreFirstTmpOutWsPos_ = 0;
};
```

### 2.3 Policy 约束接口（文档性，无基类）

每个 Policy 必须提供以下 static `__aicore__ inline` 方法：

| 方法 | 签名 | 说明 |
|------|------|------|
| `HAS_DEAL_ZERO` | `static constexpr bool` | vec=true, cube=false |
| `InitLoopState` | `void(Kernel&, TaskLoopState&)` | 初始化侧特有循环状态 |
| `GetSectionInfo` | `void(Kernel&, TaskLoopState&, uint32_t)` | 从 metadata 读取起点终点 |
| `CreateTask` | `void(Kernel&, TaskLoopState&, uint64_t, uint32_t, uint32_t, uint32_t)` | 创建任务(runinfo赋值) |
| `ExecuteTask` | `void(Kernel&, TaskLoopState&, uint64_t)` | 执行任务(ComputeMm1/Mm2 或 Vec1/Vec2) |
| `UpdateAxisInfo` | `void(Kernel&, TaskLoopState&, TASK_DEAL_MODE)` | 轴信息更新 |
| `DealZeroActSeqLen` | `void(Kernel&, uint32_t)` | 清零输出行(vec only) |
| `OnS2Start` | `void(Kernel&, TaskLoopState&)` | s2Cur==curS2Start时的hook(vec: mloop++) |
| `CalcCurS2StartEnd` | `void(Kernel&, TaskLoopState&, uint32_t, uint32_t)` | 计算当前S2起止位置 |

### 2.4 公共循环函数

```cpp
template<typename Kernel, typename Policy>
__aicore__ inline void CommonTaskLoop(Kernel &self, uint32_t sectionIdx)
{
    TaskLoopState state;
    Policy::InitLoopState(self, state);
    Policy::GetSectionInfo(self, state, sectionIdx);

    state.bN2Cur = state.bN2Start_;
    state.gS1Cur = state.gS1OStart_;
    state.s2Cur = state.s2OStart_;
    state.prevBN2Idx = state.bN2Cur;
    state.prevGS1Idx = state.gS1Cur;
    state.createdTaskCount = 0;
    state.executedTaskCount = 0;
    state.validTaskCount = 0;
    state.shouldDispatchTask = true;

    while (state.shouldDispatchTask || state.validTaskCount) {
        state.shouldDispatchTask = ShouldDispatchTask(self, state);
        if (state.shouldDispatchTask) {
            TASK_DEAL_MODE mode = GetTaskDealMode(self, state, Policy);
            if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                Policy::CreateTask(self, state, state.createdTaskCount,
                                   state.bN2Cur, state.gS1Cur, state.s2Cur);
                state.createdTaskCount++;
                state.validTaskCount++;
                Policy::UpdateAxisInfo(self, state, mode);
            } else if constexpr (Policy::HAS_DEAL_ZERO) {
                if (mode == TASK_DEAL_MODE::DEAL_ZERO) {
                    Policy::DealZeroActSeqLen(self, state.bN2Cur);
                    Policy::UpdateAxisInfo(self, state, mode);
                    continue;
                } else {
                    Policy::UpdateAxisInfo(self, state, mode);
                    continue;
                }
            } else {
                Policy::UpdateAxisInfo(self, state, mode);
                continue;
            }
        }
        if (state.validTaskCount) {
            Policy::ExecuteTask(self, state, state.executedTaskCount);
            state.executedTaskCount++;
            if (state.executedTaskCount > Kernel::PRELOAD_N) {
                state.validTaskCount--;
            }
        }
    }
}
```

### 2.5 公共逻辑函数

```cpp
// ShouldDispatchTask: 完全公共，两侧一致
template<typename Kernel>
__aicore__ inline bool ShouldDispatchTask(Kernel&, TaskLoopState &s) {
    return (s.bN2Cur != s.bN2End_ || s.gS1Cur != s.gS1OEnd_ || s.s2Cur != s.s2OEnd_);
}

// GetTaskDealMode: 公共主体 + Policy Hook
template<typename Kernel, typename Policy>
__aicore__ inline TASK_DEAL_MODE GetTaskDealMode(Kernel &self, TaskLoopState &s) {
    bool isFirstTask = (s.bN2Cur == s.bN2Start_) && (s.gS1Cur == s.gS1OStart_) && (s.s2Cur == s.s2OStart_);
    uint32_t bIdx = s.bN2Cur / self.constInfoX.n2Size;

    if (isFirstTask || s.prevBIdx != bIdx) {
        s.prevBIdx = bIdx;
        s.actSeqLensKv = self.kvSeqLensParser.GetActualSeqLength(bIdx);
        s.actSeqLensQ = self.qSeqLensParser.GetActualSeqLength(bIdx);
    }

    uint64_t s2LoopTimes = (s.actSeqLensKv + Kernel::s2BaseSize - 1) / Kernel::s2BaseSize;
    uint64_t gS1Size = s.actSeqLensQ * self.constInfoX.gSize;
    uint64_t gS1LoopTimes = (gS1Size + Kernel::mBaseSize - 1) / Kernel::mBaseSize;

    if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
        return (s.gS1Cur == 0 && s.s2Cur == 0) ?
               TASK_DEAL_MODE::DEAL_ZERO : TASK_DEAL_MODE::SKIP_ZERO;
    }

    if (isFirstTask || s.bN2Cur != s.prevBN2Idx || s.gS1Cur != s.prevGS1Idx) {
        Policy::CalcCurS2StartEnd(self, s, s.bN2Cur, s.gS1Cur);
        s.prevBN2Idx = s.bN2Cur;
        s.prevGS1Idx = s.gS1Cur;
    }

    if (s.curS2Start < s.curS2End && s.s2Cur >= s.curS2End) {
        return TASK_DEAL_MODE::S2_END;
    }
    if (s.s2Cur < s.curS2Start || s.s2Cur >= s.curS2End) {
        return TASK_DEAL_MODE::SKIP;
    }

    // Hook: vec 侧 mloop++
    if (s.s2Cur == s.curS2Start) {
        Policy::OnS2Start(self, s);
    }

    return TASK_DEAL_MODE::CREATE_TASK;
}
```

### 2.6 Cube侧 Policy 实现

```cpp
// flash_attn_gqa_cube.h
struct CubePolicy {
    static constexpr bool HAS_DEAL_ZERO = false;

    static __aicore__ inline void InitLoopState(CubeKernel&, TaskLoopState&) {}

    static __aicore__ inline void GetSectionInfo(CubeKernel &self, TaskLoopState &s, uint32_t sectionIdx) {
        s.bN2Start_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        s.gS1OStart_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s.s2OStart_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        s.bN2End_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        s.gS1OEnd_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s.s2OEnd_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXCube.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
    }

    static __aicore__ inline void CreateTask(CubeKernel &self, TaskLoopState &s,
                                              uint64_t loop, uint32_t b, uint32_t g, uint32_t s2) {
        RunInfoXCube &runInfo = self.taskRunInfo[loop & CubeKernel::PRELOAD_TASK_CACHE_MASK];
        self.CalcParams(loop, b, g, s2, runInfo);
        runInfo.isValid = true;
    }

    static __aicore__ inline void ExecuteTask(CubeKernel &self, TaskLoopState &s, uint64_t loop) {
        self.ExecuteTask(loop, self.taskRunInfo);
    }

    static __aicore__ inline void UpdateAxisInfo(CubeKernel &self, TaskLoopState &s, TASK_DEAL_MODE) {
        self.UpdateAxisInfo(s.bN2Cur, s.gS1Cur, s.s2Cur);
    }

    static __aicore__ inline void OnS2Start(CubeKernel&, TaskLoopState&) {}

    static __aicore__ inline void CalcCurS2StartEnd(CubeKernel &self, TaskLoopState &s,
                                                     uint32_t bN2Cur, uint32_t gS1Cur) {
        self.CalcCurS2StartEnd(bN2Cur, gS1Cur);  // 写 self.curS2Start/curS2End → 同步到 s
        s.curS2Start = self.curS2Start;
        s.curS2End = self.curS2End;
    }
};
```

### 2.7 Vec侧 Policy 实现

```cpp
// flash_attn_gqa_vec.h
struct VecPolicy {
    static constexpr bool HAS_DEAL_ZERO = true;

    static __aicore__ inline void InitLoopState(VecKernel &self, TaskLoopState &s) {
        s.mloop = 0;
        s.accGS1Loops = 0;
        s.headS2Split = false;
        s.tailS2Split = false;
    }

    static __aicore__ inline void GetSectionInfo(VecKernel &self, TaskLoopState &s, uint32_t sectionIdx) {
        // 与 cube 一致 + 额外读取 coreFirstTmpOutWsPos_
        s.bN2Start_ = self.faMetaDataGm.GetValue(...);
        // ...
        s.coreFirstTmpOutWsPos_ = self.faMetaDataGm.GetValue(
            self.GetFAMetaDataIndex(self.constInfoXVec.aicIdx,
                                   FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    static __aicore__ inline void CreateTask(VecKernel &self, TaskLoopState &s,
                                              uint64_t loop, uint32_t b, uint32_t g, uint32_t s2) {
        RunInfoXVec &runInfo = self.taskRunInfo[loop & VecKernel::PRELOAD_TASK_CACHE_MASK];
        self.CalcParams(loop, b, g, s2, runInfo, s);
        runInfo.isValid = true;
    }

    static __aicore__ inline void ExecuteTask(VecKernel &self, TaskLoopState &s, uint64_t loop) {
        self.ExecuteTask(loop, self.taskRunInfo);
    }

    static __aicore__ inline void UpdateAxisInfo(VecKernel &self, TaskLoopState &s, TASK_DEAL_MODE mode) {
        self.UpdateAxisInfo(mode, s.bN2Cur, s.gS1Cur, s.s2Cur);
    }

    static __aicore__ inline void DealZeroActSeqLen(VecKernel &self, uint32_t bN2Cur) {
        self.DealZeroActSeqLen(bN2Cur);
    }

    static __aicore__ inline void OnS2Start(VecKernel&, TaskLoopState &s) {
        s.mloop++;
    }

    static __aicore__ inline void CalcCurS2StartEnd(VecKernel &self, TaskLoopState &s,
                                                     uint32_t bN2Cur, uint32_t gS1Cur) {
        self.CalcCurS2StartEnd(bN2Cur, gS1Cur);
        s.curS2Start = self.curS2Start;
        s.curS2End = self.curS2End;
        // vec侧同步 headS2Split/tailS2Split 到 s
        s.headS2Split = self.headS2Split;
        s.tailS2Split = self.tailS2Split;
    }
};
```

### 2.8 优缺点

| 维度 | 评价 |
|------|------|
| **运行开销** | 0（编译期模板展开，Policy static inline 直接调用） |
| **代码清晰度** | 中等（Policy 结构体需理解模板注入机制） |
| **差异扩展性** | 加 Policy 方法即可，公共循环不变 |
| **状态管理** | `TaskLoopState` 独立，与 Kernel 成员分离 |
| **类型安全** | 高（模板强类型） |
| **vec扩展字段** | `TaskLoopState` 包含 vec 字段，cube 侧冗余 |
| **双向引用** | Policy 需要 Kernel 类型 → 需要前向声明或延迟 include |

---

## 3. 方案B: 函数指针表 + 共享上下文

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    flash_attn_task_loop_common.h                     │
│                       (公共头文件，两侧共享)                           │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐   │
│  │   TaskLoopCtx     │    │         TaskLoopOps                  │   │
│  │   (循环状态上下文)  │    │  (函数指针表: 8个函数指针 + 1个bool) │   │
│  │                    │    │                                      │   │
│  │  公共字段:         │    │  createTask     ──→ Vec/Cube实现     │   │
│  │   bN2Start_       │    │  executeTask    ──→ Vec/Cube实现     │   │
│  │   gS1OStart_      │    │  updateAxis     ──→ Vec/Cube实现     │   │
│  │   ...             │    │  dealZero       ──→ Vec only/nullptr │   │
│  │  Vec扩展字段:      │    │  onS2Start      ──→ Vec only/nullptr│   │
│  │   mloop           │    │  getSectionInfo ──→ Vec/Cube实现     │   │
│  │   accGS1Loops     │    │  calcS2StartEnd ──→ Vec/Cube实现     │   │
│  │   headS2Split     │    │  hasDealZero    ──→ true/false       │   │
│  │   tailS2Split     │    └──────────────────────────────────────┘   │
│  └──────────────────┘                                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CommonTaskLoop(ctx, ops, kernelPtr, sectionIdx)             │   │
│  │  (公共 while 循环，纯函数，无模板)                             │   │
│  │                                                              │   │
│  │  ops.getSectionInfo(ctx, kernelPtr, sectionIdx)              │   │
│  │  while (...) {                                               │   │
│  │    ShouldDispatchTask(ctx)          ← 内联公共函数            │   │
│  │    GetTaskDealMode(ctx, ops, kp)    ← 内联公共+ops.onS2Start │   │
│  │    ops.createTask(...)              ← 函数指针调用            │   │
│  │    ops.updateAxis(...)              ← 函数指针调用            │   │
│  │    ops.executeTask(...)             ← 函数指针调用            │   │
│  │  }                                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

        ↕ ops.xxx(ctx, kernelPtr, ...)   kernelPtr = (void*)this

┌──────────────────────────┐          ┌──────────────────────────────┐
│  flash_attn_gqa_cube.h   │          │  flash_attn_gqa_vec.h        │
│                          │          │                              │
│  static CubeCreateTask() │          │  static VecCreateTask()      │
│  static CubeExecuteTask()│          │  static VecExecuteTask()     │
│  static CubeUpdateAxis() │          │  static VecUpdateAxis()      │
│  static CubeGetSection() │          │  static VecGetSection()      │
│  static CubeCalcS2()     │          │  static VecCalcS2()          │
│  // dealZero=nullptr     │          │  static VecDealZero()        │
│  // onS2Start=nullptr    │          │  static VecOnS2Start()       │
│                          │          │                              │
│  Process() {             │          │  Process() {                 │
│    InitOps();            │          │    InitOps();                │
│    InitBuffers();        │          │    CrossCoreBufferInit();    │
│    AllocEventID();       │          │    InitBuffers();            │
│    for (s) {             │          │    AllocEventID();           │
│      CommonTaskLoop(     │          │    for (s) {                 │
│        ctx, ops, this, s)│          │      CommonTaskLoop(         │
│    }                     │          │        ctx, ops, this, s);   │
│    FreeEventID();        │          │      FlashDecode(s);         │
│  }                       │          │    }                         │
│                          │          │    FreeEventID();            │
└──────────────────────────┘          └──────────────────────────────┘
```

### 3.2 核心数据结构

```cpp
// TaskLoopCtx: 循环状态上下文，包含所有循环内需要的变量
struct TaskLoopCtx {
    // ──── 公共字段 ────
    uint32_t bN2Start_, gS1OStart_, s2OStart_;
    uint32_t bN2End_, gS1OEnd_, s2OEnd_;
    uint32_t bN2Cur, gS1Cur, s2Cur;
    uint32_t prevBN2Idx, prevGS1Idx, prevBIdx;
    int64_t actSeqLensKv = 0;
    int64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t createdTaskCount = 0;
    uint32_t executedTaskCount = 0;
    uint32_t validTaskCount = 0;
    bool shouldDispatchTask = true;
    // ──── Vec 扩展字段(cube侧不使用) ────
    uint32_t mloop = 0;
    uint32_t accGS1Loops = 0;
    bool headS2Split = false;
    bool tailS2Split = false;
    uint32_t coreFirstTmpOutWsPos_ = 0;
};

// TaskLoopOps: 函数指针表，运行时注入差异逻辑
struct TaskLoopOps {
    using CreateTaskFn   = void(*)(TaskLoopCtx&, void*, uint64_t, uint32_t, uint32_t, uint32_t);
    using ExecuteTaskFn  = void(*)(TaskLoopCtx&, void*, uint64_t);
    using UpdateAxisFn   = void(*)(TaskLoopCtx&, void*, TASK_DEAL_MODE);
    using DealZeroFn     = void(*)(void*, uint32_t);
    using OnS2StartFn    = void(*)(TaskLoopCtx&);
    using SectionInfoFn  = void(*)(TaskLoopCtx&, void*, uint32_t);
    using CalcS2Fn       = void(*)(TaskLoopCtx&, void*, uint32_t, uint32_t);

    CreateTaskFn   createTask;
    ExecuteTaskFn  executeTask;
    UpdateAxisFn   updateAxis;
    DealZeroFn     dealZero;       // nullptr for cube
    OnS2StartFn    onS2Start;      // nullptr for cube
    SectionInfoFn  getSectionInfo;
    CalcS2Fn       calcCurS2StartEnd;
    bool           hasDealZero;    // false for cube, true for vec
};
```

### 3.3 公共循环函数

```cpp
__aicore__ inline void CommonTaskLoop(TaskLoopCtx &ctx, TaskLoopOps &ops,
                                       void *kernelPtr, uint32_t sectionIdx)
{
    ops.getSectionInfo(ctx, kernelPtr, sectionIdx);

    ctx.bN2Cur = ctx.bN2Start_;
    ctx.gS1Cur = ctx.gS1OStart_;
    ctx.s2Cur = ctx.s2OStart_;
    ctx.prevBN2Idx = ctx.bN2Cur;
    ctx.prevGS1Idx = ctx.gS1Cur;
    ctx.createdTaskCount = 0;
    ctx.executedTaskCount = 0;
    ctx.validTaskCount = 0;
    ctx.shouldDispatchTask = true;

    while (ctx.shouldDispatchTask || ctx.validTaskCount) {
        ctx.shouldDispatchTask = ShouldDispatchTask(ctx);
        if (ctx.shouldDispatchTask) {
            TASK_DEAL_MODE mode = GetTaskDealMode(ctx, ops, kernelPtr);

            if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                ops.createTask(ctx, kernelPtr, ctx.createdTaskCount,
                               ctx.bN2Cur, ctx.gS1Cur, ctx.s2Cur);
                ctx.createdTaskCount++;
                ctx.validTaskCount++;
                ops.updateAxis(ctx, kernelPtr, mode);
            } else if (ops.hasDealZero && mode == TASK_DEAL_MODE::DEAL_ZERO) {
                ops.dealZero(kernelPtr, ctx.bN2Cur);
                ops.updateAxis(ctx, kernelPtr, mode);
                continue;
            } else {
                ops.updateAxis(ctx, kernelPtr, mode);
                continue;
            }
        }
        if (ctx.validTaskCount) {
            ops.executeTask(ctx, kernelPtr, ctx.executedTaskCount);
            ctx.executedTaskCount++;
            if (ctx.executedTaskCount > PRELOAD_N) {
                ctx.validTaskCount--;
            }
        }
    }
}
```

### 3.4 公共逻辑函数

```cpp
__aicore__ inline bool ShouldDispatchTask(TaskLoopCtx &ctx) {
    return (ctx.bN2Cur != ctx.bN2End_ || ctx.gS1Cur != ctx.gS1OEnd_ || ctx.s2Cur != ctx.s2OEnd_);
}

__aicore__ inline TASK_DEAL_MODE GetTaskDealMode(TaskLoopCtx &ctx, TaskLoopOps &ops, void *kernelPtr) {
    bool isFirstTask = (ctx.bN2Cur == ctx.bN2Start_) && (ctx.gS1Cur == ctx.gS1OStart_) && (ctx.s2Cur == ctx.s2OStart_);
    uint32_t bIdx = ctx.bN2Cur / ctx.n2Size;  // n2Size 需在 ctx 或 ops 中提供

    if (isFirstTask || ctx.prevBIdx != bIdx) {
        ctx.prevBIdx = bIdx;
        // 需通过 ops 获取 seqLens（因为依赖 Kernel 的 parser）
        ops.getSeqLens(ctx, kernelPtr, bIdx);
    }

    uint64_t s2LoopTimes = (ctx.actSeqLensKv + ctx.s2BaseSize - 1) / ctx.s2BaseSize;
    uint64_t gS1Size = ctx.actSeqLensQ * ctx.gSize;
    uint64_t gS1LoopTimes = (gS1Size + ctx.mBaseSize - 1) / ctx.mBaseSize;

    if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
        return (ctx.gS1Cur == 0 && ctx.s2Cur == 0) ?
               TASK_DEAL_MODE::DEAL_ZERO : TASK_DEAL_MODE::SKIP_ZERO;
    }

    if (isFirstTask || ctx.bN2Cur != ctx.prevBN2Idx || ctx.gS1Cur != ctx.prevGS1Idx) {
        ops.calcCurS2StartEnd(ctx, kernelPtr, ctx.bN2Cur, ctx.gS1Cur);
        ctx.prevBN2Idx = ctx.bN2Cur;
        ctx.prevGS1Idx = ctx.gS1Cur;
    }

    if (ctx.curS2Start < ctx.curS2End && ctx.s2Cur >= ctx.curS2End) {
        return TASK_DEAL_MODE::S2_END;
    }
    if (ctx.s2Cur < ctx.curS2Start || ctx.s2Cur >= ctx.curS2End) {
        return TASK_DEAL_MODE::SKIP;
    }

    // Hook: vec 侧 mloop++
    if (ctx.s2Cur == ctx.curS2Start && ops.onS2Start) {
        ops.onS2Start(ctx);
    }

    return TASK_DEAL_MODE::CREATE_TASK;
}
```

### 3.5 Cube侧 Ops 构建

```cpp
// flash_attn_gqa_cube.h
class FlashAttentionNoQuantGqaKernel {
    TaskLoopOps cubeOps_;
    TaskLoopCtx cubeCtx_;

    // 需要将编译期常量放入 ctx（因为公共函数不再有 Kernel 模板访问）
    // 或者在 TaskLoopCtx 中增加 mBaseSize/s2BaseSize/gSize/n2Size 等字段

    static __aicore__ inline void CubeCreateTask(TaskLoopCtx &ctx, void *kp,
                                                  uint64_t loop, uint32_t b, uint32_t g, uint32_t s2) {
        auto *self = static_cast<FlashAttentionNoQuantGqaKernel*>(kp);
        self->CreateTask(loop, b, g, s2, self->taskRunInfo_);
    }
    static __aicore__ inline void CubeExecuteTask(TaskLoopCtx &ctx, void *kp, uint64_t loop) {
        auto *self = static_cast<FlashAttentionNoQuantGqaKernel*>(kp);
        self->ExecuteTask(loop, self->taskRunInfo_);
    }
    static __aicore__ inline void CubeUpdateAxis(TaskLoopCtx &ctx, void *kp, TASK_DEAL_MODE) {
        auto *self = static_cast<FlashAttentionNoQuantGqaKernel*>(kp);
        self->UpdateAxisInfo(ctx.bN2Cur, ctx.gS1Cur, ctx.s2Cur);
    }
    static __aicore__ inline void CubeGetSection(TaskLoopCtx &ctx, void *kp, uint32_t sectionIdx) {
        auto *self = static_cast<FlashAttentionNoQuantGqaKernel*>(kp);
        self->GetFASectionInfo(sectionIdx);
        ctx.bN2Start_ = self->bN2Start_;
        ctx.gS1OStart_ = self->gS1OStart_;
        ctx.s2OStart_ = self->s2OStart_;
        ctx.bN2End_ = self->bN2End_;
        ctx.gS1OEnd_ = self->gS1OEnd_;
        ctx.s2OEnd_ = self->s2OEnd_;
    }
    static __aicore__ inline void CubeCalcS2(TaskLoopCtx &ctx, void *kp, uint32_t b, uint32_t g) {
        auto *self = static_cast<FlashAttentionNoQuantGqaKernel*>(kp);
        self->CalcCurS2StartEnd(b, g);
        ctx.curS2Start = self->curS2Start;
        ctx.curS2End = self->curS2End;
    }

    __aicore__ inline void InitOps() {
        cubeOps_.createTask       = CubeCreateTask;
        cubeOps_.executeTask      = CubeExecuteTask;
        cubeOps_.updateAxis       = CubeUpdateAxis;
        cubeOps_.dealZero         = nullptr;
        cubeOps_.onS2Start        = nullptr;
        cubeOps_.getSectionInfo   = CubeGetSection;
        cubeOps_.calcCurS2StartEnd = CubeCalcS2;
        cubeOps_.hasDealZero      = false;
    }

    __aicore__ inline void Process() {
        if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
            InitOps();
            InitBuffers();
            AllocEventID();
            for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
                CommonTaskLoop(cubeCtx_, cubeOps_, static_cast<void*>(this), sectionIdx);
            }
            FreeEventID();
            CrossCoreBufferUnInit();
        }
    }
};
```

### 3.6 Vec侧 Ops 构建

```cpp
// flash_attn_gqa_vec.h (类似结构，增加 dealZero/onS2Start)
static __aicore__ inline void VecDealZero(void *kp, uint32_t bN2Cur) {
    auto *self = static_cast<VecKernel*>(kp);
    self->DealZeroActSeqLen(bN2Cur);
}
static __aicore__ inline void VecOnS2Start(TaskLoopCtx &ctx) {
    ctx.mloop++;
}
static __aicore__ inline void VecUpdateAxis(TaskLoopCtx &ctx, void *kp, TASK_DEAL_MODE mode) {
    auto *self = static_cast<VecKernel*>(kp);
    self->UpdateAxisInfo(mode, ctx.bN2Cur, ctx.gS1Cur, ctx.s2Cur);
}

__aicore__ inline void InitOps() {
    vecOps_.createTask       = VecCreateTask;
    vecOps_.executeTask      = VecExecuteTask;
    vecOps_.updateAxis       = VecUpdateAxis;
    vecOps_.dealZero         = VecDealZero;
    vecOps_.onS2Start        = VecOnS2Start;
    vecOps_.getSectionInfo   = VecGetSection;
    vecOps_.calcCurS2StartEnd = VecCalcS2;
    vecOps_.hasDealZero      = true;
}
```

### 3.7 优缺点

| 维度 | 评价 |
|------|------|
| **运行开销** | ~2-4次函数指针间接调用/循环迭代（NPU hot loop 性能风险） |
| **代码清晰度** | 高（Ops/Ctx 结构直观，注入方式一目了然） |
| **差异扩展性** | 加函数指针即可 |
| **状态管理** | `TaskLoopCtx` 独立，需同步 Kernel→Ctx 的字段 |
| **类型安全** | 低（`void* kernelPtr`，static_cast 无编译期校验） |
| **vec扩展字段** | `TaskLoopCtx` 包含 vec 字段，cube 侧冗余 |
| **调试友好** | 可运行时打印 Ops 内容，便于调试 |
| **适用场景** | 快速原型、调试阶段、性能非极致敏感场景 |

---

## 4. 方案C: `if constexpr` 编译期分支 + 统一模板函数

### 4.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    flash_attn_task_loop_common.h                     │
│                       (公共头文件，两侧共享)                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CommonTaskLoop<Kernel, isCubeSide>(self, sectionIdx)        │   │
│  │  (公共 while 循环模板函数，isCubeSide = constexpr bool)       │   │
│  │                                                              │   │
│  │  self.GetFASectionInfo(sectionIdx)  ← Kernel自己的方法       │   │
│  │  self.ShouldDispatchTask(...)       ← Kernel自己的方法       │   │
│  │  self.GetTaskDealMode(...)          ← Kernel自己的方法       │   │
│  │                                                              │   │
│  │  if constexpr (!isCubeSide) {                                │   │
│  │    self.mloop = 0; ...  ← vec 初始化                        │   │
│  │  }                                                           │   │
│  │                                                              │   │
│  │  while (...) {                                               │   │
│  │    shouldDispatch = self.ShouldDispatchTask(...)             │   │
│  │    if (shouldDispatch) {                                     │   │
│  │      mode = self.GetTaskDealMode(...)                        │   │
│  │      if constexpr (!isCubeSide) {                            │   │
│  │        // vec: DEAL_ZERO 分支                                │   │
│  │        if (mode == DEAL_ZERO) {                              │   │
│  │          self.DealZeroActSeqLen(bN2Cur);                     │   │
│  │          self.UpdateAxisInfo(mode, bN2, gS1, s2);            │   │
│  │          continue;                                            │   │
│  │        } else {                                               │   │
│  │          self.UpdateAxisInfo(mode, bN2, gS1, s2);            │   │
│  │          continue;                                            │   │
│  │        }                                                      │   │
│  │      } else {                                                 │   │
│  │        // cube: 无 DEAL_ZERO                                  │   │
│  │        self.UpdateAxisInfo(bN2, gS1, s2);                    │   │
│  │        if (mode != CREATE_TASK) { continue; }                │   │
│  │      }                                                        │   │
│  │      if (mode == CREATE_TASK) {                               │   │
│  │        self.CreateTask(...);                                  │   │
│  │        if constexpr (!isCubeSide) {                           │   │
│  │          self.UpdateAxisInfo(mode, ...);                      │   │
│  │        }                                                      │   │
│  │      }                                                        │   │
│  │    }                                                          │   │
│  │    if (validTaskCount) {                                      │   │
│  │      self.ExecuteTask(...);                                   │   │
│  │    }                                                          │   │
│  │  }                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

        ↕ self.xxx()  直接调用 Kernel 成员方法

┌──────────────────────────┐          ┌──────────────────────────────┐
│  flash_attn_gqa_cube.h   │          │  flash_attn_gqa_vec.h        │
│                          │          │                              │
│  Process() {             │          │  Process() {                 │
│    InitBuffers();        │          │    CrossCoreBufferInit();    │
│    AllocEventID();       │          │    InitBuffers();            │
│    for (s) {             │          │    AllocEventID();           │
│      CommonTaskLoop      │          │    for (s) {                 │
│        <CubeKernel, true>│          │      CommonTaskLoop          │
│        (*this, s);       │          │        <VecKernel, false>    │
│    }                     │          │        (*this, s);           │
│    FreeEventID();        │          │      FlashDecode(s);         │
│  }                       │          │    }                         │
│                          │          │    FreeEventID();            │
└──────────────────────────┘          └──────────────────────────────┘
```

### 4.2 核心设计思路

- 不引入任何额外结构体（Policy/Ops/Ctx），差异全部通过 `if constexpr <isCubeSide>` 在公共函数内处理
- Kernel 类保持现有方法签名不变，公共函数直接调用 `self.xxx()`
- `isCubeSide` 为 `constexpr bool`，编译期完全消除不相关分支
- 公共函数是 Kernel 的友元模板函数或独立头文件中的模板函数

### 4.3 公共循环函数

```cpp
// flash_attn_task_loop_common.h
template<typename Kernel, bool isCubeSide>
__aicore__ inline void CommonTaskLoop(Kernel &self, uint32_t sectionIdx)
{
    self.GetFASectionInfo(sectionIdx);

    uint32_t bN2Cur = self.bN2Start_;
    uint32_t gS1Cur = self.gS1OStart_;
    uint32_t s2Cur = self.s2OStart_;
    self.prevBN2Idx = bN2Cur;
    self.prevGS1Idx = gS1Cur;

    // ──── vec 侧额外初始化 ────
    if constexpr (!isCubeSide) {
        self.mloop = 0;
        self.accGS1Loops = 0;
        self.headS2Split = false;
        self.tailS2Split = false;
    }

    uint64_t createdTaskCount = 0;
    uint32_t executedTaskCount = 0;
    bool shouldDispatchTask = true;
    uint32_t validTaskCount = 0;

    while (shouldDispatchTask || validTaskCount) {
        shouldDispatchTask = self.ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = self.GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);

            if constexpr (!isCubeSide) {
                // ──── vec 侧分支处理 ────
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    self.CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    createdTaskCount++;
                    validTaskCount++;
                    self.UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    self.DealZeroActSeqLen(bN2Cur);
                    self.UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    self.UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            } else {
                // ──── cube 侧分支处理 ────
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    self.CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    createdTaskCount++;
                    validTaskCount++;
                }
                self.UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode != TASK_DEAL_MODE::CREATE_TASK) {
                    continue;
                }
            }
        }
        if (validTaskCount) {
            self.ExecuteTask(executedTaskCount);
            executedTaskCount++;
            if (executedTaskCount > Kernel::PRELOAD_N) {
                validTaskCount--;
            }
        }
    }
}
```

### 4.4 Kernel 类适配要求

两侧 Kernel 类需保证以下方法签名可用（cube/vec 各自已有，签名差异保留）：

| 方法 | Cube签名 | Vec签名 | 说明 |
|------|---------|---------|------|
| `GetFASectionInfo` | `void(uint32_t)` | `void(uint32_t)` | 一致（vec内部多读wsPos） |
| `ShouldDispatchTask` | `bool(uint32_t,uint32_t,uint32_t)` | `bool(uint32_t,uint32_t,uint32_t)` | 一致 |
| `GetTaskDealMode` | `TASK_DEAL_MODE(uint32_t,uint32_t,uint32_t)` | `TASK_DEAL_MODE(uint32_t,uint32_t,uint32_t)` | 一致（vec内部mloop++） |
| `CreateTask` | `void(uint64_t,uint32_t,uint32_t,uint32_t)` | `void(uint64_t,uint32_t,uint32_t,uint32_t)` | 一致（内部类型不同） |
| `ExecuteTask` | `void(uint64_t)` | `void(uint64_t)` | 一致（内部逻辑不同） |
| `UpdateAxisInfo` | `void(uint32_t&,uint32_t&,uint32_t&)` | `void(TASK_DEAL_MODE,uint32_t&,uint32_t&,uint32_t&)` | **签名不同！** |

**签名差异问题**: `UpdateAxisInfo` 的签名不同（cube 3参数 vs vec 4参数），需要在公共函数中用 `if constexpr` 区分调用。

### 4.5 Process 调用方式

```cpp
// Cube Process
__aicore__ inline void Process() {
    if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
        InitBuffers();
        AllocEventID();
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            CommonTaskLoop<FlashAttentionNoQuantGqaKernel<...>, true>(*this, sectionIdx);
        }
        FreeEventID();
        CrossCoreBufferUnInit();
    }
}

// Vec Process
__aicore__ inline void Process() {
    if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
        CrossCoreBufferInit();
        InitBuffers();
        AllocEventID();
    }
    for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
        if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
            CommonTaskLoop<FlashAttentionNoQuantGqaKernel<...>, false>(*this, sectionIdx);
        }
        FlashDecode(sectionIdx);
    }
    if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
        FreeEventID();
    }
}
```

### 4.6 优缺点

| 维度 | 评价 |
|------|------|
| **运行开销** | 0（编译期模板 + if constexpr 完全消除） |
| **代码清晰度** | 高（最简洁，无额外抽象层） |
| **差异扩展性** | 低（新差异需改公共函数，加 if constexpr） |
| **状态管理** | 直接使用 Kernel 成员变量，无需同步 |
| **类型安全** | 高（模板强类型） |
| **vec扩展字段** | 不冗余（`if constexpr` 限定了作用域） |
| **公共函数膨胀** | 风险中等（当前差异点约4处，尚可控） |
| **适用场景** | 差异点少且稳定，追求快速落地 |

---

## 5. 方案D: Hook链 + 枚举驱动

### 5.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    flash_attn_task_loop_hooks.h                      │
│                       (公共头文件，两侧共享)                           │
│                                                                      │
│  enum FiaSide { CUBE, VEC }                                         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CommonTaskLoop<FiaSide::SIDE, Kernel>(self, sectionIdx)     │   │
│  │  (公共 while 循环模板函数)                                    │   │
│  │                                                              │   │
│  │  using Hooks = TaskLoopHooks<SIDE, Kernel>;                  │   │
│  │                                                              │   │
│  │  Hooks::InitLoopState(self)                                  │   │
│  │  self.GetFASectionInfo(sectionIdx)                           │   │
│  │                                                              │   │
│  │  while (...) {                                               │   │
│  │    self.ShouldDispatchTask(...)    ← 公共                    │   │
│  │    self.GetTaskDealMode(...)       ← 公共(vec内部mloop++)    │   │
│  │                                                              │   │
│  │    // ── Hook点: 任务创建 ──                                 │   │
│  │    Hooks::CreateTask(self, ...)                               │   │
│  │    // ── Hook点: 轴更新 ──                                   │   │
│  │    Hooks::UpdateAxisInfo(self, mode, ...)                     │   │
│  │    // ── Hook点: DEAL_ZERO ──                                │   │
│  │    if constexpr (Hooks::HAS_DEAL_ZERO) {                     │   │
│  │      Hooks::DealZero(self, ...)                               │   │
│  │    }                                                          │   │
│  │    // ── Hook点: 任务执行 ──                                 │   │
│  │    Hooks::ExecuteTask(self, ...)                              │   │
│  │  }                                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  template<FiaSide SIDE, typename Kernel>                     │   │
│  │  struct TaskLoopHooks;   // 声明，不定义                      │   │
│  │                                                              │   │
│  │  // 特化在各自 Kernel 头文件中提供                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

        ↕ Hooks::xxx(self, ...)  编译期静态调用

┌──────────────────────────┐          ┌──────────────────────────────┐
│  flash_attn_gqa_cube.h   │          │  flash_attn_gqa_vec.h        │
│                          │          │                              │
│  TaskLoopHooks           │          │  TaskLoopHooks               │
│    <CUBE, CubeKernel>    │          │    <VEC, VecKernel>          │
│                          │          │                              │
│  HAS_DEAL_ZERO = false   │          │  HAS_DEAL_ZERO = true        │
│  InitLoopState → {}      │          │  InitLoopState → mloop=0...  │
│  CreateTask   → Cube     │          │  CreateTask   → Vec          │
│  ExecuteTask  → Cube     │          │  ExecuteTask  → Vec          │
│  UpdateAxis   → Cube     │          │  UpdateAxis   → Vec(+mode)  │
│  DealZero     → N/A      │          │  DealZero     → DealZero..  │
│                          │          │                              │
│  Process() {             │          │  Process() {                 │
│    CommonTaskLoop        │          │    CommonTaskLoop            │
│      <CUBE, CubeKernel>  │          │      <VEC, VecKernel>       │
│      (*this, s);         │          │      (*this, s);            │
│  }                       │          │    FlashDecode(s);           │
│                          │          │  }                           │
└──────────────────────────┘          └──────────────────────────────┘
```

### 5.2 核心数据结构

```cpp
// flash_attn_task_loop_hooks.h

enum class FiaSide : uint8_t {
    CUBE = 0,
    VEC  = 1,
    // 可扩展: ANTI_QUANT, FULL_QUANT ...
};

// Hook 特化声明（不提供默认实现，强制各侧显式特化）
template<FiaSide SIDE, typename Kernel>
struct TaskLoopHooks;
```

### 5.3 Hook 约束接口（文档性）

每个 `TaskLoopHooks<SIDE, Kernel>` 特化必须提供：

| Hook | 签名 | Cube实现 | Vec实现 | 说明 |
|------|------|---------|---------|------|
| `HAS_DEAL_ZERO` | `static constexpr bool` | `false` | `true` | 编译期分支消除 |
| `InitLoopState` | `static void(Kernel&)` | 空函数 | `mloop=0; accGS1Loops=0; ...` | 循环前初始化 |
| `CreateTask` | `static void(Kernel&, uint64_t, uint32_t, uint32_t, uint32_t)` | 调用 `self.CreateTask` | 调用 `self.CreateTask` | 差异在 Kernel 内部 |
| `ExecuteTask` | `static void(Kernel&, uint64_t)` | 调用 `self.ExecuteTask` | 调用 `self.ExecuteTask` | 差异在 Kernel 内部 |
| `UpdateAxisInfo` | `static void(Kernel&, TASK_DEAL_MODE, uint32_t&, uint32_t&, uint32_t&)` | `self.UpdateAxisInfo(b,g,s)` (忽略mode) | `self.UpdateAxisInfo(mode,b,g,s)` | 签名适配 |
| `DealZeroActSeqLen` | `static void(Kernel&, uint32_t)` | 不调用(编译期消除) | `self.DealZeroActSeqLen(b)` | vec 独有 |
| `GetSectionInfo` | `static void(Kernel&, uint32_t)` | `self.GetFASectionInfo(s)` | `self.GetFASectionInfo(s)` | vec 内部多读字段 |

### 5.4 公共循环函数

```cpp
// flash_attn_task_loop_hooks.h
template<FiaSide SIDE, typename Kernel>
__aicore__ inline void CommonTaskLoop(Kernel &self, uint32_t sectionIdx)
{
    using Hooks = TaskLoopHooks<SIDE, Kernel>;

    Hooks::InitLoopState(self);
    Hooks::GetSectionInfo(self, sectionIdx);

    uint32_t bN2Cur = self.bN2Start_;
    uint32_t gS1Cur = self.gS1OStart_;
    uint32_t s2Cur = self.s2OStart_;
    self.prevBN2Idx = bN2Cur;
    self.prevGS1Idx = gS1Cur;

    uint64_t createdTaskCount = 0;
    uint32_t executedTaskCount = 0;
    bool shouldDispatchTask = true;
    uint32_t validTaskCount = 0;

    while (shouldDispatchTask || validTaskCount) {
        shouldDispatchTask = self.ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE mode = self.GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);

            // ──── Hook: DEAL_ZERO 分支(vec only) ────
            if constexpr (Hooks::HAS_DEAL_ZERO) {
                if (mode == TASK_DEAL_MODE::DEAL_ZERO) {
                    Hooks::DealZeroActSeqLen(self, bN2Cur);
                    Hooks::UpdateAxisInfo(self, mode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    Hooks::UpdateAxisInfo(self, mode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            } else {
                // cube: 无 DEAL_ZERO 分支
                if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                    Hooks::CreateTask(self, createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    createdTaskCount++;
                    validTaskCount++;
                }
                Hooks::UpdateAxisInfo(self, mode, bN2Cur, gS1Cur, s2Cur);
                if (mode != TASK_DEAL_MODE::CREATE_TASK) { continue; }
            }

            // vec: CREATE_TASK 分支（cube 已在上面处理）
            if constexpr (Hooks::HAS_DEAL_ZERO) {
                if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                    Hooks::CreateTask(self, createdTaskCount, bN2Cur, gS1Cur, s2Cur);
                    createdTaskCount++;
                    validTaskCount++;
                    Hooks::UpdateAxisInfo(self, mode, bN2Cur, gS1Cur, s2Cur);
                }
            }
        }
        if (validTaskCount) {
            Hooks::ExecuteTask(self, executedTaskCount);
            executedTaskCount++;
            if (executedTaskCount > Kernel::PRELOAD_N) { validTaskCount--; }
        }
    }
}
```

### 5.5 Cube侧 Hook 特化

```cpp
// flash_attn_gqa_cube.h
using CubeKernel = FlashAttentionNoQuantGqaKernel<INPUT_T, T, layout, ...>;

template<> struct TaskLoopHooks<FiaSide::CUBE, CubeKernel> {
    static constexpr bool HAS_DEAL_ZERO = false;

    static __aicore__ inline void InitLoopState(CubeKernel&) {}

    static __aicore__ inline void GetSectionInfo(CubeKernel &self, uint32_t sectionIdx) {
        self.GetFASectionInfo(sectionIdx);
    }

    static __aicore__ inline void CreateTask(CubeKernel &self, uint64_t loop,
                                              uint32_t b, uint32_t g, uint32_t s2) {
        self.CreateTask(loop, b, g, s2, self.taskRunInfo_);
    }

    static __aicore__ inline void ExecuteTask(CubeKernel &self, uint64_t loop) {
        self.ExecuteTask(loop, self.taskRunInfo_);
    }

    static __aicore__ inline void UpdateAxisInfo(CubeKernel &self, TASK_DEAL_MODE,
                                                  uint32_t &b, uint32_t &g, uint32_t &s) {
        self.UpdateAxisInfo(b, g, s);  // cube: 不传 mode
    }
};
```

### 5.6 Vec侧 Hook 特化

```cpp
// flash_attn_gqa_vec.h
using VecKernel = FlashAttentionNoQuantGqaKernel<INPUT_T, T, OUTPUT_T, VecFdBlockType, ...>;

template<> struct TaskLoopHooks<FiaSide::VEC, VecKernel> {
    static constexpr bool HAS_DEAL_ZERO = true;

    static __aicore__ inline void InitLoopState(VecKernel &self) {
        self.mloop = 0;
        self.accGS1Loops = 0;
        self.headS2Split = false;
        self.tailS2Split = false;
    }

    static __aicore__ inline void GetSectionInfo(VecKernel &self, uint32_t sectionIdx) {
        self.GetFASectionInfo(sectionIdx);  // vec 内部多读 coreFirstTmpOutWsPos_
    }

    static __aicore__ inline void CreateTask(VecKernel &self, uint64_t loop,
                                              uint32_t b, uint32_t g, uint32_t s2) {
        self.CreateTask(loop, b, g, s2, self.taskRunInfo_);
    }

    static __aicore__ inline void ExecuteTask(VecKernel &self, uint64_t loop) {
        self.ExecuteTask(loop, self.taskRunInfo_);
    }

    static __aicore__ inline void UpdateAxisInfo(VecKernel &self, TASK_DEAL_MODE mode,
                                                  uint32_t &b, uint32_t &g, uint32_t &s) {
        self.UpdateAxisInfo(mode, b, g, s);  // vec: 传 mode
    }

    static __aicore__ inline void DealZeroActSeqLen(VecKernel &self, uint32_t bN2Cur) {
        self.DealZeroActSeqLen(bN2Cur);
    }
};
```

### 5.7 优缺点

| 维度 | 评价 |
|------|------|
| **运行开销** | 0（编译期模板特化 + if constexpr） |
| **代码清晰度** | 高（Hook 点命名明确，职责分离清晰） |
| **差异扩展性** | 高（新增 Hook 点，不改公共循环骨架） |
| **状态管理** | 直接使用 Kernel 成员变量，无需同步 |
| **类型安全** | 高（模板特化强类型） |
| **vec扩展字段** | 不冗余（只在 vec Hook 中访问） |
| **FiaSide 可扩展** | 是（未来 ANTI_QUANT/FULL_QUANT 可加新枚举值） |
| **双向 include** | hooks.h 需前向声明 Kernel，或各侧延迟特化 |
| **适用场景** | 差异持续增长，长期维护 |

---

## 6. 方案综合对比

| 维度 | 方案A (Policy) | 方案B (FnTable) | 方案C (constexpr bool) | 方案D (Hook链) |
|------|:---------:|:---------:|:---------:|:---------:|
| **运行开销** | 0 | ~2-4次/循环 | 0 | 0 |
| **代码清晰度** | 中 | 高 | 高(简洁) | 高(结构化) |
| **差异扩展性** | 中 | 高 | 低 | 高 |
| **新增差异改公共循环** | 否(加Policy方法) | 否(加函数指针) | 是(加if constexpr) | 否(加Hook点) |
| **状态管理** | TaskLoopState独立 | TaskLoopCtx独立 | Kernel成员直用 | Kernel成员直用 |
| **类型安全** | 高(模板) | 低(void*) | 高(模板) | 高(模板特化) |
| **vec扩展字段冗余** | 是 | 是 | 否 | 否 |
| **Kernel→Ctx同步** | 需要 | 需要 | 不需要 | 不需要 |
| **方法签名适配** | Policy统一 | Ops统一 | if constexpr分支 | Hook签名统一 |
| **实现复杂度** | 中 | 低 | 低 | 中 |
| **FiaSide可扩展** | 否(Policy自由) | 否(Ops自由) | 否(bool只有两值) | 是(枚举可扩展) |
| **调试友好度** | 中 | 高(运行时查看Ops) | 低 | 中 |
| **适用场景** | 差异多且复杂 | 快速原型/调试 | 差异少且固定 | 差异持续增长 |

---

## 7. 推荐选择

### 7.1 按场景推荐

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **最快落地** | 方案C | 无额外抽象层，直接 if constexpr，改动最小 |
| **长期维护+差异增长** | 方案D | Hook链可扩展，新增差异不改公共循环 |
| **调试/单元测试需要** | 方案B | 运行时可查看/替换 Ops，最灵活 |
| **既有模板复杂度又要结构化** | 方案A | Policy 介于 C 和 D 之间 |

### 7.2 个人倾向排序

**方案D > 方案C > 方案A > 方案B**

理由：
- 方案D 兼顾零开销、结构清晰、可扩展性，最符合"控制流+函数指针"的初衷（Hook 即函数指针的编译期版本）
- 方案C 最简单，适合先落地再重构到 D
- 方案B 的 `void*` 和运行时开销在 NPU hot loop 中是硬伤
- 方案A 的 `TaskLoopState` 冗余和双向引用问题比 D 的模板特化更难管理

### 7.3 建议实施路径

```
Phase 1: 方案C 快速落地
  → 抽取 CommonTaskLoop<Kernel, bool> 公共函数
  → if constexpr 处理 DEAL_ZERO / UpdateAxisInfo 签名差异
  → 验证功能正确性

Phase 2: 重构为方案D
  → 将 if constexpr 分支提取为 Hook 点
  → 定义 TaskLoopHooks<FiaSide, Kernel> 特化
  → 公共循环改为 Hooks::xxx 调用
  → 新增差异只需加 Hook，不改骨架

Phase 3 (可选): 方案B 调试层
  → 在 D 的基础上，增加 TaskLoopOps 运行时视图
  → 用于调试时打印 Hook 配置、单步验证
```

---

## 8. 4+1视图模型

> PlantUML源文件(.puml)和PNG图片均位于 `uml_proposals/` 目录下（与本文件同级）。

### 8.1 视图总览

| 视图 | 方案A | 方案B | 方案C | 方案D | 共享 |
|------|-------|-------|-------|-------|------|
| **Logical(逻辑)** | [Structure](#方案a-逻辑视图) | [Structure](#方案b-逻辑视图) | [Structure](#方案c-逻辑视图) | [Structure](#方案d-逻辑视图) | — |
| **Process-AIC(进程·Cube)** | [AIC](#方案a-进程视图aic) | [AIC](#方案b-进程视图aic) | [AIC](#方案c-进程视图aic) | [AIC](#方案d-进程视图aic) | — |
| **Process-AIV(进程·Vec)** | [AIV](#方案a-进程视图aiv) | [AIV](#方案b-进程视图aiv) | [AIV](#方案c-进程视图aiv) | [AIV](#方案d-进程视图aiv) | — |
| **Development(开发)** | [DevView](#方案a-开发视图) | [DevView](#方案b-开发视图) | [DevView](#方案c-开发视图) | [DevView](#方案d-开发视图) | — |
| **Physical(物理)** | — | — | — | — | [PhysicalView](#物理视图所有方案共享) |
| **Scenarios(场景)** | — | — | — | — | [ScenariosView](#场景视图1) |
| **对比汇总** | — | — | — | — | [Comparison](#四方案对比汇总) |

---

### 8.2 Logical View（逻辑视图）

展示各方案的类结构、数据结构、差异注入方式和类间关系。

#### 方案A 逻辑视图

![方案A Logical View](uml_proposals/ProposalA_Structure.png)

**关键特征:** TaskLoopState独立于Kernel成员 → 需Kernel→State同步; Vec扩展字段在State中存在Cube侧冗余; Policy通过模板参数注入; HAS_DEAL_ZERO控制DEAL_ZERO分支编译期消除

#### 方案B 逻辑视图

![方案B Logical View](uml_proposals/ProposalB_Structure.png)

**关键特征:** TaskLoopCtx+TaskLoopOps(函数指针表); void* kernelPtr无类型校验; ~2-4次函数指针间接调用/迭代(NPU热循环性能风险); 运行时可查看/替换Ops(调试友好)

#### 方案C 逻辑视图

![方案C Logical View](uml_proposals/ProposalC_Structure.png)

**关键特征:** 无额外数据结构; 直接使用Kernel成员; 差异硬编码在if constexpr中(~4处); 新差异需改公共函数; 无状态同步开销; Vec字段不冗余

#### 方案D 逻辑视图

![方案D Logical View](uml_proposals/ProposalD_Structure.png)

**关键特征:** 零运行开销(static inline编译期展开); 无额外状态结构→无同步开销; TaskLoopHooks声明不定义→编译校验; FiaSide枚举可扩展; 签名适配在Hook层; Vec字段只在Vec Hook中访问→不冗余

---

### 8.3 Process View（进程视图）

> **关键:** AIC(Cube)和AIV(Vec)是**并行独立进程**，都从regbase同时启动，不是顺序执行！仅ExecuteTask阶段通过CrossCoreBuffer信号量同步（pipeline级数据依赖，非进程级顺序）。

每个方案拆分为AIC(Cube Core)和AIV(Vec Core)两张独立进程图，两者**同时从regbase启动**并行运行。

#### 方案A 进程视图·AIC

![方案A AIC Process](uml_proposals/ProposalA_AIC.png)

#### 方案A 进程视图·AIV

![方案A AIV Process](uml_proposals/ProposalA_AIV.png)

#### 方案B 进程视图·AIC

![方案B AIC Process](uml_proposals/ProposalB_AIC.png)

#### 方案B 进程视图·AIV

![方案B AIV Process](uml_proposals/ProposalB_AIV.png)

#### 方案C 进程视图·AIC

![方案C AIC Process](uml_proposals/ProposalC_AIC.png)

#### 方案C 进程视图·AIV

![方案C AIV Process](uml_proposals/ProposalC_AIV.png)

#### 方案D 进程视图·AIC

![方案D AIC Process](uml_proposals/ProposalD_AIC.png)

#### 方案D 进程视图·AIV

![方案D AIV Process](uml_proposals/ProposalD_AIV.png)

---

### 8.4 Development View（开发视图）

展示各方案的文件组织、模块依赖差异。

#### 方案A 开发视图

![方案A Dev View](uml_proposals/ProposalA_DevView.png)

**关键:** Policy需Kernel类型→前向声明或延迟include; TaskLoopState包含vec字段(cube冗余); 公共文件为 `flash_attn_task_loop_policy.h`

#### 方案B 开发视图

![方案B Dev View](uml_proposals/ProposalB_DevView.png)

**关键:** 无模板→纯函数指针; void* kernelPtr无类型校验; 需在Ctx中增加mBaseSize/s2BaseSize等常量; 公共文件为 `flash_attn_task_loop_common.h`

#### 方案C 开发视图

![方案C Dev View](uml_proposals/ProposalC_DevView.png)

**关键:** 最简洁; 无额外文件/结构体; 新差异需改公共函数; 公共文件为 `flash_attn_task_loop_common.h`

#### 方案D 开发视图

![方案D Dev View](uml_proposals/ProposalD_DevView.png)

**关键:** 声明不定义→编译校验; FiaSide可扩展; 新差异加Hook点不改骨架; 公共文件为 `flash_attn_task_loop_hooks.h`

---

### 8.5 Physical View（物理视图）

所有方案共享同一硬件部署架构。

#### 物理视图（所有方案共享）

![Physical View](uml_proposals/Shared_PhysicalView.png)

**关键执行模型:**
- AIC和AIV **并行独立启动**，都从regbase同时开始
- **不是先Cube后Vec的顺序执行！**
- CrossCoreBuffer仅用于ExecuteTask阶段的pipeline级同步:
  - AIC: Mm1/Mm2 → 写GM → 发信号 → 等AIV完成
  - AIV: 等AIC信号 → 读GM → Vec1/Vec2 → 发完成信号
- 多Core并行: 不同Core处理不同batch/head组合

---

### 8.6 Scenarios View（场景视图 +1）

关键用例，突出各方案的差异注入方式对比。

#### 场景视图（+1）

![Scenarios View](uml_proposals/Shared_ScenariosView.png)

#### UC1: Kernel并行启动（所有方案共享）

AIC和AIV**并行独立启动**，都从regbase同时开始，通过CrossCoreBuffer信号量同步（仅在ExecuteTask阶段），不是顺序执行。

#### UC2: Cube SKIP/S2_END处理（差异注入对比）

| 方案 | 调用方式 | UpdateAxisInfo参数 |
|------|---------|-------------------|
| A | Policy::UpdateAxisInfo(self, state, mode) | 3参数(b,g,s)，Policy层忽略mode |
| B | ops.updateAxis(ctx, kp, mode) → 函数指针间接 | 3参数(b,g,s) |
| C | self.UpdateAxisInfo(b,g,s) → if constexpr直接 | 3参数(b,g,s) |
| D | Hooks::UpdateAxisInfo(self, mode, b, g, s) | 3参数(b,g,s)，Hook适配忽略mode |

#### UC3: Vec DEAL_ZERO处理（差异注入对比）

| 方案 | 调用方式 | 运行开销 |
|------|---------|---------|
| A | VecPolicy::DealZeroActSeqLen(self, state.bN2Cur) → Policy静态inline | 0 |
| B | ops.dealZero(kernelPtr, ctx.bN2Cur) → 函数指针间接调用 | ~1次间接 |
| C | self.DealZeroActSeqLen(bN2Cur) → if constexpr直接 | 0 |
| D | Hooks::DealZeroActSeqLen(self, bN2Cur) → Hook静态inline | 0 |

#### UC4: 差异注入方式对比（核心区别）

![四方案对比汇总](uml_proposals/ProposalComparison.png)

| 维度 | 方案A(Policy) | 方案B(FnTable) | 方案C(constexpr) | 方案D(Hook) |
|------|:---------:|:---------:|:---------:|:---------:|
| **注入机制** | Policy static方法 | 函数指针表ops.xxx | if constexpr分支 | Hooks::xxx特化 |
| **调用方式** | Policy::xxx(self,state,...) | ops.xxx(ctx,kp,...) | self.xxx(...) | Hooks::xxx(self,...) |
| **类型安全** | 高(模板) | 低(void*) | 高(模板) | 高(模板特化) |
| **运行开销** | 0 | ~2-4次/迭代 | 0 | 0 |
| **额外结构体** | TaskLoopState | TaskLoopCtx+Ops | 无 | 无 |
| **Kernel→Ctx同步** | 需要 | 需要 | 不需要 | 不需要 |
| **Vec字段冗余** | 是(State中) | 是(Ctx中) | 否 | 否 |
| **扩展方式** | 加Policy方法 | 加函数指针 | 改公共函数 | 加Hook点 |
| **FiaSide可扩展** | 否 | 否 | 否(bool两值) | 是(枚举) |
| **调试友好** | 中 | 高(运行时看Ops) | 低 | 中 |

#### UC5: Phase演进路径

```
Phase 1: 方案C快速落地 → 验证功能
Phase 2: 重构为方案D → if constexpr提取为Hook点 → 定义Hooks特化
Phase 3: 可选方案B调试层 → TaskLoopOps运行时视图
```
