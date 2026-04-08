# MTE AllToAll 设计串讲

## 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] **设计需要修改后开发**（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

**结论**：设计整体架构合理，基于 mc2_kernel 模板进行 AllGather → AllToAll 的改造思路正确。但存在 1 个阻塞级问题（伪代码引用未定义变量）和 3 个需讨论的问题（数据对齐、类型兼容、跨核同步），需在开发前解决。

---

## 质疑清单

---

### 问题 1：Execute 伪代码引用 `execLastAivId_`，但设计未定义此变量

- **类别**：伪代码可实现性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §2.4.7（Execute 伪代码第 373 行）、§2.4.5（Execute 核分配）

**问题描述**：

Execute 伪代码中有关键逻辑：

```cpp
if ((aivId_ == execLastAivId_) && (curBlock == execAssignedBlockNums_ - 1)) {
    copyBlockNum = execTailXNums_;
}
```

此处引用 `execLastAivId_`（Execute 阶段的最后一个活跃核 ID），用于判断当前核是否需要处理尾块。但 §2.4.5 的 Execute 阶段核分配代码中只定义了：

```cpp
execRound_ = execTotalBlocks / aivNum;
execTailBlockNums_ = execTotalBlocks % aivNum;
execAssignedBlockNums_ = ...;
execXOffset_ = ...;
execTailXNums_ = ...;
```

**缺少 `execLastAivId_` 的计算逻辑。**

对比 mc2_comm.h 中的 `ComputeTailAivId()` 方法（用于 CopyDataToWin 阶段）：

```cpp
void ComputeTailAivId(uint64_t totalAivCount) {
    if (round_ == 0) {
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        lastAivId_ = totalAivCount - 1;
    }
}
```

Execute 阶段需要相同的逻辑但基于 `execRound_` 和 `execTailBlockNums_` 计算。

**Developer 视角**：实现时无法确定尾块应由哪个核处理。如果错误判断 `execLastAivId_`，会导致：要么尾块无人处理（数据丢失），要么多个核同时处理尾块（写入冲突）。

**建议方案**：在 §2.4.5 Execute 核分配部分补充 `execLastAivId_` 的计算逻辑：

```cpp
// 新增：Execute 阶段尾块核 ID
if (execRound_ == 0) {
    execLastAivId_ = execTailBlockNums_ - 1;
} else {
    execLastAivId_ = aivNum - 1;
}
```

---

### 问题 2：DataCopy 32 字节对齐截断导致非对齐数据丢失

- **类别**：API 可行性 / 精度风险
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.3（分支场景覆盖）、§2.4.7（Execute 伪代码）

**问题描述**：

DataCopy 的官方文档明确约束：**count × sizeof(T) 需 32 字节对齐，若不对齐，搬运量将对 32 字节做向下取整。**

对 int8_t（sizeof=1），即 count 必须是 32 的倍数。但在设计的尾块处理中：

```cpp
execTailXNums_ = BlockAlignMod(sendCount, X_PRE_BLOCK_NUM);
// 当 sendCount = 100 时: execTailXNums_ = 100
// DataCopy(..., 100) → 实际只拷贝 96 字节（向下取整到 32 的倍数），丢失最后 4 字节
```

同理 CopyDataToWin 阶段（mc2_comm.h 内部）也有相同问题：当 `totalElements = rankDim × sendCount` 不是 32 的倍数时，尾块数据会被截断。

**影响的测试用例**：

| 用例 | sendCount | 总数据量 | 是否对齐 | 影响 |
|------|-----------|---------|---------|------|
| T1 | 32 | 128B | ✅ 32B 对齐 | 无影响 |
| T2 | 1024 | 4KB | ✅ 1024B 对齐 | 无影响 |
| T3 | 1M | 4MB | ✅ 对齐 | 无影响 |
| **T4** | **100** | **400B** | ❌ 100% 32 ≠ 0 | **丢失 4 字节/块** |
| **T5** | **102433** | **~400KB** | ❌ 102433% 32 ≠ 0 | **丢失 17 字节/块** |

**Developer 视角**：如果按当前设计实现，T4 和 T5 测试用例必然失败。mc2_comm.h 不可修改，其中 CopyDataToWin 阶段也使用 DataCopy，同样受此限制。

**建议方案**（三选一）：

1. **添加对齐约束**（推荐，改动最小）：在设计中明确 sendCount 必须是 32 的倍数（int8_t 场景），修改 T4/T5 测试用例为对齐值
2. **使用 DataCopyPad**：在 Execute 阶段用 DataCopyPad 替代 DataCopy 处理尾块（但 mc2_comm.h 的 CopyDataToWin 阶段仍需单独处理）
3. **输入填充**：Host 端将 sendCount 向上填充到 32 的倍数，输出截断回原始大小

---

### 问题 3：DataCopy int8_t vs b8 类型兼容性

- **类别**：API 可行性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.2.1（API 语义验证）

**问题描述**：

查阅官方 API 文档（`asc-devkit/docs/api/context/基础数据搬运.md`），Ascend950PR/DT 的 DataCopy GM↔UB 支持的数据类型列为：

```
GM -> VECIN (GM -> UB):  b8, b16, b32, b64
VECOUT -> GM (UB -> GM):  b8, b16, b32, b64
```

而其他平台（Atlas A2/A3）列出的是 `int8_t, uint8_t, ...`。设计文档声称 "b8 (即 int8_t)"，但文档中 `b8` 和 `int8_t` 并未被明确标注为等价。

mc2_kernel 模板代码使用 `int8_t` 作为 DataCopy 的模板参数：

```cpp
// mc2_kernel.cpp
Mc2KernelWithMc2Context<int8_t>(...);

// mc2_kernel.h
GlobalTensor<int8_t> remoteWinXTensor_;
DataCopy(xTmpTensor, remoteWinXTensor_[srcOffset], copyBlockNum);
```

**Developer 视角**：如果 Ascend950 上 `int8_t` 和 `b8` 不是同一类型，DataCopy 模板实例化时会编译失败。模板代码是官方提供的，理应通过验证，但设计中缺少对这一类型映射的明确说明，开发者在遇到编译问题时可能缺乏排查线索。

**建议方案**：
1. 在设计文档 §1.2.1 补充说明：Ascend950 上 DataCopy 的 `b8` 类型与 `int8_t` 的映射关系
2. 如果遇到编译错误，可能需要将所有 `int8_t` 替换为 `b8`（仅 DataCopy 相关的 Tensor 类型）

---

### 问题 4：Execute 阶段成员变量来源未明确

- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.4.7（Execute 伪代码）

**问题描述**：

Execute 伪代码使用了多个成员变量，但未说明其初始化来源：

| 变量 | 推测来源 | 实际来源 |
|------|---------|---------|
| `myRankId_` | `mteComm_.hcclContext_->rankId` | 未说明 |
| `rankDim_` | `mteComm_.hcclContext_->rankDim` | 未说明 |
| `sendCount_` | `tilingDataOut->mteAllToAllTilingInfo.sendCount` | 未说明 |
| `aivId_` | `mteComm_.aivId_` | 伪代码直接用 `aivId_` |
| `winBufferFlags_` | `mteComm_.winBufferFlags_` | 未说明 |

**Developer 视角**：这些变量涉及两种来源（hcclContext 运行时获取 vs tiling 数据编译时确定），实现时需要明确区分。特别是 `aivId_` 的伪代码写法（§2.4.5 中 `aivId < execTailBlockNums_`）与实际访问路径 `mteComm_.aivId_` 不同，可能导致开发者误以为是 kernel 类的成员变量而自行定义。

**建议方案**：在 §2.4.5 或 §2.4.7 之前添加一个变量映射表，明确每个变量的来源和初始化方式：

```cpp
// Init 中初始化
myRankId_ = mteComm_.hcclContext_->rankId;
rankDim_ = mteComm_.hcclContext_->rankDim;
sendCount_ = tilingDataOut->mteAllToAllTilingInfo.sendCount;
// aivId 使用 mteComm_.aivId_
```

---

### 问题 5：mc2_comm.h 的跨核同步机制——按 core index 同步的局限性

- **类别**：MC2 多卡同步正确性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.4.6（状态同步机制）

**问题描述**：

分析 mc2_comm.h 的 ReadStatus 实现，同步机制是**按 core index 对应**的：

```cpp
// WriteStatusToWin: core i 向所有远程 rank 的状态区第 i 行写入
uint32_t coreOffset = aivId_ * hcclContext_->rankDim;
uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_UB_ALIGN_NUM;

// ReadStatus: core i 读取本地状态区第 i 行，等待所有 rank 的 core i 就绪
uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM;
```

这意味着：**core 0 只等待所有 rank 的 core 0 完成 CopyDataToWin**，不等待同 rank 的 core 1、core 2 等。

在 AllToAll 场景中，core 0 的 Execute 需要从远程 Win 读取数据。远程 Win 的数据由该 rank 的所有 core 并行写入（CopyDataToWin 按 totalElements 分块）。如果 core 0 读完状态后立即开始 Execute，而远程 rank 的 core 3 尚未完成写入其负责的数据块，core 0 可能读到不完整数据。

**Developer 视角**：这是 mc2_comm.h 框架的通用问题，AllGather 模板也有相同风险。如果框架在硬件层面有全局可见性保证（例如 MTE 引擎的写操作在 WriteStatusToWin 的 DataCopy 之前全局可见），则此问题不存在。但设计中未提及此假设。

**建议方案**：
1. 在设计文档中明确记录此同步机制的假设：**假设 MTE 引擎的 DataCopy 写操作在后续 WriteStatusToWin 的 DataCopy 之前对远程 rank 全局可见**
2. 如果此假设不成立，需要在 CopyDataToWin 后、WriteStatusToWin 前添加全局屏障

---

### 问题 6：Sum API in-place 操作违反文档约束

- **类别**：API 可行性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.6（状态同步机制）

**问题描述**：

mc2_comm.h 的 ReadStatus 中使用 Sum API 的 in-place 模式：

```cpp
Sum(statusTensor, statusTensor, sumParams);
```

但 Sum 官方文档（`Sum.md` 约束说明第 98 行）明确写道：

> **不支持源操作数与目的操作数地址重叠。**

将同一个 `statusTensor` 同时作为 src 和 dst 传入，违反此约束。

**Developer 视角**：虽然 mc2_comm.h 模板代码如此使用且可能是已验证的行为，但在 Ascend950 平台上可能有不同表现。建议在首次运行时重点验证 ReadStatus 的同步正确性。

**建议方案**：保持与模板一致的用法（因为模板已验证），但在测试阶段将 ReadStatus 的同步正确性列为重点验证项。如果出现问题，考虑分配独立的 dst tensor。

---

### 问题 7：transportUnitSize 参数描述不准确

- **类别**：MC2 mc2Context 参数
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.3（mc2Context 初始化流程）

**问题描述**：

设计文档 §2.4.3 的参数说明表：

> transportUnitSize | 8 | 传输单元大小（字节），**INT8 8 字节**

"INT8 8 字节"的描述容易让人误解为 INT8 类型占 8 字节（实际为 1 字节）。8 更可能是 HCCL 传输协议的最小传输单元大小，与数据类型无直接关系。

**Developer 视角**：此参数的正确含义影响后续扩展（如支持 FP16 时是否需要改为其他值）。

**建议方案**：修改描述为 "传输单元大小（字节），HCCL 传输协议参数，与数据类型大小无直接对应关系"。

---

### 问题 8：UB Buffer 存在优化空间

- **类别**：内存规划合理性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.5（Buffer 规划）

**问题描述**：

Buffer 规划中 xQueue（mc2_comm）和 xInQueue（kernel）分别分配 2×1024B = 4KB，占总 UB 使用量（~4.3KB）的 93%。但两者在时间上不重叠：

- xQueue 仅在 CopyDataToWin 阶段使用
- xInQueue 仅在 Execute 阶段使用

理论上两者可共享同一块 UB 空间。但受限于"mc2_comm.h 不修改"的约束，xQueue 通过 `tPipe->InitBuffer(xQueue_, ...)` 独立分配，无法与 xInQueue 共享。

**Developer 视角**：当前 ~4.3KB 远小于 248KB 限制，不存在 UB 不足的风险。此优化仅在未来需要支持更大 block size 或更多 buffer 时才有必要。

**建议方案**：当前方案合理，无需修改。在设计文档中补充说明两者的时间互斥关系，为后续优化提供参考。

---

### 问题 9：Win 数据区大小未显式约束

- **类别**：Win 窗口资源规划
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.2（Win 窗口管理策略）

**问题描述**：

CopyDataToWin 将 `totalElements = rankDim × sendCount` 字节写入本地 Win 数据区。Win 数据区大小由 HCCL 通过 `HcclAllocComResourceByTiling` 自动分配，设计未讨论：

1. Win 数据区是否足够容纳 totalElements 字节
2. 当 sendCount 很大时（如 T3: 1M），Win 窗口是否会溢出
3. Win 窗口大小与 Mc2CcTilingConfig 参数的关系

**Developer 视角**：虽然 HCCL 会根据 Tiling 配置自动分配 Win 空间，但开发者需要了解 Win 大小的上限，避免因数据量超出 Win 容量导致运行时错误。

**建议方案**：在设计文档中补充说明 Win 数据区大小由 HCCL 自动管理，并列出大数据量测试时需关注 Win 分配是否成功的检查点。

---

### 问题 10：伪代码中 `aivId` 应使用完整路径

- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.5（Execute 核分配伪代码）

**问题描述**：

§2.4.5 的 Execute 核分配代码中使用 `aivId`：

```cpp
execAssignedBlockNums_ = (aivId < execTailBlockNums_) ? execRound_ + 1 : execRound_;
uint64_t execBlockIdx = aivId * execRound_ + ...;
```

但 `aivId` 并非 kernel 类的成员变量，而是 `mteComm_.aivId_`（在 `mteComm_.InitParams()` 中初始化）。使用简写可能导致开发者误解。

**建议方案**：统一使用 `mteComm_.aivId_` 以保持一致性，或在伪代码前注明 `uint32_t aivId = mteComm_.aivId_;`。

---

## 审查统计

| 严重程度 | 数量 | 问题编号 |
|---------|------|---------|
| 🔴 阻塞 | 1 | #1 |
| 🟡 需讨论 | 4 | #2, #3, #4, #5 |
| 🟢 建议 | 5 | #6, #7, #8, #9, #10 |

## 审查维度覆盖

| 审查维度 | 发现问题 | 状态 |
|---------|---------|------|
| 1. API 可行性 | #2 (DataCopy 对齐), #3 (b8 vs int8_t), #6 (Sum in-place) | 需讨论/建议 |
| 2. 内存规划合理性 | #8 (Buffer 优化), #9 (Win 大小) | 建议 |
| 3. 多核策略可行性 | #5 (跨核同步) | 需讨论 |
| 4. 伪代码可实现性 | #1 (execLastAivId_), #4 (变量来源), #10 (aivId 路径) | 阻塞/建议 |
| 5. 精度风险 | #2 (非对齐截断) | 需讨论 |
| 6. 通信拓扑合理性 | 无问题（复用模板配置） | ✅ |
| 7. Win 窗口资源规划 | #9 (Win 大小约束) | 建议 |
| 8. 多卡同步正确性 | #5 (跨核可见性) | 需讨论 |
| 9. mc2Context 参数 | #7 (transportUnitSize 描述) | 建议 |
| 10. 多核一致性 | #1 (execLastAivId_ 影响尾块分配) | 阻塞 |

## 优先修复建议

**开发前必须解决**：
1. 问题 #1：补充 `execLastAivId_` 计算逻辑 → 阻塞开发
2. 问题 #2：确定 sendCount 对齐约束或选择非对齐处理策略 → 影响 2/5 测试用例
3. 问题 #4：补充成员变量来源映射表 → 影响实现正确性

**开发中关注**：
4. 问题 #3：编译时关注 int8_t/b8 类型兼容性
5. 问题 #5：首次运行时验证 ReadStatus 同步正确性

---

## Architect 回应

---

### 问题 1：Execute 伪代码引用 `execLastAivId_`，但设计未定义此变量

- **回应**：已修改
- **理由**：Developer 质疑完全正确。`execLastAivId_` 是 Execute 阶段尾块处理的关键变量，遗漏其计算逻辑会导致尾块无人处理或写入冲突。参考 mc2_comm.h 的 `ComputeTailAivId()` 方法，为 Execute 阶段补充了相同逻辑的计算。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_comm.h` 第 175-182 行 `ComputeTailAivId()` 方法
- **DESIGN.md 变更**：§2.4.5 Execute 阶段核分配部分，新增 `execLastAivId_` 计算逻辑：
  ```cpp
  if (execRound_ == 0) {
      execLastAivId_ = execTailBlockNums_ - 1;
  } else {
      execLastAivId_ = aivNum - 1;
  }
  ```

---

### 问题 2：DataCopy 32 字节对齐截断导致非对齐数据丢失

- **回应**：已修改
- **理由**：Developer 分析准确。DataCopy 文档明确约束 `count × sizeof(T)` 需 32 字节对齐，否则向下取整。对于 int8_t（sizeof=1），sendCount 必须是 32 的倍数。采用建议方案 1（添加对齐约束），原因：
  1. 改动最小，仅约束输入参数
  2. rankDim = 4，sendCount 为 32 的倍数时 totalElements 也满足 32 字节对齐，CopyDataToWin 和 Execute 两阶段均安全
  3. mc2_comm.h 不可修改，方案 2/3 均需修改模板或 Host 端预处理（违反 C9 约束）
- **文档依据**：`asc-devkit/docs/api/context/基础数据搬运.md` 参数说明表第 91 行："count * sizeof(T)需要32字节对齐，若不对齐，搬运量将对32字节做向下取整。"
- **DESIGN.md 变更**：§2.3 分支场景覆盖新增"对齐约束"段落，明确 sendCount 必须是 32 的倍数

---

### 问题 3：DataCopy int8_t vs b8 类型兼容性

- **回应**：部分修改
- **理由**：Developer 的分析有道理。Ascend950PR/DT 的 GM↔UB DataCopy 文档列出支持类型为 `b8, b16, b32, b64`，而非 `int8_t`。但 mc2_kernel 官方模板直接使用 `int8_t`（`GlobalTensor<int8_t>`），说明编译器在模板实例化时将 `int8_t` 隐式映射为 `b8`。这是编译器的内部行为，设计中应记录此映射关系，为开发者排查编译问题提供线索。
- **文档依据**：`asc-devkit/docs/api/context/基础数据搬运.md` 表4/表6 列出 Ascend950PR/DT 支持 `b8, b16, b32, b64`；`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/mc2_kernel.cpp` 第 52 行使用 `int8_t` 模板参数
- **DESIGN.md 变更**：§1.2.1 验证说明新增"b8 与 int8_t 的映射关系"段落，说明编译器隐式映射机制及编译错误的 fallback 方案

---

### 问题 4：Execute 阶段成员变量来源未明确

- **回应**：已修改
- **理由**：Developer 质疑合理。伪代码中使用 `aivId_`、`winBufferFlags_`、`myRankId_`、`rankDim_` 等变量但未标注来源，可能导致开发者误解为 kernel 类成员变量而自行定义，与 `mteComm_` 中的实际来源冲突。补充变量映射表后，开发者可清晰区分 hcclContext 运行时获取 vs tiling 数据编译时确定 vs mc2_comm 框架管理。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_kernel.h` 第 121-148 行 Execute() 中使用 `mteComm_.aivId_`、`mteComm_.hcclContext_->rankDim` 等完整路径
- **DESIGN.md 变更**：§2.4.7 新增"变量来源映射"表格，列出所有伪代码变量的实际来源和初始化位置；伪代码中 `aivId_`/`winBufferFlags_`/`myRankId_`/`rankDim_` 统一改为 `mteComm_.aivId_`/`mteComm_.winBufferFlags_`/`mteComm_.hcclContext_->rankId`/`mteComm_.hcclContext_->rankDim` 完整路径

---

### 问题 5：mc2_comm.h 的跨核同步机制——按 core index 同步的局限性

- **回应**：部分修改
- **理由**：Developer 的分析逻辑正确——按 core index 同步意味着 core 0 只等待所有 rank 的 core 0 就绪，不等待同 rank 的其他 core。但这是 mc2_comm.h 框架的通用设计，AllGather 模板也使用相同机制。此设计依赖一个硬件假设：MTE 引擎的 DataCopy 写操作在 WriteStatusToWin 的 DataCopy 之前对远程 rank 全局可见。依据：
  1. CopyDataToWin 末尾有 `PipeBarrier<PIPE_ALL>()` 确保本核 MTE3 写入完成
  2. WriteStatusToWin 在 CopyDataToWin 之后执行，写远程状态区的 DataCopy 隐含硬件级全局可见性保证
  3. AllGather 模板使用相同机制并已验证通过

  不修改同步机制（框架不可修改），但在设计中明确记录此假设，并将 Execute 读取远程 Win 数据完整性列为首次运行验证要点。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_comm.h` 第 207 行 `PipeBarrier<PIPE_ALL>()`；第 210-228 行 WriteStatusToWin 在 CopyDataToWin 之后执行
- **DESIGN.md 变更**：§2.4.6 新增"跨核同步假设"段落，说明同步机制的硬件依赖假设，并列出验证要点

---

### 问题 6：Sum API in-place 操作违反文档约束

- **回应**：保留原设计
- **理由**：Sum 官方文档确实标注"不支持源操作数与目的操作数地址重叠"（`Sum.md` 第 98 行）。mc2_comm.h 的 ReadStatus 使用 `Sum(statusTensor, statusTensor, sumParams)` 违反此约束。但：
  1. mc2_comm.h 是不可修改的框架代码
  2. 此用法是官方模板已验证的行为
  3. 修改 mc2_comm.h 超出本算子设计范围

  已在设计文档中将 ReadStatus 同步正确性列为验证要点。如首次运行出现同步异常，可考虑 fork mc2_comm.h 并分配独立的 dst tensor。
- **文档依据**：`asc-devkit/docs/api/context/Sum.md` 第 98 行约束说明；`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_comm.h` 第 248 行 `Sum(statusTensor, statusTensor, sumParams)`
- **DESIGN.md 变更**：§2.4.6 新增"Sum in-place 说明"段落

---

### 问题 7：transportUnitSize 参数描述不准确

- **回应**：已修改
- **理由**：Developer 分析正确。"INT8 8 字节"容易误解为 INT8 类型占 8 字节（实际为 1 字节）。8 是 HCCL 传输协议的最小传输单元大小参数，与数据类型大小无直接对应关系。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/mc2_kernel_host.h` 中 Mc2CcTilingConfig 的 transportUnitSize 参数
- **DESIGN.md 变更**：§2.4.3 参数说明表，transportUnitSize 描述从"传输单元大小（字节），INT8 8 字节"改为"传输单元大小（字节），HCCL 传输协议参数，与数据类型大小无直接对应关系"

---

### 问题 8：UB Buffer 存在优化空间

- **回应**：保留原设计
- **理由**：Developer 已正确分析：xQueue 和 xInQueue 在时间上互斥（CopyDataToWin vs Execute），理论上可共享 UB 空间。但受 mc2_comm.h 不可修改约束，xQueue 通过 `tPipe->InitBuffer(xQueue_, ...)` 独立分配，无法与 xInQueue 共享。当前 ~4.3KB 远小于 248KB 限制，无优化必要。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_comm.h` 第 154 行 `tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_BLOCK_BYTES)`
- **DESIGN.md 变更**：§1.5 新增"Buffer 时间互斥说明"段落，记录两者的互斥关系和优化空间

---

### 问题 9：Win 数据区大小未显式约束

- **回应**：已修改
- **理由**：Developer 提出了合理的关注点。虽然 Win 大小由 HCCL 通过 `HcclAllocComResourceByTiling` 自动管理，但开发者需了解 Win 分配的成功与否取决于数据量。大数据量测试用例（如 sendCount > 100K）时需关注 Win 分配是否成功。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/utils.h` 第 22-35 行 `HcclA5OpResParam` 结构体；DESIGN.md §2.4.3 `HcclAllocComResourceByTiling` 调用
- **DESIGN.md 变更**：§2.4.2 新增"Win 数据区大小说明"段落，说明 HCCL 自动管理机制和大数据量测试的检查要点

---

### 问题 10：伪代码中 `aivId` 应使用完整路径

- **回应**：已修改
- **理由**：Developer 质疑正确。§2.4.5 Execute 核分配代码使用 `aivId` 简写，而实际来源是 `mteComm_.aivId_`。与 §2.4.7 的伪代码存在一致性问题。已在 §2.4.5 添加 `uint32_t aivId = mteComm_.aivId_;` 局部变量声明，并在 §2.4.7 伪代码中统一使用 `mteComm_.aivId_` 完整路径。
- **文档依据**：`.opencode/skills/ascendc-direct-invoke-template/references/mc2_kernel/kernel/mc2_kernel.h` 第 125 行 `mteComm_.aivId_`
- **DESIGN.md 变更**：§2.4.5 添加 `uint32_t aivId = mteComm_.aivId_;` 声明；§2.4.7 伪代码统一使用 `mteComm_.aivId_`、`mteComm_.hcclContext_->rankDim` 等完整路径

---

### 回应统计

| 回应类别 | 数量 | 问题编号 |
|---------|------|---------|
| 接受 | 6 | #1, #2, #4, #7, #9, #10 |
| 保留原设计 | 2 | #6, #8 |
| 部分修改 | 2 | #3, #5 |

**DESIGN.md 已更新**，涉及 §1.2.1、§1.5、§2.3、§2.4.2、§2.4.3、§2.4.5、§2.4.6、§2.4.7 共 8 个章节。
