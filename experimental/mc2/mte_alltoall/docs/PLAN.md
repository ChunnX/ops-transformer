# MTE AllToAll 开发计划

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | mte_alltoall |
| 需求类型 | 特定用例（4 卡，INT8） |
| 通信语义 | AllToAll — 每张卡向所有卡发送不同数据块，并从所有卡接收属于自己的数据块 |
| 数据类型 | int8_t |
| 卡数 | 4 |
| 目标芯片 | Ascend950PR/Ascend950DT |
| 工程模板 | mc2_kernel（MTE Win 窗口通信） |

---

## 2. 文件清单

基于 mc2_kernel 模板，所有文件放置在 `ops/mte_alltoall/` 目录下：

```
ops/mte_alltoall/
├── kernel/
│   ├── mc2_comm.h            # MTE 通信框架（复用模板，不修改）
│   ├── mte_alltoall_kernel.h # 算子 Kernel 类（替换 Execute 逻辑）
│   ├── mte_alltoall_tiling_data.h # Tiling 数据定义（替换业务参数）
│   └── utils.h               # 通用工具函数（复用模板，不修改）
├── mte_alltoall_kernel.cpp   # Kernel 入口（<<<>>> 调用）
├── mte_alltoall_kernel_host.h # Host 调用接口声明
├── test_mte_alltoall.cpp     # 多卡多线程测试（替换验证逻辑）
├── CMakeLists.txt            # 构建配置
├── run.sh                    # 编译运行脚本
└── docs/
    ├── environment.json       # 环境信息
    ├── DESIGN.md              # 技术设计文档
    └── PLAN.md                # 本文件
```

**需要修改的文件**（相对于模板）：

| 文件 | 修改内容 |
|------|---------|
| `kernel/mte_alltoall_tiling_data.h` | 业务 Tiling 参数：bs/hiddenSize → sendCount/rankDim |
| `kernel/mte_alltoall_kernel.h` | Execute 逻辑：AllGather → AllToAll；Init 双阶段核分配 |
| `mte_alltoall_kernel.cpp` | 类名/函数名 → MteAllToAll；数据类型 int8_t |
| `mte_alltoall_kernel_host.h` | 函数声明 → MteAllToAllDemo |
| `test_mte_alltoall.cpp` | 输入数据生成、输出验证、AllToAll 语义 |
| `CMakeLists.txt` | 目标名、源文件名 |

**不需要修改的文件**：

| 文件 | 原因 |
|------|------|
| `kernel/mc2_comm.h` | MTE 通信框架通用，AllToAll 复用 CopyDataToWin/WriteStatusToWin/ReadStatus |
| `kernel/utils.h` | 通用工具函数（CeilDiv, BlockAlignMod, SyncFunc） |

---

## 3. 测试计划

### 3.1 测试环境要求

| 项目 | 要求 |
|------|------|
| NPU 设备 | 至少 4 张 Ascend950 卡 |
| HCCL | 已安装并初始化 |
| CANN | 版本匹配（与 asc-devkit 对应） |
| 编译 | 本地 macOS 设计，远程 NPU 服务器编译运行 |

### 3.2 测试用例

| # | 用例名称 | sendCount | rankDim | 总数据量(每卡) | 状态 | 结果 |
|---|---------|-----------|---------|---------------|------|------|
| T1 | 小数据量对齐 | 32 | 4 | 128B | 待测 | — |
| T2 | 中等数据量对齐 | 1024 | 4 | 4KB | 待测 | — |
| T3 | 大数据量对齐 | 1024×1024 (1M) | 4 | 4MB | 待测 | — |
| T4 | 非对齐数据量 | 100 | 4 | 400B | 待测 | — |
| T5 | 中等非对齐 | 1024×100+33 | 4 | ~400KB | 待测 | — |

### 3.3 验证方法

**Golden 计算逻辑**（Host 端验证）：

```
对于卡 myRankId:
  对每个 remoteRank ∈ [0, rankDim):
    对每个 k ∈ [0, sendCount):
      expected[remoteRank * sendCount + k] = input_of_rank(remoteRank)[myRankId * sendCount + k]
```

具体实现：
1. 每个 rank 线程生成各自的输入数据（rank 差异化，如 `input[i] = (rankId+1)*10 + i%100`）
2. 每个线程在 Host 端保存所有 rank 的输入数据（通过共享内存或文件）
3. Kernel 执行后，读取输出，按 AllToAll 语义计算期望值，逐元素比对

**精度标准**：
- INT8 类型：逐元素精确比对（atol=0, rtol=0）
- 允许的错误数：0

### 3.4 测试代码关键修改点

```cpp
// 1. 输入数据生成（每个 rank 差异化）
for (int64_t i = 0; i < totalElements; ++i) {
    xHostData[i] = static_cast<int8_t>((args.rankId + 1) * 10 + (i % 100));
}

// 2. 输出 shape（AllToAll: 与输入相同大小）
std::vector<int64_t> outputShape = {rankDim, sendCount};  // [4, sendCount]
// output 总元素数 = rankDim * sendCount = 与输入相同

// 3. 验证逻辑
for (uint32_t remoteRank = 0; remoteRank < rankDim; ++remoteRank) {
    // 本卡应该收到 remoteRank 的输入中第 myRankId 块
    for (int64_t k = 0; k < sendCount; ++k) {
        int8_t expected = static_cast<int8_t>((remoteRank + 1) * 10 + 
            ((myRankId * sendCount + k) % 100));
        int8_t actual = output_host[remoteRank * sendCount + k];
        assert(expected == actual);
    }
}
```

---

## 4. 开发阶段

### 阶段 1：项目初始化与文件创建

- [x] 从 mc2_kernel 模板创建文件到 `ops/mte_alltoall/`
- [x] 重命名文件（mc2_* → mte_alltoall_*）
- [x] 修改 CMakeLists.txt 目标名和源文件名

### 阶段 2：Tiling 数据定义

- [x] 修改 `mte_alltoall_tiling_data.h`：替换 Mc2KernelTilingInfo → MteAllToAllTilingInfo
- [x] 定义业务参数：sendCount, rankDim, aivNum

### 阶段 3：Kernel 实现

- [x] 修改 `mte_alltoall_kernel.h`：
  - [x] 替换命名空间和类名（MC2KernelImpl → MteAllToAllImpl，MC2Kernel → MteAllToAll）
  - [x] Init: 双阶段核分配计算（CopyDataToWin 按 totalElements + Execute 按 sendCount）
  - [x] Execute: AllToAll 偏移计算（srcOffset = myRankId * sendCount + execOffset）
  - [x] Process: 复用标准流程（CopyDataToWin → WriteStatusToWin → ReadStatus → Execute）
- [x] 修改 `mte_alltoall_kernel.cpp`：替换函数名（Mc2KernelGeneric → MteAllToAllGeneric）
- [x] 修改 `mte_alltoall_kernel_host.h`：替换函数声明（Mc2KernelDemo → MteAllToAllDemo）

### 阶段 4：Host 测试代码

- [x] 修改 `test_mte_alltoall.cpp`：
  - [x] CreatTilingDataAndContext: 替换 bs/hiddenSize → sendCount/rankDim
  - [x] AllocDeviceMemory: AllToAll 输入 shape 为 [rankDim * sendCount]
  - [x] LaunchOneThread: AllToAll 输出 shape 与输入相同，验证逻辑按 AllToAll 语义
  - [x] 拓扑字符串保持 AlltoAll=level0:fullmesh;level1:pairwise
- [x] 默认测试参数：sendCount=1024, rankDim=4, BLOCK_NUM=64

### 阶段 5：编译验证

- [ ] 在远程 NPU 服务器上编译
- [ ] 修复编译错误（如有）
- [ ] 确认无编译警告

### 阶段 6：功能测试

- [ ] 执行 T1（小数据量对齐）
- [ ] 执行 T2（中等数据量对齐）
- [ ] 执行 T3（大数据量对齐）
- [ ] 执行 T4（非对齐数据量）
- [ ] 执行 T5（中等非对齐）
- [ ] 所有测试通过

### 阶段 7：验收

- [ ] 精度验证：所有用例 0 错误
- [ ] 代码审查：命名规范、注释完整
- [ ] 文档归档：DESIGN.md + PLAN.md 完整

---

## 5. 已知问题和决策记录

| # | 问题 | 决策 | 日期 |
|---|------|------|------|
| 1 | NPU 在本地 macOS 不可用 | 在远程 NPU 服务器编译测试，本地仅设计 | 2026-04-08 |
| 2 | CopyDataToWin 和 Execute 核分配不同 | 双阶段核分配：CopyDataToWin 按 totalElements，Execute 按 sendCount | 2026-04-08 |
| 3 | mc2_comm.h 不修改 | 复用模板的 CopyDataToWin/WriteStatusToWin/ReadStatus，Execute 在 kernel 类中重新实现 | 2026-04-08 |
| 4 | INT8 无精度问题 | INT8 逐元素精确比对，无需浮点精度容差 | 2026-04-08 |
| 5 | Win 窗口数据布局 | AllToAll 将完整输入写入 Win（与 AllGather 相同），Execute 阶段按 myRankId 偏移读取 | 2026-04-08 |

---

## 6. 开发记录

### 6.1 实现完成情况（2026-04-08）

**代码文件清单**：

| 文件 | 状态 | 说明 |
|------|------|------|
| `kernel/mc2_comm.h` | ✅ 复用 | MTE 通信框架，未修改 |
| `kernel/utils.h` | ✅ 复用 | 工具函数（CeilDiv, BlockAlignMod, SyncFunc），未修改 |
| `kernel/mte_alltoall_tiling_data.h` | ✅ 新建 | sendCount/rankDim/aivNum 三参数 Tiling |
| `kernel/mte_alltoall_kernel.h` | ✅ 新建 | 双阶段核分配 + AllToAll Execute 偏移逻辑 |
| `mte_alltoall_kernel.cpp` | ✅ 新建 | Kernel 入口，<<<>>> 调用 |
| `mte_alltoall_kernel_host.h` | ✅ 新建 | Host 调用声明 |
| `test_mte_alltoall.cpp` | ✅ 新建 | 多卡多线程测试，AllToAll 语义验证 |
| `CMakeLists.txt` | ✅ 新建 | MC2 构建配置 |
| `run.sh` | ✅ 新建 | 编译运行脚本 |

**关键实现点**：

1. **双阶段核分配**（§2.4.5）：
   - CopyDataToWin 阶段：按 totalElements（= rankDim × sendCount）切分
   - Execute 阶段：按 sendCount 独立切分，有独立的 execRound/execTailBlockNums/execLastAivId

2. **AllToAll Execute 偏移**（§2.4.7）：
   - srcOffset = myRankId × sendCount + execOffset（从远程 Win 读取属于本卡的块）
   - dstOffset = remoteRankId × sendCount + execOffset（写入输出对应位置）

3. **测试验证逻辑**：
   - 输入数据 rank 差异化：x[i] = (rankId+1)*10 + (i%100)
   - 验证：output[remoteRank*sendCount+k] == input_of_rank(remoteRank)[myRankId*sendCount+k]
   - INT8 精确比对，atol=0, rtol=0

4. **对齐约束**：
   - sendCount 必须是 32 的倍数（DataCopy 的 count*sizeof(int8_t) 需 32B 对齐）
   - 默认测试参数 sendCount=1024 满足对齐要求

**待完成**：
- [ ] 在远程 NPU 服务器编译（本地无 CANN 环境）
- [ ] 功能测试 T1-T5
- [ ] 性能采集（NPU 可用时）

## 7. 测试结果详情

_（待在远程 NPU 服务器上执行测试后更新）_

### Round 1

| 用例 | 状态 | 错误数/总元素 | 备注 |
|------|------|-------------|------|
| T1 | 待测 | — | sendCount=32, rankDim=4 |
| T2 | 待测 | — | sendCount=1024, rankDim=4（默认参数） |
| T3 | 待测 | — | sendCount=1024×1024, rankDim=4 |
| T4 | 待测 | — | sendCount=100（非32对齐，预期DataCopy截断） |
| T5 | 待测 | — | sendCount=102400+33 |
