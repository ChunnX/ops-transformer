# quant_all_reduce 与 quant_reduce_scatter 算子优化开发测试文档

## 1. 目标概述

### 1.1 背景

优化改造 ops-transformer 仓中的 `quant_all_reduce` 与 `quant_reduce_scatter` 算子，参考 MTE skill 中的 `mte_all_gather_matmul` 参考工程的 AIV 部分实现。

- **quant_all_reduce**: 实现为 AllGather + ReduceSum
- **quant_reduce_scatter**: 实现为 AllToAll + ReduceSum

### 1.2 优化目标（三个阶段）

| 阶段 | 目标 | 验收标准 | 状态 |
|------|------|---------|------|
| **阶段 1** | 最大化 UB 空间使用 | 编译通过 + 精度通过 + git 提交 | ✅ 完成（commit `fe5a3e9f`） |
| **阶段 2** | 修改分核策略（每核负责单个 remote rank） | 编译通过 + 精度通过 + git 提交 | 🔲 待开始 |
| **阶段 3** | 采用 AIV 主核/辅核分工方式（SubBlock 机制） | 编译通过 + 精度通过 + git 提交 | 🔲 待开始 |

---

## 2. 当前算子分析

### 2.1 现有问题

#### 问题 1：UB 空间固定分配（未最大化使用）

**当前代码**（`quant_all_reduce_mte_one_shot.h` / `quant_reduce_scatter_mte.h`）：

```cpp
// 固定 1024 字节一块数据
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 穿刺取值

// UB 分配固定大小
tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 1024B
tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 32B
tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 4KB
```

**参考工程做法**（`all_gather_mte.h`）：

```cpp
constexpr static uint64_t ALLOC_UB_SPACE = 200UL * 1024UL; // UB 可用空间上限 200KB

// 动态计算 UB 分配
uint64_t usedSpace = scaleSize_ + CV_STATE_ALIGN * 2 + (rankSize_ * 2 + 1) * UB_ALIGN_BYTES;
uint64_t availableSpaceForXInQueue = (ALLOC_UB_SPACE - usedSpace) / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
uint64_t demandSpaceForXInQueue = CeilAlignU64(xSize_, X_BLOCK_BYTES);
xInQueueSize_ = availableSpaceForXInQueue < demandSpaceForXInQueue \
                ? availableSpaceForXInQueue \
                : demandSpaceForXInQueue;
```

#### 问题 2：遍历所有 rank 的分核策略（非固定分配）

**当前代码**（`quant_all_reduce_mte_one_shot.h` `ExecuteAllReduce()`）：

```cpp
// 每个核遍历所有 rank 的 Win Area
uint32_t startRankId = mteComm_.hcclContext_->rankId;
for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
    uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;
    // 从每个 remoteRank 读取数据...
}
```

**参考工程做法**（`all_gather_mte.h`）：

```cpp
// 每核独占一个 rank 通道
remoteRankId_ = aicId_ / sendCoreNumPerRank_;  // 固定分配
// sendCoreNumPerRank = aivNum / rankDim

// 只处理自己的 remoteRank
if (remoteRankId_ == mteComm_.curRankId_) {
    return;  // 跳过自己
}
```

#### 问题 3：未采用 AIV 主核/辅核分工方式

**当前代码**：
```cpp
// 纯 AIV 过程，所有核都参与工作
if ASCEND_IS_AIC {
    return;  // AIC 核直接退出
}
// 所有 AIV 核执行相同逻辑
Process();
```

**参考工程做法**（`all_gather_matmul_impl.h` AIV 侧）：
```cpp
// AIV SubBlock 分工：主核干活，辅核空转
// SubBlock 0: 主核（负责搬运、计算、状态同步）
// SubBlock 1: 辅核（仅通知 AIC，不参与实际搬运）
// 本算子不涉及 AIC，因此辅核无任何工作
```

---

## 3. 参考工程关键实现提取

### 3.1 UB 空间最大化使用

**参考文件**: `mte_all_gather_matmul/op_kernel/include/kernel/all_gather_mte.h`

**关键函数**: `InitBuffer()`

```cpp
template <AllGatherTemplateTypeClass>
__aicore__ inline void AllGatherMte<AllGatherTemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->Reset(); // UB 资源复位

    // 1. 计算除 xInQueue 外已分配的 UB 空间
    uint64_t usedSpace = scaleSize_ + CV_STATE_ALIGN * 2 + (rankSize_ * 2 + 1) * UB_ALIGN_BYTES;

    // 2. xInQueue 双 buffer，每个 buffer 按 X_BLOCK_BYTES 对齐
    uint64_t availableSpaceForXInQueue = (ALLOC_UB_SPACE - usedSpace) / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    uint64_t demandSpaceForXInQueue = CeilAlignU64(xSize_, X_BLOCK_BYTES);
    xInQueueSize_ = availableSpaceForXInQueue < demandSpaceForXInQueue \
                    ? availableSpaceForXInQueue \
                    : demandSpaceForXInQueue;

    // 3. 按计算结果分配
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xInQueueSize_);
}
```

### 3.2 每核负责单个 remote rank 的分核策略

**参考文件**: `mte_all_gather_matmul/op_kernel/include/kernel/all_gather_mte.h`

**关键参数计算**（`InitParams()`）：

```cpp
// 核分工参数
uint32_t sendCoreNumPerRank_{0};    // 每个 remote rank 分配的核数
uint32_t remoteRankId_{0};          // 当前核负责搬运的远端 rank ID

// 初始化时计算
sendCoreNumPerRank_ = tilingData->sendCoreNumPerRank;  // = aivNum / rankDim
aicId_ = mteComm_.aivId_ / GetTaskRation();  // AIV id 转换为 AIC id
remoteRankId_ = aicId_ / sendCoreNumPerRank_;  // 固定分配

// 核内索引计算
uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;
mBlockIdx_ = modCoreIndex % mDim_;
kBlockIdx_ = modCoreIndex / mDim_;
```

### 3.3 AIV 主核/辅核分工方式（SubBlock 机制）

**参考文件**: `mte_all_gather_matmul/op_kernel/include/kernel/all_gather_matmul_impl.h`

**核心思想**: 保持纯 AIV 模式，采用 SubBlock 分工机制

**AIV 侧 SubBlock 结构**:
```cpp
// SubBlock 0: 主核（GetSubBlockIdx() == 0）
InitBuffer();
CopyLocalData();        // GM → Win Area（本卡数据）
SetRemoteFlag();
WaitRemoteFlag();
ExecuteAllGather();     // 远端 Win → 本端 Win
// 如果涉及 AIC 计算，会调用 set_intra_block(AIC_NOTIFY) 通知 AIC

// SubBlock 1: 辅核（GetSubBlockIdx() == 1）
// 在参考模板中辅核调用 set_intra_block(AIC_NOTIFY + 16) 通知 AIC
// 本算子不涉及 AIC，因此辅核无任何工作，直接空转
```

**关键说明**:
- 参考模板中辅核的作用是通知 AIC 开始计算
- 本算子（quant_all_reduce / quant_reduce_scatter）不涉及 AIC Matmul 计算
- 因此辅核在本算子中无任何实际工作，仅空转
- 主核（SubBlock 0）负责全部搬运和计算工作

---

## 4. 具体优化方案

### 4.1 阶段 1：最大化 UB 空间使用

**修改文件**:
- `mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h`
- `mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h`
- `mc2/quant_reduce_scatter/op_kernel/mte_comm.h`

**修改点**:

#### 4.1.1 添加 UB 空间上限常量

```cpp
// 在 mte_comm.h 或算子头文件中添加
constexpr static uint64_t ALLOC_UB_SPACE = 200UL * 1024UL; // UB 可用空间上限 200KB
```

#### 4.1.2 动态计算 xInQueue 大小

```cpp
// Init() 函数中修改
uint64_t usedSpace = scaleSize_ + /* 其他已分配空间 */;
uint64_t availableSpaceForXInQueue = (ALLOC_UB_SPACE - usedSpace) / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
uint64_t demandSpaceForXInQueue = CeilAlignU64(xSize_, X_BLOCK_BYTES);
xInQueueSize_ = std::min(availableSpaceForXInQueue, demandSpaceForXInQueue);

tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xInQueueSize_);
```

#### 4.1.3 调整搬运循环参数

```cpp
// 基于新的 xInQueueSize_ 计算每轮搬运元素数
uint32_t xPerLoop = xInQueueSize_ / sizeof(XType);
uint32_t mLoopCnt = assignedBlockNums_ / xPerLoop;
uint32_t mTail = assignedBlockNums_ % xPerLoop;
```

### 4.2 阶段 2：每核负责单个 remote rank

**修改文件**: 同上

**修改点**:

#### 4.2.1 添加核分工参数

```cpp
// 在 MTECommunication 类或算子类中添加
uint32_t sendCoreNumPerRank_{0};    // 每个 remote rank 分配的核数
uint32_t remoteRankId_{0};          // 当前核负责的远端 rank ID
```

#### 4.2.2 InitParams() 中计算核分配

```cpp
sendCoreNumPerRank_ = aivNum_ / hcclContext_->rankDim;
remoteRankId_ = aivId_ / sendCoreNumPerRank_;

// 检查是否有工作（超出 rankDim 的核无数据）
if (remoteRankId_ >= hcclContext_->rankDim) {
    // 该核无工作，直接返回
    hasWork_ = false;
    return;
}
```

#### 4.2.3 ExecuteAllReduce() 改为单 rank 处理

```cpp
// 原代码：遍历所有 rank
for (uint32_t i = 0; i < hcclContext_->rankDim; ++i) {
    uint32_t remoteRankId = (startRankId + i) % hcclContext_->rankDim;
    ReadDataBlockReduceSum(...);
}

// 改为：只处理自己的 remoteRankId_
if (remoteRankId_ == hcclContext_->rankId) {
    // 本卡数据已在 Win 区，直接读取
} else {
    // 只从 remoteRankId_ 读取
    ReadDataBlockReduceSum(..., remoteRankId_);
}
```

### 4.3 阶段 3：采用 AIV 主核/辅核分工方式

**修改文件**:
- Kernel 入口文件（`quant_all_reduce.cpp` / `quant_reduce_scatter.cpp`）
- 算子头文件

**核心原则**: 保持纯 AIV 模式（KERNEL_TYPE_AIV_ONLY），采用 SubBlock 分工机制

**修改点**:

#### 4.3.1 保持纯 AIV Kernel TaskType

```cpp
// Kernel 入口保持纯 AIV 模式，不改为 MIX_AIC_1_2
extern "C" __global__ __aicore__ void quant_all_reduce_demo(
    GM_ADDR x, GM_ADDR scales, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 使用 KERNEL_TYPE_AIV_ONLY（纯 AIV）
    // 不需要 MIX_AIC_1_2，因为本算子不涉及 AIC 计算
}
```

#### 4.3.2 AIV 侧添加 SubBlock 分支逻辑

```cpp
// Process() 函数中
if ASCEND_IS_AIV {
    // SubBlock 0: 主核（GetSubBlockIdx() == 0）负责全部工作
    if (GetSubBlockIdx() == 0) {
        CopyDataToWin();        // 本卡数据 → Win Area
        WriteStatusToWin();     // 写状态同步
        ReadStatus();           // 等其他 rank 就绪
        ExecuteAllReduce();     // 或 ExecuteReduceScatter()
        // 不需要 set_intra_block(AIC_NOTIFY)，因为不涉及 AIC
    }
    // SubBlock 1: 辅核（GetSubBlockIdx() == 1）无工作，空转
    else {
        // 本算子不涉及 AIC，辅核无需通知 AIC
        // 辅核直接空转，不做任何操作
    }
}
```

#### 4.3.3 辅核空转说明

```cpp
// 在参考模板 mte_all_gather_matmul 中：
// - 辅核调用 set_intra_block(AIC_NOTIFY + 16) 通知 AIC 开始计算
// - AIC 侧 wait_intra_block 等待通知后执行 Matmul

// 在本算子（quant_all_reduce / quant_reduce_scatter）中：
// - 不涉及 AIC Matmul 计算
// - 辅核无需通知 AIC
// - 辅核无任何实际工作，仅空转
// - 主核（SubBlock 0）负责全部搬运和计算工作
```

---

## 5. 开发步骤

### 5.1 阶段 1 开发步骤（UB 优化）✅ 已完成

**实际实现（方案B）**：在动态 UB 分块基础上，scale 广播逻辑同步优化：

- **KG 量化**：`castScaleBuf_` 扩大到 `KG_MAX_SCALE_NUM(128) * 4B = 512B`，`BroadCast` 一次处理所有 `totalScaleNum = xNumPerBlock_/128` 个 scale，无循环
- **MX 量化**：`tempBf16ScaleBuf_` 扩大到 `MX_MAX_SCALE_NUM(256) * 2B = 512B`，`castScaleBuf_` 扩大到 `256 * 4B = 1024B`；`VF_CALL<CastVf>` 每次只能处理 32 个元素（硬件 VF 向量单元 32B/次限制，实测验证），保留分批循环；`Cast` + `BroadCast` 在循环外一次处理全量 scale
- **vecCompFixedSpace** 两个算子头文件同步更新，确保 UB 预算与实际分配一致

**精度验证**：level0 全 24 个用例 PASS（100%）

**git 记录**：`fe5a3e9f`（动态 UB 分块 + 方案B scale 广播优化，合并提交）

```
Step 1: 读取参考工程 all_gather_mte.h 的 InitBuffer() 实现          ✅
Step 2: 在 mte_comm.h 中添加 ALLOC_UB_SPACE 常量                     ✅（改为在算子头文件中动态计算）
Step 3: 修改 MTECommunication::InitBuffer() 动态计算 xQueue 大小     ✅
Step 4: 修改 quant_all_reduce_mte_one_shot.h 的 Init() 函数          ✅
Step 5: 修改 quant_reduce_scatter_mte.h 的 Init() 函数               ✅
Step 6: 调整搬运循环参数基于新 UB 大小                               ✅
Step 7: 编译这两算子的子包验证                                        ✅
Step 8: 精度验证 → mc2_test（level0 24/24 PASS）                     ✅
Step 9: Git 提交保存                                                  ✅
```

### 5.2 阶段 2 开发步骤（分核策略）

```
Step 1: 读取参考工程 all_gather_mte.h 的 InitParams() 核分工计算
Step 2: 在 mte_comm.h 中添加 sendCoreNumPerRank_ / remoteRankId_ 成员
Step 3: 修改 InitParams() 计算每核负责的 remote rank
Step 4: 修改 ExecuteAllReduce() 改为单 rank 处理
Step 5: 修改 ExecuteReduceScatter() 改为单 rank 处理
Step 6: 添加无数据核跳过分支（xOffset_ >= xNums_）
Step 7: 编译验证
Step 8: 精度验证
Step 9: Git 提交保存
```

### 5.3 阶段 3 开发步骤（AIV 主核/辅核分工）

```
Step 1: 读取参考工程 all_gather_matmul_impl.h 的 AIV SubBlock 结构
Step 2: 保持 Kernel TaskType 为 KERNEL_TYPE_AIV_ONLY（不改为 MIX）
Step 3: 在 Process() 中添加 SubBlock 分支逻辑
Step 4: SubBlock 0（主核）负责全部搬运和计算工作
Step 5: SubBlock 1（辅核）空转（本算子不涉及 AIC，无需通知）
Step 6: 编译验证
Step 7: 精度验证
Step 8: Git 提交保存
```

---

## 6. 测试流程

### 6.0 环境准备（重要）

```bash
# 1. 设置环境变量
source ~/env.sh

# 2. 检查 NPU 是否有其他进程占用（必须在跑 NPU 测试前执行）
npu-smi info

# 如果看到有其他进程占用 NPU，需要先 kill 掉
# 例如：看到进程 PID 12345 占用，则执行
sudo kill -9 12345

# 确认 NPU 空闲后再继续测试
npu-smi info  # 再次检查，确认无进程占用
```

**⚠️ 注意**：多卡环境下如果有其他进程占用 NPU，会导致通信冲突、测试失败或结果异常。务必在每次跑 NPU 测试前检查并清理。

### 6.1 编译测试

```bash
# 1. 设置环境
source ~/env.sh

# 2. 编译子包
bash build.sh --pkg --soc=ascend950 --ops="quant_all_reduce,quant_reduce_scatter"

# 3. 安装子包
bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=~/SUB

# 4. source 子包（覆盖原有环境）
source /home/m00947052/SUB/vendors/custom_transformer/bin/set_env.bash
```

### 6.2 精度测试（mc2_test）

```bash
# ⚠️ 跑 NPU 测试前必须检查 NPU 占用情况
npu-smi info
# 如有进程占用，先 sudo kill -9 <PID> 清理

# 1. 克隆测试仓库（如未克隆）
git clone https://gitcode.com/WillamShylock/mc2_test_0416.git
cd mc2_test_0416

# 2. 修改用例表中的 world_size 为 4

# 3. 跑 golden CPU 参考值（只用跑一遍）
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 5 --scene one_hccl_cpu --world_size 4

# 4. 再次检查 NPU（重要！）
npu-smi info

# 5. 跑 NPU 精度测试
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 5 --scene one_hccl_npu --world_size 4

# 6. 验证 level0 全 PASS, 两算子各有24个用例
# 可选择性校验 level1 用例
```

### 6.3 测试用例表修改

在 `david_excel/aclnnQuantAllReduce.xlsx` 中：
- 确认 `world_size` 列值修改为 4
- 开始时可先单跑一个用例验证
- 精度通过后再批跑验证

---

## 7. Git 提交规范

每个阶段完成后，保存 git 提交：

```bash
# 阶段 1 完成后
git add mc2/quant_all_reduce/op_kernel/ mc2/quant_reduce_scatter/op_kernel/
git commit -m "优化 UB 空间使用：动态计算 分块 大小，最大化利用 UB"

# 阶段 2 完成后
git add mc2/quant_all_reduce/op_kernel/ mc2/quant_reduce_scatter/op_kernel/
git commit -m "优化分核策略：每核负责单个 remote rank，避免多核竞争"

# 阶段 3 完成后
git add mc2/quant_all_reduce/op_kernel/ mc2/quant_reduce_scatter/op_kernel/
git commit -m "采用 AIV 主核/辅核分工：SubBlock 0 负责搬运，SubBlock 1 空转"
```

---

## 8. 关键文件路径

### 8.1 当前算子文件

| 算子 | 文件路径 |
|------|---------|
| quant_all_reduce | `/home/m00947052/ops-transformer/mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h` |
| quant_reduce_scatter | `/home/m00947052/ops-transformer/mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h` |
| 公共通信层 | `/home/m00947052/ops-transformer/mc2/quant_reduce_scatter/op_kernel/mte_comm.h` |
| 公共计算层 | `/home/m00947052/ops-transformer/mc2/quant_reduce_scatter/op_kernel/vec_comp.h` |

### 8.2 参考工程文件

| 内容 | 文件路径 |
|------|---------|
| AIV 核分工 | `mc2-mte-direct-invoke/references/mte_all_gather_matmul/op_kernel/include/kernel/all_gather_mte.h` |
| MIX_AIC_1_2 结构 | `mc2-mte-direct-invoke/references/mte_all_gather_matmul/op_kernel/include/kernel/all_gather_matmul_impl.h` |
| MTE 底层通信 | `mc2-mte-direct-invoke/references/mte_all_gather_matmul/op_kernel/include/kernel/all_gather_mte_comm.h` |
| 技术概览 | `mc2-mte-direct-invoke/references/mc2_mte_architecture.md` |
| 精细流水融合指南 | `mc2-mte-direct-invoke/references/mte_all_gather_matmul.md` |

---

## 9. 常见问题与解决方案

| 问题 | 解决方案 |
|------|---------|
| 编译报 UB 空间超限 | 减小 ALLOC_UB_SPACE 或检查其他 Buffer 分配 |
| 精度不通过 | 检查尾块处理逻辑、ReduceSum 累加顺序 |
| 状态同步死锁 | 检查状态区清零逻辑、WriteStatusToWin 写入顺序 |
| 多核竞争导致数据错乱 | 确认每核只访问自己负责的 remoteRank Win 区 |
| AIV 核数不足 | 检查 sendCoreNumPerRank 计算是否正确（aivNum / rankDim ≥ 1） |

---

## 10. 性能测试（暂不需要）

按照需求文件，性能测试暂不需要执行。若后续需要：

```bash
# ⚠️ 跑性能测试前必须检查 NPU 占用情况
npu-smi info
# 如有进程占用，先 sudo kill -9 <PID> 清理

# 不 source 子包，测试修改前性能
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 5 --scene one_hccl --world_size 4

# 再次检查 NPU
npu-smi info

# source 子包后，测试修改后性能
source /home/m00947052/SUB/vendors/custom_transformer/bin/set_env.bash
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 5 --scene one_hccl --world_size 4

# 性能数据在输出 CSV 文件的 mc2_prof 列
```
